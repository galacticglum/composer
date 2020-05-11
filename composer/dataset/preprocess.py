'''
Preprocesses a raw dataset so that it can be used by the models.

'''

import hashlib
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from composer.utils import parallel_process
from composer.dataset.sequence import NoteSequence
from composer.exceptions import InvalidParameterError

# The extension of preprocessed files.
_OUTPUT_EXTENSION = 'data'

def get_processed_files(dataset_path):
    '''
    Gets all the preprocessed in the specified dataset directory.

    :param dataset_path:
        The path to the directory containing the preprocessed files.
    :returns:
        A list of Path-like objects representing the paths to the preprocessed files.

    '''

    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise InvalidParameterError('\'{}\' is an invalid dataset path!'.format(dataset_path))
    
    return list(dataset_path.glob('**/*.{}'.format(_OUTPUT_EXTENSION)))

def convert_file(filepath, output_path, transform=False, time_stretch_range=(0.90, 1.10), pitch_shift_range=(-4, 4),
                 time_step_increment=10, max_time_steps=100, velocity_bins=32,
                 sustain_period_encode_mode=NoteSequence.SustainPeriodEncodeMode.EXTEND, trim_start=False):
    '''
    Converts a music file to a set of sequences.

    :param filepath:
        The path to the file to convert.
    :param output_path:
        The directory to output the file to.
    :param transform:
        Indicates whether sample transformation should be applied. If ``True``, additional samples are produced,
        one where the original sample is time stretched, pitched shifted, and a combination of both respectively.
        Defaults to ``False``.
    :param time_stretch_range:
        The range of the time stretch value. Defaults to up to 10% faster or slower (0.90 to 1.10).
    :param pitch_shift_range:
        The range of the pitch shift offset. Defaults to raising or lowering a sample by up to 4 pitch values (-4 to 4).
    :param time_step_increment:
        The number of milliseconds that a single step in time represents.
        Defaults to 10 milliseconds (i.e. one step in time is 10 milliseconds).
    :param max_time_steps:
        The maximum number of time steps that a single event can shift time by.
        Defaults to 100 (i.e. time shift can vary from 1 to 100 time steps).
        If this is ``None``, there is no limit.
    :param velocity_bins:
        The number of bins to quantize the note velocity values into. Defaults to 32.
    :param sustain_period_encode_mode:
        The way in which sustain periods should be encoded.
        Defaults to :attr:`composer.dataset.sequence.NoteSequence.SustainPeriodEncodeMode.EXTEND`.
    :param trim_start:
        Indicates whether silence from the start of the MIDI files should be trimmed away.
        Defaults to ``False``.

    '''

    filename = Path(filepath).stem
    file_id = hashlib.md5(str(filepath).encode()).hexdigest()
    file_save_path = output_path / '{}_{}.{}'.format(filename, file_id, _OUTPUT_EXTENSION)

    # Load the MIDI file into a NoteSequence and convert it into an EventSequence.
    note_sequence = NoteSequence.from_midi(filepath)
    if trim_start:
        note_sequence.trim_start()

    event_sequence = note_sequence.to_event_sequence()
    # Encode the event sequence and write to file.
    event_sequence.to_integer_encoding().to_file(file_save_path)

    if transform:
        # Convenience functions for calculating pitch shift and time stretch 
        _get_time_stretch = lambda: np.random.uniform(*time_stretch_range)

        transformed_note_sequences = []
        for pitch_shift in range(pitch_shift_range[0], pitch_shift_range[1] + 1):
            if pitch_shift == 0: continue
            transformed_note_sequences.append(note_sequence.pitch_shift(pitch_shift, inplace=False))

        transformed_note_sequences.append(note_sequence.time_stretch(_get_time_stretch(), inplace=False))
        
        # Output sequences
        for index, transformed_sequence in enumerate(transformed_note_sequences):
            destination_path = file_save_path.parent / (file_save_path.stem + '-' + str(index).zfill(2) + file_save_path.suffix)
            transformed_sequence.to_event_sequence(
                sustain_period_encode_mode=sustain_period_encode_mode
            ).to_integer_encoding().to_file(destination_path)

def _check_dataset_path(dataset_path):
    if not (dataset_path.exists() and dataset_path.is_dir()):
        logging.error('Failed preprocessing \'{}\'. The specfiied dataset path does not exist or is not a directory.')
        return False

    return True

# A tuple of currently supported file extensions.
_SUPPORTED_EXTENSIONS = ('mid', 'midi')

def _get_dataset_files(dataset_path):
    '''
    Gets all the music files in the specified directory for preprocessing.

    '''
    
    filepaths = []
    for extension in _SUPPORTED_EXTENSIONS:
        filepaths.extend(dataset_path.glob('**/*.{}'.format(extension)))
    
    return filepaths

def convert_all(config, dataset_path, output_path, sustain_period_encode_mode,
                transform, transform_percent, num_workers=16):
    '''
    Converts all music files in a dataset directory to a compact format readable by the Composer models.

    :param config:
        A :class:`composer.config.ConfigInstance` containing the configuration values.
    :param dataset_path:
        The path to the dataset directory.
    :param output_path:
        The directory where the preprocessed data will be saved.
    :param sustain_period_encode_mode:
        The way in which sustain periods should be encoded.
    :param transform:
        Indicates whether the dataset should be transformed. If ``True``, a percentage of the dataset
        is duplicated and pitch shifted and/or time-stretched.
    :param transform_percent:
        The percentage of the dataset that should be transformed.
    :param num_workers:
        The number of worker threads to spawn. Defaults to 16.

    '''

    dataset_path = Path(dataset_path)
    if not _check_dataset_path(dataset_path): return
 
    output_path = Path(dataset_path / 'processed' if output_path is None else output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    filepaths = _get_dataset_files(dataset_path)
    files_transform = {file: False for file in filepaths}
    if transform:
        transform_files_amount = int(len(filepaths) * transform_percent)
        for i in range(transform_files_amount):
            files_transform[filepaths[i]] = True

    # Run the convert_file method on each file in filepaths.
    kwargs = [{
        'filepath': file, 
        'output_path': output_path, 
        'transform': files_transform[file],
        'time_stretch_range': (config.dataset.time_stretch_range.start, config.dataset.time_stretch_range.stop),
        'pitch_shift_range': (config.dataset.pitch_shift_range.start, config.dataset.pitch_shift_range.stop),
        'time_step_increment': config.dataset.time_step_increment,
        'max_time_steps': config.dataset.max_time_steps,
        'velocity_bins': config.dataset.velocity_bins,
        'sustain_period_encode_mode': sustain_period_encode_mode,
        'trim_start': config.dataset.trim_start
    } for file in filepaths]

    parallel_process(kwargs, convert_file, use_kwargs=True)

def split_dataset(config, dataset_path, root_outpath_directory, sustain_period_encode_mode,
                  test_percent, transform, transform_percent, num_workers=16):
    '''
    Splits all music files in a dataset directory into a training and testing set based on the specified ratio.

    :param config:
        A :class:`composer.config.ConfigInstance` containing the configuration values.
    :param dataset_path:
        The path to the dataset directory.
    :param root_outpath_directory:
        The root directory where the preprocessed data will be saved.
    :param sustain_period_encode_mode:
        The way in which sustain periods should be encoded.
    :param test_percent:
        The percentage (0 to 1) of the dataset that it allocated to the test set.
    :param transform:
        Indicates whether the dataset should be transformed. If ``True``, a percentage of the dataset
        is duplicated and pitch shifted and/or time-stretched.

        Note that enabling this will ONLY transform the TRAIN set.
    :param transform_percent:
        The percentage of the dataset that should be transformed.
    :param num_workers:
        The number of worker threads to spawn. Defaults to 16.

    '''

    dataset_path = Path(dataset_path)
    if not _check_dataset_path(dataset_path): return

    filepaths = _get_dataset_files(dataset_path)

    # Split the files into train and test files.
    train_files_amount = int(len(filepaths) * (1 - test_percent))
    train_files = filepaths[:train_files_amount]
    test_files = filepaths[train_files_amount:]

    root_outpath_directory = Path(root_outpath_directory)
    train_outpath_path = root_outpath_directory / 'train'
    test_output_path = root_outpath_directory / 'test'

    # Make sure the outpaths exist...
    train_outpath_path.mkdir(exist_ok=True, parents=True)
    test_output_path.mkdir(exist_ok=True, parents=True)

    # Maps files to a boolean indicating whether the file should be transformed.
    train_files_transform = {file: False for file in train_files}
    if transform:
        transform_files_amount = int(len(train_files) * transform_percent)
        # Transform the first X percent of train files...
        # We only transform the training dataset since it doesn't make much sense
        # nor do is there much benefit in transforming the test set.
        for i in range(transform_files_amount):
            train_files_transform[train_files[i]] = True

    # Run parallel processes to convert each file.
    def _make_kwargs_set(files, files_transform, output_path):
        return [{
            'filepath': file, 
            'output_path': output_path, 
            'transform': files_transform[file] if files_transform is not None else False,
            'time_stretch_range': (config.dataset.time_stretch_range.start, config.dataset.time_stretch_range.stop),
            'pitch_shift_range': (config.dataset.pitch_shift_range.start, config.dataset.pitch_shift_range.stop),
            'time_step_increment': config.dataset.time_step_increment,
            'max_time_steps': config.dataset.max_time_steps,
            'velocity_bins': config.dataset.velocity_bins,
            'sustain_period_encode_mode': sustain_period_encode_mode,
            'trim_start': config.dataset.trim_start
        } for file in files]
    
    parallel_process(_make_kwargs_set(train_files, train_files_transform, train_outpath_path), convert_file, use_kwargs=True)
    parallel_process(_make_kwargs_set(test_files, None, test_output_path), convert_file, use_kwargs=True)