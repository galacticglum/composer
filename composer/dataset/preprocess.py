'''
Preprocesses a raw dataset so that it can be used by the models.

'''

import hashlib
import logging
from tqdm import tqdm
from pathlib import Path
from composer.utils import parallel_process
from composer.dataset.sequence import NoteSequence

def convert_file(filepath, output_path):
    '''
    Converts a music file to a set of sequences.

    :param filepath:
        The path to the file to convert.
    :param output_path:
        The directory to output the file to.

    '''

    filename = Path(filepath).stem
    file_id = hashlib.md5(str(filepath).encode()).hexdigest()
    file_save_path = output_path / '{}_{}.data'.format(filename, file_id)

    # Load the MIDI file into a NoteSequence and convert it into an EventSequence.
    event_sequence = NoteSequence.from_midi(filepath).to_event_sequence()
    # Encode the event sequence and write to file.
    event_sequence.to_integer_encoding().to_file(file_save_path)

def convert_all(dataset_path, output_path, num_workers=8):
    '''
    Converts all music files in a dataset directory to a compact format readable by the Composer models.

    :param dataset_path:
        The path to the dataset directory.
    :param output_path:
        The directory where the preprocessed data will be saved.
    :param num_workers:
        The number of worker threads to spawn. Defaults to 8.

    '''

    dataset_path = Path(dataset_path)
    if not (dataset_path.exists() and dataset_path.is_dir()):
        logging.error('Failed preprocessing \'{}\'. The specfiied dataset path does not exist or is not a directory.')
        return

    filepaths = []
    # A tuple of currently supported file extensions.
    SUPPORTED_EXTENSIONS = ('mid', 'midi')
    for extension in SUPPORTED_EXTENSIONS:
        filepaths.extend(dataset_path.glob('**/*.{}'.format(extension)))

    output_path = Path(dataset_path / 'processed' if output_path is None else output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Run the convert_file method on each file in filepaths.
    process_kwargs = [{'filepath': file, 'output_path': output_path} for file in filepaths]
    parallel_process(process_kwargs, convert_file, use_kwargs=True)

    # for i in tqdm(range(len(filepaths))):
    #     convert_file(filepaths[i], output_path)