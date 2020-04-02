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

def _check_dataset_path(dataset_path):
    if not (dataset_path.exists() and dataset_path.is_dir()):
        logging.error('Failed preprocessing \'{}\'. The specfiied dataset path does not exist or is not a directory.')
        return False

    return True

# A tuple of currently supported file extensions.
_SUPPORTED_EXTENSIONS = ('mid', 'midi')

def _get_dataset_files(dataset_path):
   filepaths = []
   for extension in _SUPPORTED_EXTENSIONS:
    filepaths.extend(dataset_path.glob('**/*.{}'.format(extension)))
    
    return filepaths

def convert_all(dataset_path, output_path, num_workers=16):
    '''
    Converts all music files in a dataset directory to a compact format readable by the Composer models.

    :param dataset_path:
        The path to the dataset directory.
    :param output_path:
        The directory where the preprocessed data will be saved.
    :param num_workers:
        The number of worker threads to spawn. Defaults to 16.

    '''

    dataset_path = Path(dataset_path)
    if not _check_dataset_path(dataset_path): return
 
    output_path = Path(dataset_path / 'processed' if output_path is None else output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    filepaths = _get_dataset_files(dataset_path)

    # Run the convert_file method on each file in filepaths.
    parallel_process([{'filepath': file, 'output_path': output_path} for file in filepaths], convert_file, use_kwargs=True)

def split_dataset(dataset_path, root_outpath_directory, test_percent, num_workers=16):
    '''
    Splits all music files in a dataset directory into a training and testing set based on the specified ratio.

    :param dataset_path:
        The path to the dataset directory.
    :param root_outpath_directory:
        The root directory where the preprocessed data will be saved.
    :param test_percent:
        The percentage (0 to 1) of the dataset that it allocated to the test set.
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

    # Run parallel processes to convert each file.
    parallel_process([{'filepath': file, 'output_path': train_outpath_path} for file in train_files], convert_file, use_kwargs=True)
    parallel_process([{'filepath': file, 'output_path': test_output_path} for file in test_files], convert_file, use_kwargs=True)

