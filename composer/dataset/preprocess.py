'''
Preprocesses a raw dataset so that it can be used by the models.

'''

import click
import pickle
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from composer.dataset.sequence import NoteSequence

def convert_file(filepath):
    '''
    Converts a music file to a set of sequences.

    '''

    pass

def convert_all(dataset_path, output_path=None, num_workers=8):
    '''
    Converts all music files in a dataset directory to a compact format readable by the Composer models.

    :param dataset_path:
        The path to the dataset directory.
    :param output_path:
        The directory where the preprocessed data will be saved. Defaults to a "processed" folder in the dataset_path directory.
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
    
    results = []
    executor = ProcessPoolExecutor(num_workers)
    for filepath in filepaths:
        results.append((filepath, executor.submit(convert_file, filepath)))
    
    with click.progressbar(length=len(results)) as bar:
        for filepath, future in results:    
            bar.label = filepath

            filename = Path(filepath).stem
            file_id = hashlib.md5(filepath.encode()).hexdigest()
            file_save_path = output_path / '{}_{}.dat'.format(filename, file_id)
            with open(file_save_path, 'wb+') as save_file_handler:
                pickle.dump(future.result(), save_file_handler)

            bar.update(1)