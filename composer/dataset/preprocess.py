'''
Preprocesses a raw dataset so that it can be used by the models.

'''

import logging
from pathlib import Path

def convert_all(directory):
    '''
    Converts all music files in a dataset directory to a compact format readable by the Composer models.

    '''

    directory = Path(directory)
    if not (directory.exists() and directory.is_dir()):
        logging.error('Failed preprocessing \'{}\'. The specfiied dataset path does not exist or is not a directory.')
        return

    files = []
    
    # A tuple of currently supported file extensions.
    SUPPORTED_EXTENSIONS = ('mid', 'midi')
    for extension in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob('**/*.{}'.format(extension)))

    print(files)