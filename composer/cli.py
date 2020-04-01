'''
The command-line interface for Composer.

'''

import click
import logging
import composer.logging_utils as logging_utils
import composer.dataset.preprocess

from pathlib import Path

def _set_verbosity_level(logger, value):
    '''
    Sets the verbosity level of the specified logger.

    '''

    x = getattr(logging, value.upper(), None)
    if x is None:
        raise click.BadParameter('Must be CRITICAL, ERROR, WARNING, INFO, or DEBUG, not \'{}\''.format(value))

    logger.setLevel(x)
        
@click.group()
@click.option('--verbosity', '-v', default='INFO', help='Either CRITICAL, ERROR, WARNING, INFO, or DEBUG.')
@click.pass_context
def cli(ctx, verbosity):
    '''
    A deep learning enabled music generator.

    '''

    logging_utils.init()
    _set_verbosity_level(logging.getLogger(), verbosity)
    
@cli.command()
@click.argument('dataset-path')
@click.option('--output-path', '-o',  default=None, \
    help='The directory where the preprocessed data will be saved. Defaults to a "processed" folder in the DATASET_PATH directory.')
@click.option('--num-workers', '-w', default=16, help='The number of worker threads to spawn. Defaults to 16.')
def preprocess(dataset_path, output_path, num_workers):
    '''
    Preprocesses a raw dataset so that it can be used by the models.

    DATASET_PATH is the path to the dataset that will be preprocessed.
    '''

    if output_path is None:
        output_path = Path(dataset_path) / 'processed'
        
    composer.dataset.preprocess.convert_all(dataset_path, output_path, num_workers)