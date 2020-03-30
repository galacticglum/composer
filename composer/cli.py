'''
The command-line interface for Composer.

'''

import click
import logging
import composer.logging_utils as logging_utils

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
@click.argument('dataset_path')
def preprocess(dataset_path):
    '''
    Preprocesses a raw dataset so that it can be used by the models.

    DATASET_PATH is the path to the dataset that will be preprocessed.
    '''

    print(dataset_path)