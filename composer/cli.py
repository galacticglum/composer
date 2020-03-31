'''
The command-line interface for Composer.

'''

import click
import logging
import composer.logging_utils as logging_utils
from composer.dataset.sequence import Note, NoteSequence, SustainPeriod
# import composer.dataset.preprocess

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
@click.option('--num-workers', '-w', default=8, help='The number of worker threads to spawn. Defaults to 8.')
def preprocess(dataset_path, output_path, num_workers):
    '''
    Preprocesses a raw dataset so that it can be used by the models.

    DATASET_PATH is the path to the dataset that will be preprocessed.
    '''

    # sequence = NoteSequence.from_midi('data/ecomp/Abdelmola01.mid')
    sequence = NoteSequence([
        Note(0, 4000, 1, 64),
        Note(0, 4000, 4, 64),
        Note(5000, 11000, 3, 37)
    ], [SustainPeriod(5000, 12000)])

    print(sequence.to_event_sequence())
    # composer.dataset.preprocess.convert_all(dataset_path, output_path, num_workers)