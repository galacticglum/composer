'''
The command-line interface for Composer.

'''

import click
import logging
import composer.logging_utils as logging_utils
from composer.dataset.sequence import Note, NoteSequence, SustainPeriod, OneHotEncodedEventSequence, IntegerEncodedEventSequence
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

    sequence = NoteSequence.from_midi('data/ecomp/Abdelmola01.mid')
    # sequence = NoteSequence([
    #     Note(0, 4000, 61, 64),
    #     Note(0, 4000, 65, 64),
    #     Note(5000, 11000, 56, 100)
    # ], [SustainPeriod(5000, 12000)])

    event_sequence = sequence.to_event_sequence(time_step_increment=1, velocity_bins=128)
    int_encoding = event_sequence.to_integer_encoding()
    int_encoding.to_file('data/Abdelmola01_int_2.data')

    note_sequence_from_file = IntegerEncodedEventSequence.from_file('data/Abdelmola01_int_2.data').decode().to_note_sequence()
    note_sequence_from_file.to_midi('data/Abdelmola01_rerender_2.mid')

    # composer.dataset.preprocess.convert_all(dataset_path, output_path, num_workers)