'''
The command-line interface for Composer.

'''

import click
import logging
import composer.config
import composer.logging_utils as logging_utils
import composer.dataset.preprocess

from pathlib import Path
from enum import Enum, unique
from composer import models
from composer.click_utils import EnumType

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


@unique
class ModelType(Enum):
    '''
    The type of the model.

    '''

    MUSIC_RNN = 'music_rnn'

    def create_model(self, config, **kwargs):
        '''
        Creates the model class associated with this :class:`ModelType` using the 
        values in the specified :class:`composer.config.ConfigInstance` object.

        :param config:
            A :class:`composer.config.ConfigInstance` containing the configuration values.
        :param **kwargs:
            External data passed to the creation method (i.e. data not in the configuration file)
        :returns:
            A :class:`tensorflow.keras.Model` object representing an instance of the specfiied model.
        '''

        def _create_music_rnn():
            return models.MusicRNN(
                kwargs['event_dimensions'], config.model.window_size, config.model.lstm_layers_count,
                config.model.lstm_layer_sizes, config.model.lstm_dropout_probability, 
                config.model.dense_layer_size, config.model.use_batch_normalization
            )

        # An easy way to map the creation functions to their respective types.
        # This is a lot better than doing something like an if/elif statement.
        function_map = {
            MUSIC_RNN: _create_music_rnn
        }

        return function_map[self]()

# The default configuration file for the MusicRNN model.
_MUSIC_RNN_DEFAULT_CONFIG = Path(__file__).parent / 'music_rnn_config.yml'

@cli.command()
@click.argument('model_type', type=EnumType(ModelType, False))
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
def train(model_type, config_filepath):
    '''
    Trains the specified model.

    '''

    if config_filepath is None:
        config_filepath = _MUSIC_RNN_DEFAULT_CONFIG

    config = composer.config.get(config_filepath)
    model = model_type.create_model(config)

    print(model)