'''
The command-line interface for Composer.

'''

import click
import logging
import numpy as np
import composer.config
import composer.logging_utils as logging_utils
import composer.dataset.preprocess

from pathlib import Path
from enum import Enum, unique
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
@click.option('--seed', default=None, help='Sets the seed of the random engine.')
@click.pass_context
def cli(ctx, verbosity, seed):
    '''
    A deep learning enabled music generator.

    '''

    np.random.seed(seed)

    logging_utils.init()
    _set_verbosity_level(logging.getLogger(), verbosity)
    
@cli.command()
@click.argument('dataset-path')
@click.argument('output_directory')
@click.option('--num-workers', '-w', default=16, help='The number of worker threads to spawn. Defaults to 16.')
@click.option('--transform/--no-transform', default=False, help='Indicates whether the dataset should be transformed. ' +
              'If true, a percentage of the dataset is duplicated and pitch shifted and/or time-stretched. Defaults to False.\n' +
              'Note: transforming a single sample produces three new samples: a pitch shifted one, time stretched one, and one with ' +
              'a combination of both. A transform percent value of 5%% means that the dataset will GROW by 3 times 5%% of the total size.')
@click.option('--transform-percent', default=0.50, help='The percentage of the dataset that should be transformed. Defaults to 50%% of the dataset.')
@click.option('--split/--no-split', default=True, help='Indicates whether the dataset should be split into train and test sets. Defaults to True.')
@click.option('--test-percent', default=0.30, help='The percentage of the dataset that is allocated to testing. Defaults to 30%%')
def preprocess(dataset_path, output_directory, num_workers, transform, transform_percent, split, test_percent):
    '''
    Preprocesses a raw dataset so that it can be used by the models.

    '''

    output_directory = Path(output_directory)

    if split:
        composer.dataset.preprocess.split_dataset(dataset_path, output_directory, test_percent, transform, transform_percent, num_workers)
    else:
        composer.dataset.preprocess.convert_all(dataset_path, output_directory, num_workers)

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
            from composer import models

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