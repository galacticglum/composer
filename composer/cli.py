'''
The command-line interface for Composer.

'''

import time
import json
import click
import logging
import datetime
import numpy as np
import composer.config
import composer.logging_utils as logging_utils
import composer.dataset.preprocess

from pathlib import Path
from enum import Enum, unique
from composer.click_utils import EnumType
from composer.exceptions import DatasetError, InvalidParameterError

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
@click.option('--seed', type=int, help='Sets the seed of the random engine.')
@click.pass_context
def cli(ctx, verbosity, seed):
    '''
    A deep learning enabled music generator.

    '''

    if seed is None:
        # We use the current time as the seed rather than letting numpy seed
        # since we want to achieve consistent results across sessions.
        # Source: https://stackoverflow.com/a/45573061/7614083
        t = int(time.time() * 1000.0)
        seed = ((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) <<  8) + ((t & 0x000000ff) << 24)

    logging_utils.init()
    _set_verbosity_level(logging.getLogger(), verbosity)

@cli.command()
@click.argument('dataset-path')
@click.argument('output-directory')
@click.option('--num-workers', '-w', default=16, help='The number of worker threads to spawn. Defaults to 16.')
@click.option('--transform/--no-transform', default=False, help='Indicates whether the dataset should be transformed. ' +
              'If true, a percentage of the dataset is duplicated and pitch shifted and/or time-stretched. Defaults to False.\n' +
              'Note: transforming a single sample produces three new samples: a pitch shifted one, time stretched one, and one with ' +
              'a combination of both. A transform percent value of 5%% means that the dataset will GROW by 3 times 5%% of the total size.')
@click.option('--transform-percent', default=0.50, help='The percentage of the dataset that should be transformed. Defaults to 50%% of the dataset.')
@click.option('--split/--no-split', default=True, help='Indicates whether the dataset should be split into train and test sets. Defaults to True.')
@click.option('--test-percent', default=0.30, help='The percentage of the dataset that is allocated to testing. Defaults to 30%%')
@click.option('--metadata/--no-metadata', 'output_metadata', default=True, help='Indicates whether to output metadata. Defaults to True.')
def preprocess(dataset_path, output_directory, num_workers, transform, transform_percent, split, test_percent, output_metadata):
    '''
    Preprocesses a raw dataset so that it can be used by the models.

    '''

    output_directory = Path(output_directory)

    if split:
        composer.dataset.preprocess.split_dataset(dataset_path, output_directory, test_percent, transform, transform_percent, num_workers)
    else:
        composer.dataset.preprocess.convert_all(dataset_path, output_directory, num_workers)

    if not output_metadata: return
    with open(output_directory / 'metadata.json', 'w+') as metadata_file:
        # The metadata file is more or less a dump of the settings used to preprocess the dataset.
        metadata = {
            'local_time': str(datetime.datetime.now()),
            'utc_time': str(datetime.datetime.utcnow()),
            'raw_dataset_path': str(Path(dataset_path).absolute()),
            'output_directory': str(output_directory.absolute()),
            'transform': transform,
            'transform_percent': transform_percent,
            'split': split,
            'test_percent': test_percent,
            'seed': int(np.random.get_state()[1][0])
        }

        json.dump(metadata, metadata_file, indent=True)

@unique
class ModelType(Enum):
    '''
    The type of the model.

    '''

    MUSIC_RNN = 'music_rnn'

    def create_model(self, config, dimensions, **kwargs):
        '''
        Creates the model class associated with this :class:`ModelType` using the 
        values in the specified :class:`composer.config.ConfigInstance` object.

        :param config:
            A :class:`composer.config.ConfigInstance` containing the configuration values.
        :param dimensions:
            A two-dimensional tuple of integers containing the input and output event dimensions;
            that is, the dimensions of a single feature and the dimensions of a single label respectively.
        :param **kwargs:
            External data passed to the creation method (i.e. data not in the configuration file)
        :returns:
            A :class:`tensorflow.keras.Model` object representing an instance of the specfiied model.
        '''

        # Creates the MusicRNN model.
        def _create_music_rnn():
            from composer import models

            return models.MusicRNN(
                *dimensions, config.model.window_size, config.model.lstm_layers_count,
                config.model.lstm_layer_sizes, config.model.lstm_dropout_probability,
                config.model.use_batch_normalization
            )

        # An easy way to map the creation functions to their respective types.
        # This is a lot better than doing something like an if/elif statement.
        function_map = {
            ModelType.MUSIC_RNN: _create_music_rnn
        }

        return function_map[self]()
        
    def get_dataset(self, dataset_path, mode, config):
        '''
        Loads a dataset for this :class:`ModelType` using the values 
        in the specified :class:`composer.config.ConfigInstance` object.

        :param dataset_path:
            The path to the preprocessed dataset organized into two subdirectories: train and test.
        :param mode:
            A string indicating the dataset mode: ``train`` or ``test``.
        :param config:
            A :class:`composer.config.ConfigInstance` containing the configuration values.
        :returns:
            A :class:`tensorflow.data.Dataset` object representing the dataset and
            a two-dimensional tuple of integers representing input and output dimensions
            of the dataset (i.e. the dimensions of a single feature and label respectively).
        
        '''

        from composer.models import load_dataset, EventEncodingType

        if mode not in ['train', 'test']:
            raise InvalidParameterError('\'{}\' is an invalid dataset mode! Must be one of: \'train\', \'test\'.'.format(mode))

        dataset_path = Path(dataset_path) / mode
        if not dataset_path.exists():
            raise DatasetError('Could not get {mode} dataset since the specified dataset directory, ' +
                               '\'{}\', has no {mode} folder.'.fromat(dataset_path, mode=mode))

        files = list(dataset_path.glob('**/*.{}'.format(composer.dataset.preprocess._OUTPUT_EXTENSION)))

        # Creates the MusicRNNDataset.
        def _load_music_rnn_dataset():
            dataset, dimensions = load_dataset(files, config.train.batch_size, config.model.window_size, input_event_encoding=EventEncodingType.ONE_HOT)

            return dataset, dimensions

        # An easy way to map the creation functions to their respective types.
        # This is a lot better than doing something like an if/elif statement.
        function_map = {
            ModelType.MUSIC_RNN: _load_music_rnn_dataset
        }

        return function_map[self]()

    def get_train_dataset(self, dataset_path, config):
        '''
        Loads the training dataset for this :class:`ModelType` using the values 
        in the specified :class:`composer.config.ConfigInstance` object.

        :param dataset_path:
            The path to the preprocessed dataset organized into two subdirectories: train and test.
        :param config:
            A :class:`composer.config.ConfigInstance` containing the configuration values.
        :returns:
            A :class:`tensorflow.data.Dataset` object representing the training dataset and
            a two-dimensional tuple of integers representing input and output dimensions
            of the dataset (i.e. the dimensions of a single feature and label respectively).
        
        '''

        return self.get_dataset(dataset_path, 'train', config)

    def get_test_dataset(self, dataset_path, config):
        '''
        Loads the testing dataset for this :class:`ModelType` using the values 
        in the specified :class:`composer.config.ConfigInstance` object.

        :param dataset_path:
            The path to the preprocessed dataset organized into two subdirectories: train and test.
        :param config:
            A :class:`composer.config.ConfigInstance` containing the configuration values.
        :returns:
            A :class:`tensorflow.data.Dataset` object representing the testing dataset and
            a two-dimensional tuple of integers representing input and output dimensions
            of the dataset (i.e. the dimensions of a single feature and label respectively).
        
        '''

        return self.get_dataset(dataset_path, 'test', config)

# The default configuration file for the MusicRNN model.
_MUSIC_RNN_DEFAULT_CONFIG = Path(__file__).parent / 'music_rnn_config.yml'

def _compile_model(model, config):
    '''
    Compiles the specified ``model``.

    '''

    from tensorflow.keras import optimizers

    optimizer = optimizers.Adam(learning_rate=config.train.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('dataset-path')
@click.option('--logdir', default='./output/logdir/', help='The root log directory. Defaults to \'./output/logdir\'.')
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
@click.option('-e', '--epochs', 'epochs', default=10, help='The number of epochs to train for. Defaults to 10.')
def train(model_type, dataset_path, logdir, config_filepath, epochs):
    '''
    Trains the specified model.

    '''

    if config_filepath is None:
        config_filepath = _MUSIC_RNN_DEFAULT_CONFIG

    config = composer.config.get(config_filepath)
    train_dataset, dimensions = model_type.get_train_dataset(dataset_path, config)
  
    # model = model_type.create_model(config, dimensions)

    # from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

    # model_logdir = Path(logdir) / '{}-{}'.format(model_type.name.lower(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # model_checkpoint_path = model_logdir / 'model-{epoch:02d}-{loss:.2f}.h5'

    # tensorboard_callback = TensorBoard(log_dir=str(model_logdir.absolute()), update_freq=25, profile_batch=0, write_graph=False, write_images=False)
    # model_checkpoint_callback = ModelCheckpoint(filepath=str(model_checkpoint_path.absolute()), monitor='loss', verbose=1, 
    #                                             save_freq=1000, save_best_only=False, mode='auto', save_weights_only=True)

    # _compile_model(model, config)
    # training_history = model.fit(train_dataset, epochs=epochs, callbacks=[tensorboard_callback, model_checkpoint_callback])

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('dataset-path')
@click.argument('restoredir')
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
def evaluate(model_type, dataset_path, restoredir, config_filepath):
    '''
    Evaluate the specified model.

    '''

    if config_filepath is None:
        config_filepath = _MUSIC_RNN_DEFAULT_CONFIG

    config = composer.config.get(config_filepath)
    test_dataset, dimensions = model_type.get_test_dataset(dataset_path, config)
    
    model = model_type.create_model(config, dimensions)
    _compile_model(model, config)
    model.build(input_shape=(config.train.batch_size, config.model.window_size, dimensions[0]))
    model.load_weights(restoredir)

    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    logging.info('- Finished evaluating model. Loss: {:.4f}, Accuracy: {:.4f}'.format(loss, accuracy))