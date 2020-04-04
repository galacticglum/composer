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
from composer.exceptions import DatasetError

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

        # Creates the MusicRNN model.
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
            ModelType.MUSIC_RNN: _create_music_rnn
        }

        return function_map[self]()
    
    def get_train_test_set(self, dataset_path, config):
        '''
        Loads the training and testing datasets for this :class:`ModelType`
        using the values in the specified :class:`composer.config.ConfigInstance` object.

        :param dataset_path:
            The path to the preprocessed dataset organized into two subdirectories: train and test.
        :param config:
            A :class:`composer.config.ConfigInstance` containing the configuration values.
        :returns:
            Two :class:`tensorflow.data.Dataset` objects representing the training and testing
            datasets respectively. 
        
        '''

        dataset_path = Path(dataset_path)
        train_dataset_path = dataset_path / 'train'
        test_dataset_path = dataset_path / 'test'
        if not train_dataset_path.exists() or not test_dataset_path.exists():
            raise DatasetError('Could not get train/test datasets since the specified dataset directory, ' +
                               '\'{}\', has no train or test folder.'.fromat(dataset_path))

        # Get all the dataset files in each directory (train and test)
        train_files = train_dataset_path.glob('**/*.{}'.format(composer.dataset.preprocess._OUTPUT_EXTENSION))
        test_files = test_dataset_path.glob('**/*.{}'.format(composer.dataset.preprocess._OUTPUT_EXTENSION))

        # Creates the MusicRNNDataset.
        def _load_music_rnn_dataset():
            from composer.models.music_rnn import create_music_rnn_dataset
            train_set = create_music_rnn_dataset(train_files, config.train.batch_size, config.model.window_size)
            test_set = create_music_rnn_dataset(test_files, config.train.batch_size, config.model.window_size)

            return train_set, test_set

        # An easy way to map the creation functions to their respective types.
        # This is a lot better than doing something like an if/elif statement.
        function_map = {
            ModelType.MUSIC_RNN: _load_music_rnn_dataset
        }

        return function_map[self]()

# The default configuration file for the MusicRNN model.
_MUSIC_RNN_DEFAULT_CONFIG = Path(__file__).parent / 'music_rnn_config.yml'

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
    train_dataset, test_dataset = model_type.get_train_test_set(dataset_path, config)

    # The event dimensions is the size of a single one-hot vector. We can use the loaded dataset and simply grab its shape.
    # The train dataset returns X and Y. The X is the input which shape (batch_size, time_steps, sequence_size),
    # where sequence_size is the size of a single input sequence (i.e. the size of the one-hot vector).
    # event_dimensions = train_dataset.take(1).shape[-1]
    # 
    # Since train_dataset.take(1) returns a TakeDataset object, we need to first convert it to an iterator.
    # Using the as_numpy_iterator method, we have a generator object that contains the first and second outputs
    # of the train_dataset generator method (these are the inputs and outputs). Using the next method, we get
    # the NEXT element in the iterator, which happens to be input, and finally we get its shape.
    event_dimensions = next(train_dataset.take(1).as_numpy_iterator())[0].shape[-1]
    model = model_type.create_model(config, event_dimensions=event_dimensions)
    
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
    from tensorflow.keras import optimizers

    model_logdir = Path(logdir) / '{}-{}'.format(model_type.name.lower(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_checkpoint_path = model_logdir / 'model-{epoch:02d}'

    tensorboard_callback = TensorBoard(log_dir=str(model_logdir.absolute()), update_freq='batch', profile_batch=0, write_graph=True, write_images=True)
    model_checkpoint_callback = ModelCheckpoint(filepath=str(model_checkpoint_path.absolute()), monitor='val_loss', 
                                                verbose=1, save_best_only=False, mode='auto')

    optimizer = optimizers.Adam(learning_rate=config.train.learning_rate, decay=config.train.decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    training_history = model.fit(train_dataset.shuffle(config.train.shuffle_buffer_size, reshuffle_each_iteration=True),
                                 epochs=epochs, callbacks=[tensorboard_callback, model_checkpoint_callback])
