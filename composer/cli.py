'''
The command-line interface for Composer.

'''

import re
import tqdm
import time
import json
import click
import logging
import requests
import datetime
import subprocess
import numpy as np

import composer.config
import composer.dataset.preprocess
import composer.logging_utils as logging_utils
from composer import ModelSaveFrequencyMode

from pathlib import Path
from enum import Enum, unique
from shutil import copy2, which
from composer.click_utils import EnumType
from composer.exceptions import DatasetError, InvalidParameterError
from composer.dataset.sequence import NoteSequence, EventSequence, OneHotEncodedEventSequence, IntegerEncodedEventSequence

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

def get_default_config():
    '''
    Gets the default configuration filepath for the specified :class:`ModelType`.

    '''

    return Path(__file__).parent / 'default_config.yml' 

@cli.command()
@click.argument('filepath')
def make_config(filepath):
    '''
    Creates a configuration file from the default configuration.

    '''

    default_config_filepath = get_default_config()
    copy2(default_config_filepath, filepath)

@unique
class ModelType(Enum):
    '''
    The type of the model.

    :cvar MUSIC_RNN:
        The type corresponding to the :class:`composer.models.MusicRNN` model.
    :cvar TRANSFORMER:
        The type corresponding to the :class:`composer.models.Transformer` model.

    '''

    MUSIC_RNN = 'music_rnn'
    TRANSFORMER = 'transformer'

def create_model(model_type, config, **kwargs):
    '''
    Creates the model class associated with this :class:`ModelType` using the 
    values in the specified :class:`composer.config.ConfigInstance` object.

    :param model_type:
        A :class:`ModelType` value representing the type of the model to create.
    :param config:
        A :class:`composer.config.ConfigInsance` containing the configuration values.
    :param \**kwargs:
        External data passed to the creation method (i.e. data not in the configuration file)
    :returns:
        A :class:`composer.models.BaseModel` object representing an instance of the specified model
        and the vocabulary size of an event in the dataset.

    '''

    from composer import models
    event_vocab_size = _get_event_vocab_size(config)

    # Creates the MusicRNN model.
    def _create_music_rnn():
        return models.MusicRNN(
            event_vocab_size, config.music_rnn.train.batch_size, config.music_rnn.model.embedding_size, 
            config.music_rnn.model.lstm_layers_count, config.music_rnn.model.lstm_layer_sizes, 
            config.music_rnn.model.lstm_dropout_probability, config.music_rnn.model.use_batch_normalization
        )

    def _create_transformer():
        return models.Transformer(
            event_vocab_size, config.transformer.model.embedding_size,
            config.transformer.model.window_size, config.transformer.model.decoder_layers_count,
            config.transformer.model.attention_head_count, config.transformer.model.use_relative_attention,
            config.transformer.model.initializer_mean, config.transformer.model.initializer_stddev,
            config.transformer.model.attention_dropout_rate, config.transformer.model.residual_dropout_rate,
            config.transformer.model.layer_normalization_epsilon, config.transformer.model.scale_attention,
            config.transformer.model.use_layer_normalization
        )

    # An easy way to map the creation functions to their respective types.
    # This is a lot better than doing something like an if/elif statement.
    function_map = {
        ModelType.MUSIC_RNN: _create_music_rnn,
        ModelType.TRANSFORMER: _create_transformer
    }

    return function_map[model_type](), event_vocab_size

def get_batch_size(model_type, config):
    '''
    Gets the batch size from the specified :class:`composer.config.ConfigInstance`
    for the given :class:`ModelType.

    '''

    if model_type == ModelType.MUSIC_RNN:
        return config.music_rnn.train.batch_size
    elif model_type == ModelType.TRANSFORMER:
        return config.transformer.train.batch_size
    else:
        raise NotImplementedError('Unrecognized model type: \'{}\'.'.format(model_type))

def get_learning_rate(model_type, config):
    '''
    Gets the learning rate from the specified :class:`composer.config.ConfigInstance`
    for the given :class:`ModelType.

    '''

    if model_type == ModelType.MUSIC_RNN:
        return config.music_rnn.train.learning_rate
    elif model_type == ModelType.TRANSFORMER:
        return config.transformer.train.learning_rate
    else:
        raise NotImplementedError('Unrecognized model type: \'{}\'.'.format(model_type))

def get_window_size(model_type, config):
    '''
    Gets the window size from the specified :class:`composer.config.ConfigInstance`
    for the given :class:`ModelType.

    '''

    if model_type == ModelType.MUSIC_RNN:
        return config.music_rnn.model.window_size
    elif model_type == ModelType.TRANSFORMER:
        return config.transformer.model.window_size
    else:
        raise NotImplementedError('Unrecognized model type: \'{}\'.'.format(model_type))

def get_dataset(model_type, dataset_path, config, mode='', use_generator=True, max_files=None,
                show_progress_bar=True, shuffle_files=True, shuffle_dataset=True):
    '''
    Loads a dataset for this :class:`ModelType` using the values 
    in the specified :class:`composer.config.ConfigInstance` object.

    :param model_type:
        A :class:`ModelType` value indicating for which model the dataset should be loaded for.
    :param dataset_path:
        The path to the preprocessed dataset organized into two subdirectories: train and test.
    :param config:
        A :class:`composer.config.ConfigInstance` containing the configuration values.
    :param mode:
        A string indicating the dataset mode: ``train`` or ``test``.
    :param use_generator:
        Indicates whether the Dataset should be given as a generator object. Defaults to ``True``.
    :param max_files:
        The maximum number of files to load. Defaults to ``None`` which means that ALL
        files will be loaded.
    :param show_progress_bar:
        Indicates whether a loading progress bar should be displayed while the dataset is loaded
        into memory. Defaults to ``True``.
    :param shuffle_files:
        Indicates whether the files should be shuffled before beginning the loading process. Defaults to ``True``.
    :param shuffle_dataset:
        Indicates whether the dataset should be shuffled. Defaults to ``True``.
    :returns:
        A :class:`tensorflow.data.Dataset` object representing the dataset.
    
    '''

    from composer.models import load_dataset, load_tfrecord_dataset, EventEncodingType

    if mode not in ['train', 'test', '']:
        raise InvalidParameterError('\'{}\' is an invalid dataset mode! Must be one of: \'train\', \'test\', or none.'.format(mode))

    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        dataset_path = dataset_path / mode
        if not dataset_path.exists():
            raise DatasetError('Could not get {mode} dataset since the specified dataset directory, ' +
                            '\'{}\', has no {mode} folder.'.format(dataset_path, mode=mode))

        files = composer.dataset.preprocess.get_processed_files(dataset_path)
        if shuffle_files:
            np.random.shuffle(files)
        
        is_dataset_tfrecord = False
    else:
        if not dataset_path.is_file() or dataset_path.suffix != '.tfrecord':
            raise InvalidParameterError(
                '\'{}\' is an invalid dataset path! The dataset can either be a '
                .format(dataset_path) + 'directory of processed MIDI files or a TFRecord file.'
            )
        
        files = [dataset_path]
        is_dataset_tfrecord = True

    if max_files is not None:
        files = files[:max_files]

    if is_dataset_tfrecord:
        dataset, header = load_tfrecord_dataset(files[0], shuffle=shuffle_dataset)
        
        # Make sure that the header metadata matches the config file
        TFRECORD_EXPORT_WARNING = 'The TFRecord file was probably exported using a different config.'
        
        dataset_model_type = ModelType(header['model_type'])
        if model_type != dataset_model_type:
            logging.warn('Model type mismatch when loading \'{}\'. Expected {} but found {}. {}'
                .format(files[0], model_type, dataset_model_type, TFRECORD_EXPORT_WARNING))

            click.confirm('Do you want to continue? This may cause errors or corrupt the training session.', abort=True)
        
        batch_size = header['batch_size']
        if get_batch_size(model_type, config) != batch_size:
            logging.error('Expected a batch size of {} but found {}. {}'.format(
                get_batch_size(model_type, config), batch_size, TFRECORD_EXPORT_WARNING))
            exit(1)
        
        window_size = header['window_size']
        if get_window_size(model_type, config) != window_size:
            logging.error('Expected a window size of {} but found {}. {}'.format(
                get_window_size(model_type, config), window_size, TFRECORD_EXPORT_WARNING))
            exit(1)
    else: 
        dataset = load_dataset(files, get_batch_size(model_type, config), 
            get_window_size(model_type, config),
            show_loading_progress_bar=show_progress_bar,
            use_generator=use_generator, shuffle=shuffle_dataset)

    return dataset

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('dataset-path')
@click.argument('output-directory')
@click.option('--num-workers', '-w', default=16, help='The number of worker threads to spawn. Defaults to 16.')
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
@click.option('--sustain-period-encode-mode', '-spe', default='extend', type=EnumType(NoteSequence.SustainPeriodEncodeMode, False), 
              help='The way in which sustain periods should be encoded. Defaults to EXTEND.\n\nRefer to NoteSequence.to_event_sequence ' +
              'documentation for more details on this parameter.')
@click.option('--transform/--no-transform', default=True, help='Indicates whether the dataset should be transformed. ' +
              'If true, a percentage of the dataset is duplicated and pitch shifted and/or time-stretched. Defaults to True.\n' +
              'Note that transforming a single sample produces many new samples: one for each pitch in the pitch shift range, and a time' +
              'stretched one (uniformly sampled from the time stretch range).')
@click.option('--transform-percent', default=1.0, help='The percentage of the dataset that should be transformed. Defaults to 100%% of the dataset.')
@click.option('--split/--no-split', default=True, help='Indicates whether the dataset should be split into train and test sets. Defaults to True.')
@click.option('--test-percent', default=0.30, help='The percentage of the dataset that is allocated to testing. Defaults to 30%%')
@click.option('--metadata/--no-metadata', 'output_metadata', default=True, help='Indicates whether to output metadata. Defaults to True.')
def preprocess(model_type, dataset_path, output_directory, num_workers, config_filepath, sustain_period_encode_mode, 
               transform, transform_percent, split, test_percent, output_metadata):
    '''
    Preprocesses a raw dataset so that it can be used by specified model type.

    '''

    config = composer.config.get(config_filepath or get_default_config())
    output_directory = Path(output_directory)

    if split:
        composer.dataset.preprocess.split_dataset(config, dataset_path, output_directory, sustain_period_encode_mode,
                                                  test_percent, transform, transform_percent, num_workers)
    else:
        composer.dataset.preprocess.convert_all(config, dataset_path, output_directory, sustain_period_encode_mode, 
                                                transform, transform_percent, num_workers)

    if not output_metadata: return
    with open(output_directory / 'metadata.json', 'w+') as metadata_file:
        # The metadata file is a dump of the settings used to preprocess the dataset.
        metadata = {
            'local_time': str(datetime.datetime.now()),
            'utc_time': str(datetime.datetime.utcnow()),
            'model_type': str(model_type),
            'raw_dataset_path': str(Path(dataset_path).absolute()),
            'output_directory': str(output_directory.absolute()),
            'sustain_period_encode_mode': str(sustain_period_encode_mode),
            'transform': transform,
            'transform_percent': transform_percent,
            'split': split,
            'test_percent': test_percent,
            'seed': int(np.random.get_state()[1][0]),
        }

        json.dump(metadata, metadata_file, indent=True)
    
    # Copy the config file used to preprocess the dataset
    copy2(config.filepath, output_directory / 'config.yml')

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('preprocessed-path')
@click.argument('output-path')
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
@click.option('--use-generator/--no-use-generator', default=False,
              help='Indicates whether the dataset should be loaded in chunks during processing ' +
              '(rather than into memory all at once). Defaults to False.')
@click.option('--max-files', default=None, help='The maximum number of files to load. Defaults to None, which means ' + 
              'that ALL files will be loaded.', type=int)
def export_dataset(model_type, preprocessed_path, output_path, config_filepath, use_generator, max_files):
    '''
    Exports a processed dataset input pipeline as a TFRecord file for fast loading times when training.

    Note that the PREPROCESSED-PATH argument refers to the path containing the preprocessed '.data' files.
    For example, this could be "dataset_parent/train", "dataset_parent/test", or simply "path/to/files".

    '''

    import tensorflow as tf
    from composer.io_utils import bytes_feature, int64_feature

    config = composer.config.get(config_filepath or get_default_config())
    dataset = get_dataset(model_type, preprocessed_path, config, shuffle_dataset=False,
                use_generator=use_generator, max_files=max_files)
 
    logging.info('Loading dataset and writing to TFRecord. This make take a while...')
    with tf.io.TFRecordWriter(output_path) as writer:
        # Serialize metadata
        batch_size, window_size = next(dataset.take(1).as_numpy_iterator())[0].shape
        writer.write(tf.train.Example(features=tf.train.Features(feature={
            'model_type': bytes_feature(model_type.value.encode('utf-8')),
            'batch_size': int64_feature(batch_size),
            'window_size': int64_feature(window_size)
        })).SerializeToString())

        for x, y in tqdm.tqdm(dataset):
            feature = {
                'x': bytes_feature(tf.io.serialize_tensor(x).numpy()),
                'y': bytes_feature(tf.io.serialize_tensor(y).numpy())
            }

            writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

    logging.info('Finished exporting \'{}\' as a TFRecord: \'{}\''.format(preprocessed_path, output_path))

def get_event_sequence_ranges(config):
    '''
    Gets the event sequence value ranges, dimensions, and ranges.

    :param config:
        A :class:`composer.config.ConfigInstance` containing the configuration values.
    :returns:
        The event value ranges, event dimensions, and event ranges.

    '''

    event_value_ranges = EventSequence._compute_event_value_ranges(config.dataset.time_step_increment, \
                                        config.dataset.max_time_steps, config.dataset.velocity_bins)
    event_dimensions = EventSequence._compute_event_dimensions(event_value_ranges)
    event_ranges = EventSequence._compute_event_ranges(event_dimensions)

    return event_value_ranges, event_dimensions, event_ranges

def _get_event_vocab_size(config):
    '''
    Computes the vocabulary size of the integer encoded events.

    :param config:
        A :class:`composer.config.ConfigInstance` containing the configuration values.
    :returns:
        The dimensions of an encoded event network input.

    '''
        
    _, _, event_ranges = get_event_sequence_ranges(config)
    return OneHotEncodedEventSequence.get_one_hot_size(event_ranges)

def decode_to_event(config, event_id):
    '''
    Decodes an encoded event to a :class:`composer.dataset.sequence.Event`
    based on the configuration values.

    '''

    event_value_ranges, event_dimensions, event_ranges = get_event_sequence_ranges(config)
    return IntegerEncodedEventSequence.id_to_event(event_id, event_ranges, event_value_ranges)

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
def summary(model_type, config_filepath):
    '''
    Prints a summary of the model.

    '''

    config = composer.config.get(config_filepath or get_default_config())

    model, _ = create_model(model_type, config)

    batch, sequence = get_batch_size(model_type, config), get_window_size(model_type, config)
    model.build(input_shape=(batch, sequence))
    model.summary()

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('dataset-path')
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.')
@click.option('--steps', default=5, help='The number of steps to visualize. Defaults to 5.')
@click.option('--decode-events/-no-decode--events', default=True, help='Indicates whether the events should be decoded ' +
              'or displayed as their raw values (i.e. as a one-hot vector or integer id).')
def visualize_training(model_type, dataset_path, config_filepath, steps, decode_events):
    '''
    Visualize how the model will train. This displays the input and expected output (features and labels) for each step
    given the dataset.

    '''

    config = composer.config.get(config_filepath or get_default_config())
    dataset = get_dataset(model_type, dataset_path, config, mode='train', use_generator=False, max_files=5, show_progress_bar=False)

    count = 0
    events = []
    if model_type == ModelType.MUSIC_RNN:
        for batch_x, batch_y in dataset:
            features = batch_x.numpy().reshape(-1)
            labels = batch_y.numpy().reshape(-1)

            assert features.shape == labels.shape
            for i in range(len(features)):
                if count == steps: break
                count += 1
                
                x, y = features[i], labels[i]
                if decode_events:
                    x = decode_to_event(config, x)
                    y = decode_to_event(config, y)

                events.append((x, y))
    
    input_header = 'Input sequence: '
    input_sequence = ', '. join(str(x) for x, _ in events) 
    output_header = 'Output sequence: '
    output_sequence = ', '. join(str(y) for _, y in events)

    divider_length = max(len(input_header) + len(input_sequence),  len(output_header) + len(output_sequence))
    print('‾' * divider_length)

    header_colourization = logging_utils.colourize_string('%s', logging_utils.colorama.Fore.GREEN)
    print('{}{}'.format(header_colourization % input_header, input_sequence))
    print('_' * divider_length)
    print('‾' * divider_length)
    print('{}{}'.format(header_colourization % output_header, output_sequence))

    print('_' * divider_length)
    
    for index, (x, y) in enumerate(events):
        print('Step {}'.format(index + 1))
        print(' - input:             {}'.format(x))
        print(' - expected output:   {}'.format(y))

def get_config_from_restoredir(restoredir):
    '''
    Gets the :class:`composer.config.ConfigInstance` object from a model checkpoint.

    :param restoredir:
        The directory of the model to restore.

    '''

    config_filepath = Path(restoredir) / 'config.yml'
    if not config_filepath.exists():
        logging.error('Failed to restore model from \'{}\'! Could not find \'config.yml\' file!'.format(restoredir))
        exit(1)
        
    return composer.config.get(config_filepath)

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('dataset-path')
@click.option('--logdir', default='./output/logdir/', help='The root log directory. Defaults to \'./output/logdir\'.')
@click.option('--restoredir', default=None, type=str, help='The directory of the model to continue training.')
@click.option('-c', '--config', 'config_filepath', default=None, 
              help='The path to the model configuration file. If unspecified, uses the default config for the model.' + 
              '\n\nIf a restoredir is specified, the configuration file in the restoredir is used instead (and this value is ignored).')
@click.option('-e', '--epochs', 'epochs', default=10, help='The number of epochs to train for. Defaults to 10.')
@click.option('--use-generator/--no-use-generator', default=False,
              help='Indicates whether the dataset should be loaded in chunks during processing ' +
              '(rather than into memory all at once). Defaults to False.')
@click.option('--max-files', default=None, help='The maximum number of files to load. Defaults to None, which means ' + 
              'that ALL files will be loaded.', type=int)
@click.option('--save-freq-mode', 'save_frequency_mode', type=EnumType(ModelSaveFrequencyMode, False),
              help='The units of the save frequency. Defaults to GLOBAL_STEP.', default='global_step')
@click.option('--save-freq', 'save_frequency', help='The frequency at which to save the model (in the units specified ' +
              'by the save frequency mode). Defaults to every 500 global steps.', type=int, default=500)
@click.option('--max-checkpoints', 'max_checkpoints', help='The maximum number of checkpoints to keep. Defaults to 3.',
              type=int, default=3)
@click.option('--show-progress-bar/--no-show-progress-bar', 'show_progress_bar', help='Indicates whether a progress bar ' +
              'will be shown to indicate epoch status. Defaults to True.', default=True)
def train(model_type, dataset_path, logdir, restoredir, config_filepath, epochs, 
          use_generator, max_files, save_frequency_mode, save_frequency,
          max_checkpoints, show_progress_bar):
    '''
    Trains the specified model.

    '''

    if restoredir is not None:
        config = get_config_from_restoredir(restoredir)
        model_logdir = None
    else:
        initial_epoch = 0
        model_logdir = Path(logdir) / '{}-{}'.format(model_type.name.lower(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        _CONFIG_COPY_FORMAT = '''
        #########################################################
        # Datetime: {datetime}.
        #########################################################
        # This is an autogenerated backup of the configuration file
        # used when invoking the train command.
        # 
        # DO NOT MODIFY THIS FILE!
        # Doing so may cause errors upon resuming training.
        #########################################################
        {config_source}
        '''

        # Remove indents caused by literal string formatting.
        _CONFIG_COPY_FORMAT = '\n'.join(line.strip() for line in _CONFIG_COPY_FORMAT.strip().split('\n'))

        model_logdir.mkdir(parents=True, exist_ok=True)
        config = composer.config.get(config_filepath or get_default_config())
        with open(config.filepath) as original_config_file, \
             open(model_logdir / 'config.yml', 'w+') as copy_config_file:
            
            out_config = _CONFIG_COPY_FORMAT.format(datetime=str(datetime.datetime.now()),
                            config_source=original_config_file.read())

            copy_config_file.write(out_config)

    model, _ = create_model(model_type, config)  

    input_shape = (get_batch_size(model_type, config), get_window_size(model_type, config))
    learning_rate = get_learning_rate(model_type, config)
    train_dataset = get_dataset(model_type, dataset_path, config, 'train', use_generator, max_files=max_files)
    model.train(
        train_dataset, input_shape, model_logdir, restoredir=restoredir, epochs=epochs,
        learning_rate=learning_rate, save_frequency_mode=save_frequency_mode,
        save_frequency=save_frequency, max_checkpoints=max_checkpoints,
        show_progress_bar=show_progress_bar
    )

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('dataset-path')
@click.argument('restoredir')
@click.option('--use-generator/--no-use-generator', default=False,
              help='Indicates whether the dataset should be loaded in chunks during processing ' +
              '(rather than into memory all at once). Defaults to False.')
@click.option('--max-files', default=None, help='The maximum number of files to load. Defaults to None, which means ' + 
              'that ALL files will be loaded.', type=int)
def evaluate(model_type, dataset_path, restoredir, use_generator, max_files):
    '''
    Evaluate the specified model.

    '''

    config = get_config_from_restoredir(restoredir)  
    model, _ = create_model(model_type, config)
    model.load_from_checkpoint(restoredir)

    model.compile(get_learning_rate(model_type, config))
    model.build(input_shape=(get_batch_size(model_type, config), None))

    test_dataset = get_dataset(model_type, dataset_path, config, 'test', use_generator, max_files=max_files, shuffle_dataset=False)
    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    logging.info('- Finished evaluating model. Loss: {:.4f}, Accuracy: {:.4f}'.format(loss, accuracy))

@cli.command()
@click.argument('model-type', type=EnumType(ModelType, False))
@click.argument('restoredir')
@click.argument('output-filepath')
@click.option('--prompt', '-p', 'prompt', default=None, help='The path of the MIDI file to prompt the network with. ' +
              'Defaults to None, meaning a random prompt will be created.')
@click.option('--prompt-length', default=10, help='Number of events to take from the start of the prompt. Defaults to 10.')
@click.option('--length', '-l', 'generate_length', default=1024, help='The length of the generated event sequence. Defaults to 1024')
@click.option('--temperature', default=1.0, help='Dictates how random the result is. Low temperature yields more predictable output. ' +
              'On the other hand, high temperature yields very random ("surprising") outputs. Defaults to 1.0.')
def generate(model_type, restoredir, output_filepath, prompt, prompt_length, generate_length, temperature):
    '''
    Generate a MIDI file.

    '''

    import tensorflow as tf

    config = get_config_from_restoredir(restoredir)
    model, _ = create_model(model_type, config)
    model.load_from_checkpoint(restoredir)

    model.compile(get_learning_rate(model_type, config))
    model.build(input_shape=(1, None))

    if prompt is None:
        raise NotImplementedError()

    prompt_note_sequence = NoteSequence.from_midi(prompt).trim_start()
    event_sequence = prompt_note_sequence.to_event_sequence(config.dataset.time_step_increment, \
                        config.dataset.max_time_steps, config.dataset.velocity_bins)

    event_sequence.events = event_sequence.events[:prompt_length]

    def _encode(event):
        return IntegerEncodedEventSequence.event_to_id(event.type, event.value, event_sequence.event_ranges, \
                event_sequence.event_value_ranges)

    def _decode(event_id):
        return IntegerEncodedEventSequence.id_to_event(event_id, event_sequence.event_ranges, \
                event_sequence.event_value_ranges)

    x = [_encode(event) for event in event_sequence.events]
    x = tf.expand_dims(x, 0)

    model.reset_states()
    for i in tqdm.tqdm(range(generate_length)):
        predictions = model(x)
        if model_type == ModelType.TRANSFORMER:
            # We only care about the first output (logits);
            # however, the transformer model outputs logits, presents, (hidden_states), (attentions).
            predictions = predictions[0]
        
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        x = tf.expand_dims([predicted_id], 0)
        event_sequence.events.append(_decode(predicted_id))

    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    event_sequence.to_note_sequence().to_midi(str(output_filepath))

@cli.command()
@click.argument('midi_filepath')
@click.option('--sf-path', 'soundfont_filepath', default=None,
              help='The filepath of the soundfont to use. If not specified, uses the default soundfont.')
@click.option('--sf-save-path', 'soundfont_save_path', default='data/soundfonts',
              help='The path to save the default soundfont to.')
@click.option('--chunk-size', 'chunk_size', default=32768,
              help='The number of bytes to download in a single chunk. Defaults to 32768.')
def synthesize(midi_filepath, soundfont_filepath, soundfont_save_path, chunk_size):
    '''
    Synthesize the specified MIDI file using a soundfont.

    The output WAVE file has the same name as the input MIDI file.

    '''

    DEFAULT_SOUNDFONT_GDRIVE_ID = '1md7ysI8JeLb6idc5ZX05_iOUTvgm_l-0'
    GDRIVE_DOWNLOAD_URL = 'https://drive.google.com/uc?export=download'

    if soundfont_filepath is None:
        soundfont_save_path = Path(soundfont_save_path)
        soundfont_save_path.mkdir(parents=True, exist_ok=True)
        soundfont_filepath = soundfont_save_path / 'default.sf2'
        if not soundfont_filepath.exists():
            logging.info('Downloading default soundfont...')

            session = requests.Session()
            response = session.get(GDRIVE_DOWNLOAD_URL, params={
                'id': DEFAULT_SOUNDFONT_GDRIVE_ID
            }, stream=True)
            
            token = next((v for k ,v in response.cookies.items() if k.startswith('download_warning')), None)
            if token:
                response = session.get(GDRIVE_DOWNLOAD_URL, params={
                    'id': DEFAULT_SOUNDFONT_GDRIVE_ID,
                    'confirm': token
                }, stream=True)

            with open(soundfont_filepath, 'wb+') as file_handle:
                total_length = response.headers.get('content-length')
                if total_length is None:
                    file_handle.write(response.content)
                else:
                    total_length = int(total_length)
                    with tqdm.tqdm(total=total_length) as bar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if not chunk: continue
                            
                            file_handle.write(chunk)
                            bar.update(len(chunk))

    # Check if fluidsynth exists...
    if which('fluidsynth') is None:
        logging.error('Could not find FluidSynth, which is required for synthesization using a soundfont.')
        exit(1)

    midi_filepath = Path(midi_filepath)
    output_filepath = midi_filepath.parent / (midi_filepath.stem + '.wav')
    subprocess.call([
        'fluidsynth', '-T', 'wav',
        '-F', str(output_filepath),
        '-ni', str(soundfont_filepath), str(midi_filepath)
    ])
