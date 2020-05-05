import array
import logging
import numpy as np
import tensorflow as tf
import composer.dataset.sequence as sequence

from enum import IntEnum
from abc import ABC, abstractmethod, abstractproperty
from composer.utils import parallel_process
from composer import ModelSaveFrequencyMode

class BaseModel(tf.keras.Model):
    '''
    A generic model interface class.

    '''

    @abstractmethod
    def train(self, dataset, input_shape, logdir, restoredir=None, epochs=None,
              learning_rate=1e-3, save_frequency_mode=ModelSaveFrequencyMode.EPOCH,
              save_frequency=1, max_checkpoints=1, checkpoint_name_format='model-{global_step}gs',
              show_progress_bar=True):
        '''
        Fit the model to the specified ``dataset``.

        :param dataset:
            An iterable object containing batched feature, label pairs (as tuples).
            The dataset shape should be (batch_size, window_size, feature_size).
        :param input_shape:
            The shape of the input data. Expects (batch_size, window_size).
        :param logdir:
            The root log directory.
        :param restoredir:
            The log directory of the model to continue training. If both ``logdir``
            and ``restoredir`` are specified, the ``restoredir`` will be used.
            Defaults to ``None``.
        :param epochs:
            The number of epochs to train for. Defaults to ``None``, meaning
            that the model will train indefinitely.
        :param learning_rate:
            The initial learning rate of the optimizer. Defaults to 1e-3.
        :param save_frequency_mode:
            A :class:`composer.models.ModelSaveFrequency` indicating the units of 
            the model save frequency. This can also be a string value corresponding
            to the enum value. Defaults to :class:`composer.ModelSaveFrequencyMode.EPOCH`.
        :param save_frequency:
            How often the model should be saved in units specified by the 
            `save_frequency_mode` parameter. Defaults to 1.
        :param max_checkpoints:
            The maximum number of checkpoints to keep. Defaults to 1.
        :param checkpoint_name_format:
            The format of the model checkpoint name. This can either be a string
            value or a method that takes in the current epoch and current global step
            and returns a string representing the checkpoint name.
            The following formatting keys are supported:
                * epochs: the current epoch (starts at 1).
                * global_step: the current global step (starts at 1).
        :param show_progress_bar:
            Indicates whether a progress bar will be shown to indicate epoch status.
            Defaults to ``True``.

        '''

        raise NotImplementedError()

    def load_from_checkpoint(self, restoredir):
        '''
        Loads a model from a checkpoint in the specified ``restoredir``.

        :param restoredir:
            The log directory to restore the checkpoint from.

        '''

        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), epoch=tf.Variable(1), model=self)

        # Restore the model, if exists
        try:
            checkpoint_path = tf.train.latest_checkpoint(str(restoredir))
            checkpoint.restore(checkpoint_path).expect_partial()
            logging.info('{} model restored from \'{}\' (global_step={}, epoch={}).'.format(
                self.__class__.__name__, checkpoint_path,
                checkpoint.step.numpy(), checkpoint.epoch.numpy()
            ))
        except:
            logging.exception('Failed to restore model from \'{}\''.format(restoredir))
            exit(1)

from composer.models.music_rnn import MusicRNN
from composer.models.transformer import Transformer

class EventEncodingType(IntEnum):
    '''
    The way that events should be encoded in a model.

    :cvar INTEGER:
        Indicates that events should be encoded as integer ids.
    :cvar ONE_HOT:
        Indicates that events should be encoded as one-hot vectors.
    
    '''

    INTEGER = 0
    ONE_HOT = 1

def _get_events_from_file(filepath, input_event_encoding):
    '''
    Gets all events from a file.

    :param filepath:
        A Path-like object representing the filepath of the encoded event sequence.
    :param input_event_encoding:
        The way that events should be encoded before being inputted into the network.

    '''

    if input_event_encoding == EventEncodingType.INTEGER:
        data, _, _, _ = sequence.IntegerEncodedEventSequence.event_ids_from_file(filepath)
    elif input_event_encoding == EventEncodingType.ONE_HOT:
        data, _, _, _ = sequence.IntegerEncodedEventSequence.one_hot_from_file(filepath, \
                            as_numpy_array=True, numpy_dtype=np.float)

    return data

def _get_events_from_file_as_generator(filepath, input_event_encoding):
    '''
    Gets all events from a file as a generator.

    :param filepath:
        A Path-like object representing the filepath of the encoded event sequence.
    :param input_event_encoding:
        The way that events should be encoded before being inputted into the network.

    '''
    
    if input_event_encoding == EventEncodingType.INTEGER:
        generator = sequence.IntegerEncodedEventSequence.event_ids_from_file_as_generator(filepath)
    elif input_event_encoding == EventEncodingType.ONE_HOT:
        generator = sequence.IntegerEncodedEventSequence.one_hot_from_file_as_generator(filepath, \
                            as_numpy_array=True, numpy_dtype=np.float)

    return generator

def _generator(filepaths, input_event_encoding):
        '''
        The generator function for loading the dataset.

        '''

        for filepath in filepaths:
            # TensorFlow automatically converts string arguments to bytes so we need to decode back to strings.
            filepath = bytes(filepath).decode('utf-8')
            events_generator = _get_events_from_file_as_generator(filepath, input_event_encoding)
            for event in events_generator:
                yield event

def load_events(filepaths, use_generator=True, show_loading_progress_bar=True, 
                input_event_encoding=EventEncodingType.INTEGER):
    '''
    Loads event sequences as :class:`composer.dataset.sequence.Event` objects
    containing in a :class:`tensorflow.data.Dataset`.

    :param filepaths:
        An array-like object of Path-like objects representing the filepaths of the encoded event sequences.
    :param use_generator:
        Indicates whether the Dataset should be given as a generator object. Defaults to ``True``.
        
        This is used when the dataset cannot be loaded entirely into memory. If it can, we recommend 
        that ``use_generator`` be set to ``False`` since it will load faster; otherwise, leave it as ``True``.
    :param show_loading_progress_bar:
        Indicates whether a loading progress bar should be displayed while the dataset is loaded into memory.
        Defaults to ``True``.

        The progress bar will only be displayed if ``use_generator`` is ``False`` (since no dataset loading
        will occur in this function if ``use_generator`` is ``True``).
    :param input_event_encoding:
        A :class:`EventEncodingType` representing the way that events should be
        encoded before being inputted into the network.
        
        If set to :attr:`EventEncodingType.ONE_HOT`, the input event sequences will
        be encoded as a series of one-hot vectors—their dimensionality determined by the value ranges 
        on the :class:`composer.dataset.sequence.EventSequence`.

        If set to :attr:`EventEncodingType.INTEGER`, the input event sequences will
        be encoded as a series of integer ids representing each event. These are fundmenetally similar 
        to the one-hot vector representation. The integer id of an event is the zero-based index of the 
        "hot" (active) bit of its one-hot vector representation.

        Defaults to :attr:`EventEncoding.INTEGER`. Due to the size a single one-hot
        vector, loading the dataset will take longer than compared to integer ids.
    :returns:
        A :class:`tensorflow.data.Dataset` object representing the dataset.

    '''

    import tensorflow as tf

    if use_generator:
        # Convert filepaths to strings because TensorFlow cannot handle Pathlib objects.
        filepaths = [str(path) for path in filepaths]
        if input_event_encoding == EventEncodingType.ONE_HOT:
            # To determine the input and output dimensions, we load up a file as if we were
            # loading it into the dataset object. We use its shape to determine the dimensions.
            # This has the disadvantage of requiring an extra (and unnecessary) load operation;
            # however, the advantage is we don't have to hard code our shapes reducing the potential
            # points of error and thus making our code more maintainable.
            _example = next(_get_events_from_file_as_generator(filepaths[0], input_event_encoding))
            if len(_example.shape) > 0:
                output_shapes = (_example.shape[-1],)
            else:
                raise Exception('Failed to load dataset as one-hot encoded events. Expected non-empty shape but got {}.'.format(_example.shape))

            ouput_types = tf.float64
        else:
            output_shapes = ()
            output_types = tf.int16

        # Create the TensorFlow dataset object
        dataset = tf.data.Dataset.from_generator(
            _generator,
            output_types=output_types,
            output_shapes=output_shapes,
            args=(filepaths, int(input_event_encoding))
        )
    else:
        _loader_func = lambda filepath: _get_events_from_file(filepath, input_event_encoding)
        logging.info('- Loading dataset (\'{}\') into memory.'.format(filepaths[0].parent))
        data = parallel_process(filepaths, _loader_func, multithread=True, n_jobs=16, front_num=0, 
                                show_progress_bar=show_loading_progress_bar, extend_result=True, initial_value=array.array('H'))

        dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset

def load_dataset(filepaths, batch_size, window_size, use_generator=True, 
                 show_loading_progress_bar=True, prefetch_buffer_size=2,
                 input_event_encoding=EventEncodingType.INTEGER, shuffle=True):
    '''
    Loads a dataset for use.

    :note:
        An input sequence consists of integers representing each event
        and the output sequence is an event encoded depending on the value
        of ``input_event_encoding``.

    :param filepaths:
        An array-like object of Path-like objects representing the filepaths of the encoded event sequences.
    :param batch_size:
        The number of samples to include a single batch.
    :param window_size:
        The number of events in a single input sequence.
    :param use_generator:
        Indicates whether the Dataset should be given as a generator object. Defaults to ``True``.
        
        This is used when the dataset cannot be loaded entirely into memory. If it can, we recommend 
        that ``use_generator`` be set to ``False`` since it will load faster; otherwise, leave it as ``True``.
    :param prefetch_buffer_size:
        The number of batches to prefetch during processing. Defaults to 2.
        This means that 2 batches will be prefetched while the current element is being processed.

        Prefetching is only used if ``use_generator`` is ``True``.
    :param show_loading_progress_bar:
        Indicates whether a loading progress bar should be displayed while the dataset is loaded into memory.
        Defaults to ``True``.

        The progress bar will only be displayed if ``use_generator`` is ``False`` (since no dataset loading
        will occur in this function if ``use_generator`` is ``True``).
    :param input_event_encoding:
        A :class:`EventEncodingType` representing the way that events should be
        encoded before being inputted into the network.
        
        If set to :attr:`EventEncodingType.ONE_HOT`, the input event sequences will
        be encoded as a series of one-hot vectors—their dimensionality determined by the value ranges 
        on the :class:`composer.dataset.sequence.EventSequence`.

        If set to :attr:`EventEncodingType.INTEGER`, the input event sequences will
        be encoded as a series of integer ids representing each event. These are fundmenetally similar 
        to the one-hot vector representation. The integer id of an event is the zero-based index of the 
        "hot" (active) bit of its one-hot vector representation.

        Defaults to :attr:`EventEncoding.INTEGER`. Due to the size a single one-hot
        vector, loading the dataset will take longer than compared to integer ids.
    :param shuffle:
        Indicates whether the dataset should be shuffled. Defaults to ``True``.
    :returns:
        A :class:`tensorflow.data.Dataset` object representing the dataset.

    '''
    
    import tensorflow as tf

    dataset = load_events(filepaths, use_generator, show_loading_progress_bar, input_event_encoding)

    # Split our dataset into sequences.
    # The input consists of window_size number of events, and the output consists of the same sequence but shifted
    # over by one timestep. In other words, each event in the sequence represents one timestep, t, and the output is
    # the event at the next timestep, t + 1. 
    dataset = dataset.batch(window_size + 1, drop_remainder=True).map(lambda x: (x[:-1], x[1:]), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        # We make the shuffle buffer big enough to fit 500 batches. After that, it will have to reshuffle.
        shuffle_buffer_size = 500 * batch_size
        dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if use_generator:
        # We only need prefetching if all the data is NOT loaded into memory
        # (i.e. when we use a generator that loads as we go).
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset

def load_tfrecord_dataset(filepath, shuffle=True):
    '''
    Loads a dataset from a TFRecord.

    :note:
        This dataset is already split into batches and sequences based
        on the config used during exporting.

    :param filepath:
        A Path-like object representing the path to the TFRecord file.
    :param shuffle:
        Indicates whether the dataset should be shuffled. Defaults to ``True``.
    :returns:
        A :class:`tensorflow.data.Dataset` object representing the dataset,
        and a ``dict`` representing the header metadata contents.

    '''

    import tensorflow as tf

    # Convert filepaths to strings because TensorFlow cannot handle Pathlib objects.
    dataset = tf.data.TFRecordDataset([str(filepath)])

    # The first example contains the metadata...
    encoded_header = next(dataset.take(1).as_numpy_iterator())
    header_example = tf.io.parse_single_example(encoded_header, {
        'model_type': tf.io.FixedLenFeature([], tf.string),
        'batch_size': tf.io.FixedLenFeature([], tf.int64),
        'window_size': tf.io.FixedLenFeature([], tf.int64)
    })

    header_example['model_type'] = header_example['model_type'].numpy().decode('utf-8')
    batch_size = header_example['batch_size'] = header_example['batch_size'].numpy()
    window_size = header_example['window_size'] = header_example['window_size'].numpy()

    target_shape = tf.TensorShape([batch_size, window_size])
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_function(proto):
        example = tf.io.parse_single_example(proto, feature_description)
        x = tf.reshape(tf.io.parse_tensor(example['x'], tf.int32), target_shape)
        y = tf.reshape(tf.io.parse_tensor(example['y'], tf.int32), target_shape)
        return x, y 
    
    dataset = dataset.skip(1).map(_parse_function)

    if shuffle:
        # We make the shuffle buffer big enough to fit 500 batches. After that, it will have to reshuffle.
        shuffle_buffer_size = 500 * batch_size
        dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    return dataset, header_example