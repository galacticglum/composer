import logging
import numpy as np
import tensorflow as tf
import composer.dataset.sequence as sequence

from enum import Enum
from composer.utils import parallel_process
from composer.models.music_rnn import MusicRNN

class EventEncodingType(Enum):
    '''
    The way that events should be encoded in a model.
    
    '''

    INTEGER = 0
    ONE_HOT = 1

def load_dataset(filepaths, batch_size, window_size, use_generator=False, 
                 show_loading_progress_bar=True, prefetch_buffer_size=2,
                 input_event_encoding=EventEncodingType.ONE_HOT):
    '''
    Loads a dataset for use.

    :note:
        An input sequence consists of integers representing each event
        and the output sequence is an event encoded as a one-hot vector.

    :param filepaths:
        An array-like object of Path-like objects representing the filepaths of the encoded event sequences.
    :param batch_size:
        The number of samples to include a single batch.
    :param window_size:
        The number of events in a single input sequence.
    :param use_generator:
        Indicates whether the Dataset should be given as a generator object. Defaults to ``False``.
    :param prefetch_buffer_size:
        The number of batches to prefetch during processing. Defaults to 2. This means that 2 batches will be
        prefetched while the current element is being processed.

        Prefetching is only used if ``use_generator`` is ``True``.
    :param show_loading_progress_bar:
        Indicates whether a loading progress bar should be displayed while the dataset is loaded into memory.
        Defaults to ``True``.

        The progress bar will only be displayed if ``use_generator`` is ``False`` (since no dataset loading
        will occur in this function if ``use_generator`` is ``True``).
    :param input_event_encoding:
        A :class:`composer.dataset.EventEncodingType` representing the way that events should be
        encoded before being inputted into the network.
        
        If set to :var:`composer.dataset.EventEncodingType.ONE_HOT`, the input event sequences will
        be encoded as a series of one-hot vectors—their dimensionality determined by the value ranges 
        on the :class:`composer.dataset.sequence.EventSequence`.

        If set to :var:`composer.dataset.EventEncodingType.INTEGER`, the input event sequences will
        be encoded as a series of integer ids representing each event. These are fundmenetally similar 
        to the one-hot vector representation. The integer id of an event is the zero-based index of the 
        "hot" (active) bit of its one-hot vector representation.

        Defaults to :var:`composer.dataset.EventEncoding.ONE_HOT`. Due to the size a single one-hot
        vector, loading the dataset will take longer than compared to integer ids.
    :returns:
        A :class:`tensorflow.data.Dataset` object representing the dataset and 
        a two-dimensional tuple of integers representing the dimensionsof a 
        single feature and label in the dataset.

    '''

    def _encode_event_as_int(event, event_ranges):
        '''
        Encodes a :class:`composer.dataset.sequence.Event` as integer.

        '''

        return event_ranges[event.type].start + (event.value or 0)

    def _get_sequences_from_file(filepath, window_size, input_event_encoding):
        '''
        Gets all sequences (of size ``window_size``) from a file.

        :param filepath:
            A Path-like object representing the filepath of the encoded event sequence.
        :param window_size:
            The number of events in a single input sequence.
        :param input_event_encoding:
            The way that events should be encoded before being inputted into the network.

        '''
        
        if input_event_encoding == EventEncodingType.INTEGER:
            data, event_value_ranges, event_ranges, sequence_settings = \
                sequence.IntegerEncodedEventSequence.event_ids_from_file(filepath, as_numpy_array=True)
            
            data = data.reshape((len(data), 1))
        elif input_event_encoding == EventEncodingType.ONE_HOT:
            data, event_value_ranges, event_ranges, sequence_settings = \
                sequence.IntegerEncodedEventSequence.one_hot_from_file(filepath, as_numpy_array=True)
        
        # The number of events that we can extract from the sample.
        # While every input sequence only contains window_size number of 
        # events, we also need an additional event for the output (label).
        # Therefore, we pull window_size + 1 events.
        extract_events_count = window_size + 1
        sequence_count = len(data) // extract_events_count
        for i in range(sequence_count):
            start, end = i * extract_events_count, extract_events_count * (i + 1)

            if input_event_encoding == EventEncodingType.ONE_HOT:
                x = data[start:end - 1]
                y = data[end - 1]
            else:
                x = data[start:end - 1]

                # We need to convert the event id to an Event object since it is
                # required by OneHotEncodedEventSequence.event_as_one_hot_vector.
                event = sequence.IntegerEncodedEventSequence.id_to_event(data[end - 1][0], event_ranges, event_value_ranges)
                y = sequence.OneHotEncodedEventSequence.event_as_one_hot_vector(event, event_ranges, event_value_ranges, as_numpy_array=True)

            yield x, y

    def _generator(filepaths, window_size):
            '''
            The generator function for loading the dataset.

            '''

            for filepath in filepaths:
                # TensorFlow automatically converts string arguments to bytes so we need to decode back to strings.
                filepath = bytes(filepath).decode('utf-8')
                for x, y in _get_sequences_from_file(filepath, window_size):
                    yield x, y

    # To determine the input and output dimensions, we load up a file as if we were
    # loading it into the dataset object. We use its shape to determine the dimensions.
    # This has the disadvantage of requiring an extra (and unnecessary) load operation;
    # however, the advantage is we don't have to hard code our shapes reducing the potential
    # points of error and thus making our code more maintainable. Besides, since 
    # _get_sequences_from_file is a generator, we only have to decode window_size number of events.
    example_sequence = next(_get_sequences_from_file(filepaths[0], window_size, input_event_encoding))
    input_event_dimensions = example_sequence[0].shape[-1]
    output_event_dimensions = example_sequence[1].shape[-1]

    shuffle_buffer_size = 50 * batch_size
    if use_generator:
        # Convert filepaths to strings because TensorFlow cannot handle Pathlib objects.
        filepaths = [str(path) for path in filepaths]

        # Create the TensorFlow dataset object
        dataset = tf.data.Dataset.from_generator(
            _generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((window_size, input_event_dimensions), (output_event_dimensions,)),
            args=(filepaths, window_size, input_event_encoding)
        )
    else:
        _loader_func = lambda filepath: list(_get_sequences_from_file(filepath, window_size, input_event_encoding))
        logging.info('- Loading dataset (\'{}\') into memory.'.format(filepaths[0].parent))
        data = parallel_process(filepaths, _loader_func, multithread=True, n_jobs=16,
                                front_num=0, show_progress_bar=show_loading_progress_bar)
        
        # Generator function that flattens a list of lists into a single list.
        # For example, suppose x = [[(1, 1), (2, 2)], [(3, 3)]].
        # The output of the flatten function is [(1, 1), (2, 2), (3, 3)].
        flatten = lambda x: [item for sublist in x for item in sublist]
        data = flatten(data)

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            output_types=(tf.float32, tf.float32),
            output_shapes=((window_size, input_event_dimensions), (output_event_dimensions,))
        )

        shuffle_buffer_size = len(data) * 2

    # Apply shuffling and batching
    dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True) \
                     .batch(batch_size, drop_remainder=True)

    if use_generator:
        # We only need prefetching if all the data is NOT loaded into memory
        # (i.e. when we use a generator that loads as we go).
        dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset, (input_event_dimensions, output_event_dimensions)