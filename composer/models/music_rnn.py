'''
MusicRNN: A recurrent neural network model designed to generate music based on an
MIDI-like event-based description language (see: ``composer.dataset.sequence`` for
more information about the event-based sequence description.)

'''

import logging
import numpy as np
import tensorflow as tf
import composer.dataset.sequence as sequence
from tensorflow.keras import Model, layers
from composer.utils import parallel_process

class MusicRNN(Model):
    '''
    A recurrent neural network model designed to generate music based on an
    MIDI-like event-based description language.
    
    :note:
        See ``composer.dataset.sequence`` for more information about the 
        event-based sequence description.
    '''

    def __init__(self, input_event_dimensions, output_event_dimensions, window_size, lstm_layers_count, 
                 lstm_layer_sizes, lstm_dropout_probability, use_batch_normalization=True):

        '''
        Initialize an instance of :class:`MusicRNN`.

        :param input_event_dimensions:
            An integer denoting the dimensions of a single feature (i.e. the size of an event sequence).
            The network takes in a sequence of these events and outputs an event denoting the next event in the sequence.
        :param output_event_dimensions:
            An integer denoting the dimensions of a single label (i.e. size of the output event sequence).
            The network takes in a sequence of input events and outputs one of these events denoting the next event in the sequence.
        :param window_size:
            The number of events (input sequences) to use to predict.
        :param lstm_layers_count:
            The number of LSTM layers.
        :param lstm_layer_sizes:
            The number of units in each LSTM layer. If this value is an integer, all LSTM layers 
            in the model will have the same size; otheriwse, it should be an array-like object
            (equal in size to the ``lstm_layers_count`` parameter) that denotes the size of each
            LSTM layer in the network.
        :param lstm_dropout_probability:
            The probability (from 0 to 1) of dropping out an LSTM unit. If this value is a number,
            the dropout will be uniform across all layers; otherwise, it should an array-like object
            (equal in size to the ``lstm_layers_count`` parameter) that denotes the dropout probability
            per LSTM layer in the network.
        :param use_batch_normalization:
            Indicates whether each LSTM layer should be followed by a :class:`tensorflow.keras.layers.BatchNormalization`
            layer. Defaults to ``True``. 
        :note:
            This sets up the model architecture and layers.

        '''

        super().__init__(name='music_rnn')
        self.lstm_layers_count = lstm_layers_count

        # Make sure that the layer sizes and dropout probabilities
        # are sized appropriately (if they are not scalar values).
        if not np.isscalar(lstm_layer_sizes):
            assert len(lstm_layer_sizes) == lstm_layers_count
        else:
            # Convert lstm_layer_sizes to a numpy array of uniform elements.
            lstm_layer_sizes = np.full(lstm_layers_count, lstm_layer_sizes)

        if not np.isscalar(lstm_dropout_probability):
            assert len(lstm_dropout_probability) == lstm_layers_count
        else:
            # Convert lstm_dropout_probability to a numpy array of uniform elements.
            lstm_dropout_probability = np.full(lstm_layers_count, lstm_dropout_probability)

        self.lstm_layers = []
        self.dropout_layers = []
        # The batch normalization layers. If None, this means that we won't use them.
        self.normalization_layers = None if not use_batch_normalization else []
        for i in range(lstm_layers_count):
            # To stack the LSTM layers, we have to set the return_sequences parameter to True
            # so that the layers return a 3-dimensional output representing the sequences.
            # All layers but the last one should do this.
            #
            # The input shape of an LSTM layer is in the form of (batch_size, time_steps, sequence_length); however,
            # the input_shape kwarg only needs (time_steps, sequence_length) since batch_size can be inferred at runtime.
            # Time steps refers to how many input sequences there are, and sequence_length is the number of units in one
            # input sequence. In our case, since the input is a one-hot vector, a single input sequence is the size of
            # this vector. Time steps is the number of one-hot vectors that we are passing in.
            self.lstm_layers.append(layers.LSTM(lstm_layer_sizes[i], input_shape=(window_size, input_event_dimensions), 
                                    return_sequences=i < lstm_layers_count - 1) )
            self.dropout_layers.append(layers.Dropout(lstm_dropout_probability[i]))

            if use_batch_normalization:
                self.normalization_layers.append(layers.BatchNormalization())


        self.output_layer = layers.Dense(output_event_dimensions, activation='softmax')

    def call(self, inputs):
        '''
        Feed forward call on this network.

        :param inputs:
            The inputs to the network. This should be an array-like object containing
            sequences (of size :var:`MusicRNN.window_size) of :var:MusicRNN.event_dimensions`
            -dimensionalone-hot vectors representing the events.

        '''

        for i in range(self.lstm_layers_count):
            # Apply LSTM layer
            inputs = self.lstm_layers[i](inputs)
            
            # Apply dropout layer
            inputs = self.dropout_layers[i](inputs)

            # If we are using batch normalization, apply it!
            if self.normalization_layers is not None:
                inputs = self.normalization_layers[i](inputs)

        inputs = self.output_layer(inputs)

        return inputs

def create_music_rnn_dataset(filepaths, batch_size, window_size, use_generator=False, 
                             show_loading_progress_bar=True, prefetch_buffer_size=2):
    '''
    Creates a dataset for the :class:`MusicRNN` model.

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

    def _get_sequences_from_file(filepath, window_size):
        '''
        Gets all sequences (of size ``window_size``) from a file.

        :param filepath:
            A Path-like object representing the filepath of the encoded event sequence.
        :param window_size:
            The number of events in a single input sequence.

        '''
        
        event_ids, event_value_ranges, event_ranges, sequence_settings = \
            sequence.IntegerEncodedEventSequence.event_ids_from_file(filepath, as_numpy_array=True)
        
        event_ids = event_ids.reshape((len(event_ids), 1))

        # The number of events that we can extract from the sample.
        # While every input sequence only contains window_size number of 
        # events, we also need an additional event for the output (label).
        # Therefore, we pull window_size + 1 events.
        extract_events_count = window_size + 1
        sequence_count = len(event_ids) // extract_events_count
        for i in range(sequence_count):
            start, end = i * extract_events_count, extract_events_count * (i + 1)
            x = event_ids[start:end-1]

            # We need to convert the event id to an Event object since it is
            # required by OneHotEncodedEventSequence.event_as_one_hot_vector.
            event = sequence.IntegerEncodedEventSequence.id_to_event(event_ids[end - 1][0], event_ranges, event_value_ranges)
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

    # The input event sequence consists of a single feature: an integer representing the event.
    input_event_dimensions = 1
    # Load an event sequence to get the output event dimensions.
    output_event_dimensions = sequence.EventSequence.from_file(filepaths[0]).to_one_hot_encoding().one_hot_size

    if use_generator:
        # Convert filepaths to strings because TensorFlow cannot handle Pathlib objects.
        filepaths = [str(path) for path in filepaths]

        # Create the TensorFlow dataset object
        dataset = tf.data.Dataset.from_generator(
            _generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=((window_size, input_event_dimensions), (output_event_dimensions,)),
            args=(filepaths, window_size)
        ).shuffle(50 * batch_size, reshuffle_each_iteration=True)
    else:
        _loader_func = lambda filepath: list(_get_sequences_from_file(filepath, window_size))
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
        ).shuffle(len(data) * 2, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    if use_generator:
        dataset = dataset.prefetch(prefetch_buffer_size)
    
    return dataset, (input_event_dimensions, output_event_dimensions)