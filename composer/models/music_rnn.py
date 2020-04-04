'''
MusicRNN: A recurrent neural network model designed to generate music based on an
MIDI-like event-based description language (see: ``composer.dataset.sequence`` for
more information about the event-based sequence description.)

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from composer.dataset.sequence import EventSequence

class MusicRNN(Model):
    '''
    A recurrent neural network model designed to generate music based on an
    MIDI-like event-based description language.
    
    :note:
        See ``composer.dataset.sequence`` for more information about the 
        event-based sequence description.
    '''

    def __init__(self, event_dimensions, window_size, lstm_layers_count, lstm_layer_sizes, 
                 lstm_dropout_probability, dense_layer_size=256, use_batch_normalization=True):

        '''
        Initialize an instance of :class:`MusicRNN`.

        :param event_dimensions:
            An integer denoting the size of the event one-hot vector. The network takes in a sequence 
            of these one-hot vectors and outputs a one-hot vector denoting the next event in the sequence.
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
        :param dense_layer_size:
            The number of units in the hidden :class:`tensorflow.keras.layers.Dense` layer preceeding the output.
            Defaults to 256.
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
            lstm_layer = layers.LSTM(lstm_layer_sizes[i], input_shape=(window_size, event_dimensions), 
                                     return_sequences=i < lstm_layers_count - 1)

            dropout_layer = layers.Dropout(lstm_dropout_probability[i])

            if use_batch_normalization:
                self.normalization_layers.append(layers.BatchNormalization())

            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(dropout_layer)
        
        self.dense_layer = layers.Dense(dense_layer_size, activation='relu')
        self.output_layer = layers.Dense(event_dimensions, activation='softmax')

    def call(self, inputs):
        '''
        Feed forward call on this network.

        :param inputs:
            The inputs to the network. This should be an array-like object containing
            sequences (of size :var:``MusicRNN.window_size``) of :var:``MusicRNN.event_dimensions``
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

        # inputs = self.dense_layer(inputs)
        inputs = self.output_layer(inputs)

        return inputs

def create_music_rnn_dataset(filepaths, batch_size, window_size):
    '''
    Creates a dataset for the :class:`MusicRNN` model.

    :param filepaths:
        An array-like object of Path-like objects representing the filepaths of the encoded event sequences.
    :param batch_size:
        The number of samples to include a single batch.
    :param window_size:
        The number of events in a single input sequence.
    :returns:
        A :class:`tensorflow.data.BatchDataset` object representing the dataset in batches.

    '''

    def _generator(filepaths, window_size):
            '''
            The generator function for loading the dataset.

            '''

            for filepath in filepaths:
                # TensorFlow automatically converts string arguments to bytes so we need to decode back to strings.
                decoded_filepath = bytes(filepath).decode('utf-8')
                one_hot_vectors = EventSequence.from_file(decoded_filepath).to_one_hot_encoding()
                
                # The number of one-hot vectors that we can extract from the sample.
                # While every input sequence only contains window_size number of one-hot vectors,
                # we also need an additional one-hot vector for the output (label).
                # Therefore, we pull window_size + 1 elements.
                extract_events_count = window_size + 1
                event_count = len(one_hot_vectors.vectors) // extract_events_count
                for i in range(event_count):
                    start, end = i * extract_events_count, extract_events_count * (i + 1)
                    vectors = one_hot_vectors.vectors[start:end]

                    # Split the vectors into inputs (x) and outputs (y)
                    x, y = np.asarray(vectors[:-1]), np.asarray(vectors[-1])

                    yield x, y

    # Create the tensorflow dataset object
    filepaths = [str(path) for path in filepaths]
    
    # Load an event sequence to get the one-hot vector size.
    one_hot_vector_size = EventSequence.from_file(filepaths[0]).to_one_hot_encoding().one_hot_size
    return tf.data.Dataset.from_generator(
        _generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=((window_size, one_hot_vector_size), (one_hot_vector_size,)),
        args=(filepaths, window_size)
    ).prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size)