'''
MusicRNN: A recurrent neural network model designed to generate music based on an
MIDI-like event-based description language (see: ``composer.dataset.sequence`` for
more information about the event-based sequence description.)

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

class MusicRNN(Model):
    '''
    A recurrent neural network model designed to generate music based on an
    MIDI-like event-based description language.
    
    :note:
        See ``composer.dataset.sequence`` for more information about the 
        event-based sequence description.
    '''

    def __init__(self, event_dimensions, batch_size, embedding_size, lstm_layers_count,
                 lstm_layer_sizes, lstm_dropout_probability, use_batch_normalization=True):

        '''
        Initialize an instance of :class:`MusicRNN`.

        :param input_event_dimensions:
            An integer denoting the dimensions of a single feature (i.e. the size of an event sequence).
            The network takes in a sequence of these events and outputs an event in the form of a sequence
            of the same size denoting the next event in the sequence.
        :param batch_size:
            The number of events in a single batch.
        :param embedding_size:
            The number of units in the embedding layer.
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

        self.embedding_layer = layers.Embedding(event_dimensions, embedding_size, batch_input_shape=(batch_size, None))
        self.lstm_layers = []
        self.dropout_layers = []
        self.normalization_layers = []]
        for i in range(lstm_layers_count):
            self.lstm_layers.append(layers.LSTM(lstm_layer_sizes[i], return_sequences=True, 
                                    stateful=True, recurrent_initializer='glorot_uniform'))

            if lstm_dropout_probability[i] > 0:
                self.dropout_layers.append(layers.Dropout(lstm_dropout_probability[i]))

            if use_batch_normalization:
                self.normalization_layers.append(layers.BatchNormalization())

        self.use_normalization_layers = len(self.normalization_layers) == self.lstm_layers
        self.use_dropout_layers = len(self.dropout_layers) == self.lstm_layers
        self.output_layer = layers.Dense(event_dimensions)

    def call(self, inputs):
        '''
        Feed forward call on this network.

        :param inputs:
            The inputs to the network. This should be an array-like object containing
            sequences (of size :var:`MusicRNN.window_size) of :var:MusicRNN.event_dimensions`
            -dimensionalone-hot vectors representing the events.

        '''

        inputs = self.embedding_layer(inputs)
        for i in range(self.lstm_layers_count):
            # Apply LSTM layer
            inputs = self.lstm_layers[i](inputs)

            if self.use_dropout_layers:
                inputs = self.dropout_layers[i](inputs)
            
            if self.use_normalization_layers:
                inputs = self.normalization_layers[i](inputs)

        inputs = self.output_layer(inputs)   
        return inputs