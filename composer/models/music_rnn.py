'''
MusicRNN: A recurrent neural network model designed to generate music based on an
MIDI-like event-based description language (see: :mod:`composer.dataset.sequence` for
more information about the event-based sequence description.)

'''

import numpy as np
import tensorflow as tf
from composer.models import BaseModel, ModelSaveFrequencyMode
from tensorflow.keras import Model, Input, layers

def _get_rnn_model(event_vocab_size, batch_size, embedding_size, lstm_layers_count,
                   lstm_layer_sizes, lstm_dropout_probability, use_batch_normalization=True):
    '''
    Build the MusicRNN model.

    :param event_vocab_size:
        The size of the MIDI-like event-based description vocabulary.
        This is the dimensionality of a one-hot vector encoded representation of an event.
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

    '''

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

    inputs = Input(batch_shape=(batch_size, None))
    embedding_layer = layers.Embedding(event_vocab_size, embedding_size)
    
    x = embedding_layer(inputs)
    for i in range(lstm_layers_count):
        lstm_layer = layers.LSTM(lstm_layer_sizes[i], return_sequences=True, 
                                 stateful=True, recurrent_initializer='glorot_uniform')

        x = lstm_layer(x)
        if lstm_dropout_probability[i] > 0:
            dropout_layer = layers.Dropout(lstm_dropout_probability[i])
            x = dropout_layer(x)

        if use_batch_normalization:
            layer_normalization = layers.BatchNormalization()
            x = layer_normalization(x)
    
    output_layer = layers.Dense(event_vocab_size)
    outputs = output_layer(x)

    return Model(inputs=inputs, outputs=outputs, name='music_rnn')

class MusicRNN(BaseModel):
    '''
    A recurrent neural network model designed to generate music based on an
    MIDI-like event-based description language.
    
    :note:
        See :mod:`composer.dataset.sequence` for more information about the 
        event-based sequence description.

    :ivar embedding_layer:
        An instance of :class:`tensorflow.keras.layers.Embedding` representing the
        embedding layer that converts integer event ids to dense float vectors.
    :ivar lstm_layers:
        A list of :class:`tensorflow.keras.layers.LSTM` layers representing each hidden
        layer of the RNN.
    :ivar dropout_layers:
        A list of :class:`tensorflow.keras.layers.Dropout` layers representing each dropout
        layer after a hidden LSTM layer.
        
        If this layer is empty or :attr:`MusicRNN.use_dropout_layers` is `False`,
        the dropout layers are not used by the :meth:`MusicRNN.call` method.
    :ivar normalization_layers:
        A list of :class:`tensorflow.keras.layers.BatchNormalization` layers representing
        each normalization layer after a hidden LSTM layer.

        If this layer is empty or :attr:`MusicRNN.use_normalization_layers` is `False`,
        the normalization layers are not used by the :meth:`MusicRNN.call` method.
    :ivar output_layer:
        An instance of :class:`tensorflow.keras.layers.Dense` representing the final fully-connected
        layer of the RNN. The output of this layer is a probability distribution among each
        event id, denoting the probability of each event occuring next.

    '''

    def __init__(self, event_vocab_size, batch_size, embedding_size, lstm_layers_count,
                 lstm_layer_sizes, lstm_dropout_probability, use_batch_normalization=True):

        '''
        Initialize an instance of :class:`MusicRNN`.

        :param event_vocab_size:
            The size of the MIDI-like event-based description vocabulary.
            This is the dimensionality of a one-hot vector encoded representation of an event.
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

        '''

        # Build the keras model
        self.model = _get_rnn_model(
            event_vocab_size, batch_size, embedding_size, lstm_layers_count,
            lstm_layer_sizes, lstm_dropout_probability, use_batch_normalization=True
        )

    def summary(self):
        '''
        Outputs a summary of the MusicRNN model.

        '''

        self.model.summary()

    def train(self, dataset, logdir, restoredir=None, epochs=None,
              save_frequency_mode=ModelSaveFrequencyMode.EPOCH, save_frequency=1):
        '''
        Fit the model to the specified ``dataset``.

        :param dataset:
            An iterable object containing feature, label pairs (as tuples).
        :param epochs:
            The number of epochs to train for. Defaults to ``None``, meaning
            that the model will train indefinitely.

        '''

        pass
