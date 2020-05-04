'''
MusicRNN: A recurrent neural network model designed to generate music based on an
MIDI-like event-based description language (see: :mod:`composer.dataset.sequence` for
more information about the event-based sequence description.)

'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from composer.models import ModelSaveFrequencyMode

class MusicRNN(Model):
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

    def __init__(self, vocab_size, batch_size, embedding_size, lstm_layers_count,
                 lstm_layer_sizes, lstm_dropout_probability, use_batch_normalization=True):

        '''
        Initialize an instance of :class:`MusicRNN`.

        :param vocab_size:
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

        self.embedding_layer = layers.Embedding(vocab_size, embedding_size, batch_input_shape=(batch_size, None))
        self.lstm_layers = []
        self.dropout_layers = []
        self.normalization_layers = []
        for i in range(lstm_layers_count):
            self.lstm_layers.append(layers.LSTM(lstm_layer_sizes[i], return_sequences=True, 
                                    stateful=True, recurrent_initializer='glorot_uniform'))

            if lstm_dropout_probability[i] > 0:
                self.dropout_layers.append(layers.Dropout(lstm_dropout_probability[i]))

            if use_batch_normalization:
                self.normalization_layers.append(layers.BatchNormalization())

        self.use_normalization_layers = len(self.normalization_layers) == lstm_layers_count
        self.use_dropout_layers = len(self.dropout_layers) == lstm_layers_count
        self.output_layer = layers.Dense(vocab_size)

    def call(self, inputs):
        '''
        Feed forward call on this network.

        :param inputs:
            The inputs to the network. This should be an array-like object containing
            sequences (of size :attr:`MusicRNN.window_size`) of :attr:`MusicRNN.event_dimensions`
            -dimensional one-hot vectors representing the events.

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

        loss = losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        # We need to build the model so that it knows about the batch size.
        self.build(input_shape=(input_shape[0], None))

        logdir = Path(logdir)
        initial_epoch = 0
        if restoredir is not None:
            checkpoint = tf.train.latest_checkpoint(restoredir)
            if checkpoint is None:
                logging.error('Failed to restore model from \'{}\'.'.format(restoredir))
                exit(1)

            self.load_weights(checkpoint)
            logdir = Path(restoredir)

            initial_epoch = int(re.search(r'(?<=model-)(.*)(?=-)', str(checkpoint)).group(0))

        tensorboard_callback = TensorBoard(log_dir=str(logdir.absolute()), update_freq=25, profile_batch=0, write_graph=False, write_images=False)
        model_checkpoint_path = logdir / 'model-{epoch:02d}-{loss:.2f}'

        is_epoch_save_freq = save_frequency_mode == ModelSaveFrequencyMode.EPOCH
        model_checkpoint_callback = ModelCheckpoint(filepath=str(model_checkpoint_path.absolute()), monitor='loss', verbose=1, 
                                                    save_freq='epoch' if is_epoch_save_freq else int(save_frequency),
                                                    period=save_frequency if is_epoch_save_freq else None, 
                                                    save_best_only=False, mode='auto', save_weights_only=True)

        self.fit(dataset, epochs=epochs + initial_epoch, initial_epoch=initial_epoch,
                       callbacks=[tensorboard_callback, model_checkpoint_callback])