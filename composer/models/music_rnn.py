'''
MusicRNN: A recurrent neural network model designed to generate music based on an
MIDI-like event-based description language (see: :mod:`composer.dataset.sequence` for
more information about the event-based sequence description.)

'''

import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from composer.models import BaseModel, ModelSaveFrequencyMode
from tensorflow.keras import layers, optimizers, losses

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

    def compile(self, learning_rate):
        '''
        Compiles this model.

        '''

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss = losses.SparseCategoricalCrossentropy(from_logits=True)

        super().compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def train(self, dataset, input_shape, logdir, restoredir=None, epochs=None,
              learning_rate=1e-3, save_frequency_mode=ModelSaveFrequencyMode.EPOCH,
              save_frequency=1, max_checkpoints=1, show_progress_bar=True):
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
        :param show_progress_bar:
            Indicates whether a progress bar will be shown to indicate epoch status.
            Defaults to ``True``.

        '''

        logdir = Path(logdir)
        if restoredir is not None:
            logdir = Path(restoredir)  

        optimizer = optimizers.Adam(learning_rate=learning_rate)
        loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)

        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), epoch=tf.Variable(1), optimizer=optimizer, model=self)
        manager = tf.train.CheckpointManager(checkpoint, logdir, max_to_keep=max_checkpoints)

        # Restore the model, if exists
        if restoredir is not None:
            try:
                checkpoint.restore(manager.latest_checkpoint)
                logging.info('Model restored from \'{}\'.'.format(manager.latest_checkpoint))
            except:
                logging.error('Failed to restore model from \'{}\'.'.format(restoredir))
                exit(1)

        # TensorBoard summary logger
        summary_log = tf.summary.create_file_writer(str(logdir / 'train'))

        steps_per_epoch = None
        save_frequency_mode = ModelSaveFrequencyMode(save_frequency_mode)

        # Build the model (i.e. inform the model of the input shape).
        self.build(input_shape=input_shape)

        while epochs is None or int(checkpoint.epoch) < epochs:
            current_epoch = int(checkpoint.epoch)
            logging.info('Epoch {}'.format(current_epoch if epochs is None else '{}/{}'.format(current_epoch, epochs)))
            with tqdm(total=steps_per_epoch, disable=not show_progress_bar) as progress_bar:
                epoch_loss_average = tf.keras.metrics.Mean()
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

                # Initialize the hidden states of the model
                self.reset_states()

                for x, y in dataset:
                    # Compute loss and optimize
                    with tf.GradientTape() as tape:
                        predictions = self(x, training=True)
                        loss = loss_object(y_true=y, y_pred=predictions)
                    
                    grads = tape.gradient(loss, self.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.trainable_variables))
                    
                    # Calculate the batch accuracy
                    _acc_pred_y = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                    _acc_y = tf.cast(y, tf.int32)
                    accuracy = tf.reduce_mean(tf.cast(tf.equal(_acc_pred_y, _acc_y), tf.float32))

                    # Update loss and accuracy metrics
                    epoch_loss_average.update_state(loss)
                    epoch_accuracy.update_state(y, predictions)

                    # Log to TensorBoard summary
                    global_step = int(checkpoint.step)
                    with summary_log.as_default():
                        tf.summary.scalar('loss', loss, step=global_step)
                        tf.summary.scalar('accuracy', accuracy, step=global_step)

                    # Update description of progress bar to show loss and accuracy statistics
                    progress_bar.set_description('- loss: {:.4f} - accuracy: {:.4f}'.format(loss, accuracy))

                    if save_frequency_mode == ModelSaveFrequencyMode.GLOBAL_STEP and global_step % save_frequency == 0:
                        save_path = manager.save()
                        progress_bar.write('Saved checkpoint for step {} at {}.'.format(global_step, save_path))

                    checkpoint.step.assign_add(1)
                    progress_bar.update(1)

                # Log to TensorBoard summary
                with summary_log.as_default():
                    tf.summary.scalar('epoch_loss', epoch_loss_average.result(), step=current_epoch)
                    tf.summary.scalar('epoch_accuracy', epoch_accuracy.result(), step=current_epoch)

                if save_frequency_mode == ModelSaveFrequencyMode.EPOCH and current_epoch % save_frequency == 0:
                    save_path = manager.save()
                    progress_bar.write('Saved checkpoint for epoch {} at {}.'.format(current_epoch, save_path))
                
                if steps_per_epoch is None:
                    steps_per_epoch = progress_bar.n

                checkpoint.epoch.assign_add(1)