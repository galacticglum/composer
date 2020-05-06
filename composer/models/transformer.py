'''
Transformer: a decoder-only Transformer model for music generation.

Source: https://github.com/openai/gpt-2/blob/master/src/model.py

The model implementation is based on the GPT-2 source code.
It is modified so that the code style is consistent and to include
a memory-efficient relative attention implementation.

'''

import math
import logging
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from composer.models import BaseModel, ModelSaveFrequencyMode
from tensorflow.keras import layers, optimizers, losses

def shape_list(x):
    '''
    Deal with dynamic shape in tensorflow cleanly.
    
    '''

    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def gelu(x):
    '''
    Gaussian Error Linear Unit (GELU) activiation function.
    '''

    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def get_padding_mask(x):
    '''
    Gets a padded mask.

    '''

    x = tf.cast(tf.math.equal(x, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return x[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def attention_mask(size):
    '''
    Creates an attention mask with the specified size.

    :note:
        If size is 4 then it returns below matrix
        [[0., 1., 1., 1.],
         [0., 0., 1., 1.],
         [0., 0., 0., 1.],
         [0., 0., 0., 0.]]

    '''

    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_attention_mask(inputs):
    '''
    Creates attention masks for the specified ``inputs``.

    '''

    att_mask = attention_mask(tf.shape(inputs)[1])
    padding_mask = get_padding_mask(inputs)
    mask = tf.maximum(padding_mask, att_mask)
    return mask

class SharedTokenEmbedding(tf.keras.layers.Layer):
    '''
    A shared token embedding layer.

    '''

    def __init__(self, vocab_size, hidden_size, initializer_mean=0, initializer_stddev=0.02, **kwargs):
        '''
        Initializes an instance of :class:`SharedTokenEmbedding`.

        :param vocab_size:
            The number of unique integer ids to expect.
        :param hidden_size:
            The number of units in the embedding layer.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.

        '''

        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer_mean = initializer_mean
        self.initializer_stddev = initializer_stddev or hidden_size**-0.5

    def build(self, input_shape):
        '''
        Builds the embedding layer and initializes weights.
        
        '''
        
        weight_shape = [self.vocab_size, self.hidden_size]
        weight_initializer = tf.keras.initializers.TruncatedNormal(mean=self.initializer_mean, stddev=self.initializer_stddev)
        self.weight = self.add_weight('weight', shape=weight_shape, initializer=weight_initializer)

        super().build(input_shape)

    def call(self, inputs, mode='embedding'):
        '''
        Get token embeddings of inputs.

        :param inputs:
            An int tensor with shape [batch_size, length].
        :param mode:
            A string indicating the behaviour of the embedding layer. Either "embedding" or "linear".
        :returns:
            If ``mode`` is "embedding", this layer returns a float32 embedding tensor with shape
            [batch_size, length, hidden_size] representing the ``inputs`` as dense float vectors.

            If ``mode`` is "linear", this layer returns a float32 linear tensor with shape
            [batch_size, length, vocab_size].

        '''

        if mode == 'embedding':
            return tf.gather(self.weight, inputs)
        elif mode == 'linear':
            input_shape = shape_list(inputs)[:-1]
            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.weight, transpose_b=True)

            return tf.reshape(logits, input_shape + [self.vocab_size])
        else:
            raise ValueError('\'{}\' is not a valid Embedding mode.'.format(mode))

class Conv1D(layers.Layer):
    '''
    A one-dimensional convolution layer as defined by Radford et al. in the GPT paper.

    :note:
        At its essence, this is a fully connected (dense) linear layer; however,
        with transposed weights. So, Y = X * W^T + B.

    '''

    def __init__(self, hidden_size, filter_size, initializer_mean=0.0,
                 initializer_stddev=0.02, **kwargs):
        '''
        Initializes an instance of :class:`Conv1D`.

        :param hidden_size:
            The number of units in the convolutional layer.
        :param filter_size:
            The size of a convolutional filter.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.

        '''

        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.initializer_mean = initializer_mean
        self.initializer_stddev = initializer_stddev

    def build(self, input_shape):
        '''
        Builds the convolutional layer and initializes weights.

        '''

        weight_initializer = tf.keras.initializers.TruncatedNormal(mean=self.initializer_mean, stddev=self.initializer_stddev)
        self.weight = self.add_weight('weight', shape=[self.hidden_size, self.filter_size], initializer=weight_initializer)
        self.bias = self.add_weight('bias', shape=[1, self.filter_size], initializer=tf.zeros_initializer())

        super().build(input_shape)

    def call(self, inputs):
        '''
        Gets convolutions on the specified ``inputs``.

        :param inputs:
            A 3-dimensional float32 tensor of shape [batch, sequence, features].
        :returns:
            A float32 tensor with shape [batch_size, length, filter_size].

        '''

        batch, sequence = shape_list(inputs)[:2]
        inputs = tf.reshape(inputs, [-1, self.hidden_size])
        inputs = tf.matmul(inputs, self.weight) + self.bias
        inputs = tf.reshape(inputs, [batch, sequence, self.filter_size])
        return inputs

class Attention(layers.Layer):
    '''
    A multihead attention layer that supports both absolute and relative attention.

    '''

    def __init__(self, hidden_size, attention_head_count, attention_dropout_rate=0.1,
                 residual_dropout_rate=0.1, scale=False, use_relative_attention=False,
                 output_attention_weights=False, initializer_mean=0.0,
                 initializer_stddev=0.02, **kwargs):
        '''
        Initialize an instance of :class:`Attention`.

        :param hidden_size:
            The number of units in a single attention head.
        :param attention_head_count:
            The number of attention heads.
        :param attention_dropout_rate:
            The dropout rate for the attention convolutional layer.
        :param residual_dropout_rate:
            The dropout rate for all fully connected (dense) linear layers.
        :param scale:
            Indicates whether the attention scores should be scaled. Defaults to ``False``.
        :param use_relative_attention:
            Indicates whether to use relative attention. Defaults to ``False``.
        :param output_attention_weights:
            Indicates whether to output the attention weights along with the scores and present states.
            Defaults to ``False``, meaning that no attention weights will be outputed.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.

        '''
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.attention_head_count = attention_head_count
        self.scale = scale
        self.use_relative_attention = use_relative_attention

        # The hidden size must be a multiple of the attention head count.
        assert hidden_size % attention_head_count == 0
        # The depth refers to the size (numbers of units) of a single head.
        self.depth = hidden_size // attention_head_count

        self.c_attn = Conv1D(
            hidden_size, hidden_size * 3,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='c_attn'
        )
        
        self.c_proj = Conv1D(
            hidden_size, hidden_size,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='c_proj'
        )

        self.attention_dropout = layers.Dropout(attention_dropout_rate)
        self.residual_dropout = layers.Dropout(residual_dropout_rate)
        self.output_attention_weights = output_attention_weights

    def build(self, input_shape):
        '''
        Builds the attention layer and initializes weights.

        '''

        if self.use_relative_attention:
            initializer = tf.keras.initializers.GlorotUniform()
            # Input shape is [batch, sequence, features] and we only need batch and sequence.
            # The query shape is [heads, batch * sequence, features], and we need our weights to match that.
            E_shape = (self.attention_head_count, input_shape[0] * input_shape[1], self.depth)
            self.E = self.add_weight('E', shape=E_shape, initializer=initializer)

        super().build(input_shape)

    def _relative_attention(self, q):
        '''
        Gets the relative attention score of a query tensor.

        '''

        # q have shape [batch, heads, sequence, features]
        batch, heads, sequence, features = shape_list(q)
        
        # [heads, batch, sequence, features]
        q_ = tf.transpose(q, [1, 0, 2, 3])
        # [heads, batch * sequence, features]
        q_ = tf.reshape(q_, [heads, batch * sequence, features])
        # [heads, batch * sequence, sequence]
        rel = tf.matmul(q_, self.E, transpose_b=True)
        # [heads, batch, sequence, sequence]
        rel = tf.reshape(rel, [heads, batch, sequence, sequence])
        # [heads, batch, sequence, 1+sequence]
        rel = tf.pad(rel, ((0, 0), (0, 0), (0, 0), (1, 0)))
        # [heads, batch, sequence+1, sequence]
        rel = tf.reshape(rel, (heads, batch, sequence+1, sequence))
        # [heads, batch, sequence, sequence]
        rel = rel[:, :, 1:]
        # [batch, heads, sequence, sequence]
        rel = tf.transpose(rel, [1, 0, 2, 3])

        return rel

    def _multihead_attention(self, q, k, v, training, mask=None):
        '''
        Gets the attention scores and weights for a query, key, and value triplet.

        '''

        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        if self.use_relative_attention:
            w = w + self._relative_attention(q)

        if self.scale:
            # scale the attention scores
            w = w * tf.math.rsqrt(tf.cast(v.shape[-1], w.dtype))

        if mask is not None:
            w += (mask * -1e9)

        w = tf.nn.softmax(w, axis=-1)  # (..., seq_len_q, seq_len_k)
        w = self.attention_dropout(w, training=training)

        output = tf.matmul(w, v)  # (..., seq_len_q, depth_v)
        return output, w

    def _split_heads(self, x):
        '''
        Splits the input tensor into a new tensor of shape [batch_size, sequence, attention_head_count, depth].

        '''

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.attention_head_count, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _merge_heads(self, x):
        '''
        Merges the input tensor, which contains data split into each attention head, into a new tensor
        of shape [batch_size, sequence, hidden_size].

        '''

        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, attention_head_count, depth)
        merged = tf.reshape(x, (batch_size, -1, self.hidden_size))
        # (batch_size, seq_len_q, hidden_size)
        return merged

    def call(self, inputs, past=None, mask=None, training=False):
        '''
        Gets the attention scores and present state of the attention layer.

        :param inputs:
            A 3-dimensional float32 tensor of shape [batch, sequence, features].
        :param past:
            The previous (past) state of this attention layer.
        :param mask:
            The attention mask. Defaults to ``None``.
        :param training:
            Indicates whether this step is training. Defaults to ``False``.
        :returns:
            The attention scores and the present state.  if ``output_attention_weights``
            is ``True``, the attention weights will be returned.

        '''

        inputs = self.c_attn(inputs)
        q, k, v = tf.split(inputs, 3, axis=2)

        # Split query, key, and value heads.
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if past is not None:
            past_k, past_v = tf.unstack(past, axis=1)
            k = tf.concat([past_k, k], axis=-2)
            v = tf.concat([past_v, v], axis=-2)

        present = tf.stack([k, v], axis=1)
        output, w = self._multihead_attention(q, k, v, training, mask)
        output = self._merge_heads(output)

        output = self.c_proj(output)  # (batch_size, seq_len_q, hidden_size)
        output = self.residual_dropout(output, training=training)
        
        result = [output, present]
        if self.output_attention_weights:
            result.append(w)
        
        return result

class MultilayerPerceptron(layers.Layer):
    '''
    A multi-layer perceptron which consists of two one-dimensional convolutional
    layers: the hidden and output layers respectively.

    '''

    def __init__(self, hidden_size, filter_size, dropout_rate=0.1,
                 initializer_mean=0.0, initializer_stddev=0.02, activation=gelu, **kwargs):
        '''
        Initializes an instance of :class:`MultilayerPerceptron`.

        :param hidden_size:
            The number of units in the hidden layer.
        :param filter_size:
            The size of a convolutional filter.
        :param dropout_rate:
            The dropout rate on the final output of this layer.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.
        :param activation:
            The activation function to use on the fully connected hidden layer.
            Defaults to the Gaussian Error Linear Unit (GELU) activiation function.

        '''

        super().__init__(**kwargs)

        self.activation = activation
        self.c_fc = Conv1D(
            filter_size, hidden_size,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='c_fc'
        )

        self.c_proj = Conv1D(
            hidden_size, filter_size,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='c_proj'
        )

        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        '''
        Gets the result of feeding the input through the multi-layer perceptron.

        '''

        inputs = self.activation(self.c_fc(inputs))
        inputs = self.c_proj(inputs)
        inputs = self.dropout(inputs, training=training)
        return inputs

class DecoderBlock(layers.Layer):
    '''
    A single Transformer-decoder block.

    '''

    def __init__(self, embedding_size, attention_head_count, use_relative_attention=False,
                 attention_dropout_rate=0.1, residual_dropout_rate=0.1, layer_normalization_epsilon=1e-5,
                 scale=False, initializer_mean=0, initializer_stddev=0.02, **kwargs):
        '''
        Initializes an instance of :class:`DecoderBlock`.

        :param embedding_size:
            The number of units in the embedding layer.
        :param attention_head_count:
            The number of attention heads.
        :param use_relative_attention:
            Indicates whether to use relative attention. Defaults to ``False``.
        :param attention_dropout_rate:
            The dropout rate for the attention convolutional layer.
        :param residual_dropout_rate:
            The dropout rate for all fully connected (dense) linear layers.
        :param layer_normalization_epsilon:
            The epsilon to use in the layer normalization layers.
        :param scale:
            Indicates whether the attention scores should be scaled. Defaults to ``False``.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.
        
        '''

        super().__init__(**kwargs)

        self.ln_1 = layers.LayerNormalization(epsilon=layer_normalization_epsilon, name='ln_1')
        self.attn = Attention(
            embedding_size, attention_head_count,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            scale=scale, use_relative_attention=use_relative_attention,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='attn'
        )

        self.ln_2 = layers.LayerNormalization(epsilon=layer_normalization_epsilon, name='ln_2')
        self.mlp = MultilayerPerceptron(
            4 * embedding_size, embedding_size,
            dropout_rate=residual_dropout_rate,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='mlp'
        )

    def call(self, inputs, mask, past=None, training=False):
        '''
        Decode the specified ``inputs``.

        '''

        x = self.ln_1(inputs)
        outputs, present = self.attn(x, past=past, mask=mask, training=training)
        x = x + outputs
        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m
        
        return x, present

class Transformer(BaseModel):
    '''
    A Transformer-decoder model designed to generate music based on an
    MIDI-like event-based description language.

    :note:
        See :mod:`composer.dataset.sequence` for more information about the 
        event-based sequence description.

    '''

    def __init__(self, vocab_size, embedding_size, window_size, decoder_layers_count,
                 attention_head_count, use_relative_attention=False, initializer_mean=0,
                 initializer_stddev=0.02, attention_dropout_rate=0.1, residual_dropout_rate=0.1,
                 layer_normalization_epsilon=1e-5, scale=True, *args, **kwargs):
        '''
        Initialize an instance of :class:`Transformer`.

        :param vocab_size:
            The size of the MIDI-like event-based description vocabulary.
            This is the dimensionality of a one-hot vector encoded representation of an event.
        :param embedding_size:
            The number of units in the embedding layer.
        :param window_size:
            The number of events in a single input sequence.
        :param decoder_layers_count:
            The number of decoder blocks.
        :param attention_head_count:
            The number of attention heads.
        :param use_relative_attention:
            Indicates whether to use relative attention. Defaults to ``False``.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.
        :param attention_dropout_rate:
            The dropout rate for the attention convolutional layer. Defaults to 0.1.
        :param residual_dropout_rate:
            The dropout rate for all fully connected (dense) linear layers. Defaults to 0.1.
        :param layer_normalization_epsilon:
            The epsilon to use in the layer normalization layers.
        :param scale:
            Indicates whether the attention scores should be scaled. Defaults to ``True``.

        '''

        super().__init__(*args, **kwargs)

        self.embedding_size = embedding_size
        self.decoder_layers_count = decoder_layers_count

        self.wte = SharedTokenEmbedding(
            vocab_size, embedding_size,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='wte'
        )
      
        embeddings_initializer = tf.keras.initializers.TruncatedNormal(mean=initializer_mean, stddev=initializer_stddev)
        self.wpe = tf.keras.layers.Embedding(
            window_size, embedding_size,
            embeddings_initializer=embeddings_initializer,
            name='wpe'
        )

        self.decoder_blocks = [DecoderBlock(
            embedding_size, attention_head_count,
            use_relative_attention=use_relative_attention,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            layer_normalization_epsilon=layer_normalization_epsilon,
            scale=scale, initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='h_%d' % (layer_index + 1)
        ) for layer_index in range(decoder_layers_count)]
        self.ln_f = layers.LayerNormalization(epsilon=layer_normalization_epsilon, name='ln_f')

    def call(self, inputs, past=None, training=False):
        '''
        Run the specified ``inputs`` through the Transformer-decoder model.

        :param inputs:
            An int tensor with shape [batch, sequence, features].
        :param past:
            The previous state of the model. Defaults to ``None``.
        :param training:
            Indicates whether this step is training. Defaults to ``False``.
        :returns:
            A probability distribution of the next feature in the sequence (given as logits)
            and the present state of the model.

        '''

        inputs = tf.cast(inputs, tf.int32)
        input_shape = shape_list(inputs)
        batch, sequence = input_shape[:2]
        if past is None:
            past_length = 0
            past = [None] * self.decoder_layers_count
        else:
            past_length = shape_list(past[0][0])[-2]

        attention_mask = create_attention_mask(inputs)
        position_ids = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]
        h = self.wte(inputs, mode='embedding') + self.wpe(position_ids)
        presents = []
        for decoder_layer, past_state in zip(self.decoder_blocks, past):
            h, present = decoder_layer(h, attention_mask, past=past_state, training=training)
            presents.append(present)
        
        h = self.ln_f(h)
        logits = self.wte(h, mode='linear')
        return logits, presents

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
        while epochs is None or int(checkpoint.epoch) < epochs:
            current_epoch = int(checkpoint.epoch)
            logging.info('Epoch {}'.format(current_epoch if epochs is None else '{}/{}'.format(current_epoch, epochs)))
            with tqdm(total=steps_per_epoch, disable=not show_progress_bar) as progress_bar:
                epoch_loss_average = tf.keras.metrics.Mean()
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

                for x, y in dataset:
                    # Compute loss and optimize
                    with tf.GradientTape() as tape:
                        predictions, _ = self(x, training=True)
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