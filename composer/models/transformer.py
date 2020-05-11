'''
Transformer: a decoder-only Transformer model for music generation.

The model implementation is based on the GPT-2 source code [1].
It is modified so that the code style is consistent and to include
a memory-efficient relative attention implementation.

The TensorFlow port is inspired by the Huggingface Transformer model [2].

Sources:
    1. https://github.com/openai/gpt-2/blob/master/src/model.py
    2. https://huggingface.co/transformers/

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

    def __init__(self, vocab_size, hidden_size, initializer_mean=0, initializer_stddev=None, **kwargs):
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

    def __init__(self, filter_size, hidden_size, initializer_mean=0.0,
                 initializer_stddev=0.02, **kwargs):
        '''
        Initializes an instance of :class:`Conv1D`.

        :param filter_size:
            The size of a convolutional filter.
        :param hidden_size:
            The number of units in the convolutional layer.
        :param initializer_mean:
            The mean of the truncated random normal initializer. Defaults to 0.
        :param initializer_stddev:
            The standard deviation of the truncated random normal initializer.
            Defaults to 0.02.

        '''

        super().__init__(**kwargs)

        self.filter_size = filter_size
        self.hidden_size = hidden_size
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

        # The number of units in this attention layer (equal to the embedding size in GPT-2).
        self.hidden_size = hidden_size
        self.attention_head_count = attention_head_count
        self.scale = scale
        self.use_relative_attention = use_relative_attention

        # The hidden size must be a multiple of the attention head count.
        assert hidden_size % attention_head_count == 0

        self.c_attn = Conv1D(
            hidden_size * 3, hidden_size,
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

    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        '''
        1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

        '''

        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)

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

    def _multihead_attention(self, inputs, training):
        '''
        Gets the attention scores and weights for a query, key, and value triplet.

        '''

        q, k, v, attention_mask, head_mask = inputs
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)

        if self.use_relative_attention:
            # Apply relative attention
            w = w + self._relative_attention(q)

        if self.scale:
            # Scale attention_scores
            dk = tf.cast(shape_list(k)[-1], tf.float32)
            w = w * tf.math.rsqrt(dk)

        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = Attention.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = tf.nn.softmax(w, axis=-1)
        w = self.attention_dropout(w, training=training)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [tf.matmul(w, v)]
        if self.output_attention_weights:
            outputs.append(w)
        
        return outputs

    def merge_heads(self, x):
        '''
        Merges the input tensor, which contains data split into each attention head, into a new tensor
        of shape [batch_size, sequence, hidden_size].

        '''

        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)

    def split_heads(self, x):
        '''
        Splits the input tensor into a new tensor of shape [batch_size, sequence, attention_head_count, depth].

        '''

        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.attention_head_count, x_shape[-1] // self.attention_head_count]
        x = tf.reshape(x, new_x_shape)
        # Output has shape (batch, head, sequence, features)
        return tf.transpose(x, (0, 2, 1, 3)) 

    def call(self, inputs, training=False):
        '''
        Gets the attention scores and present state of the attention layer.

        :param inputs:
            A list containing the input values, a 3-dimensional float32 tensor of 
            shape [batch, sequence, features], the past layer state, attention mask,
            head mask, and a boolean ``use_cache``.
        :param training:
            Indicates whether this step is training. Defaults to ``False``.
        :returns:
            The attention scores and the present state. If ``output_attention_weights``
            is ``True``, the attention  weights will be returned.

        '''

        # Decompose inputs into their respective values
        x, layer_past, attention_mask, head_mask, use_cache = inputs

        x = self.c_attn(x)
        query, key, value = tf.split(x, 3, axis=2)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        if layer_past is not None:
            past_key, past_value = tf.unstack(layer_past, axis=0)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)

        if tf.is_tensor(use_cache):
            if hasattr(use_cache, "numpy"):
                use_cache = bool(use_cache.numpy())
            else:
                use_cache = True

        if use_cache is True:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None,)

        attn_outputs = self._multihead_attention([query, key, value, attention_mask, head_mask], training=training)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.residual_dropout(a, training=training)

        outputs = [a, present] + attn_outputs[1:]
        # Output is of the form: attention scores, present state, (attention weights)
        return outputs

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
            hidden_size, filter_size,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='c_fc'
        )

        self.c_proj = Conv1D(
            filter_size, hidden_size,
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
                 scale=False, initializer_mean=0, initializer_stddev=0.02, use_layer_normalization=True,
                 output_attention_weights=False, **kwargs):
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
        :param use_layer_normalization:
            Indicates whether the inputs/outputs from layers should be normalized. Defaults to ``True``.
        :param output_attention_weights:
            Indicates whether to output the attention weights along with the scores and present states.
            Defaults to ``False``, meaning that no attention weights will be outputed.
        
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
            output_attention_weights=output_attention_weights,
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

        self.use_layer_normalization = use_layer_normalization

    def call(self, inputs, training=False):
        '''
        Decode the specified ``inputs``.

        '''

        # Decompose inputs into their respective values
        x, layer_past, attention_mask, head_mask, use_cache = inputs

        if self.use_layer_normalization:
            x = self.ln_1(x)
        
        attention_outputs = self.attn([x, layer_past, attention_mask, head_mask, use_cache], training=training)
        x = x + attention_outputs[0]

        m = x
        if self.use_layer_normalization:
            m = self.ln_2(x)
            
        m = self.mlp(m, training=training)
        x = x + m
        
        outputs = [x] + attention_outputs[1:]
        return outputs

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
                 layer_normalization_epsilon=1e-5, scale=True, use_layer_normalization=True,
                 output_hidden_states=False, output_attention_weights=False, *args, **kwargs):
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
        :param use_layer_normalization:
            Indicates whether the inputs/outputs from layers should be normalized. Defaults to ``True``.
        :param output_hidden_states:
            Indicates whether to output the hidden states of each decoder block.
            Defaults to ``False``, meaning that no hidden layer state will be outputed.
        :param output_attention_weights:
            Indicates whether to output the attention weights along with the scores and present states.
            Defaults to ``False``, meaning that no attention weights will be outputed.

        '''

        super().__init__(*args, **kwargs)

        self.embedding_size = embedding_size
        self.decoder_layers_count = decoder_layers_count
        self.use_layer_normalization = use_layer_normalization
        self.output_hidden_states = output_hidden_states
        self.output_attention_weights = output_attention_weights

        self.wte = SharedTokenEmbedding(
            vocab_size, embedding_size,
            initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            name='wte'
        )
      
        embeddings_initializer = tf.keras.initializers.TruncatedNormal(
            mean=initializer_mean, 
            stddev=initializer_stddev
        )

        self.wpe = tf.keras.layers.Embedding(
            window_size, embedding_size,
            embeddings_initializer=embeddings_initializer,
            name='wpe'
        )

        self.embedding_dropout = tf.keras.layers.Dropout(residual_dropout_rate, name='embd_dropout')
        self.decoder_blocks = [DecoderBlock(
            embedding_size, attention_head_count,
            use_relative_attention=use_relative_attention,
            attention_dropout_rate=attention_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            layer_normalization_epsilon=layer_normalization_epsilon,
            scale=scale, initializer_mean=initializer_mean,
            initializer_stddev=initializer_stddev,
            use_layer_normalization=use_layer_normalization,
            output_attention_weights=output_attention_weights,
            name='h_%d' % (layer_index + 1)
        ) for layer_index in range(decoder_layers_count)]
        self.ln_f = layers.LayerNormalization(epsilon=layer_normalization_epsilon, name='ln_f')

    def call(self, inputs, past=None, attention_mask=None, token_type_ids=None, position_ids=None,
             input_embeddings=None, use_cache=True, training=False):
        '''
        Run the specified ``inputs`` through the Transformer-decoder model.

        :param inputs:
            An int tensor with shape [batch, sequence] containing the input sequence tokens
            in the vocabulary. If ``past`` is specified, only the last ``inputs`` are used.
        :param past:
            The previous state of the model; contains pre-computed hidden-states (key and values in the attention
            layers) as computed by the previous call of this model. Defaults to ``None``.
        :param attention_mask:
            A mask to avoid performing attention on padded token indices. The mask should consist of integer values
            selected from ``[0, 1]`` where each element encodes a boolean value indicating whether or not tokens
            are masked; ``1`` for tokens that are NOT masked and ``0`` for tokens that ARE masked. Defaults to ``None``.
        :param token_type_ids:
            Token indices that are useds to indicate the first and second segments of the inputs. Consists of values
            selected from ``[0, 1]`` where ``0`` corresponds to the first segment (A) and ``1`` corresponds to the
            second segment (B). Defaults to ``None``.
        :param position_ids:
            Indicies of position for each input sequence token. This can be provided instead of computing position
            embeddings on the input token sequence. Defaults to ``None``.
        :param input_embeddings:
            The embeddings of the input token sequence. This can be provided instead of passing the input ids; useful
            when you need to control how the input token sequence is encoded into an embedding space. This is essentially
            overriding the models' internal embedding space (which is a lookup matrix for each token in the vocabulary)
            Defaults to ``None``.
        :param use_cache:
            Indicates whether to use the past key and value (this can speed up decoding). Defaults to ``True``.
        :param training:
            Indicates whether this step is training. Defaults to ``False``.
        :returns:
            A probability distribution of the next feature in the sequence (given as logits)
            and the present state of the model.

        '''

        # If we are providing the past layer state, we only
        # use the last token in the input sequence.
        if past is not None:
            if inputs is not None:
                inputs = inputs[:, -1:]

            if input_embeddings is not None:
                input_embeddings = input_embeddings[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1:]

        # Resolve the inputs and input embeddings.
        # Raise errors if both or none are specified.
        if inputs is not None and input_embeddings is not None:
            raise ValueError('You cannot specify both inputs and input embeddings.')
        elif inputs is not None:
            input_shape = shape_list(inputs)
            inputs = tf.reshape(inputs, [-1, input_shape[-1]])
        elif input_embeddings is not None:
            input_shape = shape_list(input_embeddings)[:-1]
        else:
            raise ValueError('You have to specify either inputs or input embeddings.')

        # We need to cast the inputs to integers since the operations we apply on this tensor
        # assume that the data type is an int32.
        inputs = tf.cast(inputs, tf.int32)

        if past is None:
            # We haven't been given a past state so the input to each decoder block is just None.
            past_length = 0
            past = [None] * self.decoder_layers_count
        else:
            past_length = shape_list(past[0][0])[-2]
        
        if position_ids is None:
            # Compute the position ids from the input sequence.
            # This is the input to the position embedding (which will compute positional encodings).
            position_ids = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]

        if attention_mask is not None:
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None

        # Initialize head masks to None (i.e. don't apply any masking on attention heads).
        head_mask = [None] * self.decoder_layers_count
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])

        if input_embeddings is None:
            input_embeddings = self.wte(inputs, mode='embedding')

        position_embeddings = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeddings = self.wte(token_type_ids, mode='embedding')
        else:
            token_type_embeddings = 0

        hidden_states = input_embeddings + position_embeddings + token_type_embeddings
        hidden_states = self.embedding_dropout(hidden_states, training=training)
        output_shape = input_shape + [shape_list(hidden_states)[-1]]

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.decoder_blocks, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape),)

            outputs = block([hidden_states, layer_past, attention_mask, head_mask[i], use_cache], training=training)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)

            if self.output_attention_weights:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)

        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.wte(hidden_states, mode='linear')
        outputs = (logits,)
        if use_cache is True:
            outputs = outputs + (presents,)

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)

        if self.output_attention_weights:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple(tf.reshape(t, attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)

        # logits, presents, (all hidden_states), (attentions)
        return outputs 

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