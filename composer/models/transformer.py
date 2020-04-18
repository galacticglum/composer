'''
A decoder-only Transformer model for music generation.

Source: https://github.com/openai/gpt-2/blob/master/src/model.py

The model implementation is based on the GPT-2 source code, and modified
so that the code style is consistent and to include a memory-efficient 
relative attention implementation.

'''

import tensorflow as tf

@tf.function
def shape_list(x):
    '''
    Deal with dynamic shape in tensorflow cleanly.
    
    '''

    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

@tf.function
def gelu(x):
    '''
    Gaussian Error Linear Unit (GELU) activiation function.

    '''

    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

@tf.function
def norm(x, scope, *, axis=-1, epsilon=1e-5):
    '''
    Normalize to mean = 0, std = 1, then do a diagonal affine transform.

    '''

    with tf.name_scope(scope):
        n_state = x.shape[-1].value
        
        g = tf.Variable(tf.constant_initializer(1)([n_state]), name='g')
        b = tf.Variable(tf.constant_initializer(0)([n_state]), name='b')
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

@tf.function
def split_states(x, n):
    '''
    Reshape the last dimension of x into [n, x.shape[-1] / n].

    '''

    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

@tf.function
def merge_states(x):
    '''
    Smash the last two dimensions of x into a single dimension.

    '''

    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

@tf.function
def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    '''
    One-dimensional convolution layer.

    '''

    with tf.name_scope(scope):
        *start, nx = shape_list(x)
        
        w = tf.Variable(tf.random_normal_initializer(stddev=w_init_stdev)([1, nx, nf]), name='w')
        b = tf.Variable(tf.constant_initializer(0)([nf]), name='b')
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

@tf.function
def attention_mask(nd, ns, *, dtype):
    '''
    1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    
    '''

    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

@tf.function
def attn(x, scope, n_state, attention_head_count):
    '''
    Memory-efficient relative attention unit.

    '''

    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % attention_head_count == 0

    @tf.function
    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, attention_head_count), [0, 2, 1, 3])

    @tf.function
    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    @tf.function
    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w
    
    @tf.function
    def relative_attn(q):
        # q have shape [batch, heads, sequence, features]
        batch, heads, sequence, features = shape_list(q)
        
        E = tf.Variable(tf.keras.initializers.glorot_uniform()([heads, sequence, features]), name='E')
        # [heads, batch, sequence, features]
        q_ = tf.transpose(q, [1, 0, 2, 3])
        # [heads, batch * sequence, features]
        q_ = tf.reshape(q_, [heads, batch * sequence, features])
        # [heads, batch * sequence, sequence]
        rel = tf.matmul(q_, E, transpose_b=True)
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

    @tf.function
    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w + relative_attn(q)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = tf.nn.softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.name_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)

        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

@tf.function
def mlp(x, scope, n_state):
    with tf.name_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

@tf.function
def block(x, scope, attention_head_count):
    '''
    A Transformer-decoder block.

    :note:
        These are stacked to create the final model.

    '''

    with tf.name_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, attention_head_count)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4)
        x = x + m
        return x, present

@tf.function
def expand_tile(value, size):
    '''
    Add a new axis of given size.
    
    '''

    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

class Transformer(tf.keras.Model):
    '''
    A Transformer-decoder model designed to generate music based on an
    MIDI-like event-based description language.

    :note:
        See :mod:`composer.dataset.sequence` for more information about the 
        event-based sequence description.

    :ivar event_vocab_size:
        The size of the MIDI-like event-based description vocabulary.
        This is the dimensionality of a one-hot vector encoded representation of an event.
    :ivar embedding_size:
        The number of units in the embedding layer.
    :ivar decoder_layers_count:
        The number of decoder blocks.
    :ivar attention_head_count:
        The size of an attention head unit.
    :ivar scope:
        The name of the variable scope.
    :ivar reuse_scope:
        Indicates whether values in the variable scope should be reused between runs.
        Defaults to ``False``.

    '''

    def __init__(self, event_vocab_size, embedding_size, decoder_layers_count, 
                 attention_head_count, scope='model', reuse_scope=False):
        '''
        Initialize an instance of :class:`Transformer`.

        :param event_vocab_size:
            The size of the MIDI-like event-based description vocabulary.
            This is the dimensionality of a one-hot vector encoded representation of an event.
        :param embedding_size:
            The number of units in the embedding layer.
        :param decoder_layers_count:
            The number of decoder blocks.
        :param attention_head_count:
            The size of an attention head unit.
        :param scope:
            The name of the variable scope.
        :param reuse_scope:
            Indicates whether values in the variable scope should be reused between runs.
            Defaults to ``False``.

        '''
        
        self.event_vocab_size = event_vocab_size
        self.embedding_size = embedding_size
        self.decoder_layers_count = decoder_layers_count
        self.attention_head_count = attention_head_count

        self.scope = scope
        self.reuse_scope = reuse_scope 

    def call(self, inputs):
        '''
        Feed forward call on this network.

        :param inputs:
            The inputs to the network.
        
        '''

        with tf.name_scope(self.scope, reuse=self.reuse_scope):
            batch_size, window_size = shape_list(inputs)

            wte_shape = [self.event_vocab_size, self.embedding_size]
            wte = tf.Variable(tf.random_normal_initializer(stddev=0.02)(wte_shape), name='wte')
            h = tf.gather(wte, inputs)

            # Transformer
            presents = []
            for layer in range(self.decoder_layers_count):
                h, present = block(h, 'h%d' % layer, self.attention_head_count)
                presents.append(present)
            
            outputs = {}
            outputs['present'] = tf.stack(presents, axis=1)
            h = norm(h, 'ln_f')

            # Language model loss. Do tokens <n predict token n?
            h_flat = tf.reshape(h, [batch_size * window_size, self.embedding_size])
            logits = tf.matmul(h_flat, wte, transpose_b=True)
            logits = tf.reshape(logits, [batch_size, window_size, self.event_vocab_size])

            outputs['logits'] = logits
            return outputs