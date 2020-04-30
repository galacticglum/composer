'''
A decoder-only Transformer model for music generation.

Source: https://github.com/openai/gpt-2/blob/master/src/model.py

The model implementation is based on the GPT-2 source code.
It is modified so that the code style is consistent and to include
a memory-efficient relative attention implementation.

'''

import logging
import tensorflow as tf
from pathlib import Path
from composer.models import BaseModel, ModelSaveFrequencyMode

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

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    '''
    Normalize to mean = 0, std = 1, then do a diagonal affine transform.
    '''

    with tf.compat.v1.variable_scope(scope):
        n_state = x.shape[-1].value
        
        g = tf.compat.v1.get_variable('g', [n_state], initializer=tf.compat.v1.constant_initializer(1))
        b = tf.compat.v1.get_variable('b', [n_state], initializer=tf.compat.v1.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x

def split_states(x, n):
    '''
    Reshape the last dimension of x into [n, x.shape[-1] / n].

    '''

    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])

def merge_states(x):
    '''
    Smash the last two dimensions of x into a single dimension.
    
    '''

    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    '''
    One-dimensional convolution layer.

    '''

    with tf.compat.v1.variable_scope(scope):
        *start, nx = shape_list(x)
        
        w = tf.compat.v1.get_variable('w', [1, nx, nf], initializer=tf.compat.v1.random_normal_initializer(stddev=w_init_stdev))
        b = tf.compat.v1.get_variable('b', [nf], initializer=tf.compat.v1.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
        return c

def attention_mask(nd, ns, *, dtype):
    '''
    1's in the lower triangle, counting from the lower right corner.
    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

    
    '''

    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def attn(x, scope, n_state, attention_head_count):
    '''
    Memory-efficient relative attention unit.

    '''

    # Should be [batch, sequence, features]
    assert x.shape.ndims == 3
    assert n_state % attention_head_count == 0

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, attention_head_count), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype)*(1 - b)
        return w
    
    def relative_attn(q):
        # q have shape [batch, heads, sequence, features]
        batch, heads, sequence, features = shape_list(q)
        
        E = tf.compat.v1.get_variable('E', [heads, sequence, features])
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

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w + relative_attn(q)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = tf.nn.softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.compat.v1.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)

        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present

def mlp(x, scope, n_state):
    with tf.compat.v1.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2

def block(x, scope, attention_head_count):
    '''
    A Transformer-decoder block.
    :note:
        These are stacked to create the final model.
    '''

    with tf.compat.v1.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, attention_head_count)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4)
        x = x + m
        return x, present

def expand_tile(value, size):
    '''
    Add a new axis of given size.
    
    '''

    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def _transformer_model(inputs, vocab_size, embedding_size, attention_head_count,
                       decoder_layers_count, scope='model', reuse_scope=False):
    '''
    Run a single step of the Transformer-decoder model.

    :param vocab_size:
            The size of the MIDI-like event-based description vocabulary.
            This is the dimensionality of a one-hot vector encoded representation of an event.
    :param embedding_size:
        The number of units in the embedding layer.
    :param attention_head_count:
        The number of attention heads.
    :param decoder_layers_count:
        The number of decoder blocks.
    :param scope:
        The name of the variable scope.
    :param reuse_scope:
        Indicates whether values in the variable scope should be reused between runs.
        Defaults to ``False``.
    :returns:
        The probability distribution of the next event in the sequence (given as logits)
        and a list of the present states.

    '''

    with tf.compat.v1.variable_scope(scope, reuse=reuse_scope):
        batch, sequence = shape_list(inputs)
        
        wte = tf.compat.v1.get_variable('wte', [vocab_size, embedding_size],
                                        initializer=tf.compat.v1.random_normal_initializer(stddev=0.02))        
        h = tf.gather(wte, inputs)

        # Transformer
        presents = []
        for layer in range(decoder_layers_count):
            h, present = block(h, 'h%d' % layer, attention_head_count)
            presents.append(present)
        
        presents = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')
        
        # Language model loss. Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, embedding_size])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, vocab_size])
        return logits, presents

class Transformer(BaseModel):
    '''
    A Transformer-decoder model designed to generate music based on an
    MIDI-like event-based description language.

    :note:
        See :mod:`composer.dataset.sequence` for more information about the 
        event-based sequence description.

    '''

    def __init__(self, event_vocab_size, embedding_size, attention_head_count,
                 decoder_layers_count, scope='model', reuse_scope=False):
        '''
        Initialize an instance of :class:`Transformer`.

        :param event_vocab_size:
            The size of the MIDI-like event-based description vocabulary.
            This is the dimensionality of a one-hot vector encoded representation of an event.
        :param embedding_size:
            The number of units in the embedding layer.
        :param attention_head_count:
            The number of attention heads.
        :param decoder_layers_count:
            The number of decoder blocks.
        :param scope:
            The name of the variable scope.
        :param reuse_scope:
            Indicates whether values in the variable scope should be reused between runs.
            Defaults to ``False``.

        '''

        self.event_vocab_size = event_vocab_size
        self.embedding_size = embedding_size
        self.attention_head_count = attention_head_count
        self.decoder_layers_count = decoder_layers_count
        self.scope = scope
        self.reuse_scope = reuse_scope

    def train(self, dataset, logdir, restoredir=None, epochs=None, learning_rate=1e-3,
              save_frequency_mode=ModelSaveFrequencyMode.EPOCH, save_frequency=1,
              max_checkpoints=1, checkpoint_name_format='model-{global_step}gs',
              show_progress_bar=True):
        '''
        Fit the model to the specified ``dataset``.

        :param dataset:
            An iterable object containing feature, label pairs (as tuples).
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
        :param save_frquency_mode:
            A :class:`composer.models.ModelSaveFrequency` indicating the units of 
            the model save frequency. This can also be a string value corresponding
            to the enum value. Defaults to :class:`ModelSaveFrequencyMode.EPOCH`.
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

        save_frequency = ModelSaveFrequencyMode(save_frequency_mode)
        tf.compat.v1.reset_default_graph()

        batch_size, window_size = shape_list(next(dataset)[0])
        X = tf.compat.v1.placeholder(tf.int32, [None, window_size])
        Y = tf.compat.v1.placeholder(tf.int32, [None, window_size])

        hparams = {
            'vocab_size': self.event_vocab_size,
            'embedding_size': self.embedding_size,
            'attention_head_count': self.attention_head_count,
            'decoder_layers_count': self.decoder_layers_count,
            'scope': self.model,
            'reuse_scope': self.reuse_scope
        }

        logits, _ = _transformer_model(X, **hparams)
        loss = tf.compat.v1.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits))
        accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(tf.compat.v1.equal(Y, tf.argmax(logits, 1)), tf.float32))

        summary_loss = tf.compat.v1.summary.scalar('loss', loss)
        summary_accuracy = tf.compat.v1.summary.scalar('accuracy', accuracy)

        global_step = tf.compat.v1.Variable(0, name='global_step')
        learning_rate = tf.compat.v1.Variable(learning_rate, name='learning_rate')
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

        session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto())

        # Initialize local and global variables
        session.run(tf.compat.v1.local_variables_initializer())
        session.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver(max_to_keep=max_checkpoints)

        # Restore the model, if exists
        logdir = Path(logdir)
        if restoredir is not None:
            checkpoint = tf.compat.v1.latest_checkpoint(restoredir)
            try:
                saver.restore(session, checkpoint)
                logdir = Path(restoredir)
                logging.info('Model restored from \'{}\'.'.format(restoredir))
            except:
                logging.error('Failed to restore model from \'{}\'.'.format(restoredir))
                exit(1)

        # Tensorboard logging
        summary_log = tf.compat.v1.summary.FileWriter(logdir / 'train')

        def _resolve_checkpoint_name(epoch, global_step):
            if isinstance(checkpoint_name_format, str):
                return checkpoint_name_format.format(epoch=epoch, global_step=global_step)
            else:
                # If not a string, assume it is a method... If not, it will error out.
                return checkpoint_name_format(epoch, global_step)

        current_epoch = 0
        steps_per_epoch = None
        while epochs is None or current_epoch < epochs:
            logger.info('Epoch {}'.format(str(current_epoch + 1) if epochs is None else '{}/{}'.format(current_epoch + 1, epochs)))
            with tqdm(total=steps_per_epoch, disable=not show_progress_bar) as progress_bar:
                for x, y in dataset:
                    ops = [train_step, global_step, loss, summary_loss, accuracy, summary_accuracy]
                    _, _global_step, _loss, _summary_loss, _accuracy, _summary_accuracy = session.run(ops, feed_dict={X: x, Y: Y})
                    
                    summary_log.add_summary(_summary_loss, _global_step)
                    summary_log.add_summary(_summary_accuracy, _global_step)

                    # Update description of progress bar to show loss and accuracy statistics
                    progress_bar.set_description('- loss: {:.4f} - accuracy: {:.4f}'.format(_loss, _accuracy))

                    if save_frequency_mode == ModelSaveFrequencyMode.GLOBAL_STEP and _global_step % save_frequency:
                        saver.save(session, logdir / _resolve_checkpoint_name(current_epoch, _global_step), global_step=_global_step)
                    
                    progress_bar.update(1)

                if save_frequency_mode == ModelSaveFrequencyMode.EPOCH and current_epoch % save_frequency:
                    saver.save(session, logdir / _resolve_checkpoint_name(current_epoch, _global_step), global_step=_global_step)
                
                if steps_per_epoch is None:
                    steps_per_epoch = progress_bar.n

                current_epoch += 1
