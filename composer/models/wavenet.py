'''
WaveNet: a deep generative model for raw audio. As first proposed by DeepMind
researchers Oord et al., WaveNet models perform incredibly well in audio
generation tasks with global and local conditioning (i.e. text-to-speech).

The model implementation is based on NVIDIA's implementation of the model [1].
It has been ported from PyTorch to TensorFlow (Keras). Also included is
a modified convolution module intended for fast inference (based on
"Fast Wavenet Generation Algorithm" by Paine T.L., et al. [2]).

Sources:
    1. https://github.com/NVIDIA/nv-wavenet/blob/master/pytorch/wavenet.py
    2. https://arxiv.org/abs/1611.09482

'''

# TODO: Finish WaveNet model implementation for MIDI conditioning (Midi2Wave).
# Note: the implementation of this model (and by extension the WaveNet autoencoder
#       used to synthesize MIDI files) is incomplete and in a dormant state at this time.

import tensorflow as tf
from collections import deque
from tensorflow.keras import Model, layers

class FastConv1D(layers.Layer):
    '''
    A one-dimensional convolution layer with fast inference support.

    '''

    def __init__(self, filters, kernel_size=1, strides=1, dilation_rate=1,
                 use_bias=True, is_casual=False, use_activation=False, **kwargs):
        '''
        Initializes an instance of :class:`FastConv1D`.

        :param filters:
            The dimensionality of the output space (i.e. the number of output filters in the convolution).
        :param kernel_size:
            An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
            Defaults to 1.
        :param strides:
            An integer or tuple/list of a single integer, specifying the stride length of the convolution.
            Defaults to 1.
        :param dilation_rate:
            An integer or tuple/list of a single integer, specifying the dilation rate to use for dilated convolution.
            Defaults to 1.
        :param use_bias:
            Indicates whether the layer uses a bias vector. Defaults to ``True``.
        :param is_causal:
            Indicates whether to perform causal convolutions. Defaults to ``False``.
        :param use_activation:
            Indicates whether to use an activation function (softsign). Defaults to ``False``.

        '''

        super().__init__(**kwargs)
        
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_activation = use_activation
        self.is_causal = is_causal

        self.conv1d = layers.Conv1D(filters, kernel_size, strides=strides,
                                    dilation_rate=dilation_rate, use_bias=use_bias)
    
    def call(self, inputs, training=False):
        '''
        Gets convolutions on the specified ``inputs``.

        :param inputs:
            A 3-dimensional float32 tensor of shape [batch, sequence, features].
        :param training:
            Indicates whether this step is training. Defaults to ``False``.
        :returns:
            A float32 tensor with shape [batch_size, length, kernel_size].

        '''

        if training:
            if self.is_causal:
                padding = (int((self.kernel_size - 1) * (self.dilation_rate)), 0)
                inputs = tf.pad(inputs, padding)

            inputs = self.conv1d(inputs)
            if self.use_activation:
                inputs = tf.keras.activations.softsign(inputs)
            
            return inputs
        else:
            input_shape = inputs.shape
            if len(input_shape) <= 2:
                inputs = tf.expand_dims(inputs, -1)
            
            if input_shape[-1] > 1:
                inputs = inputs[:, :, -1]
            
            if self.kernel_size == 1:
                return self.conv1d(inputs)
            elif self.is_causal:
                # Initialize the input memory
                if self.input_memory is None:
                    self.input_memory = deque()
                    for i in range(self.dilation_rate):
                        buffer = tf.zeros((inputs.shape[0], self.input_size, 1))
                        self.input_memory.append(buffer)

                self.input_memory.appendleft(tf.identity(inputs))
                x0 = self.input_memory.pop()
                return self.conv1d(tf.concat((x0, inputs), 2))

class Upsampler(layers.Layer):
    '''
    Upsamples an input tensor of shape [x, y, z] into a new tensor
    of shape [x, y, z * c] where c is the upscale ratio.

    '''

    def __init__(self, ratio):
        '''
        Initializes an instance of :class:`Upsampler`.

        :param ratio:
            The ratio by which to scale the input tensor.

        '''

        self.ratio = ratio

    def call(self, inputs):
        '''
        Upscales the input tensor.

        :param inputs:
            A tensor of shape [x, y, z].
        :returns:
            A tensor with shape [x, y, z * c].

        '''

        inputs = tf.expand_dims(inputs, 2)
        input_shape = inputs.shape
        inputs = tf.broadcast_to(inputs, (input_shape[0], input_shape[1], self.ratio]))
        inputs = tf.reshape(tf.tranpose(inputs, perm=(2, 3)), [-1])
        return inputs

class QuantizedInput(layers.Layer):
    '''
    Quantizes the inputs to this layer into 256 mu-quantized audio values.

    '''

    def __init__(self, vocab_size, embedding_size, use_activation=False):
        '''
        Initializes an instance of :class:`QuantizedInput`.

        :param vocab_size:
            An integer or tuple/list of a single integer, specifying the size of the input space
            (i.e. the number of unique integer values for which there exist a dense float vector mapping).
        :param embedding_size:
            The number of units in the embedding layer; the dimensionality of the output space
            (i.e. the dimensionality of the output densse float vectors).
        :param use_activation:
            Indicates whether to use an activation function (softsign). Defaults to ``False``.

        '''

        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.use_activation = use_activation

    def call(self, inputs):
        '''
        Quantize the specified input.

        '''

        inputs = self.embedding(inputs)
        inputs = tf.transpose(inputs, perms=(1, 2))
        if self.use_activation:
            inputs = tf.keras.activations.softsign(inputs)
        
        return inputs