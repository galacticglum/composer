'''
I/O related utility methods.

'''

import tensorflow as tf

def bytes_feature(value):
    '''
    Returns a :class:`tensorflow.train.Feature` containing a 
    bytes_list from a string / byte.

    '''

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    '''
    Returns a :class:`tensorflow.train.Feature` containing a
    float_list from a float / double.

    '''

    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    '''
    Returns a :class:`tensorflow.train.Feature` containing an 
    int64_list from a bool / enum / int / uint.

    '''

    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))