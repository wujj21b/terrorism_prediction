import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

import numpy as np

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    with open('/data/Experiment/random_seed.txt','r') as f:
        seed=int(f.read())
    tf.set_random_seed(seed)
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    with open('/data/Experiment/random_seed.txt','r') as f:
        seed=int(f.read())
    tf.set_random_seed(seed)
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    with open('/data/Experiment/random_seed.txt','r') as f:
        seed=int(f.read())
    tf.set_random_seed(seed)
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    with open('/data/Experiment/random_seed.txt','r') as f:
        seed=int(f.read())
    tf.set_random_seed(seed)
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
