import tensorflow as tf

def cosine(H_1, H_2):
    norm_H_1 = tf.nn.l2_normalize(H_1, axis=1)
    norm_H_2 = tf.nn.l2_normalize(H_2, axis=1)
    return 1-tf.reduce_sum(norm_H_1*norm_H_2, axis=1)

def euclidean(H_1, H_2):
    return tf.sqrt(tf.reduce_sum((H_1-H_2)**2, axis=1))

def manhattan(H_1, H_2):
    return tf.reduce_sum(tf.math.abs(H_1-H_2), axis=1)

class alignments():
    def __init__(self):
        self.cosine = cosine
        self.euclidean = euclidean
        self.manhattan = manhattan