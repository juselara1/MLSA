import tensorflow as tf

def kernel_mse(K, K_tilde):
    return tf.reduce_mean(-2*tf.linalg.diag(tf.matmul(K, tf.transpose(K_tilde)))\
                          +tf.linalg.diag(tf.matmul(K_tilde, tf.matmul(K, tf.transpose(K_tilde)))), axis=1)

class losses():
    def __init__(self):
        self.kernel_mse = kernel_mse