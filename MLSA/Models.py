import tensorflow as tf
import MLSA.Alignments

class LatentSemanticEmbedding(tf.keras.Model):
    def __init__(self, latent_dim, N):
        super(LatentSemanticEmbedding, self).__init__()
        
        self.encoder = tf.keras.layers.Dense(units=latent_dim, use_bias=False)
        self.decoder = tf.keras.layers.Dense(units=N, use_bias=False)

    def call(self, input_tensor, training=True):
        H = self.encoder(input_tensor, training=training)
        K_tilde = self.decoder(H, training=training)
        return K_tilde, H

class Alignment(tf.keras.layers.Layer):
    def __init__(self, align_fun=MLSA.Alignments.cosine,
                 activity_regularizer=tf.keras.regularizers.l2(0.1), **kwargs):
        super(Alignment, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.align_fun = align_fun
        
    def call(self, x):
        align = self.align_fun(x[0], x[1])
        return align
    
    def build(self, input_shape):
        super(Alignment, self).build(input_shape)
    
class MultilayerDropout(tf.keras.Model):
    def __init__(self, units, activation, rate, n_classes):
        super(MultilayerDropout, self).__init__()
        self.dense_layers = []
        for unit in units:
            self.dense_layers.append(tf.keras.layers.Dense(units=unit, activation=activation))
            self.dense_layers.append(tf.keras.layers.Dropout(rate))
        self.dense_layers.append(tf.keras.layers.Dense(units=n_classes, activation="softmax"))
    
    def call(self, input_tensor, training=True):
        y_tilde = input_tensor
        for layer in self.dense_layers:
            y_tilde = layer(y_tilde, training=training)
        return y_tilde
    
class SupervisedLatentSemanticEmbedding(tf.keras.Model):
    def __init__(self, latent_dim, N, units=[64], activation="relu", rate=0.2, n_classes=2):
        super(SupervisedLatentSemanticEmbedding, self).__init__()
        self.embedding = LatentSemanticEmbedding(latent_dim, N)
        self.mlp = MultilayerDropout(units, activation, rate, n_classes)
    
    def call(self, input_tensor, training=True):
        K_tilde, H = self.embedding(input_tensor)
        Y_tilde = self.mlp(H)
        return K_tilde, H, Y_tilde
        
class MultimodalLatentSemanticAlignment(tf.keras.Model):
    def __init__(self, latent_dim, N, align_fun=MLSA.Alignments.cosine,
                 align_regularizer=tf.keras.regularizers.l2(0.1),
                 units=[64], activation="relu", rate=0.2, n_classes=2):
        super(MultimodalLatentSemanticAlignment, self).__init__()
        self.visual_model = SupervisedLatentSemanticEmbedding(latent_dim, N, units, activation, rate, n_classes)
        self.text_model = SupervisedLatentSemanticEmbedding(latent_dim, N, units, activation, rate, n_classes)
        self.alignment = Alignment(align_fun, align_regularizer)

    def call(self, input_tensors, training=True):
        K_v_tilde, H_v, Y_tilde_v = self.visual_model(input_tensors[0])
        K_t_tilde, H_t, Y_tilde_t = self.text_model(input_tensors[1])
        align = self.alignment([H_v, H_t])
        return K_v_tilde, K_t_tilde, Y_tilde_v, Y_tilde_t
    
class models():
    def __init__(self):
        self.LatentSemanticEmbedding = LatentSemanticEmbedding
        self.Alignment = Alignment
        self.MultilayerDropout = MultilayerDropout
        self.SupervisedLatentSemanticEmbedding = SupervisedLatentSemanticEmbedding
        self.MultimodalLatentSemanticAlignment = MultimodalLatentSemanticAlignment