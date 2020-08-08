import tensorflow as tf
import numpy as np

class CrossEntropyCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, path):
        self.validation_data = validation_data
        self.path = path
        self.cur_loss = np.inf
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.validation_data[0], batch_size=self.validation_data[1].shape[0])
        loss = np.mean(tf.losses.categorical_crossentropy(self.validation_data[1], preds[2]))
        if loss<self.cur_loss:
            self.cur_loss = loss
            self.model.save_weights(self.path)
            
class CrossEntropyCallbackUnimodal(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, path):
        self.validation_data = validation_data
        self.path = path
        self.cur_loss = np.inf
    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.validation_data[0], batch_size=self.validation_data[1].shape[0])
        loss = np.mean(tf.losses.categorical_crossentropy(self.validation_data[1], preds[1]))
        if loss<self.cur_loss:
            self.cur_loss = loss
            self.model.save_weights(self.path)
        
class callbacks(object):
    def __init__(self):
        self.CrossEntropyCallback = CrossEntropyCallback
        self.CrossEntropyCallbackUnimodal = CrossEntropyCallbackUnimodal