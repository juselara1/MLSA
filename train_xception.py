import numpy as np
import tensorflow as tf
from utils.generator import ImageAugmentationSequenceH5

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
augmentation = {"preprocessing_function": tf.keras.applications.xception.preprocess_input, "apply":True,
                "random_brightness": {"max_delta": 0.3},
                "random_contrast": {"lower":0.7, "upper":1},
                "random_hue": {"max_delta": 0.1},
                "random_saturation": {"lower": 0.7, "upper":1},
                "random_rotation": {"minval": 0, "maxval": 2*np.pi},
                "horizontal_flip": True, "vertical_flip": True
               }
train_seq = ImageAugmentationSequenceH5("/tf/TCGA/images.h5", "ims_train", "y_train", num_classes=2,
                                        labels_transform=lambda i:np.int32(i>1),
                                        class_weighted=True, categorical=True, batch_size=32,
                                        augmentation = augmentation, shuffle=True)

augmentation = {"preprocessing_function": tf.keras.applications.xception.preprocess_input, "apply":False}
val_seq = ImageAugmentationSequenceH5("/tf/TCGA/images.h5", "ims_val", "y_val", num_classes=2,
                                        labels_transform=lambda i:np.int32(i>1),
                                        class_weighted=False, categorical=True, batch_size=16,
                                        augmentation = augmentation, shuffle=False)

input_tensor = tf.keras.layers.Input(shape=(256, 256, 3))
xception_base = tf.keras.applications.xception.Xception(weights="imagenet",
                                                          input_tensor=input_tensor,
                                                          include_top=False)
pool = tf.keras.layers.GlobalAveragePooling2D()(xception_base.output)
dense = tf.keras.layers.Dense(1024, activation="relu")(pool)
dropout = tf.keras.layers.Dropout(0.2)(dense)
output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
model = tf.keras.Model(inputs=input_tensor, outputs=output)
model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-6), 
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

callback = tf.keras.callbacks.ModelCheckpoint("xception.h5")
model.fit(train_seq, steps_per_epoch=len(train_seq), validation_data=val_seq, validation_steps=len(val_seq),
          epochs=10, callbacks=[callback], shuffle=False, use_multiprocessing=False)