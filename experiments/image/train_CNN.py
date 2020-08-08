import numpy as np
import tensorflow as tf
import argparse, sys
sys.path.append("../../")
from utils.generator import ImageAugmentationSequenceH5


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Trains the specified CNN.')
    parser.add_argument("--model", type=str, help="Specify the CNN architecture {inceptionv3, densenet201, inception_resnetv2, xception, resnet152v2}", default="xception")
    parser.add_argument("--save_path", type=str, help="Path to save the model.", default="models2/xception.h5")
    parser.add_argument("--data_path", type=str, help="Path for the data (hdf5).", default="/tf/TCGA/images.h5")
    
    args = parser.parse_args()
    
    input_tensor = tf.keras.layers.Input(shape=(256, 256, 3))
    base_params = {"weights": "imagenet", "input_tensor": input_tensor, "include_top": False, "pooling": "avg"}
    if args.model=="densenet201":
        preprop = tf.keras.applications.densenet.preprocess_input
        base = tf.keras.applications.densenet.DenseNet201(**base_params)
        train_batch = 32
        val_batch = 8
    elif args.model=="inceptionv3":
        preprop = tf.keras.applications.inception_v3.preprocess_input
        base = tf.keras.applications.inception_v3.InceptionV3(**base_params)
        train_batch = 64
        val_batch = 16
    elif args.model=="resnet152v2":
        preprop = tf.keras.applications.resnet_v2.preprocess_input
        base = tf.keras.applications.resnet_v2.ResNet152V2(**base_params)
        train_batch = 32
        val_batch = 16
    elif args.model=="inception_resnetv2":
        preprop = tf.keras.applications.inception_resnet_v2.preprocess_input
        base = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(**base_params)
        train_batch = 64
        val_batch = 16
    elif args.model=="xception":
        preprop = tf.keras.applications.xception.preprocess_input
        base = tf.keras.applications.xception.Xception(**base_params)
        train_batch = 32
        val_batch = 16
    augmentation = {"preprocessing_function": preprop, "apply":True,
                    "random_brightness": {"max_delta": 0.3},
                    "random_contrast": {"lower":0.7, "upper":1},
                    "random_hue": {"max_delta": 0.1},
                    "random_saturation": {"lower": 0.7, "upper":1},
                    "random_rotation": {"minval": 0, "maxval": 2*np.pi},
                    "horizontal_flip": True, "vertical_flip": True
                   }
    train_seq = ImageAugmentationSequenceH5(args.data_path, "ims_train", "y_train", num_classes=2,
                                            labels_transform=lambda i:np.int32(i>1),
                                            class_weighted=True, categorical=True, batch_size=train_batch,
                                            augmentation = augmentation, shuffle=True)

    augmentation = {"preprocessing_function": preprop, "apply":False}
    val_seq = ImageAugmentationSequenceH5(args.data_path, "ims_val", "y_val", num_classes=2,
                                          labels_transform=lambda i:np.int32(i>1),
                                          class_weighted=False, categorical=True, batch_size=val_batch,
                                          augmentation = augmentation, shuffle=False)

    dense = tf.keras.layers.Dense(1024, activation="relu")(base.output)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-7), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.save_path, monitor="val_loss", 
                                                  verbose=True, save_best_only=True,
                                                  save_weights_only=True, mode="min")
    model.fit(train_seq, steps_per_epoch=len(train_seq), validation_data=val_seq, validation_steps=len(val_seq),
              epochs=10, callbacks=[callback], shuffle=False, use_multiprocessing=False)
