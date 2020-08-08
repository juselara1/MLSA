import numpy as np
import os, h5py, gc, sys, argparse
sys.path.append("../../")
from utils.generator import ImageAugmentationSequenceH5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Extracts and saves a feature vector representation of the patches.')
    parser.add_argument("--model", type=str, help="Model to extract (e.g., inceptionv3, densenet201, inception_resnetv2, resnet152v2, xception)", default="models/inceptionv3")
    parser.add_argument("--weights", type=str, action="store", dest="weights", help="Path for model weights.", default="models/inceptionv3.h5")
    parser.add_argument("--data_path", type=str, action="store", dest="data_path", help="Data path.", default="/tf/TCGA/images.h5")
    parser.add_argument("--save_path", type=str, action="store", dest="save_path", help="Path to save the feature vectors.", default="/tf/TCGA/feature_vectors.h5")
    args = parser.parse_args()
    
    input_tensor = tf.keras.layers.Input(shape=(256, 256, 3))
    base_params = {"weights": "imagenet", "input_tensor": input_tensor, "include_top": False, "pooling": "avg"}
    if args.model=="densenet201":
        preprop = tf.keras.applications.densenet.preprocess_input
        base = tf.keras.applications.densenet.DenseNet201(**base_params)
    elif args.model=="inceptionv3":
        preprop = tf.keras.applications.inception_v3.preprocess_input
        base = tf.keras.applications.inception_v3.InceptionV3(**base_params)
    elif args.model=="resnet152v2":
        preprop = tf.keras.applications.resnet_v2.preprocess_input
        base = tf.keras.applications.resnet_v2.ResNet152V2(**base_params)
    elif args.model=="inception_resnetv2":
        preprop = tf.keras.applications.inception_resnet_v2.preprocess_input
        base = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(**base_params)
    elif args.model=="xception":
        preprop = tf.keras.applications.xception.preprocess_input
        base = tf.keras.applications.xception.Xception(**base_params)
    augmentation = {"preprocessing_function": preprop, "apply":False}
    train_seq = ImageAugmentationSequenceH5(args.data_path, "ims_train", "y_train", num_classes=2,
                                            labels_transform=lambda i:np.int32(i>1), return_y=False,
                                            class_weighted=False, categorical=True, batch_size=10,
                                            augmentation = augmentation, shuffle=False)
    val_seq = ImageAugmentationSequenceH5(args.data_path, "ims_val", "y_val", num_classes=2,
                                            labels_transform=lambda i:np.int32(i>1), return_y=False,
                                            class_weighted=False, categorical=True, batch_size=10,
                                            augmentation = augmentation, shuffle=False)
    test_seq = ImageAugmentationSequenceH5(args.data_path, "ims_test", "y_test", num_classes=2,
                                            labels_transform=lambda i:np.int32(i>1), return_y=False,
                                            class_weighted=False, categorical=True, batch_size=10,
                                            augmentation = augmentation, shuffle=False)

    partitions = [("ims_train", 268730, "y_train", "id_train", train_seq),
                  ("ims_val", 91540, "y_val", "id_val", val_seq),
                  ("ims_test", 87740, "y_test", "id_test", test_seq)]

    dt = h5py.special_dtype(vlen=str)
    with h5py.File(args.data_path, "r") as im_df:
        with h5py.File(args.save_path, "w") as df:
            for im_name, size, label_name, id_name, seq in partitions:
                df.create_dataset(im_name, shape=(size, base.output.shape[1]), dtype=np.float32)
                df.create_dataset(label_name, shape=(size, ), dtype=np.uint8)
                df.create_dataset(id_name, shape=(size, ), dtype=dt)

                df[im_name][:] = base.predict(seq, steps=len(seq), verbose=True)
                df[label_name][:] = im_df[label_name][:]
                df[id_name][:] = im_df[id_name][:]
                gc.collect()