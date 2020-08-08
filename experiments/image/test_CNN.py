import numpy as np
import tensorflow as tf
import argparse, sys, os, h5py
sys.path.append("../../")
from utils.generator import ImageAugmentationSequenceH5
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_full_model(path, base):
    dense = tf.keras.layers.Dense(1024, activation="relu")(base.output)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
    model = tf.keras.Model(inputs=input_tensor, outputs=output)
    model.load_weights(path)
    return model

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a specified model in the test set.')
    parser.add_argument("--model", type=str, help="Model to evaluate (e.g., inceptionv3, densenet201, inception_resnetv2, resnet152v2, xception)", default="inceptionv3")
    parser.add_argument("--weights", type=str, action="store", dest="weights", help="Path for model weights.", default="models/inceptionv3.h5")
    parser.add_argument("--data_path", type=str, action="store", dest="data_path", help="Data path.", default="/tf/TCGA/images.h5")
    args = parser.parse_args()
    os.system("clear")
    
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
    
    model = get_full_model(args.weights, base)
    # Defining test sequence generator    
    augmentation = {"preprocessing_function": preprop, "apply":False}
    test_seq = ImageAugmentationSequenceH5(args.data_path, "ims_test", "y_test", num_classes=2,
                                           labels_transform=lambda i:np.int32(i>1), return_y=False,
                                           class_weighted=False, categorical=True, batch_size=20,
                                           augmentation = augmentation, shuffle=False)
    os.system("clear")
    preds = model.predict(test_seq, steps=len(test_seq), verbose=1)
    y_pred = np.argmax(preds, axis=1)    
    with h5py.File("/tf/TCGA/images.h5", 'r') as df:
        y_test = df["y_test"][:]
    acc = accuracy_score(np.int32(y_test>1), y_pred)
    prec = precision_score(np.int32(y_test>1), y_pred)
    f1 = f1_score(np.int32(y_test>1), y_pred)
    rec = recall_score(np.int32(y_test>1), y_pred)
    names = ["Accuracy", "Precision", "F1", "Recall"]
    print("".join([f'{val:10}' for val in names]))
    print("".join(map(lambda i:f'{i:10}',[f'{val:.4f}' for val in [acc, prec, f1, rec]])))