import numpy as np
import pandas as pd
import sys, h5py, argparse, os
sys.path.append("../..")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import MLSA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import chi2_kernel, cosine_similarity, rbf_kernel

if __name__ == "__main__":
    os.system("clear")
    parser = argparse.ArgumentParser(description='Random search for a given combination of kernels.')
    parser.add_argument("--codebook", type=int, action="store", dest="codebook",
                        help="Codebook size of the BoVW", default=1700)
    parser.add_argument("--ngram", type=int, action="store", dest="ngram",
                        help="Number of N grams for the texts", default=2)
    parser.add_argument("--visual_kernel", type=str, action="store", dest="visual_kernel",
                        help="Kernel function for the visual modality {linear, cosine, rbf, chi2}", default="linear")
    parser.add_argument("--text_kernel", type=str, action="store", dest="text_kernel",
                        help="Kernel function for the text modality {linear, cosine, rbf, chi2}", default="linear")
    parser.add_argument("--hyperparameters", type=str, action="store", dest="hyperparameters",
                        help="Path for the CSV file of the random combination of hyperparameters.", default="params.csv")
    parser.add_argument("--training_type", type=str, action="store", dest="training_type",
                        help="Specify the type of training {unimodal, multimodal}.", default="multimodal")
    parser.add_argument("--classification_type", type=str, action="store", dest="classification_type",
                        help="Specify the classification type {binary, multiclass}.", default="binary")
    parser.add_argument("--bovw_path", type=str, action="store", dest="bovw_path",
                        help="Path of the BoVW hdf5 file.", default="/tf/TCGA/bovw.h5")
    parser.add_argument("--bow_path", type=str, action="store", dest="bow_path",
                        help="Path of the BoW hdf5 file.", default="/tf/TCGA/bow.h5")
    parser.add_argument("--save_path", type=str, action="store", dest="save_path",
                        help="Path to save the best hyperparameters", default="best_params/best_params.csv")
    
    args = parser.parse_args()
    temp_path = f"temp{args.training_type}-{args.visual_kernel}-{args.text_kernel}.h5"
    params = pd.read_csv(args.hyperparameters, index_col=0)
    with h5py.File(args.bovw_path, "r") as df:
        X_train_v = df[f"X_train{args.codebook}"][:]
        X_val_v = df[f"X_val{args.codebook}"][:]
        tfidf = TfidfTransformer(sublinear_tf=True).fit(X_train_v)
        X_train_v = tfidf.transform(X_train_v).toarray()
        X_val_v = tfidf.transform(X_val_v).toarray()

        if args.classification_type == "binary":
            y_train = df["y_train"][:]>1
            y_val = df["y_val"][:]>1
            n_classes = 2
        elif args.classification_type == "multiclass":
            y_train = df["y_train"][:]
            y_val = df["y_val"][:]
            n_classes = 5

    Y_train = tf.keras.utils.to_categorical(y_train)
    Y_val = tf.keras.utils.to_categorical(y_val)      

    with h5py.File(args.bow_path, "r") as df:
        X_train_t = df[f"X_train{args.ngram}"][:]
        X_val_t = df[f"X_val{args.ngram}"][:]
        tfidf = TfidfTransformer(sublinear_tf=True).fit(X_train_t)
        X_train_t = tfidf.transform(X_train_t).toarray()
        X_val_t = tfidf.transform(X_val_t).toarray()
        
    best_loss = np.inf
    for exp in range(params.shape[0]):
        if args.visual_kernel == "linear":
            K_train_v = np.float32(X_train_v @ X_train_v.T)
            K_val_v = np.float32(X_val_v @ X_train_v.T)
        elif args.visual_kernel == "cosine":
            K_train_v = cosine_similarity(X_train_v, X_train_v)
            K_val_v = cosine_similarity(X_val_v, X_train_v)
        elif args.visual_kernel == "rbf":
            K_train_v = rbf_kernel(X_train_v, X_train_v, gamma=params["rbf_im"].loc[exp])
            K_val_v = rbf_kernel(X_val_v, X_train_v, gamma=params["rbf_im"].loc[exp])
        elif args.visual_kernel == "chi2":
            K_train_v = chi2_kernel(X_train_v, X_train_v, gamma=params["chi_im"].loc[exp])
            K_val_v = chi2_kernel(X_val_v, X_train_v, gamma=params["chi_im"].loc[exp])

        if args.text_kernel == "linear":
            K_train_t = np.float32(X_train_t @ X_train_t.T)
            K_val_t = np.float32(X_val_t @ X_train_t.T)
        elif args.text_kernel == "cosine":
            K_train_t = cosine_similarity(X_train_t, X_train_t)
            K_val_t = cosine_similarity(X_val_t, X_train_t)
        elif args.text_kernel == "rbf":
            K_train_t = rbf_kernel(X_train_t, X_train_t, gamma=params["rbf_txt"].loc[exp])
            K_val_t = rbf_kernel(X_val_t, X_train_t, gamma=params["rbf_txt"].loc[exp])
        elif args.text_kernel == "chi2":
            K_train_t = chi2_kernel(X_train_t, X_train_t, gamma=params["chi_txt"].loc[exp])
            K_val_t = chi2_kernel(X_val_t, X_train_t, gamma=params["chi_txt"].loc[exp])

        if params["align"].loc[exp] == "cosine":
            align = MLSA.Alignments.cosine
        elif params["align"].loc[exp] == "euclidean":
            align = MLSA.Alignments.euclidean
        elif params["align"].loc[exp] == "manhattan":
            align = MLSA.Alignments.manhattan

        if args.training_type == "unimodal":
            input_v = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            K_v, H_v, Y_v = MLSA.Models.SupervisedLatentSemanticEmbedding(params["latent_dim"].loc[exp], K_train_t.shape[0],
                                                                          units=[params["units"].loc[exp]], 
                                                                          activation=params["activation"].loc[exp],
                                                                          rate=0.2, n_classes=n_classes)(input_v)
            model = tf.keras.Model(inputs=[input_v], outputs=[K_v, Y_v])
            model.compile(loss=[MLSA.Losses.kernel_mse, "categorical_crossentropy"],
                          loss_weights=[0.1, 1.], optimizer=tf.optimizers.Adam(lr=1e-4))
            callback = MLSA.Callbacks.CrossEntropyCallbackUnimodal(validation_data=(K_val_v, Y_val), path=temp_path)
            model.fit(K_train_v, [K_train_v, Y_train], epochs=1000, batch_size=K_train_t.shape[0],
                      callbacks=[callback], verbose=0)
            model.load_weights(temp_path)

            preds = model.predict(K_val_v, batch_size=Y_val.shape[0])[1]
            loss = np.mean(tf.losses.categorical_crossentropy(Y_val, preds))
        elif args.training_type == "multimodal":
            input_v = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            input_t = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            K_v, K_t, Y_v, Y_t = MLSA.Models.MultimodalLatentSemanticAlignment(params["latent_dim"].loc[exp], K_train_t.shape[0],
                                                                               align_fun=align,
                                                                               align_regularizer=tf.keras.regularizers.l2(1.0),
                                                                               units=[params["units"].loc[exp]],
                                                                               activation=params["activation"].loc[exp],
                                                                               rate=0.2, n_classes=n_classes)([input_v, input_t])
            model = tf.keras.Model(inputs=[input_v, input_t], outputs=[K_v, K_t, Y_v, Y_t])
            model.compile(loss=[MLSA.Losses.kernel_mse, MLSA.Losses.kernel_mse, "categorical_crossentropy", "categorical_crossentropy"],
                          loss_weights=[0.1, 0.1, 1., 1.], optimizer=tf.optimizers.Adam(lr=1e-4))
            callback = MLSA.Callbacks.CrossEntropyCallback(validation_data=([K_val_v, K_val_t], Y_val), path=temp_path)
            model.fit([K_train_v, K_train_t], [K_train_v, K_train_t, Y_train, Y_train], epochs=1000, batch_size=K_train_t.shape[0],
                      callbacks=[callback], verbose=0)
            model.load_weights(temp_path)

            preds = model.predict([K_val_v, K_val_t], batch_size=Y_val.shape[0])[2]
            loss = np.mean(tf.losses.categorical_crossentropy(Y_val, preds))
        print(f"Trial {exp+1}/{params.shape[0]}, saved params as: {args.save_path}", end="\r")
        if loss<best_loss:
            best_loss = loss
            best_params = params.loc[exp].copy()
            best_params["codebook"] = args.codebook
            best_params["ngram"] = args.ngram
            best_params["visual_kernel"] = args.visual_kernel
            best_params["text_kernel"] = args.text_kernel
            best_params.to_csv(args.save_path)
    os.remove(temp_path)