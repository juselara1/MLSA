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
from utils import retrieval
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import chi2_kernel, cosine_similarity, rbf_kernel
from sklearn.utils import class_weight
np.random.seed(0)
tf.random.set_seed(0)

if __name__ == "__main__":
    os.system("clear")
    parser = argparse.ArgumentParser(description='Evaluates a given model on the test set.')
    parser.add_argument("--hyperparameters", type=str, action="store", dest="hyperparameters",
                        help="CSV file with the model's hyperparameters.", default="best_params/linear_unimodal.csv")
    parser.add_argument("--training_type", type=str, action="store", dest="training_type",
                        help="Specify the type of training {unimodal, multimodal}.", default="multimodal")
    parser.add_argument("--retrieval_type", type=str, action="store", dest="retrieval_type",
                        help="Specify the which representation is used for retrieval {H, Y}.", default="Y")
    parser.add_argument("--bovw_path", type=str, action="store", dest="bovw_path",
                        help="Path of the BoVW hdf5 file.", default="/tf/TCGA/bovw.h5")
    parser.add_argument("--bow_path", type=str, action="store", dest="bow_path",
                        help="Path of the BoW hdf5 file.", default="/tf/TCGA/bow.h5")
    parser.add_argument("--trials", type=int, action="store", dest="trials",
                        help="Number of trials.", default=10)
    args = parser.parse_args()
    params = pd.read_csv(args.hyperparameters, index_col=0)
    
    temp_path = f"temp{args.training_type}-{params.loc['visual_kernel'][0]}-{params.loc['text_kernel'][0]}.h5"
    
    with h5py.File(args.bovw_path, "r") as df:
        X_train_v = df[f"X_train{params.loc['codebook'][0]}"][:]
        X_val_v = df[f"X_val{params.loc['codebook'][0]}"][:]
        X_test_v = df[f"X_test{params.loc['codebook'][0]}"][:]
        tfidf = TfidfTransformer(sublinear_tf=True).fit(X_train_v)
        X_train_v = tfidf.transform(X_train_v).toarray()
        X_val_v = tfidf.transform(X_val_v).toarray()
        X_test_v = tfidf.transform(X_test_v).toarray()
        
        y_train = df["y_train"][:]
        y_val = df["y_val"][:]
        y_test = df["y_test"][:]
        n_classes = 5
    Y_train = tf.keras.utils.to_categorical(y_train)
    Y_val = tf.keras.utils.to_categorical(y_val)
    Y_test = tf.keras.utils.to_categorical(y_test)
    
    rank_test = cosine_similarity(Y_test, Y_train)
    
    with h5py.File(args.bow_path, "r") as df:
        X_train_t = df[f"X_train{params.loc['ngram'][0]}"][:]
        X_val_t = df[f"X_val{params.loc['ngram'][0]}"][:]
        X_test_t = df[f"X_test{params.loc['ngram'][0]}"][:]
        tfidf = TfidfTransformer(sublinear_tf=True).fit(X_train_t)
        X_train_t = tfidf.transform(X_train_t).toarray()
        X_val_t = tfidf.transform(X_val_t).toarray()
        X_test_t = tfidf.transform(X_test_t).toarray()
    
    if params.loc['visual_kernel'][0] == "linear":
        K_train_v = np.float32(X_train_v @ X_train_v.T)
        K_val_v = np.float32(X_val_v @ X_train_v.T)
        K_test_v = np.float32(X_test_v @ X_train_v.T)
    elif params.loc['visual_kernel'][0] == "cosine":
        K_train_v = cosine_similarity(X_train_v, X_train_v)
        K_val_v = cosine_similarity(X_val_v, X_train_v)
        K_test_v = cosine_similarity(X_test_v, X_train_v)
    elif params.loc['visual_kernel'][0] == "rbf":
        K_train_v = rbf_kernel(X_train_v, X_train_v, gamma=float(params.loc["rbf_im"][0]))
        K_val_v = rbf_kernel(X_val_v, X_train_v, gamma=float(params.loc["rbf_im"][0]))
        K_test_v = rbf_kernel(X_test_v, X_train_v, gamma=float(params.loc["rbf_im"][0]))
    elif params.loc['visual_kernel'][0] == "chi2":
        K_train_v = chi2_kernel(X_train_v, X_train_v, gamma=float(params.loc["chi_im"][0]))
        K_val_v = chi2_kernel(X_val_v, X_train_v, gamma=float(params.loc["chi_im"][0]))
        K_test_v = chi2_kernel(X_test_v, X_train_v, gamma=float(params.loc["chi_im"][0]))

    if params.loc['text_kernel'][0] == "linear":
        K_train_t = np.float32(X_train_t @ X_train_t.T)
        K_val_t = np.float32(X_val_t @ X_train_t.T)
        K_test_t = np.float32(X_test_t @ X_train_t.T)
    elif params.loc['text_kernel'][0] == "cosine":
        K_train_t = cosine_similarity(X_train_t, X_train_t)
        K_val_t = cosine_similarity(X_val_t, X_train_t)
        K_test_t = cosine_similarity(X_test_t, X_train_t)
    elif params.loc['text_kernel'][0] == "rbf":
        K_train_t = rbf_kernel(X_train_t, X_train_t, gamma=float(params.loc["rbf_txt"][0]))
        K_val_t = rbf_kernel(X_val_t, X_train_t, gamma=float(params.loc["rbf_txt"][0]))
        K_test_t = rbf_kernel(X_test_t, X_train_t, gamma=float(params.loc["rbf_txt"][0]))
    elif params.loc['text_kernel'][0] == "chi2":
        K_train_t = chi2_kernel(X_train_t, X_train_t, gamma=float(params.loc["chi_txt"][0]))
        K_val_t = chi2_kernel(X_val_t, X_train_t, gamma=float(params.loc["chi_txt"][0]))
        K_test_t = chi2_kernel(X_test_t, X_train_t, gamma=float(params.loc["chi_txt"][0]))
        
    if params.loc["align"][0] == "cosine":
        align = MLSA.Alignments.cosine
    elif params.loc["align"][0] == "euclidean":
        align = MLSA.Alignments.euclidean
    elif params.loc["align"][0] == "manhattan":
        align = MLSA.Alignments.manhattan
    names = ["map", "gm_map", "P_10", "P_30"]
    print("".join(map(lambda name: f"{name:20}", names)))
    
    results = []
    for trial in range(args.trials):
        if args.training_type == "unimodal":
            input_v = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            K_v, H_v, Y_v = MLSA.Models.SupervisedLatentSemanticEmbedding(params.loc["latent_dim"][0], K_train_t.shape[0],
                                                                          units=[params.loc["units"][0]], 
                                                                          activation=params.loc["activation"][0],
                                                                          rate=0.2, n_classes=n_classes)(input_v)
            model = tf.keras.Model(inputs=[input_v], outputs=[K_v, Y_v])
            model.compile(loss=[MLSA.Losses.kernel_mse, "categorical_crossentropy"],
                          loss_weights=[0.1, 1.], optimizer=tf.optimizers.Adam(lr=1e-4))
            callback = MLSA.Callbacks.CrossEntropyCallbackUnimodal(validation_data=(K_val_v, Y_val), path=temp_path)
            model.fit(K_train_v, [K_train_v, Y_train], epochs=1000, batch_size=K_train_t.shape[0],
                      callbacks=[callback], verbose=0)
            model.load_weights(temp_path)
            preds_model = tf.keras.Model(inputs=[input_v], outputs=[K_v, H_v, Y_v])
            if args.retrieval_type == "H":
                query = preds_model.predict(K_test_v, batch_size=Y_test.shape[0])[1]
                target = preds_model.predict(K_train_v, batch_size=Y_train.shape[0])[1]
            elif args.retrieval_type == "Y":
                query = preds_model.predict(K_test_v, batch_size=Y_test.shape[0])[2]
                target = Y_train
        elif args.training_type == "multimodal":
            input_v = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            input_t = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            K_v, K_t, Y_v, Y_t = MLSA.Models.MultimodalLatentSemanticAlignment(params.loc["latent_dim"][0], K_train_t.shape[0],
                                                                               align_fun=align,
                                                                               align_regularizer=tf.keras.regularizers.l2(1.0),
                                                                               units=[params.loc["units"][0]],
                                                                               activation=params.loc["activation"][0],
                                                                               rate=0.2, n_classes=n_classes)([input_v, input_t])
            model = tf.keras.Model(inputs=[input_v, input_t], outputs=[K_v, K_t, Y_v, Y_t])
            model.compile(loss=[MLSA.Losses.kernel_mse, MLSA.Losses.kernel_mse, "categorical_crossentropy", "categorical_crossentropy"],
                          loss_weights=[0.1, 0.1, 1., 1.], optimizer=tf.optimizers.Adam(lr=1e-4))
            callback = MLSA.Callbacks.CrossEntropyCallback(validation_data=([K_val_v, K_val_t], Y_val), path=temp_path)
            model.fit([K_train_v, K_train_t], [K_train_v, K_train_t, Y_train, Y_train], epochs=1000, batch_size=K_train_t.shape[0],
                      callbacks=[callback], verbose=0)
            model.load_weights(temp_path)
            
            input_preds = tf.keras.layers.Input(shape=(K_train_t.shape[0], ))
            output = model.layers[-1].layers[0](input_preds)
            preds_model = tf.keras.Model(inputs=input_preds, outputs=output)
            if args.retrieval_type == "H":
                query = preds_model.predict(K_test_v, batch_size=Y_test.shape[0])[1]
                target = preds_model.predict(K_train_v, batch_size=Y_train.shape[0])[1]
            elif args.retrieval_type == "Y":
                query = preds_model.predict(K_test_v, batch_size=Y_test.shape[0])[2]
                target = Y_train
        rank_pred = cosine_similarity(query, target)
        scores = retrieval.retrieval_metrics(rank_test, rank_pred,
                                             f"qrels-{args.training_type}-{params.loc['visual_kernel'][0]}-{params.loc['text_kernel'][0]}.out", f"preds-{args.training_type}-{params.loc['visual_kernel'][0]}-{params.loc['text_kernel'][0]}.out")
        scores = [scores[key] for key in names]
        results.append(scores)
        print("".join(map(lambda score: f"{score:20}", [f"{i:.4f}" for i in scores])))
    means = np.array(results).mean(axis=0)
    devs = np.array(results).std(axis=0)
    print("".join(map(lambda i:f'{i:20}',[f'{avg:.4f}'+u"\u00B1"+f'{std:.4f}' for avg, std in zip(means, devs)])))
    os.remove(temp_path)