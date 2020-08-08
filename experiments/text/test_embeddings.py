import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.feature_extraction.text import TfidfTransformer
import os, argparse, h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
def create_model(units, act, dim):
    model = tf.keras.models.Sequential()
    # Primer capa intermedia con dropout
    model.add(tf.keras.layers.Dense(units, activation=act, input_shape=(dim, )))
    model.add(tf.keras.layers.Dropout(0.2))
    # Segunda capa intermedia con dropout
    model.add(tf.keras.layers.Dense(units, activation=act))
    model.add(tf.keras.layers.Dropout(0.2))
    # Capa de salida
    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    # Compilamos el modelo
    model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-4))
    return model

def train_histogram(path, trials):
    names = ["Ngram", "Accuracy", "Precision", "F1", "Recall"]
    print("".join([f'{val:20}' for val in names]))
    with h5py.File(path, "r") as df:
        y_test = df["y_test"][:]>1
        Y_train = tf.keras.utils.to_categorical(df["y_train"][:]>1)
        Y_val = tf.keras.utils.to_categorical(df["y_val"][:]>1)
        for k in range(1, 8):    
            X_train = df[f"X_train{k}"][:]
            X_val = df[f"X_val{k}"][:]
            X_test = df[f"X_test{k}"][:]

            acc = []; prec = []; f1 = []; rec = []
            for i in range(trials):
                model = create_model(32, "relu", X_train.shape[1])
                cb = tf.keras.callbacks.ModelCheckpoint("histogram.h5")
                model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[cb], epochs=100,
                          batch_size=16, verbose=False)
                model.load_weights("histogram.h5")

                preds = model.predict(X_test)
                y_pred = np.argmax(preds, axis=1)
                acc.append(accuracy_score(y_test, y_pred))
                prec.append(precision_score(y_test, y_pred, average="weighted"))
                f1.append(f1_score(y_test, y_pred, average="weighted"))
                rec.append(recall_score(y_test, y_pred, average="weighted"))
            print(f'{k:<20}'+"".join(map(lambda i:f'{i:20}',[f'{avg:.4f}'+u"\u00B1"+f'{std:.4f}' for avg, std in zip([np.mean(acc), np.mean(prec), np.mean(f1), np.mean(rec)], [np.std(acc), np.std(prec), np.std(f1), np.std(rec)])])))  
        
def train_tfidf(path, trials):
    names = ["Ngram", "Accuracy", "Precision", "F1", "Recall"]
    print("".join([f'{val:20}' for val in names]))
    
    with h5py.File(path, "r") as df:
        y_test = df[f"y_test"][:]>1
        Y_train = tf.keras.utils.to_categorical(df["y_train"][:]>1)
        Y_val = tf.keras.utils.to_categorical(df["y_val"][:]>1)
        for k in range(1, 8):
            X_train = df[f"X_train{k}"][:]
            X_val = df[f"X_val{k}"][:]
            X_test = df[f"X_test{k}"][:]
            
            tfidf = TfidfTransformer(sublinear_tf=True).fit(X_train)
            
            X_train = tfidf.transform(X_train).toarray()
            X_val = tfidf.transform(X_val).toarray()
            X_test = tfidf.transform(X_test).toarray()

            acc = []; prec = []; f1 = []; rec = []
            for i in range(trials):
                model = create_model(32, "relu", X_train.shape[1])
                cb = tf.keras.callbacks.ModelCheckpoint("tfidf.h5")
                model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[cb], epochs=100,
                          batch_size=16, verbose=False)
                model.load_weights("tfidf.h5")

                preds = model.predict(X_test)
                y_pred = np.argmax(preds, axis=1)
                acc.append(accuracy_score(y_test, y_pred))
                prec.append(precision_score(y_test, y_pred))
                f1.append(f1_score(y_test, y_pred))
                rec.append(recall_score(y_test, y_pred))
            print(f'{k:<20}'+"".join(map(lambda i:f'{i:20}',[f'{avg:.4f}'+u"\u00B1"+f'{std:.4f}' for avg, std in zip([np.mean(acc), np.mean(prec), np.mean(f1), np.mean(rec)], [np.std(acc), np.std(prec), np.std(f1), np.std(rec)])]))) 
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate all the BoW representations')
    parser.add_argument("--repr", type=str, help="Representation to evaluate {histogram, tfidf}", default="histogram")
    parser.add_argument("--bow_path", type=str, help="Path of the bag of words.", default="/tf/TCGA/bow.h5")
    parser.add_argument("--trials", type=int, help="Number of trials", default=30)
    args = parser.parse_args()
    os.system("clear")
    if args.repr=="histogram":
        train_histogram(args.bow_path, args.trials)
    elif args.repr=="tfidf":
        train_tfidf(args.bow_path, args.trials)