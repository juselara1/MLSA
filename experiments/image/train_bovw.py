# Importing libraries
import numpy as np
np.random.seed(0)
import h5py, gc, pickle, os, argparse
from sklearn.cluster import MiniBatchKMeans

# Function to build a BoVW representation
def build_bow(preds, y_full, ids, k):
    uniques = np.unique(ids)
    X = np.zeros((uniques.size, k))
    y = np.zeros((uniques.size, ))
    for i, patient in enumerate(uniques):
        assignments, counts = np.unique(preds[ids==patient], return_counts=True)
        X[i, assignments] = counts
        y[i] = y_full[ids==patient][0]
    return X, y

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Evaluate a specified model in the test set.')
    parser.add_argument("--feature_vectors", type=str, action="store", dest="feature_vectors",
                        help="Path for feature vectors.", default="/tf/TCGA/feature_vectors.h5")
    parser.add_argument("--save_path", type=str, action="store", dest="save_path", help="Path to save the BoVW representations",
                        default="/tf/TCGA/bovw.h5")
    parser.add_argument("--model_path", type=str, action="store", dest="model_path", help="Path to save the K-Means models",
                        default="km_models")
    args = parser.parse_args()
    os.system("clear")
    # Loading data
    with h5py.File(args.feature_vectors, "r") as df:
        X_train = df["ims_train"][:]
        y_train = df["y_train"][:]
        id_train = df["id_train"][:]

        X_val = df["ims_val"][:]
        y_val = df["y_val"][:]
        id_val = df["id_val"][:]

        X_test = df["ims_test"][:]
        y_test = df["y_test"][:]
        id_test = df["id_test"][:]

    # Training k-means models to construct the codebook
    for k in np.arange(100, 2100, 100):
        print(f"Using {k} clusters", end="\r")
        km_model = MiniBatchKMeans(n_clusters=k, batch_size=1024)
        km_model.fit(X_train)
        with open(f"{args.model_path}/kmeans{k}.pkl", "wb") as f:
            pickle.dump(km_model, f)
    os.system("clear")
    # Computing and saving the BoVW representations
    with h5py.File(args.save_path, "w") as df:
        for k in np.arange(100, 2100, 100):
            with open(f"{args.model_path}/kmeans{k}.pkl", "rb") as f:
                km_model = pickle.load(f)
            X_train_bow, y_train_bow = build_bow(km_model.predict(X_train), y_train, id_train, k)
            X_val_bow, y_val_bow = build_bow(km_model.predict(X_val), y_val, id_val, k)
            X_test_bow, y_test_bow = build_bow(km_model.predict(X_test), y_test, id_test, k)
            df[f"X_train{k}"] = X_train_bow
            df[f"X_val{k}"] = X_val_bow
            df[f"X_test{k}"] = X_test_bow

            print(f"Created BoVW representation using a codebook of {k}", end="\r")
        df[f"y_train"] = y_train_bow
        df[f"y_val"] = y_val_bow
        df[f"y_test"] = y_test_bow