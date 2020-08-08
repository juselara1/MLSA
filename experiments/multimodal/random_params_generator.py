import numpy as np
import random, argparse
import pandas as pd

def create_params(n_exp, **intervals):
    keys=list(intervals.keys())
    params=[]
    for exp in range(n_exp):
        combination=[]
        for i in keys:
            if i in ["rbf_im", "chi_im", "rbf_txt", "chi_txt"]:
                combination.append(10**np.random.uniform(intervals[i][0],intervals[i][1]))
            else:
                combination.append(random.choice(intervals[i]))
        params.append(combination)
    return pd.DataFrame(params,columns=keys)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Creates a CSV file with random combination of hyperparameters.')
    parser.add_argument("--save_path", type=str, action="store", dest="save_path",
                        help="Path to save the CSV file.", default="params.csv")
    parser.add_argument("--seed", type=int, action="store", dest="seed",
                        help="Random seed.", default=0)
    parser.add_argument("--trials", type=int, action="store", dest="trials",
                        help="Number of combinations to generate.", default=100)
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    params = {"rbf_im": [-2,2], "chi_im":[-3,1], "rbf_txt": [-3,1], "chi_txt": [-4,-1],
              "latent_dim": [600, 700, 800, 900, 1000], "align": ["cosine", "manhattan", "euclidean"],
              "activation": ["relu", "sigmoid", "linear"], "units": [32, 64, 128]}
    df = create_params(args.trials, **params)
    df.to_csv(args.save_path)