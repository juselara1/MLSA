import numpy as np
import os, re

def to_qrels(save_path, rank_true):
    iter_id = "0"
    with open(save_path, "w") as f:
        for i in range(rank_true.shape[0]):
            for j in range(rank_true.shape[1]):
                query_id = "0"+str(i+1) if i<9 else str(i+1)
                target_id = str(j)
                f.write(" ".join([query_id, iter_id, target_id, str(int(rank_true[i, j])), "\n"]))

def to_output(save_path, rank_pred):
    iter_id = "0"; run_id = "0"; rank_ignored = "1"
    with open(save_path, "w") as f:
        for i in range(rank_pred.shape[0]):
            for j in range(rank_pred.shape[1]):
                query_id = "0"+str(i+1) if i<9 else str(i+1)
                target_id = str(j)
                f.write(" ".join([query_id, iter_id, target_id, rank_ignored, str(rank_pred[i, j]), run_id, "\n"]))
                        
def retrieval_metrics(rank_true, rank_pred, qrels_name="rank_qrels.out", output_name="rank_pred.out"):
    to_qrels(qrels_name, rank_true)
    to_output(output_name, rank_pred)
    
    stream = os.popen(f"trec_eval {qrels_name} {output_name}")
    results = stream.read().split("\n")
    results = list(map(lambda i: i.split("\tall\t"), results))
    results = list(map(lambda i: [re.sub(r"\s*", "", i[0]), float(i[1])], results[:-1]))
    os.remove(qrels_name); os.remove(output_name)
    return dict(results)