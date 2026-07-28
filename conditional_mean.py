import pandas as pd
import pickle
import numpy as np

from itertools import product
from tqdm import tqdm


def cosine_similarity_matrices(matrix_a, matrix_b):
    """"""
    norm_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    normalized_a = matrix_a / norm_a

    norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    normalized_b = matrix_b / norm_b

    return np.dot(normalized_a, normalized_b.T)


def calculate_conditional_mean(focal_df, focal_mat, ref_mat, window, quantiles, result_dict):
    """Average cosine similarity among the top-q items in the ref"""
    cos_sim = cosine_similarity_matrices(focal_mat, ref_mat)
    assert cos_sim.shape == (focal_mat.shape[0], ref_mat.shape[0])
    m = ref_mat.shape[0]
    for q in quantiles:
        k = int(np.ceil(m * q))
        topk = np.partition(cos_sim, m-k, axis=1)[:, -k:] # time-consuming
        assert topk.shape == (focal_df.shape[0], k)
        for pid, s in zip(focal_df.PaperID, np.round(topk.mean(axis=1), 6)):
            if pid not in result_dict:
                result_dict[pid] = {}
            result_dict[pid][window, int(q*100)] = s


def calculate_sim_by_year(paper_df_path, paper_embedding_path, windows=[1,3], quantiles=[0.01,0.05,0.1]):
    """"""
    paper_df = pd.read_pickle(paper_df_path)
    paper_df.drop(columns=['Abstract', 'AbstractCleaned'], inplace=True)
    print('%d papers'%paper_df.shape[0], '\n', 'papers per year:', '\n', paper_df.Year.value_counts().sort_index(), '\n', sep='')
    with open(paper_embedding_path, 'rb') as fin:
        pid_to_embedding = pickle.load(fin)
    print('%d papers with embeddings' % len(pid_to_embedding))

    year_min, year_max = paper_df.Year.min(), paper_df.Year.max()
    result_past = {}
    result_future = {}
    for focal_year in sorted(paper_df.Year.unique()):
        focal_df = paper_df[paper_df.Year == focal_year]
        focal_mat = np.concatenate([pid_to_embedding[pid] for pid in focal_df.PaperID], axis=0)
        for w in tqdm(windows, desc='%d'%focal_year):
            if (focal_year - year_min) >= w:
                past_df = paper_df[paper_df.Year.between(focal_year-w, focal_year-1)]
                past_mat = np.concatenate([pid_to_embedding[pid] for pid in past_df.PaperID], axis=0)
                calculate_conditional_mean(focal_df, focal_mat, past_mat, w, quantiles, result_past)
            
            if (year_max - focal_year) >= w:
                future_df = paper_df[paper_df.Year.between(focal_year+1, focal_year+w)]
                future_mat = np.concatenate([pid_to_embedding[pid] for pid in future_df.PaperID], axis=0)
                calculate_conditional_mean(focal_df, focal_mat, future_mat, w, quantiles, result_future)
    
    for w, q in product(windows, quantiles):
        q = int(q*100)
        paper_df[f"past_window{w}_quant{q}%"] = paper_df.PaperID.apply(lambda x: result_past.get(x, {}).get((w,q), None))
        paper_df[f"future_window{w}_quant{q}%"] = paper_df.PaperID.apply(lambda x: result_future.get(x, {}).get((w,q), None))
    return paper_df

