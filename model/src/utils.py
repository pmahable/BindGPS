import numpy as np
from scipy import sparse

def load_kmer_counts(path: str = "../data/sparse_kmer_counts.npz", drop_unk: bool = True):
    mat = sparse.load_npz(path)
    if drop_unk:
        mat = mat[:, :-1]
    return mat

def svm_project_kmer_vec(svd_model, full_dataset, batch_size=128):
    projected_vecs = []
    for i in range(0, len(full_dataset), batch_size):
        batch = full_dataset[i:i+batch_size]
        projected_vecs.append(batch @ svd_model.components_.T)
    return np.vstack(projected_vecs)

def load_svd(n_components: int = 128, path: str = None):
    if path is None:
        path = f"../models/seq_svm/svd_model_{n_components}.joblib"
    return joblib.load(path)