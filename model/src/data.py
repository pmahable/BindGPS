from itertools import product
import numpy as np
from scipy.sparse import csr_array
from scipy import sparse

import pandas as pd

from torch.utils.data import Dataset, DataLoader

    
class KmerTokenizer:
    def __init__(self, k=8, alphabet=('A','C','G','T')):
        self.k = k
        # build full vocabulary of all possible k-mers
        kmers = (''.join(p) for p in product(alphabet, repeat=k))
        self.vocab = {kmer: idx for idx, kmer in enumerate(kmers)}
        # optional: reserve an index for unknowns (e.g. containing “N”)
        self.unk_token = '<UNK>'
        self.vocab[self.unk_token] = len(self.vocab)

    def tokenize(self, seq: str) -> list[int]:
        """
        Slide a window of length k across seq and convert each k-mer to its index.
        Unknown k-mers map to the UNK token.
        """
        tokens = []
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            tokens.append(self.vocab.get(kmer, self.vocab[self.unk_token]))
        return tokens

    def detokenize(self, token_ids: list[int]) -> list[str]:
        """Reverse mapping: token IDs back to k-mer strings."""
        inv_vocab = {idx: kmer for kmer, idx in self.vocab.items()}
        return [inv_vocab.get(i, self.unk_token) for i in token_ids]
    
    def seq_to_vec(self, seq: str) -> np.ndarray:
        """Convert a sequence to a vector represenation"""
        
        tokens = self.tokenize(seq)
        vec = np.zeros(len(self.vocab))
        
        for token in tokens:
            vec[token] += 1
        
        # Return a sparse array
        return csr_array(vec)
        

class DNASequenceDataset(Dataset):
    def __init__(self, 
        sequences: np.ndarray,
        gene_labels: np.ndarray,
        mre_labels: np.ndarray, 
        vecs: np.ndarray,
        metadata: pd.DataFrame,
    ):
        self.sequences = sequences
        self.gene_labels = gene_labels
        self.mre_labels = mre_labels
        self.vecs = vecs
        self.metadata = metadata

    def get_full_item(self, idx):
        return self.sequences.iloc[idx], self.gene_labels[idx], self.mre_labels[idx], self.vecs[idx], self.metadata.iloc[idx]
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.gene_labels[idx], self.mre_labels[idx], self.vecs[idx]
        
