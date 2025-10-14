from itertools import product
import numpy as np
from scipy.sparse import csr_array
from scipy import sparse

import pandas as pd

import os
from typing import Dict, Iterable, Tuple, List
import numpy as np
import pandas as pd
import torch 

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

class DatasetLoader:
    """
    Load the Node and Graph datasets into Pandas DataFrames
    *Organized by resolution and p-value (graph)
    """
    def __init__(self, base_path: str = '/oscar/data/larschan/shared_data/BindGPS/data/datasets'):
        self.base_path = base_path
    def load(self, p_value: str, resolution: str):
        """Load node and edge datasets"""
        node_df = pd.read_pickle(f"{self.base_path}/node_dataset_{resolution}.pkl")
        edge_df = pd.read_pickle(f"{self.base_path}/edge_dataset_{p_value}_{resolution}.pkl")
        return node_df, edge_df

class GraphParamBuilder:
    """
    Process and Node and Graph Features into Tensors. 
    Input
    node_df : node dataset (resolution)
    edge_df : edge dataset (resolution/p_value)
    target : target variable from node dataset
    mask : column to organize train/val/test sets
    input_features : subset of chip-seq signals to embed node information
    seed : integer for reproducible splits
    
    Output: dictionary with keys containing tensor values
    
    """
    def __init__(self,
        node_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        target: pd.Series, 
        mask: pd.Series,
        input_features: pd.DataFrame,
        seed: int
    ):
        self.node_df = node_df
        self.edge_df = edge_df
        self.target = target
        self.mask = mask
        self.input_features = input_features
        self.seed = seed

    def _split_indices(self, indices, ratios, seed):
        """
        Split indices into train/val/test sets by chromosome
        """
        #Specify the Chromosome Range
        chrom_ranges = {
            "chr2L": (0, 23513),
            "chr2R": (23514, 48800),
            "chr3L": (48801, 76911),
            "chr3R": (76912, 108991),
            "chr4": (108992, 110340),
            "chrX": (110341, 133883),
            "chrY": (133884, 137551),
        }
        
        #lists of each training/validation/testing set
        train_all: List[int] = []
        val_all:   List[int] = []
        test_all:  List[int] = []
        
        rng = np.random.default_rng(seed)

        for chrom, (start, end) in chrom_ranges.items():
            #range filter to filter indices within the chromosome bounds
            mask = (indices >= start) & (indices <= end)
            chrom_idx = indices[mask].to_numpy()
            if chrom_idx.size == 0: #indices do not fall within chromosome category
                continue

            rng.shuffle(chrom_idx)
            
            n = chrom_idx.size
            n_train = int(np.floor(n * ratios[0]))
            n_val = int(np.floor(n * ratios[1]))
            n_test = n - n_train - n_val #ensure total values matches original indexing

            train_all.extend(chrom_idx[:n_train])
            val_all.extend(chrom_idx[n_train:n_train + n_val])
            test_all.extend(chrom_idx[n_train + n_val:])

        return {
            #the sorted function places the idx values in chronological order
            "train": pd.Index(sorted(train_all), dtype="int64"),
            "val": pd.Index(sorted(val_all), dtype="int64"),
            "test": pd.Index(sorted(test_all), dtype="int64"),
        }

    def convert_to_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Convert all data into torch tensors
        
        Input
        y : target variable from node dataset
        all_mask : column from node dataset to organize test/train/val sets
        input_features: subset of chip-seq signals
        
        Output
        X : subset of chip-seq signals (input features)
        y : target variable from node dataset
        all_mask : column from node dataset to organize test/train/val set [True if contains a value / False if not]
        train_mask : training nodes [True if included in training set / False if not]
        val_mask : validation nodes (see above)
        test_mask : testing nodes (see above)
        edge_index : bin1 and bin2 labels
        edge_weight : -log10 transformed p-value
        
        """
        y = self.target
        all_mask = self.mask
        input_features = self.input_features

        mask_idx = all_mask.index
        
        #define the ratio of splitting
        splits = self._split_indices(mask_idx, ratios=(0.70, 0.15, 0.15), seed=self.seed)

        train_mask = np.full(all_mask.shape[0], False)
        val_mask = np.full(all_mask.shape[0], False)
        test_mask = np.full(all_mask.shape[0], False)

        train_mask[splits["train"]] = True
        val_mask[splits["val"]] = True
        test_mask[splits["test"]] = True

        #edge information
        edges = self.edge_df[['bin1', 'bin2']]
        edge_pvalue_transformed = self.edge_df['p-value_transformed']
        edge_contactCount = self.edge_df['contactCount']

        # transformed loop size
        self.edge_df['loop_size_transformed'] = np.log10(self.edge_df['loop_size'] + 1e-8)
        edge_loop_size_transformed = self.edge_df['loop_size_transformed']

        #return tensors
        return {
            "X": torch.tensor(input_features.to_numpy(), dtype=torch.float32),
            "y": torch.tensor(y.to_numpy(), dtype=torch.float32),
            "all_mask": torch.tensor(all_mask.to_numpy(), dtype=torch.bool),
            "train_mask": torch.tensor(train_mask, dtype=torch.bool),
            "val_mask": torch.tensor(val_mask, dtype=torch.bool),
            "test_mask": torch.tensor(test_mask, dtype=torch.bool),
            "edge_index": torch.tensor(edges.transpose().to_numpy(), dtype=torch.long),
            "edge_pvalue_transformed": torch.tensor(edge_pvalue_transformed.to_numpy(), dtype=torch.float32),
            "edge_contactCount": torch.tensor(edge_contactCount.to_numpy(), dtype=torch.float32),
            "edge_loop_size_transformed": torch.tensor(edge_loop_size_transformed.to_numpy(), dtype=torch.float32),
        }

        
