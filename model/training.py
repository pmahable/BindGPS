import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, Tuple
import json

from src.wandb_utils import WandbLogger
from src.metrics import compute_metrics
from src.models import KmerSequenceModel


# TODO: @Romer, feel free to port the notebook code into this format to make it easier...
class SeqSVMTrainer:
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dims: Tuple[int, ...] = (256, 128),
                 num_classes: int = 3,
                 dropout: float = 0.3,
                 wandb_logger: Optional[WandbLogger] = None,
                 device: Optional[torch.device] = None):
                 ):
        pass

    def train():
        pass

    def evaluate():
        pass

    def input_gradients():
        pass