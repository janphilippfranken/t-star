import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    
    def __init__(self, n_logprobs=50, n_labels=2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_logprobs, n_labels))
        self.b = nn.Parameter(torch.empty(n_logprobs))