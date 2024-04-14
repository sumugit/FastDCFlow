import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class CausalLoss(nn.Module):

    def __init__(self, monotonous_dims: list):
        super().__init__()
        self.monotonous_dims = monotonous_dims

    def forward(self, x, x_cf):
        monotonous_loss = 0
        
        for dim in self.monotonous_dims:
            monotonous_loss += torch.mean(torch.max(x[:, dim] - x_cf[:, dim], torch.zeros_like(x[:, dim])))
        monotonous_loss /= len(self.monotonous_dims)
        
        return monotonous_loss