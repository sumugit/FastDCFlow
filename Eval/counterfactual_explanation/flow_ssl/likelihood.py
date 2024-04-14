import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


class Likelihood(nn.Module):
    """Get the NLL loss for a RealNVP model.
    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior, k=256):
        super().__init__()
        self.k = k
        self.prior = prior

    def forward(self, z, sldj, y=None):
        z = z.reshape((z.shape[0], -1))
        # 事前分布の対数尤度を計算
        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)
        corrected_prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:])
        # 対数尤度を計算
        ll = corrected_prior_ll + sldj

        return -ll