from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..realnvp.utils import checkerboard_mask
from ..resnet_realnvp import ResNet


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1
    TABULAR = 2


class MaskChannelwise:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHANNEL_WISE
        self.reverse_mask = reverse_mask

    def mask(self, x):
        if self.reverse_mask:
            x_id, x_change = x.chunk(2, dim=1)
        else:
            x_change, x_id = x.chunk(2, dim=1)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        if self.reverse_mask:
            return torch.cat((x_id, x_change), dim=1)
        else:
            return torch.cat((x_change, x_id), dim=1)
        
    def mask_st_output(self, s, t):
        return s, t


class MaskCheckerboard:
    def __init__(self, reverse_mask):
        self.type = MaskType.CHECKERBOARD
        self.reverse_mask = reverse_mask

    def mask(self, x):
        self.b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)


class MaskTabular:
    # mask convolutional for tabular data
    def __init__(self, reverse_mask):
        self.type = MaskType.TABULAR
        self.reverse_mask = reverse_mask

    def mask(self, x):
        #PAVEL: should we pass x sizes to __init__ and only create mask once?
        dim = x.size(1)
        split = dim // 2
        self.b = torch.zeros((1, dim), dtype=torch.float).to(x.device)
        if self.reverse_mask:
            self.b[:, split:] = 1.
        else:
            self.b[:, :split] = 1.
        x_id = x * self.b
        x_change = x * (1 - self.b)
        return x_id, x_change

    def unmask(self, x_id, x_change):
        return x_id * self.b + x_change * (1 - self.b)
    
    def mask_st_output(self, s, t):
        return s * (1 - self.b), t * (1 - self.b)
    

class CouplingLayerBase(nn.Module):
    """Coupling layer base class in RealNVP.
    
    must define self.mask, self.st_net, self.rescale
    """

    def _get_st(self, x):
        x_id, x_change = self.mask.mask(x)
        st = self.st_net(x_id) # architecture is defined in the child class
        s, t = st.chunk(2, dim=1)
        s = self.rescale(torch.tanh(s)) # rescale s to [-1, 1] and then to [0, 1]
        return s, t, x_id, x_change

    def forward(self, x, sldj=None, reverse=True):
        s, t, x_id, x_change = self._get_st(x)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        if torch.isnan(exp_s).any():
            # exp_s = torch.inf
            raise RuntimeError('Scale factor has NaN entries')
        x_change = (x_change + t) * exp_s
        self._logdet = s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)
        return x

    def inverse(self, y):
        s, t, x_id, x_change = self._get_st(y)
        s, t = self.mask.mask_st_output(s, t)
        exp_s = s.exp()
        inv_exp_s = s.mul(-1).exp()
        if torch.isnan(inv_exp_s).any():
            raise RuntimeError('Scale factor has NaN entries')
        x_change = x_change * inv_exp_s - t
        #         self._logdet = -s.view(s.size(0), -1).sum(-1)
        x = self.mask.unmask(x_id, x_change)
        return x

    def logdet(self):
        return self._logdet


class CouplingLayer(CouplingLayerBase):
    """Coupling layer in RealNVP for image data.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask (MaskChannelWise or MaskChannelWise): mask.
    """

    def __init__(self, in_channels, mid_channels, num_blocks, mask, init_zeros=False):
        super(CouplingLayer, self).__init__()

        self.mask = mask

        # Build scale and translate network
        if self.mask.type == MaskType.CHANNEL_WISE:
            in_channels //= 2

        # Pavel: reuse Marc's ResNet block?
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1,
                             double_after_norm=(self.mask.type == MaskType.CHECKERBOARD),
                             init_zeros=init_zeros)

        # Learnable scale for s
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.

    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x


class CouplingLayerTabular(CouplingLayerBase):

    def __init__(self, in_dim, mid_dim, num_layers, mask, init_zeros=False,dropout=False):
        """Coupling layer in RealNVP for tabular data.
        in_dim: input dimension
        mid_dim: dimension of hidden layers
        num_layers: number of hidden layers
        mask: mask
        init_zeros: whether to initialize the last layer to zeros
        dropout: whether to use dropout
        """
        super(CouplingLayerTabular, self).__init__()
        self.mask = mask
        self.layers = [
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(.5) if dropout else nn.Sequential(),
            *self._inner_seq(num_layers, mid_dim),
        ]
        last_layer = nn.Linear(mid_dim, in_dim*2) # in_dim*2 for s and t (scale and translate)
        if init_zeros:
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)
        self.layers.append(last_layer)

        self.st_net = nn.Sequential(*self.layers)
        self.rescale = nn.utils.weight_norm(RescaleTabular(in_dim))
       
    @staticmethod
    def _inner_seq(num_layers, mid_dim):
        # add additional mid layers
        res = []
        for _ in range(num_layers):
            res.append(nn.Linear(mid_dim, mid_dim))
            res.append(nn.ReLU())
        return res


class RescaleTabular(nn.Module):
    def __init__(self, D):
        super(RescaleTabular, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        x = self.weight * x
        return x


class Dequantization(nn.Module):
    def __init__(self, alpha=1e-5, quants=256):
        """
        Args:
            alpha: small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants: Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        self.quants = torch.Tensor(np.array([8, 5, 6])).cuda()

    def forward(self, z, reverse=False):
        """離散データを連続データに変換
        """
        if not reverse:
            # # 元のデータ空間から潜在空間へのマッピングを行う際に利用
            z = self.dequant(z) # Transform discrete values to continuous volumes
            z, ldj = self.sigmoid(z, reverse=True) # Scale to [-inf, inf]
            return z, ldj
        else:
            # 潜在空間から元のデータ空間へのマッピングを行う際に利用
            z = self.sigmoid(z, reverse=False) # scale to [0, 1]
            z = z * self.quants
            z[:, 0] = torch.floor(z[:, 0]).clamp(min=0, max=self.quants[0] - 1).to(torch.int32)
            z[:, 1] = torch.floor(z[:, 1]).clamp(min=0, max=self.quants[1] - 1).to(torch.int32)
            z[:, 2] = torch.floor(z[:, 2]).clamp(min=0, max=self.quants[2] - 1).to(torch.int32)
            # z = torch.floor(z).clamp(min=0, max=self.quants - 1).to(torch.int32)
            return z

    def sigmoid(self, z, ldj=0, reverse=False):
        if not reverse:
            # シグモイド関数を適用
            z = torch.sigmoid(z) # scale to [0, 1]
            ldj += (-z - 2 * F.softplus(-z)) # log det jacobian of sigmoid
            return z, ldj
        else:
            # シグモイド関数の逆関数を適用
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            z = torch.log(z) - torch.log(1 - z) # logit (sigmoidの逆関数)
            return z

    def dequant(self, z, ldj=0):
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        return z


class DequantizationOriginal(nn.Module):
    def __init__(self, alpha=1e-5, quants=256):
        """Adult-Income データセットに対する離散値の処理 (特徴量追加で修正必要)
        Args:
            alpha: small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants: Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = alpha
        # self.quants = quants
        self.quants = torch.Tensor(np.array([8, 5, 6])).cuda()

    def forward(self, z, ldj=None, reverse=False):
        """離散値と連続値の相互変換
        z: input tensor
        ldj: log-determinant of Jacobian
        reverse: whether to perform the forward or inverse operation
        """
        if not reverse:
            # 離散値 -> 連続値
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            # 連続値 -> 離散値
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            # ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z[:, 0] = torch.floor(z[:, 0]).clamp(min=0, max=self.quants[0] - 1).to(torch.int32)
            z[:, 1] = torch.floor(z[:, 1]).clamp(min=0, max=self.quants[1] - 1).to(torch.int32)
            z[:, 2] = torch.floor(z[:, 2]).clamp(min=0, max=self.quants[2] - 1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj=None, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            # ldj += (-z - 2 * F.softplus(-z)).sum()
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            # ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            # ldj += (-torch.log(z) - torch.log(1 - z)).sum()
            z = torch.log(z) - torch.log(1 - z)
        return z, ldj

    def dequant(self, z, ldj=None):
        """Transform discrete values to continuous volumes
        """
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach() # add Uniform noise
        z = z / self.quants # Normalize to [0, 1]
        # ldj -= np.log(8) * np.prod(z.shape[1:])
        return z, ldj
