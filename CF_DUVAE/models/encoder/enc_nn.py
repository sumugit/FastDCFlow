import torch
import torch.nn as nn
import math

class GaussianNNEncoder(nn.Module):
    def __init__(self, encoder_normalize_data_catalog, args):
        super(GaussianNNEncoder, self).__init__()
        self.encoded_size = args.encoded_size
        self.data_size = len(encoder_normalize_data_catalog.data_frame.columns)
        self.hidden_size1 = args.hidden_size1
        self.hidden_size2 = args.hidden_size2
        self.args = args
        
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size, self.hidden_size1),
            nn.BatchNorm1d(self.hidden_size1),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.BatchNorm1d(self.hidden_size2),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        
        self.fc_mean = nn.Linear(self.hidden_size2, self.encoded_size)
        self.fc_logvar = nn.Linear(self.hidden_size2, self.encoded_size)
        
        # DU-VAE proposal: BN
        self.mean_bn = nn.BatchNorm1d(self.encoded_size)
        self.reset_parameters()

    def reset_parameters(self, reset=False):
        """initialize mean_bn
        """
        if not reset:
            nn.init.constant_(self.mean_bn.weight, self.args.gamma)
        else:
            print('reset BN !')
            if self.args.gamma_train:
                nn.init.constant_(self.mean_bn.weight, self.args.gamma)
            else:
                self.mean_bn.weight.fill_(self.args.gamma)
            nn.init.constant_(self.mean_bn.bias, 0.0)


    def forward(self, x, c, training):
        x = torch.cat((x, c), dim=1)
        h = self.encoder(x)
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        
        # 以下のときmeanから余分な次元を除去する
        # args.gamma <= 0のとき, Batch Normalizationは有効なスケール因子を持たない
        # ミニバッチのサイズが1のとき, バッチ正規化を計算するための十分なデータがない
        if self.args.gamma <= 0 or (z_mean.squeeze(0).size(0) == 1 and training):
            z_mean = z_mean.squeeze(0)
        else:
            # DU-VAE proposal: BN
            self.mean_bn.weight.requires_grad = True
            # self.mean_bn.weight.data: gamma_mud in equ(9)
            ss = torch.mean(self.mean_bn.weight.data ** 2) ** 0.5
            self.mean_bn.weight.data = self.mean_bn.weight.data * self.args.gamma / ss
            if training:
                z_mean = self.mean_bn(z_mean.squeeze(0)) # 前後でz_meanの次元は変わらない
            else:
                z_mean = self.mean_bn(z_mean)
        
        # DU-VAE proposal: dropout when training
        if self.args.p_drop > 0:
            # equ(6) : alpha = 1.0 / (2 * math.e * math.pi)
            var = z_logvar.exp() - self.args.delta_rate * 1.0 / (2 * math.e * math.pi)
            var = torch.dropout(var, p=self.args.p_drop, train=training)
            z_logvar = torch.log(var + self.args.delta_rate * 1.0 / (2 * math.e * math.pi))


        return z_mean, z_logvar