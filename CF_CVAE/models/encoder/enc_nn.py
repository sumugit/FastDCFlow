import torch
import torch.nn as nn


class GaussianNNEncoder(nn.Module):
    def __init__(self, encoder_normalize_data_catalog, encoder_onehotenc_catalog, args):
        super(GaussianNNEncoder, self).__init__()
        self.encoded_size = args.cvae_encoded_size
        self.data_size = len(encoder_normalize_data_catalog.data_frame.columns)
        self.num_labels = len(encoder_onehotenc_catalog.encoded_feature_name)
        self.hidden_size1 = args.cvae_hidden_size1
        self.hidden_size2 = args.cvae_hidden_size2
        self.hidden_size3 = args.cvae_hidden_size3
        
        self.encoder = nn.Sequential(
            nn.Linear(self.data_size + self.num_labels, self.hidden_size1),
            nn.BatchNorm1d(self.hidden_size1),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.BatchNorm1d(self.hidden_size2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, self.hidden_size3),
            nn.BatchNorm1d(self.hidden_size3),
            nn.Dropout(0.1),
            nn.ReLU(),
            )
        
        self.fc_mean = nn.Linear(self.hidden_size3, self.encoded_size)
        self.fc_logvar = nn.Linear(self.hidden_size3, self.encoded_size)


    def forward(self, x, c, label):
        x = torch.cat((x, c, label), dim=1)
        h = self.encoder(x)
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)

        return z_mean, z_logvar