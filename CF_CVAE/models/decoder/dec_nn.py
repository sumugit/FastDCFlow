import torch
import torch.nn as nn


class NNDecoder(nn.Module):
    def __init__(self, encoder_normalize_data_catalog, encoder_onehotenc_catalog, args):
        super(NNDecoder, self).__init__()
        self.encoded_size = args.cvae_encoded_size
        self.data_size = len(encoder_normalize_data_catalog.data_frame.columns)
        self.num_labels = len(encoder_onehotenc_catalog.encoded_feature_name)
        self.hidden_size1 = args.cvae_hidden_size1
        self.hidden_size2 = args.cvae_hidden_size2
        self.hidden_size3 = args.cvae_hidden_size3

        self.decoder = nn.Sequential(
            nn.Linear(self.encoded_size + 1 + self.num_labels, self.hidden_size3),
            nn.BatchNorm1d(self.hidden_size3),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size3, self.hidden_size2),
            nn.BatchNorm1d(self.hidden_size2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, self.hidden_size1),
            nn.BatchNorm1d(self.hidden_size1),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.data_size-1),
            nn.Sigmoid()
            )
    
    def forward(self, z, c, label):
        z_concat = torch.cat((z, c, label), dim=1)
        x_cf = self.decoder(z_concat)
        
        return x_cf