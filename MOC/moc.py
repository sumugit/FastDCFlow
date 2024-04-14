import torch
from torch import nn


class MOC():

    def __init__(self, encoder_normalize_data_catalog, predictive_model, query_instance, total_CFs):
        self.encoder_normalize_data_catalog = encoder_normalize_data_catalog
        self.feature_names = self.encoder_normalize_data_catalog.feature_names
        self.pred_model = predictive_model
        self.query_instance = query_instance
        self.train_dataset = encoder_normalize_data_catalog.data_frame
        self.total_CFs = total_CFs
        
        # Perturbation range for each variable
        self.feature_range_min = {}
        self.feature_range_max = {}
        for feature in self.encoder_normalize_data_catalog.feature_names:
            self.feature_range_min[feature] = self.train_dataset[feature].min()
            self.feature_range_max[feature] = self.train_dataset[feature].max()
        
        self.loss_y = nn.BCELoss(reduction='none')
        self.loss_prox = nn.MSELoss(reduction='none')


    def compute_yloss(self, cfs):
        """Computes the first part (y-loss) of the loss function."""
        cfs = torch.tensor(cfs, dtype=torch.float32)
        y_cf = self.pred_model(cfs)
        y_expected = torch.ones(y_cf.shape)
        yloss = self.loss_y(y_cf, y_expected)
        
        return yloss.detach().numpy()

    def compute_proximity_loss(self, cfs, query_instance):
        """Compute weighted distance between two vectors."""
        cfs = torch.tensor(cfs, dtype=torch.float32, requires_grad=False)
        query_instance = torch.tensor(query_instance, dtype=torch.float32, requires_grad=False)
        proximity_loss = self.loss_prox(cfs, query_instance).sum(dim=1)

        return proximity_loss.detach().numpy()

    def predict_fn(self, cfs):
        cfs = torch.tensor(cfs, dtype=torch.float32)
        y_cf = self.pred_model(cfs)
        
        return y_cf.detach().numpy()   
