import torch
import numpy as np
import timeit
from collections import defaultdict


class DicePyTorch():
    def __init__(self, encoder_normalize_data_catalog, predictive_model):
        self.encoder_normalize_data_catalog = encoder_normalize_data_catalog
        self.pred_model = predictive_model
        self.targetenc_dataset = self.encoder_normalize_data_catalog.data_frame
        self.history = defaultdict(lambda: [])

    def generate_counterfactuals(self, query_instance, total_CFs, max_iter, learning_rate,
                                 diversity_loss_type="dpp_style:inverse_dist", optimizer="pytorch:adam"):
        self.total_CFs = total_CFs
        self.do_cf_initializations(self.total_CFs)
        self.do_loss_initializations(diversity_loss_type)
        final_cfs = self.find_counterfactuals(query_instance, optimizer, learning_rate, max_iter)
        final_cfs = np.array(final_cfs)
        final_cfs = torch.tensor(final_cfs, dtype=torch.float32)
        
        return final_cfs
        
    def do_cf_initializations(self, num_inits):
            self.cfs = []
            kx = 0
            while kx < num_inits:
                one_init = np.zeros(len(self.encoder_normalize_data_catalog.feature_names))
                for jx, feature in enumerate(self.encoder_normalize_data_catalog.feature_names):
                    one_init[jx] = np.random.uniform(
                        self.targetenc_dataset[feature].min(), self.targetenc_dataset[feature].max())
                cf = torch.tensor(one_init).float()
                cf.requires_grad = True
                self.cfs.append(cf)
                kx += 1


    def do_loss_initializations(self, diversity_loss_type):
        self.diversity_loss_type = diversity_loss_type
        self.yloss_opt = torch.nn.BCELoss()

    def do_optimizer_initializations(self, optimizer, learning_rate):
        opt_method = optimizer.split(':')[1]
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(self.cfs, lr=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.cfs, lr=learning_rate)

    def compute_yloss(self):
        yloss = 0.0
        criterion = torch.nn.BCELoss()
        for i in range(self.total_CFs):
            temp_loss = criterion(self.pred_model(self.cfs[i]), torch.tensor([self.target_cf_class]))
            yloss += temp_loss

        return yloss

    def compute_dist(self, x_hat, x1):
        criterion = torch.nn.MSELoss()
        return criterion(x_hat.float(), x1.float())

    def compute_proximity_loss(self):
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        return proximity_loss

    def dpp_style(self, submethod):
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i, j)] = 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
                    if i == j:
                        det_entries[(i, j)] += 0.0001
        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i, j)] = 1.0/(torch.exp(self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        diversity_loss = torch.logdet(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss()
        self.diversity_loss = self.compute_diversity_loss()
        self.loss = self.yloss + self.proximity_loss - self.diversity_loss
        self.history['yloss'].append(self.yloss.item())
        self.history['proximity_loss'].append(self.proximity_loss.item())
        self.history['diversity_loss'].append(self.diversity_loss.item())
        self.history['total_loss'].append(self.loss.item())

    def find_counterfactuals(self, query_instance, optimizer, learning_rate, max_iter):
        # Prepares user defined query_instance for DiCE.
        self.x1 = torch.tensor(query_instance, requires_grad=False).float()
        test_pred = self.pred_model(self.x1)
        desired_class = 1.0
        self.target_cf_class = torch.tensor(desired_class).float()
        self.max_iter = max_iter
        # running optimization steps
        start_time = timeit.default_timer()
        final_cfs = []

        # initialize optimizer
        self.do_optimizer_initializations(optimizer, learning_rate)
        
        for iter in range(self.max_iter):
            # zero all existing gradients
            self.optimizer.zero_grad()
            self.pred_model.zero_grad()
            # get loss and backpropogate
            self.compute_loss()
            self.loss.backward()
            # update the variables
            self.optimizer.step()
        # storing final CFs
        for j in range(0, self.total_CFs):
            temp = self.cfs[j].detach().clone().numpy()
            final_cfs.append(temp)

        self.elapsed = timeit.default_timer() - start_time
        m, s = divmod(self.elapsed, 60)
        
        return final_cfs