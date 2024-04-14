from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('/workspace/Eval/')
from counterfactual_explanation.utils.mlcatalog import (
    get_latent_representation_from_flow,
    get_latent_representation_from_flow_mixed_type,
    original_space_value_from_latent_representation,
    original_space_value_from_latent_representation_mixed_type)


class FindCounterfactualSample(ABC):
    @abstractmethod
    def initialize_latent_representation(self):
        pass

    @abstractmethod
    def distance_loss(self):
        pass

    @abstractmethod
    def prediction_loss(self):
        pass

    @abstractmethod
    def fair_loss(self):
        pass


class CounterfactualSimpleBn(FindCounterfactualSample):
    def __init__(self, predictive_model, flow_model, weight):
        self.flow_model = flow_model
        self.predictive_model = predictive_model
        self.distance_loss_func = torch.nn.MSELoss()
        self.predictive_loss_func = torch.nn.BCELoss()
        self.lr = 1e-3
        self.n_steps = 100
        self.weight = weight

    @property
    def _flow_model(self):
        return self.flow_model

    @property
    def _predictive_model(self):
        return self.predictive_model

    def initialize_latent_representation(self):
        pass

    def distance_loss(self, factual, counterfactual):
        return self.distance_loss_func(factual, counterfactual)

    def prediction_loss(self, representation_counterfactual):
        counterfactual = self._original_space_value_from_latent_representation(
            representation_counterfactual) # inverse transformation
        yhat = self._predictive_model(counterfactual).reshape(-1)
        yexpected = torch.ones(
            yhat.shape, dtype=torch.float).reshape(-1)
        self.predictive_loss_func(yhat, yexpected)
        return self.predictive_loss_func(yhat, yexpected)

    def fair_loss(self):
        return 0

    def combine_loss(self, factual, counterfactual):
        return self.weight * self.prediction_loss(counterfactual) + (1 - self.weight) * self.distance_loss(factual,
                                                                                                           counterfactual)

    def make_perturbation(self, z_value, delta_value):
        return z_value + delta_value

    def _get_latent_representation_from_flow(self, input_value):
        return get_latent_representation_from_flow(self.flow_model, input_value)

    def _original_space_value_from_latent_representation(self, z_value):
        return original_space_value_from_latent_representation(self.flow_model, z_value)
    

    def find_counterfactual_via_optimizer(self, factual):
        z_value = self._get_latent_representation_from_flow(factual)
        delta_value = nn.Parameter(torch.zeros(z_value.shape))

        representation_factual = self._get_latent_representation_from_flow(
            factual)
        z_hat = self.make_perturbation(z_value, delta_value) # add perturbation
        x_hat = self._original_space_value_from_latent_representation(
            z_hat)
        optimizer = optim.Adam([delta_value], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
        for epoch in (range(self.n_steps)):
            epoch += 1
            z_hat = self.make_perturbation(z_value, delta_value) # add perturbation
            x_hat = self._original_space_value_from_latent_representation(
                z_hat)
            total_loss = self.combine_loss(representation_factual, z_hat)
            optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            optimizer.step()
            
            if epoch % 10 == 0:
                scheduler.step()
                cur_lr = scheduler.optimizer.param_groups[0]['lr']
                # print("\n Epoch {}, Loss {:.4f}, Learning rate {:.4f}".format(epoch, total_loss, cur_lr))

        return x_hat.detach()
