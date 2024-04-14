import torch
import torch.utils.data
from torch import nn
import warnings
warnings.filterwarnings('ignore')


class CF_VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CF_VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_y = nn.BCELoss() # prediction loss
        self.loss_prox = nn.MSELoss() # distance loss

    def sample_latent_code(self, mean, logvar, temperature=1.0):
        std = logvar.mul(0.5).exp() * temperature
        eps = torch.normal(mean=0, std=1, size=std.shape)
        return mean + torch.mul(eps, std)

    def compute_elbo(self, x, expected_outcome, predictive_model, temperature=1.0):
        # resize c for compute encoder and decoder
        expected_outcome = expected_outcome.view(expected_outcome.shape[0], 1)
        # encoder
        mean, logvar = self.encoder(x, expected_outcome)
        kl_divergence = torch.mean(0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1))        
        # decoder
        z = self.sample_latent_code(mean, logvar, temperature)
        x_cf = self.decoder(z, expected_outcome)
        # Proximity
        recon_err = self.loss_prox(x, x_cf)
        # yloss
        y_cf = predictive_model(x_cf)
        validity_loss = self.loss_y(y_cf, expected_outcome)
        total_loss = recon_err + kl_divergence + validity_loss        
        
        return total_loss, x_cf, predictive_model(x_cf)

    def compute_loss(self, x, expected_outcome, predictive_model, temperature=1.0):
        # resize c for compute encoder and decoder
        expected_outcome = expected_outcome.view(expected_outcome.shape[0], 1)
        # encoder
        mean, logvar = self.encoder(x, expected_outcome)
        kl_divergence = torch.mean(0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).sum(dim=1))
        # decoder
        z = self.sample_latent_code(mean, logvar, temperature)
        x_cf = self.decoder(z, expected_outcome) # (batch_size, data_size)
        # Proximity
        recon_err = self.loss_prox(x, x_cf)
        # yloss
        y_cf = predictive_model(x_cf)
        validity_loss = self.loss_y(y_cf, expected_outcome)

        # print(f'kl_divergence: {kl_divergence}, recon_err: {recon_err}, validity_loss: {validity_loss}')
        return recon_err + kl_divergence + validity_loss, recon_err, kl_divergence, validity_loss