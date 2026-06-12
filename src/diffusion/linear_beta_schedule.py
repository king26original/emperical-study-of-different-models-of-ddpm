import torch

timesteps = 1000

linear_beta=torch.linspace(
    1e-4,
    0.02,
    timesteps
)

linear_alpha=1.0-linear_beta

linear_alpha_bar=torch.cumprod(linear_alpha, dim=0)

linear_posterior_variance=linear_beta.clone()

linear_posterior_variance[1:]=(linear_beta[1:]*(1-linear_alpha_bar[:-1])/(1-linear_alpha_bar[1:]))
