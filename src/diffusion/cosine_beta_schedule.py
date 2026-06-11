import math

timesteps=1000

s=0.008

steps=timesteps+1

x=torch.linspace(
    0,
    timesteps,
    steps,
)

cosine_alpha_bar=torch.cos(((x/timesteps+s)/(1+s))*math.pi*0.5)**2

cosine_alpha_bar=cosine_alpha_bar/cosine_alpha_bar[0]

cosine_beta=1-(cosine_alpha_bar[1:]/cosine_alpha_bar[:-1])

cosine_beta=torch.clamp(cosine_beta, 1e-4, 0.999)

cosine_alpha=1.0-cosine_beta

cosine_alpha_bar=torch.cumprod(cosine_alpha, dim=0)

cosine_posterior_variance=cosine_beta.clone()

cosine_posterior_variance[1:]=(cosine_beta[1:]*(1-cosine_alpha_bar[:-1])/(1-cosine_alpha_bar[1:]))
