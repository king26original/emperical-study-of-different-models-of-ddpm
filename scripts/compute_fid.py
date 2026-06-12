'''
run this file either after training the models, or after downloading the checkpoints 
'''

from src.evaluation.fid import fid, fid_fn
from src.diffusion.cosine_beta_schedule import cosine_alpha, cosine_alpha_bar, cosine_posterior_variance
from src.diffusion.linear_beta_schedule import linear_alpha, linear_alpha_bar, linear_posterior_variance
from checkpoints.load import model1, model2, ema_model3, ema_model4

fid_score_model1=fid_fn(model1, fid, linear_alpha, linear_alpha_bar, linear_posterior_variance)
fid_score_model2=fid_fn(model2, fid, linear_alpha, linear_alpha_bar, linear_posterior_variance)
fid_score_model3=fid_fn(ema_model3, fid, cosine_alpha, cosine_alpha_bar, cosine_posterior_variance)
fid_score_model4=fid_fn(ema_model4, fid, cosine_alpha, cosine_alpha_bar, cosine_posterior_variance)
