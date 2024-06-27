import torch.nn.functional as F
import torch

import src.config as conf
from src.scheduler import create_noise_schedule, add_noise


def get_loss(model, x_0, t):
    x_noisy, noise = add_noise(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


def improved_loss(model, x_0, t):
    x_noisy, noise = add_noise(x_0, t)
    output = model(x_noisy, t)
    x_theta, v = torch.split(output, output.shape[1] // 2, dim=1)
    
    scheduler = create_noise_schedule(conf.T)
    betas = scheduler['betas']
    alpha_bar = scheduler['alphas_cumprod']
    
    beta_tilde = ((1 - alpha_bar[t-1]) / (1 - alpha_bar[t])) * betas[t]
    
    # Reshape betas[t] and beta_tilde to match v's shape
    betas_t = betas[t].view(-1, 1, 1, 1).expand_as(v)
    beta_tilde = beta_tilde.view(-1, 1, 1, 1).expand_as(v)
    
    # Compute KL divergence
    epsilon = 1e-8
    kl_div = 0.5 * (-1 + v - torch.exp(-v) + ((1 - betas_t) / (beta_tilde + epsilon)) * torch.exp(v))
    
    # Compute MSE loss for x_theta
    mse_loss = F.mse_loss(x_theta, noise)
    
    kl_factor = 0.001
    return mse_loss + kl_factor * kl_div.mean()