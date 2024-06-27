import torch.nn.functional as F
from src.utils import forward_diffusion_sample
import src.config as conf
from src.scheduler import get_scheduler
import torch


def get_loss(model, x_0, t):
    scheduler = get_scheduler(conf.T)
    x_noisy, noise = forward_diffusion_sample(
        x_0,
        t,
        scheduler["sqrt_alphas_cumprod"],
        scheduler["sqrt_one_minus_alphas_cumprod"],
    )
    x_noisy = torch.clamp(x_noisy, -1.0, 1.0)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)
