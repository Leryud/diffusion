import torch.nn.functional as F
import torch

import src.config as conf
from src.utils import add_noise
from src.scheduler import create_noise_schedule


def get_loss(model, x_0, t):
    scheduler = create_noise_schedule(conf.T)
    x_noisy, noise = add_noise(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

