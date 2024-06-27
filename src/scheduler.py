import torch
import torch.nn.functional as F
import numpy as np


def adaptive_noise_schedule(
    t, image_size, schedule_type="cosine", temperature=1.0, b=1.0
):
    if schedule_type == "cosine":
        return np.cos(((t / image_size) + 0.008) / 1.008 * np.pi * 0.5) ** 2
    elif schedule_type == "sigmoid":
        return 1 / (1 + np.exp(-temperature * (t - 0.5)))
    elif schedule_type == "linear":
        return 1 - t
    else:
        raise ValueError("Unknown schedule type")


def get_noise_schedule(
    num_timesteps, image_size, schedule_type="cosine", temperature=1.0, b=1.0
):
    t = np.linspace(0, 1, num_timesteps)
    alphas = adaptive_noise_schedule(t, image_size, schedule_type, temperature, b)
    alphas = b * alphas  # Apply input scaling
    betas = 1 - (alphas[1:] / alphas[:-1])
    return np.clip(betas, 0, 0.999)


def get_scheduler(T):
    betas = get_noise_schedule(
        num_timesteps=T, image_size=256, schedule_type="cosine", temperature=2.0, b=1.2
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }
