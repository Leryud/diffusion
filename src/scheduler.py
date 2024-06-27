import torch
import torch.nn.functional as F
import src.config as conf

def calculate_alphas(betas):
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(conf.device), alphas_cumprod[:-1]])
    return alphas, alphas_cumprod, alphas_cumprod_prev

def betas_for_alpha_bar(num_steps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    """
    t = torch.linspace(0, 1, num_steps + 1).to(conf.device)
    alpha_bar_vals = alpha_bar(t)
    betas = 1 - (alpha_bar_vals[1:] / alpha_bar_vals[:-1])
    return torch.clamp(betas, max=max_beta)

def create_noise_schedule(num_steps, schedule_type="cosine"):
    scale = 1000 / num_steps
    beta_start = scale * conf.BETA_START
    beta_end = scale * conf.BETA_END
    
    if schedule_type == "cosine":
        alpha_bar = lambda t: torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
        betas = betas_for_alpha_bar(num_steps, alpha_bar)
        
    elif schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps)
    else:
        raise ValueError("Unknown schedule type")
    
    alphas, alphas_cumprod, alphas_cumprod_prev = calculate_alphas(betas)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "posterior_variance": betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    }

def add_noise(x, t):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    t = t.to(conf.device)
    scheduler = create_noise_schedule(conf.T)
    sqrt_alphas_cumprod_t = scheduler["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = scheduler["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    
    noise = torch.randn_like(x).to(conf.device)
    
    mean = sqrt_alphas_cumprod_t.to(conf.device) * x.to(conf.device)
    var = sqrt_one_minus_alphas_cumprod_t.to(conf.device) * noise.to(conf.device)
    
    noisy_image = mean + var
    
    return torch.clamp(noisy_image, -1.0, 1.0), noise.to(conf.device)