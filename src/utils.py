import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import src.config as conf


def show_images(data, num_samples=4, cols=4, run=None):
    """Plots some samples from the dataset and logs to Neptune"""
    fig = plt.figure(figsize=(15, 15))
    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.imshow(img[0])
        plt.axis("off")

    if run:
        # Log the figure to Neptune
        run["dataset/samples"].upload(fig)
    plt.close(fig)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]

    return reverse_transforms(image)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals while considering the batch dimension
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def plot_noise_schedule(diffusion_schedule, num_steps, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(num_steps-1), diffusion_schedule["betas"], label="Beta")
    ax.plot(range(num_steps-1), diffusion_schedule["alphas_cumprod"], label="Alpha Cumulative Product")
    ax.plot(range(num_steps-1), diffusion_schedule["sqrt_one_minus_alphas_cumprod"], label="Sqrt(1 - Alpha Cumulative Product)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.set_title("Noise Schedule")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_diffusion_process(dataloader, num_steps, diffusion_schedule, save_path):
    image = next(iter(dataloader))[0]
    
    num_images = 10
    step_size = num_steps // num_images
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    fig.suptitle("Forward Diffusion Process")
    
    axes[0, 0].imshow(show_tensor_image(image[0]))
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original")
    
    for idx, timestep in enumerate(range(0, num_steps, step_size)):
        t = torch.tensor([timestep]).type(torch.int64)
        noisy_image, _ = add_noise(image, t, diffusion_schedule, image.device)
        
        row, col = divmod(idx + 1, 6)
        axes[row, col].imshow(show_tensor_image(noisy_image[0]))
        axes[row, col].axis("off")
        axes[row, col].set_title(f"t={timestep}")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()