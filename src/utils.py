import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import src.config as conf
from neptune.types import File
import io


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
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        run["dataset/samples"].append(File.as_image(buf))
    plt.close(fig)


def show_tensor_image(image, run=None, step=None):
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

    pil_image = reverse_transforms(image)

    if run:
        # Log the PIL image to Neptune
        run["generated_samples"].append(File.as_image(pil_image), step=step)
    return pil_image


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals while considering the batch dimension
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(
    x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device=conf.device
):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    noisy_image = sqrt_alphas_cumprod_t.to(device) * x_0.to(
        device
    ) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

    return torch.clamp(noisy_image, -1.0, 1.0), noise.to(device)


def test_show_diffusion(
    dataloader,
    T,
    forward_diffusion_sample,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
):
    # Simulate forward diffusion
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
        img, noise = forward_diffusion_sample(
            image, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
        )
        show_tensor_image(img)

    plt.show()
