import matplotlib.pyplot as plt
import neptune
import torch
import torchvision
from neptune.types import File
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import v2
from tqdm.auto import tqdm

import src.config as conf
from src.loss import get_loss, improved_loss
from src.scheduler import create_noise_schedule
from src.unet import SimpleUnet
from src.utils import show_tensor_image


def load_transformed_dataset(img_size):
    data_transforms = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((img_size, img_size)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda t: (t * 2) - 1),  # Scales to -1;1
        ]
    )
    data_transforms = data_transforms
    train = torchvision.datasets.StanfordCars(
        root="/diffusion_data/data/", download=False, transform=data_transforms
    )
    test = torchvision.datasets.StanfordCars(
        root="/diffusion_data/data/",
        download=False,
        transform=data_transforms,
        split="test",
    )

    return torch.utils.data.ConcatDataset([train, test])


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns the denoised image.
    Applies noise to this images if we are not in the last step yet.
    """
    scheduler = create_noise_schedule(conf.T)

    betas_t = scheduler["betas"][t]
    sqrt_one_minus_alphas_cumprod_t = scheduler["sqrt_one_minus_alphas_cumprod"][t]
    sqrt_recip_alphas_t = scheduler["sqrt_recip_alphas"][t]

    # Call model (current image - noise_prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    posterior_variance_t = scheduler["posterior_variance"][t]

    if t == 0:
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sampling_step(model, x, t):
    output = model(x, t)
    x_theta, v = torch.split(output, output.shape[1] // 2, dim=1)

    scheduler = create_noise_schedule(conf.T)
    betas = scheduler["betas"]
    alpha_bar = scheduler["alphas_cumprod"]

    beta_tilde = ((1 - alpha_bar[t - 1]) / (1 - alpha_bar[t])) * betas[t]

    # Compute learned variance
    variance = torch.exp(v * torch.log(betas[t]) + (1 - v) * torch.log(beta_tilde))

    # Compute mean
    mean = (x - betas[t] * x_theta / torch.sqrt(1 - alpha_bar[t])) / torch.sqrt(alpha_bar[t])

    if t > 0:
        noise = torch.randn_like(x)
        return mean + torch.sqrt(variance) * noise
    else:
        return mean


@torch.no_grad()
def sample_plot_image(model, epoch, step, run=None):
    img_size = conf.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=conf.device)
    fig, axes = plt.subplots(1, 20, figsize=(20, 2))
    plt.axis("off")
    num_images = 20
    stepsize = int(conf.T / num_images)

    sample_images = []

    for i in range(0, conf.T)[::-1]:
        t = torch.full((1,), i, device=conf.device)
        img = sampling_step(model, img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            sample_images.append(img.cpu().squeeze(0))

            # For visualization
            idx = int(i / stepsize)
            axes[idx].imshow(show_tensor_image(img.detach().cpu()))
            axes[idx].axis("off")

    if run:
        # Log the sampling process figure to Neptune
        run["sampling_process"].upload(fig)

        # Log individual generated images
        for idx, img in enumerate(sample_images):
            run["generated_samples"].append(
                File.as_image(show_tensor_image(img)),
                description=f"Epoch {epoch}, Step {step}",
            )

    plt.close(fig)
    return sample_images


def main():
    # Initialize Neptune run
    # Previous API key has been revoked
    run = neptune.init_run(
        project="leryud/diffusion",
        api_token="API_TOKEN",
    )

    # Log configuration parameters
    run["parameters"] = {
        "img_size": conf.IMG_SIZE,
        "batch_size": conf.BATCH_SIZE,
        "epochs": conf.epochs,
        "learning_rate": conf.lr,
        "T": conf.T,
        "device": str(conf.device),
    }

    data = load_transformed_dataset(img_size=conf.IMG_SIZE)
    dataloader = DataLoader(
        data,
        batch_size=conf.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        pin_memory=True,
    )

    model = SimpleUnet()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    torch.compile(model)
    model.to(conf.device)

    optimizer = Adam(model.parameters(), lr=conf.lr)
    scaler = GradScaler()

    epochs = conf.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Create a TQDM progress bar for epochs
    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0

        # Create a TQDM progress bar for batches
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for step, batch in enumerate(batch_bar):
            for param in model.parameters():
                param.grad = None

            with autocast():
                t = torch.randint(0, conf.T, (conf.BATCH_SIZE,), device=conf.device).long()
                loss = improved_loss(model, batch[0], t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update batch progress bar
            batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Log batch-level metrics
            run["train/batch/loss"].append(loss.item())
            run["train/batch/step"].append(step + epoch * len(dataloader))

        scheduler.step()
        # Calculate and log epoch-level metrics
        avg_epoch_loss = epoch_loss / len(dataloader)
        run["train/epoch/loss"].append(avg_epoch_loss)
        run["train/epoch/number"].append(epoch)

        # Update epoch progress bar
        epoch_bar.set_postfix({"avg_loss": f"{avg_epoch_loss:.4f}"})

        # Sample and log images every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_images = sample_plot_image(model, epoch, step, run)

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"/diffusion/results/train/model_checkpoint_epoch_{epoch+1}.pt"
            torch.save(model, checkpoint_path)
            run[f"checkpoints/model_checkpoint_epoch_{epoch+1}"].upload(checkpoint_path)

    # End the Neptune run
    run.stop()


if __name__ == "__main__":
    main()
