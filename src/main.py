import matplotlib.pyplot as plt
import neptune
import torch
import torchvision
from neptune.types import File
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import io

import src.config as conf
from src.loss import get_loss
from src.scheduler import get_scheduler
from src.unet import SimpleUnet
from src.utils import get_index_from_list, show_tensor_image

run = neptune.init_run(
    project="leryud/diffusion",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZmE2NmFhMy0wODUzLTQ1YmItYjg3Zi1iMjU1NWQzNzg5YmUifQ==",
)


params = {"learning_rate": conf.lr, "optimizer": "Adam"}
run["parameters"] = params


def load_transformed_dataset(img_size):
    data_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales to 0;1
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scales to -1;1
    ]

    data_transforms = transforms.Compose(data_transforms)
    train = torchvision.datasets.StanfordCars(
        root="data/", download=False, transform=data_transforms
    )
    test = torchvision.datasets.StanfordCars(
        root="data/", download=False, transform=data_transforms, split="test"
    )

    return torch.utils.data.ConcatDataset([train, test])


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns the denoised image.
    Applies noise to this images if we are not in the last step yet.
    """
    scheduler = get_scheduler(conf.T)
    betas_t = get_index_from_list(scheduler["betas"], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        scheduler["sqrt_one_minus_alphas_cumprod"], t, x.shape
    )
    sqrt_recip_alpas_t = get_index_from_list(scheduler["sqrt_recip_alphas"], t, x.shape)

    # Call model (current image - noise_prediction)
    model_mean = sqrt_recip_alpas_t * (x - betas_t) * model(x, t) / sqrt_one_minus_alphas_cumprod_t

    posterior_variance_t = get_index_from_list(scheduler["posterior_variance"], t, x.shape)

    if t == 0:
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, epoch, run=None):
    # Sample noise
    img_size = conf.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=conf.device)
    fig = plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(conf.T / num_images)

    generated_images = []

    for i in range(0, conf.T)[::-1]:
        t = torch.full((1,), i, device=conf.device)
        img = sample_timestep(model, img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            pil_image = show_tensor_image(img.detach().cpu())
            generated_images.append(pil_image)

    if run:
        # Log the figure to Neptune
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        run["sampling_process"].append(File.as_image(buf), description=f"Epoch {epoch}")

        # Log individual generated images
        for idx, img in enumerate(generated_images):
            run["generated_samples"].append(
                File.as_image(img), description=f"Epoch {epoch}, Step {idx}"
            )

    plt.close(fig)
    return generated_images


def main():
    # Initialize Neptune run
    run = neptune.init_run(
        project="your_project_name", name="diffusion_model_training", tags=["diffusion", "unet"]
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
    dataloader = DataLoader(data, batch_size=conf.BATCH_SIZE, shuffle=True, drop_last=True)

    model = SimpleUnet()
    model.to(conf.device)
    optimizer = Adam(model.parameters(), lr=conf.lr)
    epochs = conf.epochs

    # Create a TQDM progress bar for epochs
    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0

        # Create a TQDM progress bar for batches
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for step, batch in enumerate(batch_bar):
            optimizer.zero_grad()

            t = torch.randint(0, conf.T, (conf.BATCH_SIZE,), device=conf.device).long()
            loss = get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()

            # Update batch progress bar
            batch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Log batch-level metrics
            run["train/batch/loss"].append(loss.item())
            run["train/batch/step"].append(step + epoch * len(dataloader))

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
                sample_images = sample_plot_image(model, epoch)
                image_grid = make_grid(sample_images, nrow=4)

                # Log the image grid to Neptune
                run["train/sample_images"].append(
                    File.as_image(image_grid.permute(1, 2, 0).cpu().numpy()),
                    description=f"Epoch {epoch}",
                )

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"model_checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            run["checkpoints"].append(File(checkpoint_path))

    # End the Neptune run
    run.stop()


if __name__ == "__main__":
    main()
    run.stop()
