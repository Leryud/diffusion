import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import Adam

from src.loss import get_loss
import src.config as conf
from src.scheduler import get_scheduler
from src.utils import (
    get_index_from_list,
    show_tensor_image,
)
from src.unet import SimpleUnet

import neptune

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
def sample_plot_image(model):
    # Sample noise
    img_size = conf.IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=conf.device)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(conf.T / num_images)

    for i in range(0, conf.T)[::-1]:
        t = torch.full((1,), i, device=conf.device)
        img = sample_timestep(model, img, t)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
    plt.show()


def main():
    data = load_transformed_dataset(img_size=conf.IMG_SIZE)
    dataloader = DataLoader(data, batch_size=conf.BATCH_SIZE, shuffle=True, drop_last=True)

    model = SimpleUnet()
    model.to(conf.device)
    optimizer = Adam(model.parameters(), lr=conf.lr)
    epochs = conf.epochs

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, conf.T, (conf.BATCH_SIZE,), device=conf.device).long()
            loss = get_loss(model, batch[0], t)
            run["train/loss"].append(loss)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} | loss {loss.item()}")
                sample_plot_image(model)


if __name__ == "__main__":
    main()
    run.stop()
