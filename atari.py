import os

import numpy as np

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib

from model import ConvAE, ConvVAE, ConvVQVAE, ConvVQVAECos

matplotlib.use('TkAgg')

import einops


env_id = "Seaquest-v4"
n_itr = 10000
dim_latent = 256
n_category = 512


data = np.load(f"dataset/{env_id}.npy")
print(data.shape)

device = torch.device(0 if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(size=(84, 84)),
])


def get_batch(batch_size):
    with torch.no_grad():
        index = np.random.choice(range(data.shape[0]), batch_size)
        im_batch = torch.FloatTensor(einops.rearrange(data[index], "b h w c -> b c h w"))
        im_batch = transform(im_batch) / 255.
    return im_batch


# model = ConvAE(dim_latent=dim_latent)
# model = ConvVAE(dim_latent=dim_latent)
# model = ConvVQVAE(dim_latent=dim_latent, n_category=n_category)
model = ConvVQVAECos(dim_latent=dim_latent, n_category=n_category).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.003)

plt.figure()
hist_loss = []

for i in range(n_itr):

    model.train()

    x = get_batch(128).to(device)

    _, _, loss, _ = model(x)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    hist_loss.append(loss.item())
    print(f"{i + 1}-th itr loss: {hist_loss[-1]}")

    if (i + 1) % 10 == 0:

        model.eval()

        with torch.no_grad():
            x_test = get_batch(1).to(device)
            x_pred, z, _, z_index = model(x_test)

        plt.subplot(221)
        plt.cla()
        plt.imshow(einops.rearrange(x_test, "b c h w -> h (b w) c").cpu().numpy())
        plt.title("input")

        plt.subplot(222)
        plt.cla()
        if isinstance(model, ConvAE) or isinstance(model, ConvVAE):
            plt.imshow(z, cmap="gray")
        elif isinstance(model, ConvVQVAE) or isinstance(model, ConvVQVAECos):
            plt.imshow((z_index[0]).cpu().numpy() / n_category, cmap="nipy_spectral")
        plt.clim(0, 1)
        plt.colorbar()
        plt.title("latent")

        plt.subplot(223)
        plt.cla()
        plt.imshow(einops.rearrange(x_pred, "b c h w -> h (b w) c").cpu().numpy())
        plt.title("reconstruction")

        plt.subplot(224)
        plt.cla()
        plt.plot(hist_loss)
        plt.yscale("log")
        plt.title("Loss")

        plt.tight_layout()

        plt.pause(0.0001)

os.makedirs("saved_model", exist_ok=True)
torch.save(model.to("cpu").state_dict(), f"saved_model/vqvae_{env_id}.pth")

plt.show()
