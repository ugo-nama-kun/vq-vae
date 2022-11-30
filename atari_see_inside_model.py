import einops
import gym
import torch
from torchvision.transforms import transforms

from model import ConvVQVAECos, ConvAE, ConvVAE, ConvVQVAE
import matplotlib.pyplot as plt


env_id = "Seaquest-v4" # "SpaceInvaders-v4"
model_path = f"vqvae_{env_id}.pth"

dim_latent = 256
n_category = 512

device = torch.device(0 if torch.cuda.is_available() else "cpu")

model = ConvVQVAECos(dim_latent=dim_latent, n_category=n_category).to(device)
model.load_state_dict(torch.load(f"saved_model/{model_path}"))


transform = transforms.Compose([
    transforms.Resize(size=(84, 84)),
])

env = gym.make(env_id, render_mode="rgb_array")
img = env.reset()

while True:
    env.reset()
    done = False
    while not done:
        obs, r, done, _ = env.step(env.action_space.sample())

        img = env.render("rgb_array")
        img = img[None]
        img = torch.FloatTensor(einops.rearrange(img, "b h w c -> b c h w")).to(device)
        img = transform(img) / 255.

        model.eval()

        with torch.no_grad():
            x_pred, z, _, z_index = model(img)

        plt.subplot(131)
        plt.cla()
        plt.imshow(einops.rearrange(img, "b c h w -> h (b w) c").cpu().numpy())
        plt.title("input")

        plt.subplot(132)
        plt.cla()
        if isinstance(model, ConvAE) or isinstance(model, ConvVAE):
            plt.imshow(z, cmap="gray")
        elif isinstance(model, ConvVQVAE) or isinstance(model, ConvVQVAECos):
            plt.imshow((z_index[0]).cpu().numpy() / n_category, cmap="nipy_spectral")
        plt.title("latent")

        plt.subplot(133)
        plt.cla()
        plt.imshow(einops.rearrange(x_pred, "b c h w -> h (b w) c").cpu().numpy())
        plt.title("reconstruction")

        plt.tight_layout()

        plt.pause(0.0001)
