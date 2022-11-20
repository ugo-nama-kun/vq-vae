
import numpy as np

import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib

from model import ConvAE, ConvVAE

matplotlib.use('TkAgg')

import einops

data = np.load("dataset/Freeway-v4.npy")
print(data.shape)

transform = transforms.Compose([
    transforms.Resize(size=(84, 84)),
])


def get_batch(batch_size):
    with torch.no_grad():
        index = np.random.choice(range(data.shape[0]), batch_size)
        im_batch = torch.FloatTensor(einops.rearrange(data[index], "b h w c -> b c h w"))
        im_batch = transform(im_batch) / 255.
    return im_batch


# im = einops.rearrange(get_batch(5).numpy(), "b c h w -> h (b w) c")
# plt.imshow(im, interpolation="nearest")
# plt.pause(0.1)


if __name__ == '__main__':
    
    dim_latent = 10
    n_category = 5
    n_query = 7
    
    model = ConvAE(dim_latent=dim_latent * n_category)
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.003)
    
    n_itr = 10000
    
    plt.figure()
    hist_loss = []
    
    for i in range(n_itr):
        
        model.train()
        
        x = get_batch(64)
        
        _, _, loss = model(x)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        hist_loss.append(loss.item())
        print(f"{i + 1}-th itr loss: {hist_loss[-1]}")
        
        if (i + 1) % 10 == 0:
            
            model.eval()
            
            with torch.no_grad():
                x_test = get_batch(5)
                x_pred, z, _ = model(x_test)
            
            plt.clf()
            
            plt.subplot(411)
            plt.imshow(einops.rearrange(x_test, "b c h w -> h (b w) c").numpy())
            plt.title("input")
            
            plt.subplot(412)
            if isinstance(model, ConvAE) or isinstance(model, ConvVAE):
                plt.imshow(einops.rearrange(z, "b (h w) -> h (b w)", h=n_category, w=dim_latent).numpy(), cmap="gray")
            # elif isinstance(model, MinVQVAE1D):
            #     plt.imshow(einops.rearrange(z, "b (h w) -> h (b w)", h=n_category, w=1).numpy(), cmap="gray")
            # elif isinstance(model, MinVQVAE) or isinstance(model, MinVQVAECos):
            #     plt.imshow(einops.rearrange(z, "b h w -> h (b w)").numpy(), cmap="gray")
            plt.title("latent")
            
            plt.subplot(413)
            plt.imshow(einops.rearrange(x_pred, "b c h w -> h (b w) c").numpy())
            plt.title("reconstruction")
            
            plt.subplot(414)
            plt.plot(hist_loss)
            plt.yscale("log")
            plt.title("Loss")
            
            plt.tight_layout()
            
            plt.pause(0.0001)

plt.show()
