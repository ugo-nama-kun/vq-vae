import numpy as np
from sklearn import datasets

from model import AE, MinVQVAE, MinVQVAE_MultiQuery, MinVQVAE_Cos_MultiQuery, MinVQVAE_Cos

import torch
import einops

import matplotlib.pyplot as plt


class NistHandle:
    def __init__(self, flat=False):
        digits = datasets.load_digits()
        self.digit_dict = {}
        for n, target in enumerate(digits.target):
            image = digits.images[n] if not flat else digits.data[n]
            if self.digit_dict.get(target) is None:
                self.digit_dict[target] = [image]
            else:
                self.digit_dict[target].append(image)

    def get(self, target: int):
        digit_list = self.digit_dict[target]
        image_index = np.random.choice(len(digit_list))
        return digit_list[image_index]

    def get_batch(self, n_batch):
        batch = np.array([self.get(target=np.random.randint(10)) for _ in range(n_batch)])
        return batch
    
    def get_test(self):
        d = np.array([self.get(target=i) for i in range(10)])
        return d


if __name__ == '__main__':
    nist_handle = NistHandle(flat=True)
    
    print(nist_handle.get(0))
    
    dim_latent = 20
    n_category = 20
    n_query = 7
    
    # model = AE(input_dim=64, dim_latent=dim_latent * n_category)
    model = MinVQVAE(input_dim=64, dim_latent=dim_latent, n_category=n_category)
    # model = MinVQVAE_Cos(input_dim=64, dim_latent=dim_latent, n_category=n_category)
    # model = MinVQVAE_MultiQuery(input_dim=64, dim_latent=dim_latent, n_category=n_category, n_query=n_query)
    # model = MinVQVAE_Cos_MultiQuery(input_dim=64, dim_latent=dim_latent, n_category=n_category, n_query=n_query)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.003)
    
    n_itr = 10000
    
    plt.figure()
    hist_loss = []
    
    for i in range(n_itr):
        
        model.train()
        
        x = torch.FloatTensor(nist_handle.get_batch(n_batch=64) / 16)
        
        _, _, loss = model(x)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        hist_loss.append(loss.item())
        print(f"{i + 1}-th itr loss: {hist_loss[-1]}")
        
        if (i + 1) % 100 == 0:
    
            model.eval()
            
            with torch.no_grad():
                x_test = torch.FloatTensor(nist_handle.get_test() / 16)
                x_pred, z, _ = model(x_test)
                
            plt.clf()

            plt.subplot(411)
            plt.imshow(einops.rearrange(x_test, "b (h w) -> h (b w)", h=8, w=8).numpy(), cmap="gray")
            plt.title("input")

            plt.subplot(412)
            if isinstance(model, AE):
                plt.imshow(einops.rearrange(z, "b (h w) -> h (b w)", h=n_category, w=dim_latent).numpy(), cmap="gray")
            elif isinstance(model, MinVQVAE) or isinstance(model, MinVQVAE_Cos):
                plt.imshow(einops.rearrange(z, "b (h w) -> h (b w)", h=n_category, w=1).numpy().transpose(), cmap="gray")
            elif isinstance(model, MinVQVAE_MultiQuery) or isinstance(model, MinVQVAE_Cos_MultiQuery):
                plt.imshow(einops.rearrange(z, "b h w -> h (b w)").numpy(), cmap="gray")
            plt.title("latent")

            plt.subplot(413)
            plt.imshow(einops.rearrange(x_pred, "b (h w) -> h (b w)", h=8, w=8).numpy(), cmap="gray")
            plt.title("reconstruction")
            
            plt.subplot(414)
            plt.plot(hist_loss)
            plt.yscale("log")
            plt.title("Loss")
            
            plt.tight_layout()
            
            plt.pause(0.0001)


plt.show()
