# Minimum VQ-VAE
A mininum implementation of VQ-VAE using NIST image dataset.

Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

### nist.py

![image](https://user-images.githubusercontent.com/1684732/202863376-a800987d-7b45-443b-8699-2a5679142106.png)

### atari.py
![vqvae](https://user-images.githubusercontent.com/1684732/202904512-f4d3c756-f854-4325-8de7-92335644cd28.png)

### One-more-thing

The original VQ-VAE used the squared error in the latent embedding selection $z_q$. The implementation of this distance can be complicated and hence tale time to compute. Then we additionally implemented the cosine distance-based VQ-VAE. This seemingly works well and very efficient to compute.


### nist.py
![image](https://user-images.githubusercontent.com/1684732/202863684-1709e49d-51fe-41ad-9357-64461252416e.png)

### atari.py
![vqvae-cos](https://user-images.githubusercontent.com/1684732/202904575-b8fe07bd-7dfb-47da-84ea-9c1327721652.png)
