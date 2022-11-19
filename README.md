# Minimum VQ-VAE
A mininum implementation of VQ-VAE using NIST image dataset.

Van Den Oord, Aaron, and Oriol Vinyals. "Neural discrete representation learning." Advances in neural information processing systems 30 (2017).

![image](https://user-images.githubusercontent.com/1684732/202863376-a800987d-7b45-443b-8699-2a5679142106.png)

### One-more-thing

The original VQ-VAE used the squared error in the latent embedding selection $z_q$. The implementation of this distance can be complicated and hence tale time to compute. Then we additionally implemented the cosine distance-based VQ-VAE. This seemingly works well and very efficient to compute.

