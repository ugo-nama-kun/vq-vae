import einops
import torch
from torch import nn
from einops.layers.torch import Rearrange


# from https://pystyle.info/pytorch-resnet/
def conv3x3(in_channels, out_channels, stride=1):
	return nn.Conv2d(
		in_channels,
		out_channels,
		kernel_size=3,
		stride=stride,
		padding=1,
		bias=False,
	)


def conv1x1(in_channels, out_channels, stride=1):
	return nn.Conv2d(
		in_channels, out_channels, kernel_size=1, stride=stride, bias=False
	)


class BasicBlock(nn.Module):
	expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

	def __init__(
			self,
			in_channels,
			channels,
			stride=1
	):
		super().__init__()
		self.conv1 = conv3x3(in_channels, channels, stride)
		self.bn1 = nn.BatchNorm2d(channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(channels, channels)
		self.bn2 = nn.BatchNorm2d(channels)

		# 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
		if in_channels != channels * self.expansion:
			self.shortcut = nn.Sequential(
				conv1x1(in_channels, channels * self.expansion, stride),
				nn.BatchNorm2d(channels * self.expansion),
			)
		else:
			self.shortcut = nn.Sequential()

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		out += self.shortcut(x)

		out = self.relu(out)

		return out


class AE(nn.Module):
	def __init__(self, input_dim, dim_latent, n_hidden=100):
		super(AE, self).__init__()
		
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, n_hidden),
			nn.GELU(),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Linear(n_hidden, dim_latent)
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(dim_latent, n_hidden),
			nn.GELU(),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Linear(n_hidden, input_dim),
			nn.Sigmoid()
		)
		
		for l in self.encoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
		
		for l in self.decoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
		
	def forward(self, x):
		z = self.encoder(x)
		x_pred = self.decoder(z)
		
		loss = torch.mean((x - x_pred) ** 2) / x.shape[0]
		
		return x_pred, z, loss


class MinVQVAE(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_hidden=100):
		super(MinVQVAE, self).__init__()
		
		self.n_category = n_category
		
		self.embed_pool = nn.Parameter(torch.randn((n_category, dim_latent), dtype=torch.float32))
		
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, dim_latent)
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(dim_latent, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, input_dim),
			nn.Sigmoid()
		)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
		
		for l in self.encoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
		
		for l in self.decoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
	
	def forward(self, x):
		z_e = self.encoder(x)
		
		with torch.no_grad():
			factor = torch.zeros((x.shape[0], self.n_category))
			for n in range(z_e.shape[0]):
				for k in range(self.n_category):
					factor[n, k] = torch.norm(z_e[n] - self.embed_pool[k])
			
			z_index = torch.argmin(factor, dim=1)

			z_discrete = nn.functional.one_hot(z_index, self.n_category)
			
		z_q = self.embed_pool[z_index]
	
		# Straight-through estimator
		z_ss = z_e - z_e.detach() + z_q.detach()
		
		x_pred = self.decoder(z_ss)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += torch.mean((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.mean((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss


class MinVQVAE_Cos(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_hidden=100):
		super(MinVQVAE_Cos, self).__init__()
		
		self.n_category = n_category
		
		self.embed_pool = nn.Parameter(torch.randn((n_category, dim_latent), dtype=torch.float32))
		
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, dim_latent),
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(dim_latent, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, input_dim),
			nn.Sigmoid()
		)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
		
		for l in self.encoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
		
		for l in self.decoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
	
	def forward(self, x):
		z_e = self.encoder(x)
		
		with torch.no_grad():
			# Cosine distance
			factor = torch.einsum("c d, b d -> b c", self.embed_pool, z_e)
			
			z_index = torch.argmax(factor, dim=1)
			
			z_discrete = nn.functional.one_hot(z_index, self.n_category)
			
		z_q = self.embed_pool[z_index]
		
		# Straight-through estimator
		z_ss = z_e - z_e.detach() + z_q.detach()
		
		x_pred = self.decoder(z_ss)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += torch.mean((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.mean((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss


class MinVQVAE_MultiQuery(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_query, n_hidden=100):
		super(MinVQVAE_MultiQuery, self).__init__()
		
		self.n_category = n_category
		self.dim_latent = dim_latent
		self.n_query = n_query
		
		self.embed_pool = nn.Parameter(torch.randn((n_category, dim_latent), dtype=torch.float32))
		
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_query * dim_latent)
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(n_query * dim_latent, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, input_dim),
			nn.Sigmoid()
		)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
		
		for l in self.encoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)

		for l in self.decoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)

	def forward(self, x):
		z_e = self.encoder(x)
		z_e = einops.rearrange(z_e, "b (i j) -> b i j", i=self.n_query, j=self.dim_latent)
		
		with torch.no_grad():
			factor = torch.zeros((x.shape[0], self.n_query, self.n_category))
			for n in range(x.shape[0]):
				for n_q in range(self.n_query):
					for k in range(self.n_category):
						factor[n, n_q, k] = torch.norm(z_e[n, n_q] - self.embed_pool[k])
			
			z_index = torch.argmin(factor, dim=2)
			
			z_discrete = nn.functional.one_hot(z_index, self.n_category)
		
		z_q = self.embed_pool[z_index]
		
		# Straight-through estimator
		z_ss = z_e - z_e.detach() + z_q.detach()
		z_ss = einops.rearrange(z_ss, "b i j -> b (i j)")
		
		x_pred = self.decoder(z_ss)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += torch.mean((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.mean((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss


class MinVQVAE_Cos_MultiQuery(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_query, n_hidden=100):
		super(MinVQVAE_Cos_MultiQuery, self).__init__()
		
		self.n_category = n_category
		self.dim_latent = dim_latent
		self.n_query = n_query
		
		self.embed_pool = nn.Parameter(torch.randn((n_category, dim_latent), dtype=torch.float32))
		
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_query * dim_latent)
		)
		
		self.decoder = nn.Sequential(
			nn.Linear(n_query * dim_latent, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, n_hidden),
			nn.GELU(),
			nn.Dropout(p=0.5),
			nn.Linear(n_hidden, input_dim),
			nn.Sigmoid()
		)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
		
		for l in self.encoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
		
		for l in self.decoder:
			if isinstance(l, nn.Linear):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
	
	def forward(self, x):
		z_e = self.encoder(x)
		z_e = einops.rearrange(z_e, "b (n d) -> b n d", n=self.n_query, d=self.dim_latent)
		
		with torch.no_grad():
			# Cosine distance
			factor = torch.einsum("c d, b n d -> b n c", self.embed_pool, z_e)
			
			z_index = torch.argmax(factor, dim=2)
			
			z_discrete = nn.functional.one_hot(z_index, self.n_category)
		
		z_q = self.embed_pool[z_index]
		
		# Straight-through estimator
		z_ss = z_e - z_e.detach() + z_q.detach()
		z_ss = einops.rearrange(z_ss, "b i j -> b (i j)")
		
		x_pred = self.decoder(z_ss)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += torch.mean((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.mean((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss


class ConvAE(nn.Module):

	def __init__(self, dim_latent):
		super(ConvAE, self).__init__()

		self.e_cnn = nn.Sequential(
			nn.Conv2d(3, 32, 3, stride=2, padding=1),
			nn.GELU(),
			nn.Conv2d(32, 32, 3, stride=2, padding=1),
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			Rearrange("b c h w -> b (c h w)"),  # flatten
			nn.Dropout(),
			nn.Linear(21 * 21 * 32, dim_latent),
		)
		
		self.d_cnn = nn.Sequential(
			nn.Linear(dim_latent, 21 * 21 * 32),
			nn.GELU(),
			nn.Dropout(),
			Rearrange("b (c h w) -> b c h w", c=32, h=21, w=21),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0, output_padding=0),
			nn.GELU(),
			nn.ConvTranspose2d(32, 3, 3, stride=2, padding=2, output_padding=1),
			nn.Sigmoid()
		)
		
		for l in self.e_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
				
		for l in self.d_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.ConvTranspose2d):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)

	def forward(self, x):
		z = self.e_cnn(x)
		x_pred = self.d_cnn(z)
		loss = torch.mean((x - x_pred) ** 2) / x.shape[0]
		return x_pred, z, loss, None


class ConvVAE(nn.Module):
	
	def __init__(self, dim_latent):
		super(ConvVAE, self).__init__()
		
		self.e_cnn = nn.Sequential(
			nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
			nn.GELU(),
			nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			Rearrange("b c h w -> b (c h w)"),  # flatten
			nn.Dropout(),
		)
		
		self.e_mean = nn.Linear(21 * 21 * 32, dim_latent)
		self.e_scale = nn.Linear(21 * 21 * 32, dim_latent)

		self.d_cnn = nn.Sequential(
			nn.Linear(dim_latent, 21 * 21 * 32),
			nn.GELU(),
			nn.Dropout(),
			Rearrange("b (c h w) -> b c h w", c=32, h=21, w=21),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0, output_padding=0, bias=False),
			nn.GELU(),
			nn.ConvTranspose2d(32, 3, 3, stride=2, padding=2, output_padding=1, bias=False),
			nn.Sigmoid()
		)
		
		for l in self.e_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
		
		for l in self.d_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.ConvTranspose2d):
				nn.init.orthogonal_(l.weight, gain=1)
				nn.init.zeros_(l.bias)
				
		nn.init.orthogonal_(self.e_mean.weight, gain=1)
		nn.init.zeros_(self.e_mean.weight)

		nn.init.orthogonal_(self.e_scale.weight, gain=1)
		nn.init.zeros_(self.e_scale.weight)

	def forward(self, x):
		h = self.e_cnn(x)
		
		mean = self.e_mean(h)
		scale = torch.exp(self.e_scale(h))
		z = mean + scale * torch.randn(mean.shape)
		
		kl = (scale ** 2 + mean ** 2 - torch.log(scale) - 0.5).sum(dim=1)
		
		x_pred = self.d_cnn(z)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += kl.mean()
		loss /= x.shape[0]
		
		return x_pred, z, loss, None


class ConvVQVAE(nn.Module):
	def __init__(self, n_category, dim_latent):
		super(ConvVQVAE, self).__init__()
		
		self.n_category = n_category
		self.dim_latent = dim_latent
		
		self.embed_pool = nn.Parameter(torch.randn((n_category, dim_latent), dtype=torch.float32))
		
		self.e_cnn = nn.Sequential(
			nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),  # 84x84x3 -> 42x42x32
			nn.GELU(),
			nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),  # 42x42x32 -> 21x21x32
			nn.GELU(),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			nn.Dropout2d(0.1),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			nn.Dropout2d(0.1),
			nn.Conv2d(32, self.dim_latent, 1, stride=1, bias=False),  #
		)
		
		self.d_cnn = nn.Sequential(
			BasicBlock(in_channels=self.dim_latent, channels=32),
			nn.GELU(),
			nn.Dropout2d(0.1),
			BasicBlock(in_channels=32, channels=32),
			nn.GELU(),
			nn.Dropout2d(0.1),
			nn.ConvTranspose2d(32, 32, 3, stride=2, padding=0, output_padding=0, bias=False),
			nn.GELU(),
			nn.ConvTranspose2d(32, 3, 3, stride=2, padding=2, output_padding=1, bias=False),
			nn.Sigmoid()
		)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
		
		for l in self.e_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
				nn.init.orthogonal_(l.weight, gain=1)
				if l.bias is not None:
					nn.init.zeros_(l.bias)
		
		for l in self.d_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.ConvTranspose2d):
				nn.init.orthogonal_(l.weight, gain=1)
				if l.bias is not None:
					nn.init.zeros_(l.bias)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
	
	def forward(self, x):
		z_e = self.e_cnn(x)  # (batch, dim_latent, h, w)
		z_tmp = einops.rearrange(z_e, "b d h w -> b h w d")
		
		with torch.no_grad():
			factor = torch.zeros((x.shape[0], self.n_category,) + z_tmp.shape[1:3])
			for n in range(x.shape[0]):
				for k in range(self.n_category):
					factor[n, k] = torch.norm(z_tmp[n] - self.embed_pool[k], dim=2)
			
			factor = einops.rearrange(factor, "b d h w -> b h w d")
			z_index = torch.argmin(factor, dim=3)
			
			z_discrete = nn.functional.one_hot(z_index, self.n_category)
		
		# print("z_index: ", z_index.shape)
		z_q = self.embed_pool[z_index]
		z_q = einops.rearrange(z_q, "b h w d -> b d h w")

		# Straight-through estimator
		# print("z_e: ", z_e.shape, ", z_q: ", z_q.shape)
		z_ss = z_e - z_e.detach() + z_q.detach()
		
		x_pred = self.d_cnn(z_ss)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += torch.mean((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.mean((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss, z_index


class ConvVQVAECos(nn.Module):
	def __init__(self, n_category, dim_latent):
		super(ConvVQVAECos, self).__init__()
		
		self.n_category = n_category
		self.dim_latent = dim_latent
		
		self.embed_pool = nn.Parameter(torch.randn((n_category, dim_latent), dtype=torch.float32))
		
		self.e_cnn = nn.Sequential(
			nn.Conv2d(3, 128, 3, stride=2, padding=1, bias=False),  # 84x84x3 -> 42x42x32
			nn.GELU(),
			nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),  # 42x42x32 -> 21x21x32
			nn.GELU(),
			BasicBlock(in_channels=128, channels=128),
			nn.GELU(),
			nn.Dropout2d(0.5),
			BasicBlock(in_channels=128, channels=128),
			nn.GELU(),
			nn.Dropout2d(0.5),
			nn.Conv2d(128, self.dim_latent, 1, stride=1, bias=False),
		)
		
		self.d_cnn = nn.Sequential(
			nn.ConvTranspose2d(self.dim_latent, 128, 1, stride=1, bias=False),
			nn.GELU(),
			nn.Dropout2d(0.5),
			BasicBlock(in_channels=128, channels=128),
			nn.GELU(),
			nn.Dropout2d(0.5),
			BasicBlock(in_channels=128, channels=128),
			nn.GELU(),
			nn.Dropout2d(0.1),
			nn.ConvTranspose2d(128, 128, 3, stride=2, padding=0, output_padding=0, bias=False),
			nn.GELU(),
			nn.ConvTranspose2d(128, 3, 3, stride=2, padding=2, output_padding=1, bias=False),
			nn.Sigmoid()
		)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
		
		for l in self.e_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
				nn.init.orthogonal_(l.weight, gain=1)
				if l.bias is not None:
					nn.init.zeros_(l.bias)
		
		for l in self.d_cnn:
			if isinstance(l, nn.Linear) or isinstance(l, nn.ConvTranspose2d):
				nn.init.orthogonal_(l.weight, gain=1)
				if l.bias is not None:
					nn.init.zeros_(l.bias)
		
		nn.init.orthogonal_(self.embed_pool, gain=1)
	
	def forward(self, x):
		z_e = self.e_cnn(x)  # (batch, dim_latent, h, w)
		
		with torch.no_grad():
			# Cosine distance
			factor = torch.einsum("c d, b d h w -> b h w c", self.embed_pool, z_e)
			
			z_index = torch.argmax(factor, dim=3)
			
			z_discrete = nn.functional.one_hot(z_index, self.n_category)
		
		# print("z_index: ", z_index.shape)
		z_q = self.embed_pool[z_index]
		z_q = einops.rearrange(z_q, "b h w d -> b d h w")
		
		# Straight-through estimator
		# print("z_e: ", z_e.shape, ", z_q: ", z_q.shape)
		z_ss = z_e - z_e.detach() + z_q.detach()
		
		x_pred = self.d_cnn(z_ss)
		
		loss = torch.mean((x - x_pred) ** 2)
		loss += torch.mean((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.mean((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss, z_index
