import einops
import torch
from torch import nn


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
		
		loss = torch.sum((x - x_pred) ** 2) / x.shape[0]
		
		return x_pred, z, loss


class MinVQVAE1D(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_hidden=100):
		super(MinVQVAE1D, self).__init__()
		
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
		
		loss = torch.sum((x - x_pred) ** 2)
		loss += torch.sum((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.sum((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss


class MinVQVAE(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_query, n_hidden=100):
		super(MinVQVAE, self).__init__()
		
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
		
		loss = torch.sum((x - x_pred) ** 2)
		loss += torch.sum((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.sum((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss


class MinVQVAECos(nn.Module):
	def __init__(self, input_dim, n_category, dim_latent, n_query, n_hidden=100):
		super(MinVQVAECos, self).__init__()
		
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
		
		loss = torch.sum((x - x_pred) ** 2)
		loss += torch.sum((z_e.detach() - z_q) ** 2)
		loss += 0.25 * torch.sum((z_e - z_q.detach()) ** 2)
		loss /= x.shape[0]
		
		return x_pred, z_discrete, loss
