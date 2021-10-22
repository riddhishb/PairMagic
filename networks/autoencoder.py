import torch.nn as nn
import torch.nn.functional as F

'''
TODO: generalize this to handle multiple layers
'''

class fc_autonet(nn.Module):
	def __init__(self, dim, bottleneck):
		super(fc_autonet, self).__init__()
		self.fc_e_1 = nn.Linear(dim, 128)
		self.fc_e_2 = nn.Linear(128, 64)
		self.fc_e_3 = nn.Linear(64, bottleneck)
		self.fc_d_1 = nn.Linear(bottleneck, 64)
		self.fc_d_2 = nn.Linear(64, 128)
		self.fc_d_3 = nn.Linear(128, dim)

	def encoder(self, z):
		z = F.relu(self.fc_e_1(z))
		z = F.relu(self.fc_e_2(z))
		z = F.relu(self.fc_e_3(z))
		return z

	def decoder(self, z):
		z = F.relu(self.fc_d_1(z))
		z = F.relu(self.fc_d_2(z))
		z = self.fc_d_3(z)
		return z

	def forward(self, x):
		lat = self.encoder(x)
		out = self.decoder(lat)
		return out
