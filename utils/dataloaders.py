import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import itertools

def data_model(indata, batchSz, type, combfactors, train=False):

	if type == "lccomb":
		dataset = dataloader_lccomb(indata, combfactors)
	if type == "mixup":
		# here combfactors is actually the alpha for beta distribution
		dataset = dataloader_mixup(indata, combfactors)

	return DataLoader(
	  dataset,
	  batch_size=batchSz,
	  shuffle=train,
	  pin_memory=True,
	  drop_last=True
	)

class dataloader_lccomb(Dataset):

	def __init__(self, X, combfactors):
		self.data = X
		self.size = self.data.shape[0]
		self.combfactors = combfactors

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		land = self.data[index%self.size, :]
		diff_pts = land
		if len(self.combfactors) > 0:
			for fct in self.combfactors:
				tmpid = torch.randperm(self.size)
				tmpland = self.data[tmpid[0] % self.size, :]
				diff_pts = diff_pts + fct*tmpland

		return torch.from_numpy(diff_pts)

class dataloader_mixup(Dataset):

	def __init__(self, X, alpha):
		self.data = X
		self.size = self.data.shape[0]
		self.alpha = alpha

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		land = self.data[index%self.size, :]
		tmpid = torch.randperm(self.size)
		tmpland = self.data[tmpid[0] % self.size, :]
		if self.alpha > 0:
			lam = np.random.beta(self.alpha, self.alpha)	
		else:
			lam = 1
		outpts = lam*land + (1-lam)*tmpland
		return torch.from_numpy(outpts)

