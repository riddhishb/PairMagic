import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import itertools

def data_model(indata, batchSz, type, combfactors, train=False):

	if type == "lccomb":
		dataset = dataloader_lccomb(indata, combfactors)

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
