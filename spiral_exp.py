import numpy as np
import json
import os
import torch
from torch import optim
from utils import data_generator, dataloaders
from networks import autoencoder
from inference import lc_infer
from argparse import ArgumentParser
import matplotlib.pyplot as plt


# scaled_z = (theta - theta.min()) / theta.ptp()
# colors = plt.cm.coolwarm(scaled_z)

# # latent space analysis
# latent_space = []
# if diff:
# 	# keep a reference, the median particle.
# 	refpt = inputData[50, ...]
# 	for i in range(inputData.shape[0]):
# 		inpt = inputData[i, ...]
# 		diffin = torch.from_numpy((inpt - refpt)).to(device).float()
# 		lat = model.encoder(diffin)
# 		latent_space.append(lat.detach().cpu().numpy())
# else:
# 	for i in range(inputData.shape[0]):
# 		indata = torch.from_numpy(inputData[i, ...]).to(device).float()
# 		lat = model.encoder(indata)
# 		latent_space.append(lat.detach().cpu().numpy())

# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(inputData[..., 0], inputData[..., 1], inputData[..., 2], c=theta, cmap=plt.cm.Spectral)
# ax = fig.add_subplot(122)
# ax.scatter(np.arange(inputData.shape[0]), np.array(latent_space), c=theta, cmap=plt.cm.Spectral)
# plt.savefig(out_dir + '/latent_space.png')
# plt.cla()

def runFunction(config_file):
	config = json.load(open(config_file))
	
	'''Generate the Data and create dataloaders'''

	if config['data_type'] == '2d':
		[inputData, theta] = data_generator.spiral_2d(T=5, M=20)
		dim = 2
	else:
		[inputData, theta] = data_generator.corkscrew_3d(T=5, M=20)
		dim = 3
	
	# setting up train validation split
	numtrain = int(len(inputData)*0.8)
	idx = np.arange(inputData.shape[0])
	np.random.shuffle(idx)
	trainidx = idx[:numtrain]
	validx = idx[numtrain:]
	
	# create the dataloaders
	batch_size = config['batch_size']
	combfactors = config['lin_combo']
	if config['model_type'] == "baseline":
		train_loader = dataloaders.data_model(inputData[trainidx, :], batch_size, "lccomb",  combfactors, train=True)
		val_loader = dataloaders.data_model(inputData[validx, :], batch_size, "lccomb",  combfactors, train=False)
	# TODO: add stuff for the mixup and other methods
	
	'''Load the model'''
	
	num_epochs = config['num_epochs']
	lr = config['learning_rate']
	wd = config['weight_decay']
	bottleneck = config['bottleneck']
	save_dir = config['save_dir']
	if not (os.path.exists(save_dir)):
		os.mkdir(save_dir)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# define model and optimizer
	model = autoencoder.fc_autonet(dim, bottleneck).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

	''' Model Training '''

	model.train()
	best_val = 10000
	train_loss= []
	val_loss= []

	for e in range(num_epochs):
		avg_train_loss = 0
		trcount = 0
		for x in train_loader:
			x= x.to(device)
			if config['noise']:
				x = x + config['noise_sig']*torch.rand(x.shape).to(device)

			outdata = model(x.float())
			loss = torch.mean((outdata - x.float())**2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			avg_train_loss += loss.item()
			trcount += 1
		avg_train_loss = avg_train_loss/trcount
		train_loss.append(avg_train_loss)
		avg_val_loss = 0
		valcount = 0
		
		for x in val_loader:
			x = x.to(device)
			outdata = model(x.float())
			loss = torch.mean((outdata - x.float())**2)
			avg_val_loss += loss.item()
			valcount += 1
		
		avg_val_loss = avg_val_loss /valcount
		val_loss.append(avg_val_loss)
		if avg_val_loss < best_val:
			best_val = avg_val_loss
			torch.save(model, save_dir + '/best_model.pt')

		print("Epoch, Train Loss, Val Loss ", e, avg_train_loss, avg_val_loss)

	torch.save(model, save_dir + '/final_model.pt')
	
	''' Model Evaluation and Inference '''
	model.eval()

	# plot the loss curve and the reconstruction error
	fig, ax = plt.subplots()
	ax.plot(train_loss, c= 'b')
	ax.plot(val_loss, c= 'r')
	ax.legend(['train', 'val'])
	ax.set_xlabel('epochs')
	ax.set_ylabel('Reconstruction Loss')
	plt.savefig(save_dir + '/lossplot.png')
	plt.cla()

	out_train_data = lc_infer.perdictions(model, inputData[trainidx, :], inputData, combfactors, device)
	out_val_data = lc_infer.perdictions(model, inputData[validx, :], inputData, combfactors, device)
	lc_infer.reconstruction_errors_plot(inputData[trainidx, :], inputData[validx, :], out_train_data, out_val_data, save_dir)
	lc_infer.prediction_single_viz(model, inputData, [0], combfactors, save_dir + '/recons_0.png', device)
	lc_infer.prediction_single_viz(model, inputData, [10], combfactors, save_dir + '/recons_10.png', device)
	lc_infer.prediction_single_viz(model, inputData, [20], combfactors, save_dir + '/recons_20.png', device)
	lc_infer.prediction_single_viz(model, inputData, [30], combfactors, save_dir + '/recons_30.png', device)
	lc_infer.prediction_single_viz(model, inputData, [40], combfactors, save_dir + '/recons_40.png', device)
	lc_infer.prediction_single_viz(model, inputData, [50], combfactors, save_dir + '/recons_50.png', device)
	lc_infer.prediction_single_viz(model, inputData, [60], combfactors, save_dir + '/recons_60.png', device)
	lc_infer.prediction_single_viz(model, inputData, [70], combfactors, save_dir + '/recons_70.png', device)
	lc_infer.prediction_single_viz(model, inputData, [80], combfactors, save_dir + '/recons_80.png', device)
	lc_infer.prediction_single_viz(model, inputData, [90], combfactors, save_dir + '/recons_90.png', device)

	# lc_infer.prediction_single_viz(model, inputData, [80], combfactors, save_dir + '/recons_80.png', device)
	#TODO: plot the latent space


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_file", type=str, dest="config_file", help="path to config file for the parameters")
    args = parser.parse_args()
    runFunction(**vars(args))
