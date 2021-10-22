import torch
import numpy as np
import matplotlib.pyplot as plt


def perdictions(model, data_to_infer, train_data, lc_combo, device):

	N = 20 # number of references to compare againts and average
	out_data = np.zeros(data_to_infer.shape)
	trlen = train_data.shape[0]
	for i in range(data_to_infer.shape[0]):
		cur_data = data_to_infer[i, :]
		
		if lc_combo > 0:
			for j in range(N):
				ids = torch.randperm(trlen)
				for k in range(len(lc_combo)):
					fct = lc_combo[k]
					cur_data = cur_data + fct*train_data[ids[k], :]
				
				tmp = model(torch.from_numpy(cur_data).to(device).float())
				tmp = tmp.detach().cpu().numpy()
				for k in range(len(lc_combo)):
					fct = lc_combo[k]
					tmp = tmp - fct*train_data[ids[k], :]
				out_data[i, ...] += tmp
			out_data[i, ...] = out_data[i, ...] / N

		else:
			tmp = model(torch.from_numpy(cur_data).to(device).float())
			out_data[i, ...] = tmp.detach().cpu().numpy()
	
	return out_data

def reconstruction_errors_plot(in_data_train, in_data_val, out_data_train, out_data_val, save_dir):
	
	fig = plt.figure()
	dim = in_data_train.shape[1]
	if dim == 3:
		ax = fig.add_subplot(121, projection='3d')
		ax.scatter(in_data_val[:, 0], in_data_val[:, 1], in_data_val[:, 2], color='r')
		ax.set_title('GT Validation')
		ax = fig.add_subplot(122, projection='3d')
		ax.scatter(out_data_val[:, 0], out_data_val[:, 1], out_data_val[:, 2], color='b')
		ax.set_title('Reconstructed Validation')
		plt.savefig(save_dir + '/val_reconstruction.png')

		plt.clf()

		fig = plt.figure()
		ax = fig.add_subplot(121, projection='3d')
		ax.scatter(in_data_train[:, 0], in_data_train[:, 1], in_data_train[:, 2], color='r')
		ax.set_title('GT Training')
		ax = fig.add_subplot(122, projection='3d')
		ax.scatter(out_data_train[:, 0], out_data_train[:, 1], out_data_train[:, 2], color='b')
		ax.set_title('Reconstructed Training')
		plt.savefig(save_dir + '/train_reconstruction.png')
		plt.clf()
	
	if dim == 2:
		ax = fig.add_subplot(121)
		ax.scatter(in_data_val[:, 0], in_data_val[:, 1], color='r')
		ax.set_title('GT Validation')
		ax = fig.add_subplot(122)
		ax.scatter(out_data_val[:, 0], out_data_val[:, 1], color='b')
		ax.set_title('Reconstructed Validation')
		plt.savefig(save_dir + '/val_reconstruction.png')

		plt.clf()

		fig = plt.figure()
		ax = fig.add_subplot(121)
		ax.scatter(in_data_train[:, 0], in_data_train[:, 1], color='r')
		ax.set_title('GT Training')
		ax = fig.add_subplot(122)
		ax.scatter(out_data_train[:, 0], out_data_train[:, 1], color='b')
		ax.set_title('Reconstructed Training')
		plt.savefig(save_dir + '/train_reconstruction.png')
		plt.clf()

	# box plots for the reconstruction error
	train_err = (in_data_train - out_data_train)**2
	train_err_avg = np.sqrt(train_err.mean(1))
	val_err = (in_data_val - out_data_val)**2
	val_err_avg = np.sqrt(val_err.mean(1))

	fig = plt.figure()
	ax = fig.add_subplot(121)
	data_lists = {}
	data_lists["train"] = (train_err_avg).tolist()
	data_lists["validation"] = (val_err_avg).tolist()
	bp = ax.boxplot(data_lists.values(), patch_artist=True, meanprops=dict(color='purple'), meanline=True, showmeans=True, medianprops=dict(color='k'))
	ax.set_xticklabels(data_lists.keys())
	colors = ['pink', 'lightblue']

	for patch, color in zip(bp['boxes'], colors):
	    patch.set_facecolor(color)

	plt.savefig(save_dir + '/reconstruction_errors.png')
	plt.clf()