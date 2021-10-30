import torch
import numpy as np
import matplotlib.pyplot as plt

def prediction_single_viz(model, all_data, ids, lc_combo, save_file, device):
	
	out_data = np.zeros(all_data.shape)
	trlen = all_data.shape[0]
	for i in range(trlen):
		cur_data = all_data[i, :]
		tmp = torch.from_numpy(cur_data).to(device)
		outpttmp = torch.from_numpy(np.zeros(cur_data.shape)).to(device).float()
		indiff = torch.from_numpy(np.zeros(cur_data.shape)).to(device).float()
		indiff = tmp
		for j, fct in enumerate(lc_combo):
			tmp2 = torch.from_numpy(all_data[ids[j], :]).to(device)
			indiff = indiff + fct*tmp2

		outdiff = model(indiff.float())
		outpttmp = outdiff
		for j, fct in enumerate(lc_combo):
			tmp2 = torch.from_numpy(all_data[ids[j], :]).to(device)
			outpttmp = outpttmp - fct*tmp2.float()
		
		out_data[i, :] = outpttmp.detach().cpu().numpy()
	
	# now plot the data
	fig = plt.figure()
	ax = fig.add_subplot(121)
	import pdb; pdb.set_trace()
	colors = plt.cm.coolwarm(np.linspace(0, 1, all_data.shape[0]))
	ax.scatter(all_data[:, 0], all_data[:, 1], color=colors)
	ax.scatter(all_data[ids[0], 0], all_data[ids[0], 1], color='k')
	ax.set_title('GT Validation')
	ax = fig.add_subplot(122)
	ax.scatter(out_data[:, 0], out_data[:, 1], color=colors)
	ax.set_title('Reconstructed Validation')
	plt.savefig(save_file)

	plt.clf()
	

def perdictions(model, data_to_infer, train_data, lc_combo, device):

	out_data = np.zeros(data_to_infer.shape)
	trlen = train_data.shape[0]
	for i in range(data_to_infer.shape[0]):
		cur_data = data_to_infer[i, :]
		N = 1000
		if len(lc_combo) > 0:
			tmp = torch.from_numpy(cur_data).to(device)
			outpttmp = torch.from_numpy(np.zeros(cur_data.shape)).to(device).float()
			indiff = torch.from_numpy(np.zeros(cur_data.shape)).to(device).float()
			count = 0
			
			rdids = np.random.choice(train_data.shape[0], [N, len(lc_combo)])
			while count < N:
				indiff = tmp
				for j, fct in enumerate(lc_combo):
					curid = rdids[count, j]
					indiff = indiff + fct*torch.from_numpy(train_data[curid, :]).to(device)

				outdiff = model(indiff.float())
				for j, fct in enumerate(lc_combo):
					curid = rdids[count, j]
					outdiff = outdiff - fct*torch.from_numpy(train_data[curid, :]).to(device)
				outpttmp += outdiff
				count += 1

			outpts = outpttmp / N 
			out_data[i, :] = outpts.detach().cpu().numpy()

		else:
			tmp = torch.from_numpy(cur_data).to(device)
			outpts = model(tmp.float())
			out_data[i, :] = outpts.detach().cpu().numpy()
	
	return out_data

def reconstruction_errors_plot(in_data, out_data, trainidx, validx, save_dir, color=[]):
	
	fig = plt.figure()
	in_data_train = in_data[trainidx, :]
	in_data_val = in_data[validx, :]
	out_data_train = out_data[trainidx, :]
	out_data_val = out_data[validx, :]
	
	if len(color) == 0:
		color = np.linspace(0, 1, in_data.shape[0])
		color_train = color[trainidx]
		color_val = color[validx]
	else:
		color_train = color[trainidx]
		color_val = color[validx]

	dim = in_data_train.shape[1]

	if dim == 3:
		ax = fig.add_subplot(121, projection='3d')
		ax.scatter(in_data_val[:, 0], in_data_val[:, 1], in_data_val[:, 2], c=color_val, cmap=plt.cm.Spectral)
		ax.set_title('GT Validation')
		ax = fig.add_subplot(122, projection='3d')
		ax.scatter(out_data_val[:, 0], out_data_val[:, 1], out_data_val[:, 2], c=color_val, cmap=plt.cm.Spectral)
		ax.set_title('Reconstructed Validation')
		plt.savefig(save_dir + '/val_reconstruction.png')

		plt.clf()

		fig = plt.figure()
		ax = fig.add_subplot(121, projection='3d')
		ax.scatter(in_data_train[:, 0], in_data_train[:, 1], in_data_train[:, 2], c=color_train, cmap=plt.cm.Spectral)
		ax.set_title('GT Training')
		ax = fig.add_subplot(122, projection='3d')
		ax.scatter(out_data_train[:, 0], out_data_train[:, 1], out_data_train[:, 2], c=color_train, cmap=plt.cm.Spectral)
		ax.set_title('Reconstructed Training')
		plt.savefig(save_dir + '/train_reconstruction.png')
		plt.clf()
	
	if dim == 2:
		ax = fig.add_subplot(121)
		ax.scatter(in_data_val[:, 0], in_data_val[:, 1], color=color_val)
		ax.set_title('GT Validation')
		ax = fig.add_subplot(122)
		ax.scatter(out_data_val[:, 0], out_data_val[:, 1], color=color_val)
		ax.set_title('Reconstructed Validation')
		plt.savefig(save_dir + '/val_reconstruction.png')

		plt.clf()

		fig = plt.figure()
		ax = fig.add_subplot(121)
		ax.scatter(in_data_train[:, 0], in_data_train[:, 1], color=color_train)
		ax.set_title('GT Training')
		ax = fig.add_subplot(122)
		ax.scatter(out_data_train[:, 0], out_data_train[:, 1], color=color_train)
		ax.set_title('Reconstructed Training')
		plt.savefig(save_dir + '/train_reconstruction.png')
		plt.clf()

	# box plots for the reconstruction error
	train_err = (in_data_train - out_data_train)**2
	train_err_avg = np.sqrt(train_err.mean(1))
	val_err = (in_data_val - out_data_val)**2
	val_err_avg = np.sqrt(val_err.mean(1))

	fig = plt.figure()
	ax = fig.add_subplot(111)
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