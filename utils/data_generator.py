'''
Generator for toy dataset used for experiments
'''
import numpy as np

def spiral_2d(T, M, a=1, b=1):
	
	'''
	a, b: parameters for the spiral
	T : number of turns
	M: samples per turn

	returns: the generated data and the associated latent parameters.
	'''

	N = M*T
	w = 3.14
	theta = np.linspace(0, T, N)
	inputData = np.zeros([N, 2])
	inputData[..., 0] = (a + b*theta)*np.cos(w*theta)
	inputData[..., 1] = (a + b*theta)*np.sin(w*theta)
	return [inputData, theta]


def corkscrew_3d(T, M, a=1, b=6):
	
	'''
	a, b: parameters for the spiral
	T : number of turns
	M: samples per turn

	returns: the generated data and the associated latent parameters.
	'''

	w = 3.14
	N = M*T
	theta = np.linspace(0, T, N)
	inputData = np.zeros([N, 3])
	inputData[..., 0] = (a - w*theta)*np.cos(w*theta)
	inputData[..., 1] = b*w*theta
	inputData[..., 2] = (a + w*theta)*np.sin(w*theta)
	return [inputData, theta]