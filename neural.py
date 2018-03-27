import numpy as np
import random

def sigmoid(x):
	"""
	Returns the sigmoid of the data 
	
	"""

	return (1 / (1 + np.exp(-x)))

def sigmoidprime(x):
	return sigmoid(x) * (1 - sigmoid(x))

class OneNeuronNeuralNetwork:
	"""

	A no-layer neural network with a single neuron/perceptron
	----------------------------------------------------------------
	data = The set of data you wish to pass into the network

	"""
	def __init__(self, data):
		self.weights = np.array([np.random.randn() for i in range(len(data[0]))])
		self.bias = np.random.randn()
	
	def fit(self, data, classes, learnrate = 0.001):
		"""

		Changes the weights and bias using backpropagation
		----------------------------------------------------------------
		data = Set of data to be working with

		classes = Set of classes corresponding to the data given

		learnrate (default = 0.001) = Specifies the learning rate of the algorithm
		or the rate in which cost should be brought to a minimum

		"""
		for a in range(10000):
			randomindex = int(random.uniform(1, len(data)))
			point = data[randomindex]
			z = np.sum(point * self.weights) + self.bias
			pred = sigmoid(z)
			cost = (pred - classes[randomindex]) ** 2
			dcost_dpred = 2 * (pred - classes[randomindex])
			dpred_dz = sigmoidprime(z)
			dz_dw = point
			dz_db = 1.0
			dcost_dz = dcost_dpred * dpred_dz
			dcost_dw = dcost_dz * dz_dw
			dcost_db = dcost_dz * dz_db

			self.weights = self.weights - learnrate * dcost_dw
			self.bias = self.bias - learnrate * dcost_db

	def predict(self, inputs):
		"""

		Predicts the output of a given input. Can be called ONLY after calling the
		fit function
		-------------------------------------------------------------------
		inputs = Set of test inputs to extract output according to the neural net
		model

		"""
		outputs = []
		for a in inputs:
			pred = sigmoid(np.sum(self.weights * a) + self.bias)
			outputs.append(int(round(pred)))
		return outputs
