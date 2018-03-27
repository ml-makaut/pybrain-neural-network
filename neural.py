import numpy as np
import random

def sigmoid(x):
	return (1 / (1 + np.exp(-x)))

def sigmoidprime(x):
	return sigmoid(x) * (1 - sigmoid(x))

# A simple no-layer neural network

class NoLayerNeuralNetwork:
	def __init__(self, data):
		self.weights = np.array([np.random.randn() for i in range(len(data[0]))])
		self.bias = np.random.randn()
	def fit(self, data, classes, learnrate = 0.001):
		for a in range(10):
			randomindex = int(random.uniform(1, len(data)))
			point = data[randomindex]
			z = np.sum(point * self.weights) + self.bias
			pred = sigmoid(z)
			cost = (pred - classes[randomindex]) ** 2
			dcost_dpred = 2 * (pred - classes[randomindex])
			dpred_dz = sigmoidprime(z)
			dz_dw = point
			dz_db = 1
			dcost_dz = dcost_dpred * dpred_dz
			dcost_dw = dcost_dz * dz_dw
			dcost_db = dcost_dz * dz_db

			self.weights = self.weights - learnrate * dcost_dw
			self.bias = self.bias - learnrate * dcost_db

	def predict(self, inputs):
		outputs = []
		for a in inputs:
			pred = sigmoid(np.sum(self.weights * a) + self.bias)
			outputs.append(int(round(pred)))
		return outputs

irisdata = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.1, 2.9, 4.7, 1.4]])
classes = [0, 0, 1, 1]
nn = NoLayerNeuralNetwork(irisdata)
nn.fit(irisdata, classes, learnrate = 0.07)
print(nn.predict([[7.1, 3.3, 4.8, 1.35], [4.6, 3.1, 1.5, 0.2]]))
