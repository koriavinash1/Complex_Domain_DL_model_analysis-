import tensorflow as tf
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

weight_block0 = np.load("./block0.npy")
weight_block1 = np.load("./block1.npy")

matrix = np.zeros(weight_block0.shape[0], weight_block0.shape[1], weight_block0.shape[3])
reg = linear_model.LinearRegression(	)
# ts list
for model in range(weight_block0.shape[3]):
	for i in range(weight_block0.shape[0]):
		for j in range(weight_block0.shape[1]):
			matrix[i, j, model] = weight_block0[i, j, :, model]
