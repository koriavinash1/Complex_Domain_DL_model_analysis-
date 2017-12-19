import tensorflow as tf
from keras.datasets import mnist, cifar10
import time
import keras
from keras.layers import Dropout, Dense, Input, Activation, Conv2D
from keras.models import Model, Sequential, load_model
from keras import optimizers
from keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint
import sys
import os
import numpy as np

# data
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
def generate_dataset(output_dim = 8,num_examples=1000):
    def int2vec(x,dim=output_dim):
        out = np.zeros(dim)
        binrep = np.array(list(np.binary_repr(x))).astype('int')
        out[-len(binrep):] = binrep
        return out

    x_left_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    x_right_int = (np.random.rand(num_examples) * 2**(output_dim - 1)).astype('int')
    y_int = x_left_int + x_right_int

    x = list()
    for i in range(len(x_left_int)):
        x.append(np.concatenate((int2vec(x_left_int[i]),int2vec(x_right_int[i]))))

    y = list()
    for i in range(len(y_int)):
        y.append(int2vec(y_int[i]))

    x = np.array(x)
    y = np.array(y)
    
    return (x,y)


# hyper params
image_size = 32
batch_size = 2
image_channels = 3
num_classes = 10
epochs = 500
data_augmentation = True


x_train, y_train = generate_dataset(num_examples=2000, output_dim = 2)

#####

layer_1_dim = 12
layer_2_dim = 6
# layer_3_dim = 128
# layer_4_dim = 64

# input_dim = image_size*image_size*image_channels
# output_dim = num_classes

input_dim = len(x_train[0])
output_dim = len(y_train[0])



# normalize data
# x_train = (x_train.astype('float32')/255).reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3])
# x_test = (x_test.astype('float32')/255).reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3])


# # onehot converstion
# y_train = (keras.utils.to_categorical(y_train, num_classes))
# y_test = (keras.utils.to_categorical(y_test, num_classes))


# print status
print(x_train.shape)
print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

# graph setup

x = tf.placeholder(tf.float32, shape=(None, input_dim))
y = tf.placeholder(tf.float32, shape=(None, output_dim))

class DataSet(object):
	def __init__(self, images, labels):
		self._num_examples = len(images)
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	def shuffle_in_unison(self, a, b):
		a = np.array(a)
		b = np.array(b, ndmin=2)
		assert len(a) == len(b)
		shuffled_a = np.empty(a.shape, dtype=a.dtype)
		shuffled_b = np.empty(b.shape, dtype=b.dtype)
		permutation = np.random.permutation(len(a))
		for old_index, new_index in enumerate(permutation):
			shuffled_a[new_index] = a[old_index]
			shuffled_b[new_index] = b[old_index]
		return shuffled_a, shuffled_b

	def next_batch(self, batch_size):
		if self._index_in_epoch > len(self._images)-batch_size:
			self._index_in_epoch = 0
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		end = self._index_in_epoch
		return self.shuffle_in_unison(self._images[start:end], self._labels[start:end])



def define_variable(shape, name): 
	initializer = tf.contrib.layers.variance_scaling_initializer()
	return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

def fx_pluse_b(inx, mu, sd, b):
	l = int(mu.get_shape()[0])
	l_1 = int(mu.get_shape()[1])
	l_0 = batch_size
	print l,l_1,l_0
	
	updated = tf.Variable([l_0, l_1])
	for i in range(0,l_0):
		for j in range(0, l_1):
			distri = tf.div(tf.exp(-0.5*tf.div(tf.multiply((inx[i,]-tf.transpose(mu)[j,]), (inx[i,]-tf.transpose(mu)[j,])), 
								tf.multiply(tf.transpose(sd)[j,], tf.transpose(sd)[j,]))), tf.transpose(sd)[j,])
			updated[i,j].assign(tf.reduce_sum(distri))

	# tfx = np.exp(-0.5*np.matmul((mu/sd).T, mu/sd))
	return tf.add(updated, b)

def nonlinear(x):
	return tf.nn.sigmoid(x)

# assuming all weights and biase follow standard normal distribution
class Layer:
	def __init__(self, input_dim, output_dim, name = None):
		self.mean = define_variable([input_dim, output_dim], name+'mean')
		self.sd = define_variable([input_dim, output_dim], name+'sd')
		self.biase = define_variable([output_dim], name+"biase")

	def forward(self, layer_input):
		self.input = layer_input
		self.fx = fx_pluse_b(self.input, self.mean, self.sd, self.biase)
		return nonlinear(self.fx)


layer_1 = Layer(input_dim, layer_1_dim, "layer1")
layer_2 = Layer(layer_1_dim, layer_2_dim, "layer2")
output_layer = Layer(layer_2_dim, output_dim, "layerout")


# graph building 

layer_1_out = layer_1.forward(x)
layer_2_out = layer_2.forward(layer_1_out)
pred = output_layer.forward(layer_2_out)
print "asdfghjkl;"
print pred
print y

# compiling... 
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
cost = tf.nn.l2_loss(pred-y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


cifar10_data = DataSet(x_train, y_train)

with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step <= epochs * 60000/ batch_size:
		batch_x, batch_y = cifar10_data.next_batch(batch_size)

		_, ly1, ly2 = sess.run([optimizer,layer_1_out,layer_2_out], feed_dict={x: batch_x, y: batch_y})

		# print(ly1, ly2)
		
		if step*batch_size % 1000 == 0:
			loss, acc, yas = sess.run([cost, accuracy, pred], feed_dict={x: batch_x, y: batch_y})
		    	print "TRAINING IMAGE= {:.1f}".format(step*batch_size)+", EPOCH= {:.5f}".format(step*batch_size//60000)+", loss= {:.6f}".format(loss) + ", Accuracy= {:.5f}".format(acc)+ ", output= {}".format(yas)+", true data = {}".format(batch_y)
		step += 1
	print "Optimization Finished!"