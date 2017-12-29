import numpy as np
import tensorflow as tf

def generate_dataset(num_examples=1000, t='one'):
	x_left_int = (np.random.rand(num_examples)*5 + 2).astype('int')
	x_right_int = (np.random.rand(num_examples)*5 + 2).astype('int')

	x = np.vstack([x_left_int, x_right_int])

	if t == "one":
		y = (x_left_int + x_right_int)*(x_left_int + x_right_int)
	elif t == "two":
		y = (x_left_int - x_right_int)*(x_left_int - x_right_int)
	else:
		y = x_left_int * x_right_int

	return (x.T, y.reshape(num_examples, 1))

R = 500
batch_size = 50
hidden_units = 3
iterations = 150
input_units = 2
output_units = 1
models = 3

external_memory_block_1 = np.zeros((input_units, hidden_units, R, models), dtype="float32")
external_memory_block_2 = np.zeros((hidden_units, output_units, R, models), dtype="float32")

# useful functions...
def define_variable(shape, name): 
	initializer = tf.contrib.layers.variance_scaling_initializer()
	return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

def nonlinear(x):
	return tf.nn.relu(x)

# layer ppty...
class Layer:
	def __init__(self, input_dim, output_dim, name = None):
		self.weights = define_variable([input_dim, output_dim], name+"weights")
		# self.biase = define_variable([output_dim], name+"biase")

	def forward(self, layer_input):
		self.input = layer_input
		self.fx = tf.matmul(self.input, self.weights)
		return nonlinear(self.fx)

	def weights(self):
		return self.weights


# graph 
class Network(object):
	def __init__(self, layers, x, y, batch_size, iterations):
		self.layers = layers
		self.trainX = x
		self.trainY = y
		self.batch_size = batch_size
		self.iterations = iterations

	def cost_fn(self, pred, true):
		a = np.sqrt((true - pred)*(true - pred))
		return a

	def train(self):
		input_ph_x = tf.placeholder(tf.float32, shape=(None, input_units))
		output_ph_y = tf.placeholder(tf.float32, shape=(None, output_units))

		temp = input_ph_x
		for i in range(0, len(self.layers)):
			temp = self.layers[i].forward(temp)

		# compiling... 
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		cost = tf.nn.l2_loss(temp - output_ph_y)
		optimizer =tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(cost)
		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			step = 0
			while step <= iterations / batch_size:
				batch_x, batch_y = self.trainX[step*batch_size: (step+1)*batch_size], self.trainY[step*batch_size: (step+1)*batch_size]
				# print batch_x, batch_y
				sess.run(optimizer, feed_dict={input_ph_x: batch_x, output_ph_y: batch_y})

				
				if step*batch_size % 100 == 0:
					loss = sess.run(cost, feed_dict={input_ph_x: batch_x, output_ph_y: batch_y})
				    	print "TRAINING IMAGE= {:.1f}".format(step*batch_size)+", EPOCH= {:.5f}".format(step*batch_size//60000)+", loss= {:.6f}".format(loss)
				step += 1
			print "Optimization Finished!"

			weights = [sess.run(self.layers[0].weights), sess.run(self.layers[1].weights)]
		return weights


		
	def test(self):
		pass


# for monte-carlo simulations..
# R number of realizations
# each realizations with 500 iterations....

class simulations(object):
	def __init__ (self, model, x, y, bsize, iterations, R, mod=0):
		self.model = model
		self.realizations = R
		self.mod = mod

	def layer_init(self, r):
		layer_1 = Layer(input_units, hidden_units, "layer1_"+str(r)+"_"+str(self.mod))
		layer_2 = Layer(hidden_units, output_units, "layer2_"+str(r)+"_"+str(self.mod))
		layers = [layer_1, layer_2]
		return layers

	def save_weights(self, weights, i):
		# print weights[0].shape, weights[1].shape
		# print external_memory_block_2[:, i, self.mod].shape

		external_memory_block_1[:,:, i, self.mod] = weights[0]
		external_memory_block_2[:,:, i, self.mod] = weights[1]

	def run_simulations(self):
		for i in range(self.realizations):
			layers = self.layer_init(i)
			layers = self.model(layers, x, y, batch_size, iterations).train()
			self.save_weights(layers, i)
			print("*"*50)
			print("simulation number: {}".format(i))


# layer building

x,y = generate_dataset(t="one")
# build graph ...
sim = simulations(Network, x, y, batch_size, iterations, R, 0)
sim.run_simulations()

print ("#"*50)
print("mod B")
x,y = generate_dataset(t="two")
# build graph ...
sim = simulations(Network, x, y, batch_size, iterations, R, 1)
sim.run_simulations()


print ("#"*50)
print("mod C")
x,y = generate_dataset(t="three")
# build graph ...
sim = simulations(Network, x, y, batch_size, iterations, R, 2)
sim.run_simulations()


np.save("./block0.npy", external_memory_block_1)
np.save("./block1.npy", external_memory_block_2)

















# for iter in range(iterations):
# 			error = 0

# 			for batch_i in range(int(len(x) / batch_size)):
# 				batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
# 				batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]

# 				# # forward pass
# 				# temp = batch_x
# 				# for i in range(0, len(self.layers)):
# 				# 	temp = self.layers[i].forward(temp)

# 				# # loss comutation...
# 				# cost = self.cost_fn(temp, batch_y)

# 				# delta = cost
# 				# # backward pass...
# 				# for i in range(0, len(self.layers)):
# 				# 	delta = self.layers[len(self.layers) -1 -i].backward(delta)

# 				# # weight updates
# 				# for layer in self.layers:
# 				# 	layer.update()

# 				layer_1_out = self.layers[0].forward(batch_x)
# 				layer_2_out = self.layers[1].forward(layer_1_out)

# 				output_layer_delta = self.cost_fn(layer_2_out, batch_y)

# 				layer_1_delta = self.layers[1].backward(output_layer_delta)
# 				self.layers[0].backward(layer_1_delta)

# 				self.layers[0].update()
# 				self.layers[1].update()

# 				error += np.sum(np.abs(cost*temp*(1-temp)))
# 				# error += np.sum(np.abs(output_layer_out))

# 			# print status...
# 			if(iter % 10 == 0):
# 				sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error) )
# 				print("")
# 		return self.layers