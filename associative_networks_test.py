import numpy as np

def generate_dataset(num_examples=1000, t='one'):
	x_left_int = np.random.rand(num_examples).astype('int')
	x_right_int = np.random.rand(num_examples).astype('int')

	x = np.hstack([x_left_int, x_right_int])

	if t == "one":
		y = (x_left_int + x_right_int)*(x_left_int + x_right_int)
	elif t == "two":
		y = (x_left_int - x_right_int)*(x_left_int - x_right_int)
	else:
		y = x_left_int * x_right_int

	return (x,y.reshape(num_examples,1))

R = 1
batch_size = 10
hidden_units = 5
iterations = 500
input_units = 2
output_units = 1
models = 3

external_memory_block_1 = np.zeros((input_units, hidden_units, R, models), dtype="float32")
external_memory_block_2 = np.zeros((hidden_units, output_units, R, models), dtype="float32")


# useful functions...
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_out2deriv(out):
	return out * (1 - out)

# layer ppty...
class Layer(object):
	def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv, alpha = 10):
		self.weight = np.random.randn(input_dim, output_dim)
		self.nonlin = nonlin
		self.nonlin_deriv = nonlin_deriv
		self.alpha = alpha

	def forward(self, input):
		self.input = input
		self.output = self.nonlin(self.input)
		return self.output

	def backward(self, output_delta):
		# print output_delta.shape
		self.weight_output_delta = output_delta*self.nonlin_deriv(self.output)
		# print self.weight_output_delta.shape
		return self.weight_output_delta.dot(self.weight.T)

	def update(self):
		self.weight -= self.input.T.dot(self.weight_output_delta) * self.alpha

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
		for iter in range(iterations):
			error = 0

			for batch_i in range(int(len(x) / batch_size)):
				batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
				batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]

				# # forward pass
				# temp = batch_x
				# for i in range(0, len(self.layers)):
				# 	temp = self.layers[i].forward(temp)

				# # loss comutation...
				# cost = self.cost_fn(temp, batch_y)

				# delta = cost
				# # backward pass...
				# for i in range(0, len(self.layers)):
				# 	delta = self.layers[len(self.layers) -1 -i].backward(delta)

				# # weight updates
				# for layer in self.layers:
				# 	layer.update()

				layer_1_out = self.layers[0].forward(batch_x)
				layer_2_out = self.layers[1].forward(layer_1_out)

				output_layer_delta = self.cost_fn(layer_2_out, batch_y)

				layer_1_delta = self.layers[1].backward(output_layer_delta)
				self.layers[0].backward(layer_1_delta)

				self.layers[0].update()
				self.layers[1].update()

				error += np.sum(np.abs(cost*temp*(1-temp)))
				# error += np.sum(np.abs(output_layer_out))

			# print status...
			if(iter % 10 == 0):
				sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error) )
				print("")
		return self.layers

	def test(self):
		pass


class simulations(object):
	def __init__ (self, model, x, y, bsize, iterations, R, mod=0):
		self.model = model
		self.realizations = R

	def layer_init(self):
		layer_1 = Layer(input_units, hidden_units, sigmoid, sigmoid_out2deriv)
		layer_2 = Layer(hidden_units, output_units, sigmoid,sigmoid_out2deriv)
		layers = [layer_1, layer_2]
		return layers

	def save_weights(self, layers, i):
		external_memory_block_1[:, i, mod] = layers[0].weights()
		external_memory_block_2[:, i, mod] = layers[1].weights()

	def run_simulations(self):
		for i in range(self.realizations):
			layers = self.layer_init()
			layers = self.model(layers, x, y, batch_size, iterations).train()
			self.save_weights(layers, i)
			print("*"*50)
			print("simulation number: {}".format(i))


# layer building

x,y = generate_dataset(t="one")
# build graph ...
sim = simulations(Network, x, y, batch_size, iterations, R, 0)
sim.run_simulations()

print ("#"*100)
print("mod B")
x,y = generate_dataset(t="two")
# build graph ...
sim = simulations(Network, x, y, batch_size, iterations, R, 1)
sim.run_simulations()


print ("#"*100)
print("mod C")
x,y = generate_dataset(t="three")
# build graph ...
sim = simulations(Network, x, y, batch_size, iterations, R, 2)
sim.run_simulations()


np.save("./block0.npy", external_memory_block_1)
np.save("./block1.npy", external_memory_block_2)