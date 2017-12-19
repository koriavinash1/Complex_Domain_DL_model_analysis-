import numpy as np
import sys

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

def define_comp_tensor(input_dim, output_dim):
	return np.random.randn(input_dim, output_dim)+i*

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_out2deriv(out):
	return out * (1 - out)



def cost_fn_complex(pred, true):
	true = true*np.exp(1i  * np.arrange(len(true)) / len(true))
	angle_loss = true[]
	pass


#############################################################################
def cost_fn(pred, true):
	a = np.sqrt((true - pred)*(true - pred))
	return a

# normal distribution function for probabilistic nature 
def fx_pluse_b(inx, mu, sd, b=0):
	# s = np.reciprocal(sd)
	l = mu.shape[0]
	l_1 = mu.shape[1]
	l_0 = inx.shape[0]

	updated = np.zeros((l_0, l_1))
	for i in range(0,l_0):
		for j in range(0, l_1):
    			updated[i,j] = np.sum(np.exp(-0.5*(inx[i,]-mu.T[j,])*(inx[i,]-mu.T[j,])/sd.T[j,]/sd.T[j,])/sd.T[j,])

	# tfx = np.exp(-0.5*np.matmul((mu/sd).T, mu/sd))
	return updated

def fx_deriv_mu(a, x,mu,sd):

	l = mu.shape[0]
	l_1 = mu.shape[1]
	l_0 = x.shape[0]

	updated = np.zeros((l_0, l_1))
	for i in range(0,l_0):
	    	for j in range(0, l_1):
	        		updated[i,j] = np.sum((x[i,] -mu.T[j,])/sd.T[j,])

	return -2*a*updated

def fx_deriv_sd(a, x,mu,sd):
	l = mu.shape[0]
	l_1 = mu.shape[1]
	l_0 = x.shape[0]

	updated = np.zeros((l_0, l_1))
	for i in range(0,l_0):
		for j in range(0, l_1):
			updated[i,j] = np.sum((x[i,] -mu.T[j,])/(sd.T[j,]*sd.T[j,]*sd.T[j,]) - 1/ sd.T[j,])

	return a*updated
#################################################################################


# class Layer(object):
    
# 	def __init__(self,input_dim, output_dim,nonlin,nonlin_deriv, alpha = 10):
# 	    self.mean = np.random.randn(input_dim, output_dim)
# 	    self.sd = np.random.randn(input_dim, output_dim)
# 	    # self.biase = np.random.randn(output_dim)
# 	    self.nonlin = nonlin
# 	    self.nonlin_deriv = nonlin_deriv
# 	    self.alpha = alpha

# 	def forward(self, input):
# 	    self.input = input
# 	    self.fx = fx_pluse_b(self.input, self.mean, self.sd)
# 	    self.output = self.nonlin(self.fx)
# 	    return self.output

# 	def backward(self, output_delta):
# 	    self.mean_output_delta = output_delta[0]*self.nonlin_deriv(self.output)*fx_deriv_mu(self.fx, self.input, self.mean, self.sd)
# 	    self.sd_output_delta = output_delta[1]*self.nonlin_deriv(self.output)*fx_deriv_sd(self.fx, self.input, self.mean, self.sd)
# 	    return [self.mean_output_delta.dot(self.mean.T), self.sd_output_delta.dot(self.sd.T)]

# 	def update(self):
# 	    self.mean -= self.input.T.dot(self.mean_output_delta) * self.alpha
# 	    self.sd -= self.input.T.dot(self.sd_output_delta) * self.alpha 
        

num_examples = 100
output_dim = 2
iterations = 1000

x,y = generate_dataset(num_examples=num_examples, output_dim = output_dim)

batch_size = 5
input_dim = len(x[0])
layer_1_dim = 12
layer_2_dim = 6


# layer building

layer_1 = Layer(input_dim, layer_1_dim, sigmoid, sigmoid_out2deriv)
layer_2 = Layer(layer_1_dim, layer_2_dim, sigmoid,sigmoid_out2deriv)
output_layer = Layer(layer_2_dim, output_dim, sigmoid, sigmoid_out2deriv)



# graph building...
for iter in range(iterations):
    error = 0

    for batch_i in range(int(len(x) / batch_size)):
        batch_x = x[(batch_i * batch_size):(batch_i+1)*batch_size]
        batch_y = y[(batch_i * batch_size):(batch_i+1)*batch_size]
        
        layer_1_out = layer_1.forward(batch_x)
        layer_2_out = layer_2.forward(layer_1_out)
        output_layer_out = output_layer.forward(layer_2_out)

        output_layer_delta = cost_fn(output_layer_out, batch_y)

        layer_2_delta = output_layer.backward(output_layer_delta)
        layer_1_delta = layer_2.backward(layer_2_delta)
        layer_1.backward(layer_1_delta)

        layer_1.update()
        layer_2.update()
        output_layer.update()
        
        # error += np.sum(np.abs(output_layer_delta*output_layer_out*(1-output_layer_out)))
        error += np.sum(np.abs(output_layer_out))


    # print status...
    if(iter % 10 == 0):
        sys.stdout.write("\rIter:" + str(iter) + " Loss:" + str(error) )
        print("")
    if(iter % 100 == 0):
        print("\n")
        print("*"*50)   
        print(batch_y, output_layer_out)



# function for loss calculations
# in complex domain
def complex_cost_fn(true, pred):
	return
	

class complexLayer(object):
	def __init__(self, input_dim, output_dim, alpha = 10):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self,alpha = alpha	

	def define_comp_layer(self, input_dim, output_dim):
		return np.random.randn(input_dim, output_layer) + \
			1i *  np.random.randn(input_dim, output_dim)

	def wx_pluse_b(self, w,x, b=None):
		return np.add(np.multiply(w,x), b)

	# activation function sigmoid in this case
	# add multiple options as relu, tanh, elu, leaky relu
	def nonlinear(self, x):
		return 1 / (1 + np.exp(-x))

	# deravative of an activation funciton sigmoid in this case
	# add multiple options as relu, tanh, elu. leaky relu 
	def nonlinear_derivative_w(self, out):
		return return out * (1 - out)

	# forward pass for network 
	def forward(self, input):
		self.input = input
		self.weight = self.define_comp_layer(self.input_dim, self.output_dim)
		self.wx = self.wx_pluse_b(self.input, self.weight)
		self.output = self.nonlinear(self.fx)
		return self.output

	# backward pass for network
	def backward(self):
		self.mean_output_delta = output_delta[0]*self.nonlin_deriv(self.output)*fx_deriv_mu(self.fx, self.input, self.mean, self.sd)
		self.sd_output_delta = output_delta[1]*self.nonlin_deriv(self.output)*fx_deriv_sd(self.fx, self.input, self.mean, self.sd)
		return [self.mean_output_delta.dot(self.mean.T), self.sd_output_delta.dot(self.sd.T)]		

	# weight updates
	def update(self):
		pass 