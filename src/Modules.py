import numpy as np


##### MODULES

class Linear:
	def __init__(self,n_input,n_output):
		"""
		n_input = input dimension
		n_output = output dimension = nb of neurons of this layer
		"""
		self._linear = True
		self._input = None
		self._output = None
		self._weights = 2 * (np.random.rand(n_input,n_output) - 0.5) #shape (n_input,n_output)
		#self._weights = np.random.uniform(-np.sqrt(6/(n_input+n_output)),np.sqrt(6/(n_input+n_output)),size=(n_input,n_output))
		self._biases = np.zeros((1,n_output)) #shape (1,n_output)
		self._dweights = np.zeros((n_input,n_output)) #shape (n_input,n_output)
		self._dbiases = np.zeros((1,n_output)) #shape (1,n_output)
		self._dinput = None

	def forward(self, X):
		"""
		array(n_sample,n_input) -> array(n_sample,n_output)
		"""
		self._input = X
		self._output = np.dot(X,self._weights) + self._biases

	def zero_grad(self):
		"""
		Resets gradients to 0
		"""
		self._dweights = np.zeros(self._dweights.shape)
		self._dbiases = np.zeros(self._dbiases.shape)

	def backward(self,delta):
		"""
		array() -> array()
		"""
		self._dweights = np.dot(self._input.T,delta)
		self._dbiases = delta.sum(axis=0,keepdims=True)
		self._dinput = np.dot(delta,self._weights.T)


class TanH:

	def __init__(self):
		self._linear = False
		self._input = None
		self._output = None
		self._dinput = None

	def forward(self, X):
		"""
		X : array(n_samples,d)
		Calcule le passe forward
		"""
		self._input = X
		self._output = np.tanh(X)

	def backward(self, delta):
		self._dinput = (1-np.tanh(self._input)**2) * delta

	def predictions(self,output,class0=0,class1=1):
		"""
		array(n_samples) -> array(n_samples)
		Determines predicted classes using the output of the last module
		"""
		return np.where(output < 0,class0,class1)


class Sigmoid:

	def __init__(self):
		self._linear = False
		self._input = None
		self._output = None
		self._dinput = None

	def forward(self,X):
		"""
		array(n_samples,n_input) -> array(n_samples,n_input)
		Computes the Sigmoid of X
		"""
		self._input = X
		self._output = (1 / (1 + np.exp(-X)))

	def backward(self, delta):
		"""
		array(n_samples,n_input) * array(n_input,1) -> array(n_input,1)
		Computes the gradient on the output of this layer (which is the input of the next layer)
		"""
		self._dinput = delta * (1 - self._output) * self._output

	def predictions(self,output,class0=0,class1=1):
		"""
		array(n_samples) -> array(n_samples)
		Determines predicted classes using the output of the last module
		"""
		return np.where(output < 0.5,class0,class1)


class ReLU:
	def __init__(self):
		self._linear = False
		self._input = None
		self._output = None
		self._dinput = None

	def forward(self,X):
		"""
		array(n_samples,n_input) -> array(n_samples,n_input)
		Computes the ReLU of X
		"""
		self._input = X
		self._output = np.maximum(0.0,X)

	def backward(self, delta):
		"""
		array(n_samples,n_input) * array(n_input,1) -> array(n_input,1)
		Computes the gradient on the output of this layer (which is the input of the next layer)
		"""
		#return np.where(input <= 0, 0.0, 1.0)
		self._dinput = np.where(self._input <= 0, 0.0, 1.0) * delta

	def predictions(self,output):
		"""
		array(n_samples) -> array(n_samples)
		Determines predicted classes using the output of the last module
		"""
		return output


class Softmax:

	def __init__(self):
		self._linear = False
		self._input = None
		self._output = None
		self._dinput = None

	def forward(self,X):
		"""
		array(n_samples,n_input) -> array(n_samples,n_input)
		Computes the softmax of X
		"""
		self._input = X
		expX = np.exp(X-X.max(axis=1,keepdims=True))
		self._output = expX / (expX.sum(axis=1,keepdims=True))

	def backward(self,delta):
		"""
		array(n_samples,n_input) * array(n_input,1) -> array(n_input,1)
		Computes the gradient on the output of this layer (which is the input of the next layer)
		"""
		# softmax = self.forward(input)
		# return softmax * (1-softmax) * delta

		self._dinput = np.zeros_like(delta)

		for i,(single_output,single_delta) in enumerate(zip(self._output,delta)):  # for each sample
			single_output = single_output.reshape(-1,1) # flatten
			jacobian = np.diagflat(single_output) - np.dot(single_output,single_output.T)   # Jacobian matrix (dS_ij/dz_ik)
			self._dinput[i] = np.dot(jacobian,single_delta)

	def predictions(self,output):
		"""
		array(n_samples,n_classes) -> array(n_samples,1)
		Calculate predictions for probabilities in proba
		"""
		return np.argmax(output,axis=1)