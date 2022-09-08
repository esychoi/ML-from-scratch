##### OPTIMIZERS

# class Optim:
#     def __init__(self,net,loss,eps=1e-3):
#         self.net = net
#         self.loss = loss
#         self.eps = eps

#     def step(self,batch_x,batch_y):
#         pass

class Optimizer_SGD:
	def __init__(self, learning_rate=0.001):
		self._learning_rate = learning_rate
	
	def update_parameters(self,layer):
		if layer._linear:
			layer._weights -= self._learning_rate * layer._dweights
			layer._biases -= self._learning_rate * layer._dbiases
