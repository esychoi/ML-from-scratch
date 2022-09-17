##### LOSS CLASSES

import numpy as np
from src.Modules import Softmax


class Loss(object):

	def __init__(self):
		self._dinput = None

	def forward(self, y, yhat):
		pass

	def backward(self, y, yhat):
		pass


class MSELoss(Loss):

	def __init__(self):
		super().__init__()

	def forward(self,y,yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		Returns the MSE loss
		"""
		y = y.reshape(-1,1)
		yhat = yhat.reshape(-1,1)
		return np.power((y-yhat),2).sum(axis=1)

	def backward(self,y,yhat):
		"""
		array(n_samples,dim) * array(n_samples,dim) -> array(n_samples,1)
		Returns the gradient of the MSE loss
		"""
		y = y.reshape(-1,1)
		yhat = yhat.reshape(-1,1)
		n_samples,dim = y.shape
		self._dinput = ((2*(yhat-y))/dim)


class CELoss(Loss):

	def __init__(self):
		super().__init__()

	def forward(self, y, yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		args:
			y: sparse ground truth
			yhat: output of the last module
		Computes the categorical cross entropy
		"""
		# c = np.argmax(y,axis=1) # reverse one hot encoding of y
		# return np.log(np.exp(yhat).sum(axis=1)) - np.choose(c,yhat.T)

		# Clip both sides to prevent division by 0 and not drag mean towards any value
		yhat_clipped = np.clip(yhat,1e-7,1-1e-7)

		confidences = yhat_clipped[range(yhat.shape[0]),y]

		# if y is one hot encoded
		# if len(y.shape) == 2:
		# 	confidences = np.sum(yhat_clipped*y,axis=1)

		return -np.log(confidences)

	def backward(self, y, yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		args:
			y: sparse ground truth
			yhat: output of the last module
		Computes and normalizes the gradient of the cross entropy
		"""
		n_samples,n_classes = yhat.shape

		y_true = np.eye(n_classes)[y]
		self._dinput = -y_true/yhat
		self._dinput = self._dinput


class BCELoss(Loss):

	def __init__(self):
		super().__init__()

	def forward(self,y,yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		args:
			y: ground truth (sparse (1 binary class) or multi-label (several binary classes))
			yhat: output of the last activation module
		Computes the categorical cross entropy wrt the softmax on yhat
		"""
		if len(y.shape) == 1:
			y_true = y.reshape(-1,1)
		else:
			y_true = y
		yhat_clipped = np.clip(yhat,1e-7,1-1e-7)
		return -(y_true * np.log(yhat_clipped) + (1-y_true) * np.log(1-yhat_clipped))

	def backward(self,y,yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		args:
			y: ground truth (sparse (1 binary class) or multi-label (several binary classes))
			yhat: output of the last activation module
		Computes the gradient of the cross entropy
		"""
		if len(y.shape) == 1:
			y_true = y.reshape(-1,1)
		else:
			y_true = y
		n_classes = yhat.shape[1]
		yhat_clipped = np.clip(yhat,1e-7,1-1e-7)
		self._dinput = -((y_true/yhat_clipped) - ((1-y_true)/(1-yhat_clipped)))/n_classes # gradient

class Softmax_CELoss(Loss): # CELoss with softmax input
	"""
	More efficient to use directly this loss without the final Softmax layer
	because the final formula is much simpler and the backward function in the Softmax class even contains a loop
	"""

	def __init__(self,n_classes=2):
		super().__init__()
		self.activation = Softmax()
		self.loss = CELoss()
		self._input = None
		self._output = None
		self._dinput = None
		self._loss_value = None

	def forward(self,y,X):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		args:
			y: sparse ground truth
			X: input of the softmax layer = output of the last linear unit
		Computes the categorical cross entropy wrt the softmax on X
		"""
		self._input = X
		self.activation.forward(X)
		self._output = self.activation._output
		self._loss_value = self.loss.forward(y,self._output)

	def backward(self, y, yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
		args:
			y: sparse ground truth
			yhat: output of the softmax module
		Computes and normalizes the gradient of the categorical cross entropy wrt yhat
		"""
		n_samples = y.shape[0]
		self._dinput = yhat.copy()
		self._dinput[range(n_samples),y] -= 1
		self._dinput = self._dinput