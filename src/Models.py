import numpy as np
from src.Modules import *
from src.Loss import *
from sklearn.utils import gen_batches
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

##### NEURAL NETWORKS

class Sequential:
	def __init__(self,layers,loss=None,optimizer=None,accuracy=None):
		self.layers = layers    	#list of layers
		self.loss = loss    		# Loss object
		self.optimizer = optimizer  # Optimizer object
		self.accuracy = accuracy	# "classification" or "regression"

	def add_module(self,M):
		self.layers.append(M)

	def set(self,loss=None,optimizer=None):
		if loss is not None:
			self.loss = loss
		if optimizer is not None:
			self.optimizer = optimizer

	def forward_pass(self,X,y=None):
		"""
		array(n_samples,n_input) -> array(n_samples,n_output)
		args:
			X: input data
		Performs a forward pass and returns the final result
		"""
		H = len(self.layers) #nb of layers

		self.layers[0].forward(X)   # calculate output of the first module

		for h in range(1,H):    # going through every layer
			self.layers[h].forward(self.layers[h-1]._output)

		if isinstance(self.loss,Softmax_CELoss):
			self.loss.forward(y,self.layers[-1]._output)
			return self.loss._output
		return self.layers[-1]._output

	def backward_pass(self,y,yhat):
		"""
		array(n_samples,n_classes) * array(n_samples,n_classes) -> array()
		args:
			y : one hot encoded ground truth vectors
			yhat : output of the last module
		Performs a backward pass
		"""
		H = len(self.layers) # nb of layers

		self.loss.backward(y,yhat)  # calculate dinput of loss
		self.layers[-1].backward(self.loss._dinput) # calculate dinput of the last module

		for h in range(H-2,0,-1):   # going through every layer in reversed order
			self.layers[h].backward(self.layers[h+1]._dinput)
			
	def predict(self,data,accuracy):
		"""
		(array(n_samples,n_classes) * array(n_samples,n_classes)) * str -> array(n_samples,n_output) * array(n_samples,1) * array(n_sample)
		args:
			data: couple (X,y) where X in the input data and y the corresponding labels (sparse or multi-label)
			accuracy: "classification" or "regression"
		Returns the final output of the model, and the corresponding loss and accuracy
		"""
		X,y = data

		if isinstance(self.loss,Softmax_CELoss):
			output = self.forward_pass(X,y)
			if accuracy == "classification":
				y_pred = self.loss.activation.predictions(output)
				acc = accuracy_score(y,y_pred)
			if accuracy == "regression":
				acc = mean_absolute_error(y,output)
			return output, self.loss._loss_value, acc

		output = self.forward_pass(X)
		l = self.loss.forward(y,output)
		if accuracy == "classification":
			y_pred = self.layers[-1].predictions(output)
			acc = accuracy_score(y,y_pred)
		if accuracy == "regression":
			acc = mean_absolute_error(y,output)
		return output, l, acc

	def fit(self,X,y,epochs=1,batch_size=100,verbose=True,val_data=None):
		"""
		array(n_samples,n_input) * array(n_samples,n_classes) * int * int * array(n_val,n_input)
		args:
			X: training data
			y: one hot encoded ground truth vectors
			epochs: number of epochs
			batch_size: number of samples of data per gradient update
			verbose: True = print summary at every step, False = no print
			val_data: validation data (val_X,val_y)

		Trains the model with the settings given by the arguments.

		returns:
			train_losses: array(epochs,steps) containing all training loss values
			train_accuracy: array(epochs,steps) containing all training accuracy values
			val_losses: array(epochs) containing all validation loss values
			val_accuracy: array(epochs) containing all validation accuracy values
		"""
		slices = list(gen_batches(X.shape[0],batch_size))   # create slices of indices for batches
		steps = len(slices) # number of training steps during each epoch
		train_losses = np.zeros((epochs,steps))     # array containing all training loss values
		train_accuracy = np.zeros((epochs,steps))   # array containing all training accuracy values
		val_losses = np.zeros(epochs)   # array containing all validation loss values
		val_accuracy = np.zeros(epochs) # array containing all validation accuracy values

		#y_true = np.argmax(y,axis=1)    # discrete version of y

		for epoch in range(epochs):
			if verbose:
				print("\n\n")
				print("-"*40)
				print("EPOCH {}".format(epoch))
				print("-"*40)
			
			step = 0
			for s in slices:
				batch_X = X[s]
				batch_y = y[s]

				# Forward pass
				if isinstance(self.loss,Softmax_CELoss):
					output = self.forward_pass(batch_X,batch_y)

					# Calculate loss and accuracy
					train_losses[epoch,step] = self.loss._loss_value.mean()
					if self.accuracy == "classification":
						y_pred = self.loss.activation.predictions(output)
						train_accuracy[epoch,step] = accuracy_score(batch_y,y_pred)
					if self.accuracy == "regression":
						train_accuracy[epoch,step] = mean_absolute_error(batch_y,output)
				else:
					output = self.forward_pass(batch_X)

					# Calculate loss and accuracy
					train_losses[epoch,step] = self.loss.forward(batch_y,output).mean()
					if self.accuracy == "classification":
						y_pred = self.layers[-1].predictions(output)
						train_accuracy[epoch,step] = accuracy_score(batch_y,y_pred)
					if self.accuracy == "regression":
						train_accuracy[epoch,step] = mean_absolute_error(batch_y,output)
					
				# Backward pass
				self.backward_pass(batch_y,output)

				# Update parameters using the optimizer
				for layer in self.layers:
					self.optimizer.update_parameters(layer)

				if verbose: # print step summary
					print("Step {} of epoch {}".format(step,epoch))
					print("\tLoss :",train_losses[epoch,step])
					print("\tAccuracy :",train_accuracy[epoch,step])

				step += 1 

			if verbose: # print epoch summary
				print("-"*20)
				print("Summary of epoch {}".format(epoch))
				print("\tTraining loss :",train_losses[epoch].mean())
				print("\tTraining accuracy :",train_accuracy[epoch].mean())

			if val_data is not None:
				_, val_loss, val_acc = self.predict(val_data,self.accuracy)
				val_losses[epoch] = val_loss.mean()
				val_accuracy[epoch] = val_acc.mean()

				if verbose:
					print("\tValidation loss :",val_loss.mean())
					print("\tValidation accuracy :",val_acc.mean())

		print("Last training loss :",train_losses[-1].mean())
		print("Last training accuracy :",train_accuracy[-1].mean())
		if val_data is not None:
			print("Last validation loss :",val_losses[-1].mean())
			print("Last validation accuracy :",val_accuracy[-1].mean())

		return train_losses,train_accuracy,val_losses,val_accuracy



##### AUTOENCODER

class AutoEncoder:
	def __init__(self,encoder,decoder,loss=None,optimizer=None,accuracy=None):
		self.encoder = encoder # Sequential
		self.decoder = decoder # Sequential
		self.autoencoder = Sequential(encoder.layers+decoder.layers,loss=loss,optimizer=optimizer,accuracy=accuracy)

	def set(self,loss=None,optimizer=None,accuracy=None):
		if loss is not None:
			self.autoencoder.loss = loss
		if optimizer is not None:
			self.autoencoder.optimizer = optimizer
		if accuracy is not None:
			self.autoencoder.accuracy = accuracy
		
	def encode(self,X):
		return self.encoder.forward_pass(X)

	def decode(self,X):
		return self.decoder.forward_pass(X)

	def predict(self,X):
		encoded = self.encoder.forward_pass(X)
		return self.decoder.forward_pass(encoded)

	def fit(self,X,y,epochs=1,batch_size=10,verbose=True,val_data=None):
		return self.autoencoder.fit(X,y,epochs,batch_size,verbose,val_data)