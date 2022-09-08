import numpy as np
import matplotlib.pyplot as plt
from src.utils import *
from src.Modules import *
from src.Loss import *
from src.Optimizers import *
from src.Models import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from keras.datasets import mnist
from sklearn.datasets import make_moons, make_blobs

def one_hot(y,d):
	"""
	array(n_samples,1) -> array(n_samples,n_classes)
	Transforms y into a one hot encoded vector
	"""
	return np.eye(d)[y]


if __name__ == '__main__':

	moons = False
	if moons :
		# Create dataset
		X, y = make_moons(n_samples=6000, noise=0.1, random_state=42)
		
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

		# plt.figure()
		# plt.scatter(X[:,0],X[:,1])
		# plt.show()

		# Building model 1
		dense1 = Linear(2,4)
		activation1 = TanH()
		dense2 = Linear(4,6)
		activation2 = TanH()
		dense3 = Linear(6,1)
		activation3 = Sigmoid()
		loss = BCELoss()
		optimizer=Optimizer_SGD(learning_rate=0.03)
		model = Sequential([dense1,activation1,dense2,activation2,dense3,activation3],loss=loss,optimizer=optimizer,accuracy="classification")

		# Training model 1
		train_losses,train_accuracy,val_losses,val_accuracy = model.fit(X_train,y_train,epochs=128,batch_size=32,verbose=False,val_data=(X_val,y_val))

		plt.figure()
		plot_frontiere(X_val,lambda x : model.layers[-1].predictions(model.forward_pass(x)),step=100)
		plot_data(X_val,y_val)
		plt.title("Decision boundary for validation points")

		# plt.figure()
		# plt.imshow(train_losses)
		# plt.title("Training loss heatmap")
		# plt.xlabel("steps")
		# plt.ylabel("epochs")
		# plt.colorbar()

		# plt.figure()
		# plt.imshow(train_accuracy)
		# plt.title("Training accuracy heatmap")
		# plt.xlabel("steps")
		# plt.ylabel("epochs")
		# plt.colorbar()

		plt.figure()
		plt.plot(train_losses.mean(axis=1),label="Train")
		plt.plot(val_losses,label="Validation")
		plt.legend()
		plt.title("Loss")
		plt.xlabel("epochs")

		plt.figure()
		plt.plot(train_accuracy.mean(axis=1),label="Train")
		plt.plot(val_accuracy,label="Validation")
		plt.legend()
		plt.title("Accuracy")
		plt.xlabel("epochs")

		plt.show()

	blobs = False
	if blobs :
		# Create dataset
		X, y = make_blobs(n_samples=2000, centers=3, cluster_std=3, random_state=42)
		
		X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

		# plt.figure()
		# plt.scatter(X[:,0],X[:,1],c=y)
		# plt.show()

		# Building model 2
		dense1 = Linear(2,4)
		activation1 = Sigmoid()
		dense2 = Linear(4,6)
		activation2 = Sigmoid()
		dense3 = Linear(6,3)
		activ_loss = Softmax_CELoss()
		optimizer=Optimizer_SGD(learning_rate=0.25)
		model = Sequential([dense1,activation1,dense2,activation2,dense3],loss=activ_loss,optimizer=optimizer,accuracy="classification")

		# Training model 2
		train_losses,train_accuracy,val_losses,val_accuracy = model.fit(X_train,y_train,epochs=64,batch_size=32,verbose=False,val_data=(X_val,y_val))

		plt.figure()
		plt.plot(train_losses.mean(axis=1),label="Train")
		plt.plot(val_losses,label="Validation")
		plt.legend()
		plt.title("Loss")
		plt.xlabel("epochs")

		plt.figure()
		plt.plot(train_accuracy.mean(axis=1),label="Train")
		plt.plot(val_accuracy,label="Validation")
		plt.legend()
		plt.title("Accuracy")
		plt.xlabel("epochs")

		plt.show()


	autoenc = True
	if autoenc: # MNIST DATASET
		(X_train,y_true),(X_test,y_test) = mnist.load_data()
		X_train,X_test = X_train.reshape(-1,784),X_test.reshape(-1,784)

		# Normalisation to have values in [0,1]
		X_train = X_train.astype('float32') / 255.0
		X_test = X_test.astype('float32') / 255.0

		# Noising data
		noise_factor = 0.3
		X_train_noisy = np.clip(X_train + noise_factor * np.random.randn(X_train.shape[0],X_train.shape[1]),0.0,1.0)
		X_test_noisy = np.clip(X_test + noise_factor * np.random.randn(X_test.shape[0],X_test.shape[1]),0.0,1.0)

		# Auto Encoder
		encdense = Linear(784,128)
		encactivation = ReLU()
		encoder = Sequential([encdense, encactivation])

		decdense = Linear(128,784)
		decactivation = Sigmoid()
		decoder = Sequential([decdense, decactivation])

		loss = BCELoss()
		optimizer = Optimizer_SGD(learning_rate=0.2)
		autoencoder = AutoEncoder(encoder,decoder,loss,optimizer=optimizer,accuracy="regression")
		
		train_losses,train_accuracy,val_losses,val_accuracy = autoencoder.fit(X_train_noisy,X_train,epochs=100,batch_size=128,verbose=False,val_data=(X_test_noisy,X_test))

		# Plot loss
		plt.figure()
		plt.plot(train_losses.mean(axis=1),label="Train")
		plt.plot(val_losses,label="Validation")
		plt.legend()
		plt.title("Loss")
		plt.xlabel("epochs")

		plt.figure()
		plt.plot(train_accuracy.mean(axis=1),label="Train")
		plt.plot(val_accuracy,label="Validation")
		plt.legend()
		plt.title("Accuracy")
		plt.xlabel("epochs")

		
		# Show a reconstructed image
		plot_digits(autoencoder,X_test,X_test_noisy,prediction=True)

		plt.show()

