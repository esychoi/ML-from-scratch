import numpy as np
from src.Optimizers import Optimizer_SGD
from sklearn.utils import gen_batches
from sklearn.metrics import accuracy_score


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

    # def update_parameters(self, learning_rate=1e-3):
    #     """
    #     Updates parameters (weights and biases) wrt the gradients and the learning rate and resets gradients
    #     """
    #     self._weights -= learning_rate*self._dweights
    #     self._biases -= learning_rate*self._dbiases
    #     # self.zero_grad()

    # def backward_update_gradient(self, input, delta):
    #     """
    #     array(n_samples,n_input) * array(n_output,1)
    #     Updates the gradients on parameters
    #     """
    #     self._dweights += np.dot(input.T,delta)
    #     self._dbiases += delta.sum(axis=0,keepdims=True)


    # def backward_delta(self, input, delta):
    #     """
    #     array(n_samples,n_input) * array(n_output,1)
    #     Computes the gradient on values
    #     """
    #     # n_samples,n_input = input.shape #nb examples, input dimension
    #     # n_output = delta.shape[0] #output dimension
    #     #
    #     # W = self._parameters.reshape(n_input,n_output)
    #     self._dinput = np.dot(delta,self._weights.T)

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
        self._input = input
        self._output = np.tanh(X)

    def backward(self, input, delta):
        self._dinput = (1-np.tanh(input)**2) * delta

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
        #self._dinput = self._output * (1 - self._output) * delta
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
        self._output = expX / (expX.sum(axis=1,keepdims=True)+1e-8) # +1e-8 to avoid division by 0

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


        

##### NEURAL NETWORKS

class Sequential:
    def __init__(self,layers,loss=None,optimizer=None):
        self.layers = layers    #list of layers
        self.loss = loss    # Loss object
        self.optimizer = optimizer  # Optimizer object

    def add_module(self,M):
        self.layers.append(M)

    def set(self,loss=None,optimizer=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer

    def forward_pass(self,X):
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
            

    def evaluate(self,X):
        return 0,0

    def fit(self,X,y,epochs=1,batch_size=100,verbose=True,val_data=None):
        """
        array(n_samples,n_input) * array(n_samples,n_classes) * int * int * array(n_val,n_input)
        args:
            X: training data
            y: one hot encoded ground truth vectors
            epochs: number of epochs
            batch_size: number of samples of data per gradient update
            verbose: True = print summary at every step, False = no print
            val_data: validation data

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

        y_true = np.argmax(y,axis=1)    # discrete version of y

        for epoch in range(epochs):
            if verbose:
                print("\n\n")
                print("-"*40)
                print("EPOCH {}".format(epoch))
                print("-"*40)

            # Reset loss and accuracy
            
            step = 0
            for s in slices:
                batch_X = X[s]
                batch_y = y[s]

                # Forward pass
                output = self.forward_pass(batch_X)

                # Calculate loss and accuracy
                train_losses[epoch,step] = self.loss.forward(batch_y,output).mean()
                y_pred = self.layers[-1].predictions(output)
                train_accuracy[epoch,step] = accuracy_score(y_true[s],y_pred)

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

            if val_data:
                val_loss, val_acc = self.evaluate(val_data)
                val_losses[epoch] = val_loss
                val_accuracy[epoch] = val_acc

                if verbose:
                    print("\tValidation loss :",val_loss)
                    print("\tValidation accuracy :",val_acc)

        return train_losses,train_accuracy,val_losses,val_accuracy



##### AUTOENCODER

class AutoEncoder:
    def __init__(self,encoder,decoder,loss,eps=1e-3):
        self.encoder = encoder #Sequential
        self.decoder = decoder #Sequential
        self.loss = loss
        self.eps = eps
        #self.autoenc = Optim(Sequential(self.encoder.layers + self.decoder.layers),self.loss,self.eps)

    def encode(self,X):
        return self.encoder.forward_pass(X)

    def decode(self,X):
        return self.decoder.forward_pass(X)

    def predict(self,X):
        encoded = self.encoder.forward_pass(X)
        return self.decoder.forward_pass(encoded)

    def fit(self,X,y,batch,niter):
        loss_train = Optimizer_SGD(self.autoenc,X,y,batch,niter)
        return loss_train
