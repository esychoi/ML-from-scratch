import numpy as np
from sklearn.utils import gen_batches
from sklearn.metrics import accuracy_score


##### MODULES

# class Module(object):
#     def __init__(self):
#         self._parameters = None
#         self._gradient = None

#     def zero_grad(self):
#         """
#         Reset gradients to zero
#         """
#         pass

#     def forward(self, X):
#         """
#         Forward pass
#         """
#         pass

#     def update_parameters(self, learning_rate=1e-3):
#         """
#         Updates parameters wrt to gradient and the learning rate
#         """
#         self._parameters -= learning_rate*self._gradient

#     def backward_update_gradient(self, input, delta):
#         """
#         Met a jour la valeur du gradient
#         """
#         pass

#     def backward_delta(self, input, delta):
#         """
#         Calcule la derivee de l'erreur
#         """
#         pass


class Linear:
    def __init__(self,n_input,n_output):
        """
        n_input = input dimension
        n_output = output dimension = nb of neurons of this layer
        """
        self._weights = 2 * (np.random.rand(n_input,n_output) - 0.5) #shape (n_input,n_output)
        self._biases = np.zeros((1,n_output)) #shape (1,n_output)
        self._dweights = np.zeros((n_input,n_output)) #shape (n_input,n_output)
        self._dbiases = np.zeros((1,n_output)) #shape (1,n_output)

    def forward(self, X):
        """
        array(n_sample,n_input) -> array(n_sample,n_output)
        Forward pass
        """
        return np.dot(X,self._weights) + self._biases

    def zero_grad(self):
        """
        Resets gradients to 0
        """
        self._dweights = np.zeros(self._dweights.shape)
        self._dbiases = np.zeros(self._dbiases.shape)

    def update_parameters(self, learning_rate=1e-3):
        """
        Updates parameters (weights and biases) wrt the gradients and the learning rate and resets gradients
        """
        self._weights -= learning_rate*self._dweights
        self._biases -= learning_rate*self._dbiases
        self.zero_grad()

    def backward_update_gradient(self, input, delta):
        """
        array(n_samples,n_input) * array(n_output,1)
        Updates the gradients on parameters
        """
        self._dweights += np.dot(input.T,delta)
        self._dbiases += delta.sum(axis=0,keepdims=True)


    def backward_delta(self, input, delta):
        """
        array(n_samples,n_input) * array(n_output,1)
        Computes the gradient on values
        """
        # n_samples,n_input = input.shape #nb examples, input dimension
        # n_output = delta.shape[0] #output dimension
        #
        # W = self._parameters.reshape(n_input,n_output)
        return np.dot(delta,self._weights.T)


class TanH:

    def forward(self, X):
        """
        X : array(n_samples,d)
        Calcule le passe forward
        """
        return np.tanh(X)

    def backward(self, input, delta):
        return (1-np.tanh(input)**2) * delta


class Sigmoide:

    def forward(self,X):
        return (1 / (1 + np.exp(-X)))

    def backward(self, input, delta):
        s = (1 / (1 + np.exp(-input)))
        return s * (1 - s) * delta


class ReLU:

    def forward(self,X):
        """
        array(n_samples,n_input) -> array(n_samples,n_input)
        Computes the ReLU of X
        """
        return np.maximum(0.0,X)

    def backward(self, input, delta):
        """
        array(n_samples,n_input) * array(n_input,1) -> array(n_input,1)
        Computes the gradient on the output of this layer (which is the input of the next layer)
        """
        #return np.where(input <= 0, 0.0, 1.0)
        return np.where(input <= 0, 0.0, 1.0) * delta


class Softmax:

    def forward(self,X):
        """
        array(n_samples,n_input) -> array(n_samples,n_input)
        Computes the softmax of X
        """
        expX = np.exp(X-X.max(axis=1,keepdims=True))
        return expX / (expX.sum(axis=1,keepdims=True)+1e-8) # +1e-8 to avoid division by 0

    def backward(self,output,delta):
        """
        array(n_samples,n_input) * array(n_input,1) -> array(n_input,1)
        Computes the gradient on the output of this layer (which is the input of the next layer)
        """
        # softmax = self.forward(input)
        # return softmax * (1-softmax) * delta

        dinput = np.zeros_like(delta)

        for i,(single_output,single_delta) in enumerate(zip(output,delta)):  # for each sample
            single_output = single_output.reshape(-1,1) # flatten
            jacobian = np.diagflat(single_output) - np.dot(single_output,single_output.T)   # Jacobian matrix (dS_ij/dz_ik)
            dinput[i] = np.dot(jacobian,single_delta)

        return dinput

        


##### NEURAL NETWORKS

class Sequentiel:
    def __init__(self,layers):
        self.layers = layers #list of layers
        self.inputs = [[]] * len(layers) #list of inputs of each module
        self.outputs = [[]] * len(layers) #list of outputs of each module

    def add_module(self,M):
            self.layers.append(M)

    def forward_pass(self,X):
        H = len(self.layers) #nb of layers
        z = X.copy()
        for h in range(H):
            self.inputs[h] = z.copy()
            z = self.layers[h].forward(z)
            self.outputs[h] = z.copy()
        return z

    def backpropagation(self,input,delta):
        H = len(self.layers) #nb of layers
        deltah = delta
        for h in range(H-1, 0, -1):
            self.layers[h].backward_update_gradient(self.inputs[h],deltah)
            deltah = self.layers[h].backward_delta(self.inputs[h],deltah)
        return deltah

    def update_parameters(self,learning_rate = 0.001):
        for h in range(len(self.layers)):
            self.layers[h].update_parameters(learning_rate)




##### AUTOENCODER

class AutoEncoder:
    def __init__(self,encoder,decoder,loss,eps=1e-3):
        self.encoder = encoder #Sequentiel
        self.decoder = decoder #Sequentiel
        self.loss = loss
        self.eps = eps
        self.autoenc = Optim(Sequentiel(self.encoder.layers + self.decoder.layers),self.loss,self.eps)

    def encode(self,X):
        return self.encoder.forward_pass(X)

    def decode(self,X):
        return self.decoder.forward_pass(X)

    def predict(self,X):
        encoded = self.encoder.forward_pass(X)
        return self.decoder.forward_pass(encoded)

    def fit(self,X,y,batch,niter):
        loss_train = SGD(self.autoenc,X,y,batch,niter)
        return loss_train
