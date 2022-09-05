##### LOSS CLASSES

from turtle import forward
import numpy as np



class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class MSELoss(Loss):
    def forward(self,y,yhat):
        """
        array(batch,d) * array(batch,d) -> array(batch,1)
        return the MSE loss
        """
        y = y.reshape(-1,1)
        yhat = yhat.reshape(-1,1)
        return np.power((y-yhat),2).sum(axis=1)

    def backward(self,y,yhat):
        """
        array(batch,d) * array(batch,d) -> array(batch,1)
        return the gradient of the MSE loss
        """
        y = y.reshape(-1,1)
        yhat = yhat.reshape(-1,1)
        return 2*(yhat-y)


class CELoss(Loss):
    def forward(self, y, yhat):
        """
        array(n_samples,nb_class) * array(n_samples,nb_class) -> array(n_samples,1)
        y : one hot encoded vectors (ground truth)
        yhat : predicted classes = output of the last module
        Computes the categorical cross entropy
        """
        # c = np.argmax(y,axis=1) # reverse one hot encoding of y
        # return np.log(np.exp(yhat).sum(axis=1)) - np.choose(c,yhat.T)

        # Clip to prevent division by 0
        yhat_clipped = np.clip(yhat,1e-7,1-1e-7)

        confidences = np.sum(yhat_clipped*y,axis=1)
        return -np.log(confidences) # or return mean



    def backward(self, y, yhat):
        """
        array(n_samples,nb_class) * array(n_samples,nb_class) -> array(n_samples,1)
        y : one hot encoded vectors (ground truth)
        yhat : predicted classes = output of the last module
        Computes the gradient of the cross entropy
        """
        # softmax = np.exp(yhat) / (np.exp(yhat).sum(axis=1).reshape(-1,1)+1e-8)
        # return softmax - y
        return (-y/yhat)  # and normalize by divinding by n_samples ?


class BCELoss(Loss):
    def forward(self,y,yhat):
        #return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))
        return -(y*np.maximum(-100, np.log(yhat)) + (1-y)*np.maximum(-100, np.log(1-yhat)))

    def backward(self,y,yhat):
        return -(y/np.maximum(np.exp(-100),yhat) - (1-y)/np.maximum(np.exp(-100), 1-yhat))


class Softmax_CELoss(Loss): # CELoss with softmax input
    """
    More efficient to use directly this loss without the final Softmax layer
    because the final formula is much simpler and the backward function in the Softmax class even contains a loop
    """

    def forward(self,y,yhat):
        """
        array(n_samples,nb_class) * array(n_samples,nb_class) -> array(n_samples,1)
        y : one hot encoded vectors (ground truth)
        yhat : output of the last linear unit
        Computes the categorical cross entropy wrt the softmax on yhat
        """
        softmax =  np.exp(yhat-yhat.max(axis=1,keepdims=True))
        softmax =  softmax / (softmax.sum(axis=1,keepdims=True)+1e-8)
        self._softmax_outputs = softmax

        softmax_clipped = np.clip(yhat,1e-7,1-1e-7)
        confidences = np.sum(softmax_clipped*y,axis=1)
        return -np.log(confidences) # or return mean

    def backward(self, y, yhat):
        """
        array(n_samples,nb_class) * array(n_samples,nb_class) -> array(n_samples,1)
        y : one hot encoded vectors (ground truth)
        yhat : output of the last linear unit
        Computes the gradient of the categorical cross entropy wrt yhat
        """
        y_true = np.argmax(y,axis=1)    # turn the one hot encoded y into discrete values
        dinput = yhat.copy()
        dinput[range(y.shape[0]),y_true] -= 1
        return dinput # normalize ?