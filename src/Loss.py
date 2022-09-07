##### LOSS CLASSES

import numpy as np



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
        self._dinput = ((2*(yhat-y))/dim)/n_samples


class CELoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, y, yhat):
        """
        array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
        args:
            y: one hot encoded vectors (ground truth)
            yhat: output of the last module
        Computes the categorical cross entropy
        """
        # c = np.argmax(y,axis=1) # reverse one hot encoding of y
        # return np.log(np.exp(yhat).sum(axis=1)) - np.choose(c,yhat.T)

        # Clip both sides to prevent division by 0 and not drag mean towards any value
        yhat_clipped = np.clip(yhat,1e-7,1-1e-7)

        confidences = np.sum(yhat_clipped*y,axis=1)
        return -np.log(confidences) # or return mean

    def backward(self, y, yhat):
        """
        array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
        args:
            y: one hot encoded vectors (ground truth)
            yhat: output of the last module
        Computes and normalizes the gradient of the cross entropy
        """
        # softmax = np.exp(yhat) / (np.exp(yhat).sum(axis=1).reshape(-1,1)+1e-8)
        # return softmax - y
        n_samples = y.shape[0]
        self._dinput = (-y/yhat)/n_samples


class BCELoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self,y,yhat):
        """
        array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
        args:
            y: one hot encoded vectors (ground truth)
            yhat: output of the last module
        Computes the categorical cross entropy wrt the softmax on yhat
        """
        #return -(y*np.log(yhat) + (1-y)*np.log(1-yhat))
        yhat_clipped = np.clip(yhat,1e-7,1-1e-7)
        return -(y * np.log(yhat_clipped) + (1-y) * np.log(1-yhat_clipped))

    def backward(self,y,yhat):
        """
        array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
        args:
            y: one hot encoded vectors (ground truth)
            yhat: output of the last module
        Computes and normalizes the gradient of the cross entropy
        """
        n_samples,n_classes = y.shape
        yhat_clipped = np.clip(yhat,1e-7,1-1e-7)
        self._dinput = -((y/yhat_clipped) - ((1-y)/(1-yhat_clipped)))/n_classes # gradient
        self._dinput = self._dinput/n_samples   #normalization

class Softmax_CELoss(Loss): # CELoss with softmax input
    """
    More efficient to use directly this loss without the final Softmax layer
    because the final formula is much simpler and the backward function in the Softmax class even contains a loop
    """

    def __init__(self):
        super().__init__()
        self._softmax_outputs = None

    def forward(self,y,X):
        """
        array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
        args:
            y: one hot encoded vectors (ground truth)
            X: input of the softmax layer = output of the last linear unit
        Computes the categorical cross entropy wrt the softmax on X
        """
        softmax =  np.exp(X-X.max(axis=1,keepdims=True))
        softmax =  softmax / (softmax.sum(axis=1,keepdims=True)+1e-8)
        self._softmax_outputs = softmax

        softmax_clipped = np.clip(X,1e-7,1-1e-7)
        confidences = np.sum(softmax_clipped*y,axis=1)
        return -np.log(confidences) # or return mean

    def backward(self, y, yhat):
        """
        array(n_samples,n_classes) * array(n_samples,n_classes) -> array(n_samples,1)
        args:
            y: one hot encoded vectors (ground truth)
            yhat: output of the softmax module
        Computes and normalizes the gradient of the categorical cross entropy wrt yhat
        """
        n_samples = y.shape[0]
        y_true = np.argmax(y,axis=1)    # turn the one hot encoded y into discrete values
        self._dinput = yhat.copy()
        self._dinput[range(n_samples),y_true] -= 1
        self._dinput = self._dinput/n_samples