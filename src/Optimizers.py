##### OPTIMIZERS

class Optim:
    def __init__(self,net,loss,eps=1e-3):
        self.net = net
        self.loss = loss
        self.eps = eps

    def step(self,batch_x,batch_y):
        # Forward pass
        z = self.net.forward_pass(batch_x)
        # Loss
        loss = self.loss.forward(batch_y,z)
        # Gradient of the loss
        g_loss = self.loss.backward(batch_y,z)
        # Backpropagation
        delta = self.net.backpropagation(batch_x,g_loss)
        # Update parameters
        self.net.update_parameters(learning_rate=self.eps)

        return z,loss

class Optimizer_SGD:
    def __init__(self,learning_rate=0.01):
        self._learning_rate = learning_rate

    

# def SGD(optim,X,y,batch,niter):
#     """
#     neural network (Optim) * array(batch,d) * int * int
#     Training net on X for niter epochs
#     """
#     loss_list = []

#     for i in range(niter):
#         loss = []

#         slices = gen_batches(X.shape[0],batch) # Create batches

#         for s in slices:
#             batch_x = X[s]
#             batch_y = y[s]

#             y_pred,l = optim.step(batch_x,batch_y)
#             loss.append(l.mean())

#         loss_list.append(np.array(loss).mean())
#     return loss_list