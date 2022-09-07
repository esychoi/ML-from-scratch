import numpy as np
import matplotlib.pyplot as plt
from src.utils import *
from src.Modules import *
from src.Loss import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from keras.datasets import mnist
from sklearn.datasets import make_moons

def one_hot(y,d):
    """
    array(n_samples,1) -> array(n_samples,n_classes)
    Transforms y into a one hot encoded vector
    """
    return np.eye(d)[y]

def prediction_mnist(yhat):
    """
    array(batch,nb class) -> array(batch,1)
    Return the final prediction (highest probability)
    """
    return np.argmax(yhat,axis=1)

def prediction_toy(z,activation='Sigmoide'):
    """
    activation = last layer : Sigmoide or TanH
    """
    if activation == 'Sigmoide':
        return np.where(z < 0.5,-1,1)
    elif activation == 'TanH':
        return np.where(z < 0,-1,1)



if __name__ == '__main__':

    # Create dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

    # plt.figure()
    # plt.scatter(X[:,0],X[:,1],c=y)
    # plt.show()

    y = one_hot(y,2)

    # Building model
    dense1 = Linear(2,4)
    activation1 = ReLU()
    dense2 = Linear(4,2)
    activation2 = Softmax()
    loss = BCELoss()
    optimizer=Optimizer_SGD(learning_rate=0.001)
    model = Sequential([dense1,activation1,dense2,activation2],loss=loss,optimizer=optimizer)

    # Training model
    train_losses,train_accuracy,val_losses,val_accuracy = model.fit(X,y,epochs=20,batch_size=10,verbose=False)

    # print(train_losses)
    # print(train_accuracy)

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
    plt.plot(train_losses.mean(axis=1))
    plt.title("Training loss")
    plt.xlabel("epochs")

    plt.figure()
    plt.plot(train_accuracy.mean(axis=1))
    plt.title("Training accuracy")
    plt.xlabel("epochs")

    plt.show()




    # MNIST DATASET
    digits = False
    if digits:
        niter = 10
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
        X_train,X_test = X_train.reshape(-1,784),X_test.reshape(-1,784)
        y_train_OH,y_test_OH = one_hot(y_train,10),one_hot(y_test,10)

        # #plt.figure()
        # #plt.bar(np.arange(10),np.histogram(y_train)[0])
        # #plt.bar(np.arange(10),np.histogram(y_test)[0])

        # # Balancing data
        # m = np.min(np.histogram(y_train)[0])
        # count = {i : 0 for i in range(10)}
        # X_train_balanced = []
        # y_train_balanced = []
        # for i in range(X_train.shape[0]):
        #     if count[y_train[i]] < m:
        #         X_train_balanced.append(X_train[i])
        #         y_train_balanced.append(y_train[i])
        #         count[y_train[i]] += 1
        # X_train_balanced = np.array(X_train_balanced)
        # y_train_balanced_OH = one_hot(np.array(y_train_balanced),10)

        # #plt.figure()
        # #plt.bar(np.arange(10),np.histogram(y_train_balanced)[0])

        # layer1 = Linear(784,512)
        # hiddenlayer1 = TanH()
        # layer2 = Linear(512,256)
        # hiddenlayer2 = TanH()
        # layer3 = Linear(256,10)
        # hiddenlayer3 = SoftMax()

        # nn = Sequentiel([layer1, hiddenlayer1, layer2, hiddenlayer2, layer3, hiddenlayer3])
        # L = CELoss()

        # loss_train = []
        # loss_test = []
        # for i in range(niter):
        #     y_pred = nn.forward_pass(X_test)
        #     loss_test.append(L.forward(y_test_OH,y_pred).mean())
        #     slices = list(gen_batches(X_train.shape[0],10)) # Create batches
        #     loss = 0.0
        #     for s in slices:
        #         batch_x = X_train[s]
        #         batch_y = y_train_OH[s]

        #         # Forward pass
        #         z = nn.forward_pass(batch_x)
        #         # Loss
        #         loss += L.forward(batch_y,z).mean()
        #         # Gradient of the loss
        #         g_loss = L.backward(batch_y,z)
        #         # Backpropagation
        #         delta = nn.backpropagation(batch_x,g_loss)
        #         # Update parameters
        #         nn.update_parameters(gradient_step=0.001)

        #     loss_train.append(loss/len(slices))

        # optim = Optim(nn,L)
        # loss_train = SGD(optim,X_train,y_train_OH,10,niter)

        # # Plot loss
        # plt.figure()
        # plt.plot(loss_train,label='Train')
        # #plt.plot(loss_test,label='Test')
        # plt.xlabel('Epochs')
        # plt.title('Evolution of the Loss (CE)')
        # plt.legend()

        # y_pred = prediction_mnist(nn.forward_pass(X_test))
        # cm = confusion_matrix(y_test, y_pred, labels=np.arange(10))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.arange(10))
        # disp.plot()
        # print("Taux de bonne classification :",100*cm.diagonal().sum()/X_test.shape[0],"%")


    autoenc = False
    if autoenc: # MNIST DATASET
        niter = 15
        (X_train,y_true),(X_test,y_test) = mnist.load_data()
        X_train,X_test = X_train.reshape(-1,784),X_test.reshape(-1,784)
        y_train = one_hot(y_true,10)

        # Normalisation to have values in [0,1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # Noising data
        noise_factor = 0.3
        X_train_noisy = np.clip(X_train + noise_factor * np.random.randn(X_train.shape[0],X_train.shape[1]),0.0,1.0)
        X_test_noisy = np.clip(X_test + noise_factor * np.random.randn(X_test.shape[0],X_test.shape[1]),0.0,1.0)

        # Auto Encoder
        layer1 = Linear(784,400)
        hiddenlayer1 = TanH()
        layer2 = Linear(256,256)
        hiddenlayer2 = TanH()
        layer3 = Linear(784,128)
        hiddenlayer3 = TanH()

        #encoder = Sequentiel([layer1, hiddenlayer1, layer2, hiddenlayer2, layer3, hiddenlayer3])
        #encoder = Sequentiel([layer1, hiddenlayer1, layer3, hiddenlayer3])
        encoder = Sequential([layer3, hiddenlayer3])

        layer1bis = Linear(200,400)
        hiddenlayer1bis = TanH()
        layer2bis = Linear(256,512)
        hiddenlayer2bis = TanH()
        layer3bis = Linear(128,784)
        hiddenlayer3bis = Sigmoid()

        #decoder = Sequentiel([layer1bis, hiddenlayer1bis, layer2bis, hiddenlayer2bis, layer3bis, hiddenlayer3bis])
        #decoder = Sequentiel([layer1bis, hiddenlayer1bis, layer3bis, hiddenlayer3bis])
        decoder = Sequential([layer3bis, hiddenlayer3bis])

        L = BCELoss()
        autoEncoder = AutoEncoder(encoder,decoder,L)
        #optim = Optim(nn,L)

        loss_train = []
        loss_test = []
        for i in range(niter):
            loss_test.append(L.forward(X_train_noisy,autoEncoder.predict(X_train_noisy)).mean())
            slices = list(gen_batches(X_train.shape[0],30)) # Create batches
            loss = 0.0
            for s in slices:
                batch_x = X_train_noisy[s]
                batch_y = X_train[s]

                z,l = autoEncoder.autoenc.step(batch_x,batch_y)
                loss += l.mean()

            loss_train.append(loss/len(slices))

        #loss_train = SGD(optim,X_train,y_train_OH,10,niter)
        #loss_train = autoEncoder.fit(X_train_noisy,X_train,30,niter)

        # Plot loss
        plt.figure()
        plt.plot(loss_train,label='Train')
        plt.plot(loss_test,label='Test')
        plt.xlabel('Epochs')
        plt.title('Evolution of the Loss (BCE)')
        plt.legend()

        # Show a reconstructed image
        plot_digits(autoEncoder,X_test,X_test_noisy,prediction=True)



    plt.show()
