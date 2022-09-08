# ML-from-scratch
Implementing a neural network from scratch.

## Introduction
The objective of this work is to implement a **neural network from scratch** with no framework (only using Python and Numpy).

We will then use it to build an **autoencoder** model to perform image denoising on the MNIST dataset. 

An autoencoder is a type of neural network mainly used to learn an encoding of the input data, usually reducing dimensions. It is an unsupervised learning technique.

Autoencoders consist of 2 components :
- an encoder that compresses the input to obtain its *code*
- a decoder that reconstruct the input from the code.

They can be used for several tasks : inpainting, clustering, visualisation, denoising images...
We will focus on the latter application.

All the math involved in the process is detailed below

## Getting started

This project was built using Pyhon 3.9.

Run the following command to install dependencies :
```
$ pip install -r requirements.txt
```
The Python files contains a neural network model coded from scratch that you can run with the following command :
```
python main.py
```
The notebook contains the same model implemented with Keras.

## The math

To train a neural network, we start by doing a forward pass through all the layers to compute the final output. We then evaluate the loss whose derivative tells us how to change the parameters (weights and biases of linear units) by calculating all gradients of all layers going backwards (that's the backpropagation) using the chain rule.

Here, we will focus on the formulas of the different gradients involved in the process.

We will note $X$ the input of the considered layer, $Z$ the output and $\nabla$ the gradient of the next layer.

#### Linear unit
* $Z = XW+B$ where $W$ and $B$ are respectively the weights and the bias of the linear unit.
* $\frac{dZ}{dW} = X^T\nabla$
* $\frac{dZ}{dB} = sum_{columns}\nabla$
* $\frac{dZ}{dX} = \nabla W^T$

#### ReLU
* $Z = max(X,0)$
* $\frac{dZ}{dX} = \mathbb{1}_{[X>0]}$

#### TanH
* $Z = \tanh(X)$
* $\frac{dZ}{dX} = (1-\tanh(X)^2) * \nabla$ where $*$ denotes the element-wise multiplication

#### Sigmoid
* $Z = S(X) = \frac{1}{1+\exp(-X)}$
* $\frac{dZ}{dX} = \nabla * (1-S(X)) * S(X)$

#### Softmax
* $Z = (\frac{\exp(X)}{\sum_{cols} \exp(X))})_{columns}$
* $\frac{dZ}{dX} = \hat{Z} - Z$ where $\hat{Z} is the ouput of the softmax function when used with the cross entropy loss

