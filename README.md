# ImageDenoiser
Denoising images using autoencoders coded from scratch.

## Introduction

An autoencoder is a type of neural network mainly used to learn an encoding of the input data, usually reducing dimensions. It is an unsupervised learning technique.

Autoencoders consist of 2 components :
- an encoder that compresses the input to obtain its *code*
- a decoder that reconstruct the input from the code.

They can be used for several tasks : inpainting, clustering, visualisation, denoising images...
We will focus on the latter application.

The objective of this project is to train an autoencoder to denoise images. We will use the MNIST dataset.
We will first implement a model using Keras to see the feasibility of the project. We will then try to do the same thing but from scratch, in order to focus on the maths that is involved in the process.

For more details, please refer to the report associated with this project.

## Getting started

This project was built using Pyhon 3.9.

Run the following command to get the required Python libraries :
`pip install -r requirements.txt`

The Python files contains a neural network model coded from scratch and the notebook contains the same model implemented with Keras.

