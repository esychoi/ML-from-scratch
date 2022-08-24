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

## Requirements

## Usage

After downloading this repository, you can use one of the two following methods to run the code :
1. You can run the provided notebook that contains some details on the implementation.
2. You can run the main file by running the command `python3 main.py`.
