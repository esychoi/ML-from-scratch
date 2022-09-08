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

(coming soon)
