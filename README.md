# OCR Machine Learning
This is a simple program that can recognize 26 English letters. The program can be trained itself to recognize different English letters by implementing Neural Network Model. 

## Working Process
The Neural Network Model is the most popular machine learning model in Artificial Intelligence industry, which it is the simulation of human's neural network, the web of interconnected neurons, which contains different weights and biases. 

Different neurons come up with layers to interconnect to form proper and functional neural network. There are three basic layers, an input layer, inner layers, and an output layer. At this point, all the data input into the the input layer, and use the forward methods to inner layers to process and calculate, then go into the output layer to get the final result.

The training process is very essential, which is known as the backpropagation to adjust the weights and biases in the inner layers.

## Demo
Here is the result of a successful demostration to indentify the distroted letter "A". The image is pained in MS Paint.

![](https://github.com/JP-25/OCR_MachineLearning/blob/master/demo_pic/A_demo.png)

## Sources
Training Data can be downloaded from https://github.com/JP-25/OCR_MachineLearning/blob/master/train.npy.

Testing Data can be downloaded from https://github.com/JP-25/OCR_MachineLearning/blob/master/test.npy .

Validating Data can be downloaded from https://github.com/JP-25/OCR_MachineLearning/blob/master/validate.npy .

final_project.py is the training program, the trained results will be saved in weights.npy and biases.npy in order to serve for the application.py to achiwve the goal of recognizing different English letters. 

The test_images contains images for testing, which are different from the testing images during training process.

## Paper
Here is the link of the paper of this machine learning project.
https://github.com/JP-25/OCR_MachineLearning/blob/master/machineLearning.pdf

## Feature Implementation
The preprocessing function in application.py can be improved to be able to deal with RGB images. Your constructive comments and suggestions are welcome!
