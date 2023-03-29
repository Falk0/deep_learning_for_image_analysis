import numpy as np
import pandas as pd
import load_mnist as lm
import matplotlib.pyplot as plt

class NeuralNetwork:
    # Create a neural network with #hidden layers and #neurons in each layer
    def __init__(self, features, learningRate, *args):
        self.features = features
        self.args = args
        self.learningRate = learningRate
        self.weights = None
        self.bias = None
        self.dW = None
        self.dB = None
        self.training_history = []
       
    # Create arrays with random parameters 
    def initiliaze_parameters(self):
        self.weights = np.zeros((self.features,1))
        self.bias = 0

    
    # Run model forward with the input x 
    def model_forward(self, X): 
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


    def compute_cost(self, Y_pred, Y):
        return (Y_pred - Y)


    #Calculate the gradients 
    def model_backward(self, X, Y):
        samples, features = X.shape
        Y_pred = self.model_forward(X)
        cost = self.compute_cost(Y_pred, Y)
        
        self.dW = (2/samples) * np.dot(X.T, cost) 
        self.dB = (2/samples) * np.sum(cost)
        self.training_history.append(np.mean(np.abs(cost)))

    #Update the weight and bias with the pre-calculated gradients
    def update_parameters(self):
        self.weights -= self.learningRate * self.dW
        self.bias -= self.learningRate * self.dB

    #Forward run
    def predict(self, X):
        return self.model_forward(X)

    #Train the network
    def train_linear_model(self, X, Y, iterations):
        for i in range(iterations):
            self.model_backward(X, Y)
            self.update_parameters()


    def print_parameters(self):
        print(self.weights)
        print(self.bias)
    
    def history(self):
       return self.training_history
        

X_train, Y_train, X_test, Y_test = lm.load_mnist()

