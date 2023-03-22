import numpy as np

class NeuralNetwork:
    # Create a neural network with #hidden layers and #neurons in each layer
    def __init__(self, features, *args):
        self.features = features
        self.args = args
        self.weights = None
        self.bias = None
        dW = None
        dB = None
        
       
    # Create arrays with random parameters 
    def initiliaze_parameters(self):
        self.weights = np.zeros((self.features,1))
        self.bias = 0

    
    # Run model forward with the input x 
    def model_forward(self, X): 
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def compute_cost(y, y_true):
        return ((y_true - y) ** 2)



    def model_backward(self, X, Y):
        samples, features = X.shape
        Y_pred = self.model_forward(X)
        self.dW = (1/samples) * np.dot(X.T, Y_pred - Y) 
        self.dB = (1/samples) * np.sum(Y_pred - Y)


    def update_parameters(self):
        alpha = 0.1
        self.weights -= alpha * self.dW
        self.bias -= alpha * self.dB


    def predict():
        pass

    def train_linear_model():
        pass

    def print_parameters(self):
        print(self.weights)
        print(self.bias)

X = np.transpose(np.array([[1, 2, 3]]))
Y = 3 * X + 1
print(Y)

nn = NeuralNetwork(1,1)
nn.initiliaze_parameters()
for x in range(1000):
    nn.model_backward(X, Y)
    nn.update_parameters()

nn.print_parameters()