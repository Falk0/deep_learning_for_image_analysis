import numpy as np

class NeuralNetwork:
    # Create a neural network with #hidden layers and #neurons in each layer
    def __init__(self, hiddenLayers, *args):
        self.hiddenLayers = hiddenLayers
        self.args = args
        self.weights = []
        self.bias = []
        
       
    # Create arrays with random parameters 
    def initiliaze_parameters(self):
        lst = [1]
        for arg in self.args:
            lst.append(arg)

        for i in range(1,len(lst),1):
            self.weights.append(np.random.rand(lst[i-1],lst[i]))
            self.bias.append(np.random.rand(1,lst[i]))

    
    def model_forward():
        pass    

    def compute_cost():
        pass

    def model_backward():
        pass

    def update_parameters():
        pass

    def predict():
        pass

    def train_linear_model():
        pass

    def print_parameters(self):
        print(self.hiddenLayers)
        print(self.args)
        print(self.weights)
        print(self.bias)

nn = NeuralNetwork(4,10,11)
nn.initiliaze_parameters()
nn.print_parameters()

