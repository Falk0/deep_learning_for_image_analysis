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
        self.loss= None
        self.weights1 = None
        self.bias1 = None
        self.dL = None
        self.bA1 = None
        self.dZ1 = None
        self.dW1 = None
        self.dB1 = None
        self.training_history = []
       
    # Create arrays with random parameters 
    def initiliaze_parameters(self):
       # np.random.seed(42)
        self.weights1 = np.random.rand(10, 784)
        self.bias1 = np.random.rand(10, 1)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def sigmoid(self, Z):
        A = np.exp(Z)/(1 + np.exp(Z))
        return A
    
    def relu(self, Z):
        A = np.maximum(Z,0)
        return A


    # Run model forward with the input x 
    def model_forward(self, X): 
        Z1 = np.dot(self.weights1, X) + self.bias1  # 10, m
        A1 = self.softmax(Z1)                   # 10, m
        #print('model forward')
        #print(Z1.shape)
        #print(A1.shape)
        return Z1, A1


    def compute_cost(self, Y_pred, Y):
        return (Y_pred - Y)

    def cross_entropy(self, Y_pred, Y):
        cost = -np.sum(Y * np.log(Y_pred + 1e-10))
        #print(cost)
        #print('cross entropy')
        #print(cost.shape)
        return cost


    #Calculate the gradients 
    def model_backward(self, X, Y):
        samples, features  = X.shape
        Z1, A1 = self.model_forward(X) 
        cost = self.cross_entropy(A1, Y)

        self.dZ1 =  (-Y + A1)          # 10 m

        self.dW1 = (1/samples) * np.dot(self.dZ1, X.T)  # 10, 784  

        self.dB1 = (1/samples) * np.reshape(np.sum(self.dZ1,1),(10,1)) # 10, 1

        self.training_history.append(np.mean(np.abs(cost)))

    #Update the weight and bias with the pre-calciulated gradients
    def update_parameters(self):
        self.weights1 -= self.learningRate * self.dW1
        self.bias1 -= self.learningRate * self.dB1

    #Forward run
    def predict(self, X):
        Z1, A1 = self.model_forward(X)
        return A1
    

    #Train the network
    def train_linear_model(self, X, Y, iterations):
        for i in range(iterations):
            self.model_backward(X, Y)
            self.update_parameters()


    def print_parameters(self):
        print(self.weights1.shape)
        print(self.bias1)
    
    def history(self):
       return self.training_history
    
    def weights_as_image(self):
        fig, ax = plt.subplots(2,5, dpi=200)   
        for x in range(5):
            im1 = self.weights1[x,:].reshape((28,28))
            im2 = self.weights1[5+x,:].reshape((28,28))
            ax[0,x].imshow(im1)
            ax[0,x].set_xticks([]) 
            ax[0,x].set_yticks([])
            ax[1,x].imshow(im2)
            ax[1,x].set_xticks([]) 
            ax[1,x].set_yticks([]) 
        plt.show()

X_train, Y_train, X_test, Y_test = lm.load_mnist()


assert X_train.shape[0] == Y_train.shape[0], "Mismatch in the number of samples between data and labels."

def shuffle_data_and_labels(data, labels):
    # Get the number of samples
    n_samples = data.shape[0]

    # Create an array of indices representing the samples
    indices = np.arange(n_samples)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Rearrange the data and labels using the shuffled indices
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    return shuffled_data, shuffled_labels

# Shuffle the data and labels
shuffled_data, shuffled_labels = shuffle_data_and_labels(X_train, Y_train)
shuffled__testdata, shuffled_testlabels = shuffle_data_and_labels(X_test, Y_test)


nn = NeuralNetwork(784, 1e-2, 1)
nn.initiliaze_parameters()
nn.train_linear_model(shuffled_data.T, shuffled_labels.T, 1000)
nn.weights_as_image()
plt.plot(nn.history())
plt.show()



arr1 = shuffled_testlabels.T
arr2 = nn.predict(shuffled__testdata.T)

def compare_max_indices(arr1, arr2):
    # Make sure both arrays have the same shape
    assert arr1.shape == arr2.shape, "Arrays must have the same shape."

    # Get the indices of the maximum values along axis 0 (rows)
    max_indices_arr1 = np.argmax(arr1, axis=0)
    max_indices_arr2 = np.argmax(arr2, axis=0)

    # Compare the indices and count the matches
    matches = np.sum(max_indices_arr1 == max_indices_arr2)

    # Calculate the percentage of correct matches
    percentage = (matches / arr1.shape[1]) * 100

    return percentage

# Calculate the percentage of correct matches
correct_percentage = compare_max_indices(arr1, arr2)

print("Percentage of correct matches:", correct_percentage)