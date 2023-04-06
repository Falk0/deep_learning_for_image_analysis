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
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.dZ1 = None
        self.dW1 = None
        self.dB1 = None
        self.dZ2 = None
        self.dW2 = None
        self.dB2 = None
        self.layers = []
        self.training_history = []
        self.training_history_test = []
       

    # Create arrays with random parameters 
    def initiliaze_parameters(self):
       # np.random.seed(42)
        self.weights1 = np.random.rand(100, 784) * np.sqrt(1 / 784)
        self.bias1 = np.random.rand(100, 1)
        self.weights2 = np.random.rand(10, 100) * np.sqrt(1 / 100)
        self.bias2 = np.random.rand(10, 1)



    def softmax(self, Z):
        Z_max = np.max(Z, axis=0)
        Z_shifted = Z - Z_max
        A = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=0)
        return A


    def sigmoid(self, Z):
        A = np.exp(Z)/(1 + np.exp(Z))
        return A

    def relu(self, Z):
        return np.maximum(Z,0)

    def relu_grad(self, Z):
        return Z > 0

    # Run model forward with the input x 
    def model_forward(self, X): 
        Z1 = np.dot(self.weights1, X) + self.bias1  # 10, m
        A1 = self.relu(Z1)                   # 10, m
        Z2 = np.dot(self.weights2, A1) + self.bias2
        A2 = self.softmax(Z2)
 
        return Z1, A1, Z2, A2


    def compute_cost(self, Y_pred, Y):
        return (Y_pred - Y)


    def cross_entropy(self, Y_pred, Y):
        cost = -np.sum(Y * np.log(Y_pred + 1e-10))
        return cost


    #Calculate the gradients 
    def model_backward(self, X, Y, X_test, Y_test):
        features, samples  = X.shape
        features_test, samples_test  = X_test.shape
        
        # Calculate cost and save history
        #Z1_test, A1_test = self.model_forward(X_test)
        #cost_test = self.cross_entropy(A1_test, Y_test)/samples_test
        #self.training_history_test.append(cost_test)
        
        Z1_train, A1_train, Z2_train, A2_train = self.model_forward(X)
        cost_train = self.cross_entropy(A2_train, Y)/samples
        #self.training_history.append(cost_train)
        # Calculate gradients
        self.dZ2 = (-Y + A2_train)        
        self.dW2 = (1/samples) * np.dot(self.dZ2, A1_train.T) 
        self.dB2 = (1/samples) * np.reshape(np.sum(self.dZ2,1),(10,1)) 

        self.dZ1 = np.dot(self.weights2.T,self.dZ2) * self.relu_grad(Z1_train)          
        self.dW1 = (1/samples) * np.dot(self.dZ1, X.T)  
        self.dB1 = (1/samples) * np.reshape(np.sum(self.dZ1,1),(100,1))

    def create_layers(self, nodes, activation):
        weights = np.random.rand(nodes[1], nodes[0]) * np.sqrt(1 / 784) 
        bias = np.random.rand(nodes[1], 1)
        self.layers.append([weights, bias, activation])
        self.variables.append([None, None, None])

    def linear_backward(self):
        pass

    def relu_backward(self):
        pass

    def sigmoid_backward(self):
        pass

    def update_parameters_new(self):
        pass


    #Update the weight and bias with the pre-calciulated gradients
    def update_parameters(self):
        self.weights1 -= self.learningRate * self.dW1
        self.bias1 -= self.learningRate * self.dB1
        self.weights2 -= self.learningRate * self.dW2
        self.bias2 -= self.learningRate * self.dB2

    
    def compare(self, arr1, arr2):
        # Get the indices of the maximum values along axis 0 (rows)
        max_indices_arr1 = np.argmax(arr1, axis=0)
        max_indices_arr2 = np.argmax(arr2, axis=0)

        # Compare the indices and count the matches
        matches = np.sum(max_indices_arr1 == max_indices_arr2)

        # Calculate the percentage of correct matches
        percentage = (matches / arr1.shape[1]) * 100

        return percentage


       
     #Predict and calculate percentage correct
    def predict(self, X, Y):
        Z1, A1, Z2, A2 = self.model_forward(X)
        percent = self.compare(A2, Y)
        return percent
    
    #Train the network
    def train_linear_model(self, X, Y, X_test, Y_test, iterations):
        self.model_backward(X, Y, X_test, Y_test)
        self.update_parameters()


    def print_parameters(self):
        print(self.weights1.shape)
        print(self.bias1.shape)
        print(self.weights2.shape)
        print(self.bias2.shape)
    
    def history(self):
       return self.training_history, self.training_history_test
    
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


def shuffle_data_and_labels(data, labels):
    # Get the number of samples
    n_samples = data.shape[0]
    indices = np.arange(n_samples)

    # Shuffle the indices
    np.random.shuffle(indices)

    # Rearrange the data and labels using the shuffled indices
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    return shuffled_data, shuffled_labels


def create_mini_batches(data, labels, num_batches):

    # Calculate the number of batches
    batch_size = data.shape[0] // num_batches



    # Create mini-batches
    mini_batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size

        data_batch = data[start_index:end_index]
        labels_batch = labels[start_index:end_index]

        mini_batches.append((data_batch, labels_batch))

    # If there are remaining samples, add them as an additional batch
    if data.shape[0] % batch_size != 0:
        start_index = num_batches * batch_size
        data_batch = data[start_index:]
        labels_batch = labels[start_index:]

        mini_batches.append((data_batch, labels_batch))

    return mini_batches




# Load training and test data
X_train, Y_train, X_test, Y_test = lm.load_mnist()


# Shuffle the data and labels
shuffled_data, shuffled_labels = shuffle_data_and_labels(X_train, Y_train)
shuffled_testdata, shuffled_testlabels = shuffle_data_and_labels(X_test, Y_test)

# Create mini-batches
number_of_batches = 100
mini_batches = create_mini_batches(shuffled_data, shuffled_labels, number_of_batches)
mini_batches_test = create_mini_batches(shuffled_testdata, shuffled_testlabels, number_of_batches)



# Create neural network
nn = NeuralNetwork(784, 1e-2, 1)
nn.initiliaze_parameters()


epochs = 100
history = []
history_test = []

# Train the network for number of epochs
for y in range(epochs):
    print(str(y) + ' out of ' + str(epochs) + ' epochs')
    for x in range(len(mini_batches)):
        nn.train_linear_model(mini_batches[x][0].T, mini_batches[x][1].T, mini_batches_test[x][0].T, mini_batches_test[x][1].T,   1)
        #hist, hist_test = nn.history()
        #history.append(np.mean(hist))
        #history_test.append(np.mean(hist_test))


# plot the weights as images
nn.weights_as_image()

#hist, hist_test = nn.history()

# Plot the training history
#plt.plot(history, label='training_loss')
#plt.plot(history_test, label='test_loss')
#plt.legend()
#plt.show()

print(nn.predict(shuffled_testdata.T, shuffled_testlabels.T))