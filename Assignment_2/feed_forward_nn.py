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
        self.training_historyTest = []
       
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

        return Z1, A1


    def compute_cost(self, Y_pred, Y):
        return (Y_pred - Y)

    def cross_entropy(self, Y_pred, Y):
        cost = -np.sum(Y * np.log(Y_pred + 1e-10))

        return cost


    #Calculate the gradients 
    def model_backward(self, X, Y, X_test, Y_test):
        features, samples  = X.shape
        features_test, samples_test  = X.shape
        Z1, A1 = self.model_forward(X) 
        Z1test, A1test = self.model_forward(X_test)
        cost = self.cross_entropy(A1, Y)
        costTest = self.cross_entropy(A1test, Y_test)

        self.dZ1 =  (-Y + A1)          # 10 m

        self.dW1 = (1/samples) * np.dot(self.dZ1, X.T)  # 10, 784  

        self.dB1 = (1/samples) * np.reshape(np.sum(self.dZ1,1),(10,1)) # 10, 1

        self.training_history.append(np.mean(np.abs(cost))/samples)
        self.training_historyTest.append(np.mean(np.abs(costTest))/samples_test)

    #Update the weight and bias with the pre-calciulated gradients
    def update_parameters(self):
        self.weights1 -= self.learningRate * self.dW1
        self.bias1 -= self.learningRate * self.dB1

    #Forward run
    def predict(self, X):
        Z1, A1 = self.model_forward(X)
        return A1
    

    #Train the network
    def train_linear_model(self, X, Y, X_test, Y_test, iterations):
        for i in range(iterations):
            self.model_backward(X, Y, X_test, Y_test)
            self.update_parameters()


    def print_parameters(self):
        print(self.weights1.shape)
        print(self.bias1)
    
    def history(self):
       return self.training_history, self.training_historyTest
    
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


def create_mini_batches(data, labels, batch_size):

    # Calculate the number of batches
    num_batches = data.shape[0] // batch_size

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

def compare_max_indices(arr1, arr2):

    # Get the indices of the maximum values along axis 0 (rows)
    max_indices_arr1 = np.argmax(arr1, axis=0)
    max_indices_arr2 = np.argmax(arr2, axis=0)

    # Compare the indices and count the matches
    matches = np.sum(max_indices_arr1 == max_indices_arr2)

    # Calculate the percentage of correct matches
    percentage = (matches / arr1.shape[1]) * 100

    return percentage


# Shuffle the data and labels
shuffled_data, shuffled_labels = shuffle_data_and_labels(X_train, Y_train)
shuffled_testdata, shuffled_testlabels = shuffle_data_and_labels(X_test, Y_test)

# Create mini-batches
batch_size = 500
mini_batches = create_mini_batches(shuffled_data, shuffled_labels, batch_size)
mini_batches_test = create_mini_batches(shuffled_data, shuffled_labels, batch_size)

# TODO fix so equal number of batches to simplify training and test plot of history

# Create neural network
nn = NeuralNetwork(784, 1e-2, 1)

nn.initiliaze_parameters()

# Train the network for number of epochs
epochs = 300

for y in range(epochs):
    for x in range(len(mini_batches)):
        nn.train_linear_model(mini_batches[x][0].T, mini_batches[x][1].T, mini_batches_test[x][0].T, mini_batches_test[x][1].T,   1)

# plot the weights as images
nn.weights_as_image()

# Plot the training history
hist, hist_test = nn.history()
plt.plot(hist, label='training_loss')
plt.plot(hist_test, label='test_loss')
plt.legend()
plt.show()



arr1 = shuffled_testlabels.T
arr2 = nn.predict(shuffled_testdata.T)

# Calculate the percentage of correct matches
correct_percentage = compare_max_indices(arr1, arr2)

print("Percentage of correct matches:", correct_percentage)