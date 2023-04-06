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
        self.layersIO = []
        self.layerGradients = []

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

    def linar_forward(self, layer , X):
        #print(self.layers[layer][0].shape)
        #print(X.shape)
        Z = np.dot(self.layers[layer][0], X) + self.layers[layer][1]   
        self.layersIO[layer][0] = X #input
        self.layersIO[layer][1] = Z #dotproduct
        
    
    def activation_forward(self, layer):
        if self.layers[layer][2] == 'relu':
            A = self.relu(self.layersIO[layer][1]) #activation of dotproduct
            self.layersIO[layer][2] = A #save activated dotproduct array

        elif self.layers[layer][2] == 'sigmoid':
            A = self.sigmoid(self.layersIO[layer][1])
            self.layersIO[layer][2] = A
        
        elif self.layers[layer][2] == 'softmax':
            A = self.softmax(self.layersIO[layer][1])
            self.layersIO[layer][2] = A
         
   
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
        self.dW2 = (1/samples) * np.dot(self.dZ2, A1_train.T) #dZ2 * input
        self.dB2 = (1/samples) * np.reshape(np.sum(self.dZ2,1),(10,1)) 

        self.dZ1 = np.dot(self.weights2.T,self.dZ2) * self.relu_grad(Z1_train)  #W2 * dZ2 * dA1        
        self.dW1 = (1/samples) * np.dot(self.dZ1, X.T)  #dZ1 * Input
        self.dB1 = (1/samples) * np.reshape(np.sum(self.dZ1,1),(100,1)) #dZ1


    def create_layer(self, nodes, activation):
        weights = np.random.rand(nodes[1], nodes[0]) * np.sqrt(1 / 784) 
        bias = np.random.rand(nodes[1], 1)
        self.layers.append([weights, bias, activation]) #[weights, bias, activation function]
        self.layersIO.append([None, None, None]) # [input, Z, A]
        self.layerGradients.append([None, None, None, None]) #[dZ, dW, dB, dA]

    def activation_backward(self, layer, Y):
        if self.layers[layer][2] == 'relu':
            dZ = self.relu_backward(self.layersIO[layer][1])
            self.layerGradients[layer][0] = dZ

        elif self.layers[layer][2] == 'sigmoid':
           pass

        elif self.layers[layer][2] == 'softmax':
            dA = - Y + self.layersIO[layer][2]
            self.layerGradients[layer][3] = dA
            


        


    def linear_backward(self, layer):
        if self.layers[layer][2] == 'softmax':
            dZ = self.layerGradients[layer][3]
            dW = (1/600) * np.dot(dZ, self.layersIO[layer-1][1].T)
            dB = (1/600) * np.reshape(np.sum(dZ,1), (10,1))

            self.layerGradients[layer] = [dZ, dW, dB]

        else:
            dZ = np.dot(self.layers[layer+1][0].T, self.layerGradients[layer+1][0]) * self.layerGradients[layer][0]
            dW = (1/600) * np.dot(self.layerGradients[layer][0], self.layersIO[layer][0].T)
            dB = (1/600) * np.reshape(np.sum(self.layerGradients[layer][0],1), (100,1))
       
            self.layerGradients[layer] = [dZ, dW, dB]

    def softmax_backward(self,layer, Y):
        pass


    def relu_backward(self, Z):
        return Z > 0


    def sigmoid_backward(self):
        pass


    def update_parameters_new(self):
        for i in range(len(self.layers)):
            print(i)
            self.layers[i][0] -= self.learningRate * self.layerGradients[i][1]
            self.layers[i][1] -= self.learningRate * self.layerGradients[i][2]

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
    
    def print_layer(self):
        for i in range(len(self.layers)):
            print(self.layers[i][0].shape)


    def history(self):
       return self.training_history, self.training_history_test
    

    def weights_as_image(self):
        fig, ax = plt.subplots(2,5, dpi=200)   
        for x in range(5):
            im1 = self.layers[0][0][x,:].reshape((28,28))
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

nn.create_layer([784, 100], 'relu')
nn.create_layer([100, 10], 'softmax')

nn.print_layer()


nn.linar_forward(0, mini_batches[0][0].T)
nn.activation_forward(0)
nn.linar_forward(1, nn.layersIO[0][2])
nn.activation_forward(1)
nn.activation_backward(1, mini_batches[0][1].T)
nn.linear_backward(1)
nn.activation_backward(0, mini_batches[0][1].T)
nn.linear_backward(0)
nn.update_parameters_new()
nn.print_layer()

#TODO one backward pass works, index out of range in the second???


epochs = 1
history = []
history_test = []

# Train the network for number of epochs
for y in range(epochs):
    print(str(y+1) + ' out of ' + str(epochs) + ' epochs')
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