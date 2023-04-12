import numpy as np
import pandas as pd
import load_mnist as lm
import matplotlib.pyplot as plt


class NeuralNetwork:
    # Create a neural network with #hidden layers and #neurons in each layer
    def __init__(self, features, learningRate, X_train, Y_train, X_test,Y_test):
        self.features = features
        self.learningRate = learningRate
        self.layers = []
        self.layersIO = []
        self.layerGradients = []
        self.training_history = []
        self.test_history = []
        self.accuracy = []
        self.test_accuracy = []
        self.iterations = []
       
    #def cross_entropy(self, Y_pred, Y):
    def cross_entropy(self,X, Y):
        self.model_forward(X)
        batch_size = Y.shape[1]
        cost = -np.sum(Y * np.log(self.layersIO[-1][2] + 1e-10)) / batch_size
        return cost


    def create_layer(self, nodes, activation):
        np.random.seed(42)
        weights = np.random.rand(nodes[1], nodes[0]) *  0.01 
        bias = np.random.rand(nodes[1], 1) * np.sqrt(1 / (nodes[0]))
        self.layers.append([weights, bias, activation]) #[weights, bias, activation function]
        self.layersIO.append([None, None, None]) # [input, Z, A]
        self.layerGradients.append([None, None, None, None]) #[dZ, dW, dB, dA]


    def linear_forward(self, layer , input):
        Z = np.dot(self.layers[layer][0], input) + self.layers[layer][1]   
        self.layersIO[layer][0] = input #input
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
         

    def activation_backward(self, layer, Y):
        if self.layers[layer][2] == 'relu':
            dA = self.relu_backward(self.layersIO[layer][2])
            self.layerGradients[layer][3] = dA

        elif self.layers[layer][2] == 'sigmoid':
            dA = self.sigmoid_backward(self.layersIO[layer][2])
            self.layerGradients[layer][3] = dA
       

    def linear_backward(self, layer, Y):
        batch_size = self.layersIO[layer][0].shape[1]

        if self.layers[layer][2] == 'softmax':
            dZ = - Y + self.layersIO[layer][2]
            dW = (1/batch_size) * np.dot(dZ, self.layersIO[layer-1][2].T)
            dB = (1/batch_size) * np.reshape(np.sum(dZ,1), (dZ.shape[0],1))

            self.layerGradients[layer][0:3] = dZ, dW, dB

        else:
            dA = self.layerGradients[layer][3]
            dZ = np.dot(self.layers[layer+1][0].T, self.layerGradients[layer+1][0]) * dA
            dW = (1/batch_size) * np.dot(dZ, self.layersIO[layer][0].T)
            dB = (1/batch_size) * np.reshape(np.sum(dZ,1), (dZ.shape[0],1))
       
            self.layerGradients[layer][0:3] = dZ, dW, dB
       
       
    def softmax(self, Z):
        Z_max = np.max(Z, axis=0)
        Z_shifted = Z - Z_max
        A = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=0)
        return A


    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A


    def relu(self, Z):
        return np.maximum(Z,0)


    def relu_backward(self, Z):
        return Z > 0


    def sigmoid_backward(self, A):
        return A * (1 - A)
    
        
    def model_forward(self, minibatch_X): 
         self.linear_forward(0, minibatch_X)
         self.activation_forward(0)
         for i in range(1,len(self.layers),1):
            self.linear_forward(i, self.layersIO[i-1][2])
            self.activation_forward(i)
        
        

    def model_backward(self, minibatch_Y):
         for i in range(len(self.layers)-1,-1, -1):
            self.activation_backward(i, minibatch_Y)
            self.linear_backward(i, minibatch_Y)


    def update_parameters(self):
        for i in range(len(self.layers)):
            self.layers[i][0] -= self.learningRate * self.layerGradients[i][1]
            self.layers[i][1] -= self.learningRate * self.layerGradients[i][2]
        

    def compare(self, arr1, arr2):
        # Get the indices of the maximum values along axis 0 (rows)
        max_indices_arr1 = np.argmax(arr1, axis=0)
        max_indices_arr2 = np.argmax(arr2, axis=0)

        # Compare the indices and count the matches
        matches = np.sum(max_indices_arr1 == max_indices_arr2)

        # Calculate the percentage of correct matches
        percentage = (matches / arr1.shape[1]) * 100

        return percentage


    def train_model(self, mini_batches, epochs):
        iter = 0
        k = 200
        for y in range(epochs):
            print(str(y+1) + ' out of ' + str(epochs) + ' epochs')
            for x in range(len(mini_batches)): 
                self.model_forward(mini_batches[x][0].T)
                self.model_backward(mini_batches[x][1].T)
                self.update_parameters()
                if x % k == 0:
                    self.training_history.append(self.cross_entropy(mini_batches[x][0].T, mini_batches[x][1].T))
                    self.test_history.append(self.cross_entropy(X_test.T,Y_test.T))
                    self.test_accuracy.append(nn.predict(X_train,Y_train))
                    self.accuracy.append(nn.predict(mini_batches[x][0],mini_batches[x][1])) 
                    iter += k
                    self.iterations.append(iter)
                    
            print(nn.predict(X_test,Y_test))

    def predict(self, data, labels):
        self.model_forward(data.T)
        percent = self.compare(self.layersIO[-1][2], labels.T)
        return percent


    def print_layer(self):
        for i in range(len(self.layers)):
            print(self.layers[i][0].shape)


    def history(self):
       return self.training_history, self.training_history_test
    

    def weights_as_image(self):
        fig, ax = plt.subplots(2,5)  
        for x in range(5):
            im1 = self.layers[0][0][x,:].reshape((28,28))
            im2 = self.layers[0][0][5+x,:].reshape((28,28))
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

    # Calculate the batch size
    batch_size = data.shape[0] // num_batches

    mini_batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size

        data_batch = data[start_index:end_index]
        labels_batch = labels[start_index:end_index]

        mini_batches.append((data_batch, labels_batch))

    # Take the remaining and make a batch
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
number_of_batches = 2000
mini_batches = create_mini_batches(shuffled_data, shuffled_labels, number_of_batches)
mini_batches_test = create_mini_batches(shuffled_testdata, shuffled_testlabels, number_of_batches)



# Create neural network
nn = NeuralNetwork(784, 1e-2, X_train, Y_train, X_test, Y_test)
nn.create_layer([784, 128], 'relu')
nn.create_layer([128,64], 'relu')
nn.create_layer([64, 10], 'softmax')
nn.print_layer()

#Train the network and print test accuarcy while training
epochs = 50
nn.train_model(mini_batches,epochs)   
print(nn.predict(shuffled_testdata,shuffled_testlabels))


#Plot training cost and accuarcy history after training
fig1, (ax1, ax2) = plt.subplots(1, 2)
fig1.set_figwidth(10)

ax1.set_title('Cost')
ax1.set_xlabel('Iterations')
ax1.plot(nn.iterations, nn.training_history, label='training cost')
ax1.plot(nn.iterations, nn.test_history, label='test cost')
ax1.grid()
ax1.legend()

ax2.set_title('Accuracy')
ax2.set_xlabel('Iterations')
ax2.plot(nn.iterations, nn.accuracy, label= 'accuracy')
ax2.plot(nn.iterations, nn.test_accuracy, label= 'test accuracy')
ax2.grid()
ax2.legend()

plt.show()


nn.weights_as_image()