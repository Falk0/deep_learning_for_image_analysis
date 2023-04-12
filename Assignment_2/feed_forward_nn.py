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
        self.bA1 = None
        self.dZ1 = None
        self.dW1 = None
        self.dB1 = None
        self.A1 = None
        self.Z2 = None
        self.training_history = []
        self.training_history_test = []
       

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
        self.Z1 = np.dot(self.weights1, X) + self.bias1  # 10, m
        self.A1 = self.softmax(self.Z1)                   # 10, m



    def cross_entropy(self, X, Y):
        self.model_forward(X)
        batch_size = Y.shape[1]
        cost = -np.sum(Y * np.log(self.A1 + 1e-10)) / batch_size
        return cost


    #Calculate the gradients 
    def model_backward(self, X, Y):
        features, samples  = X.shape
        features_test, samples_test  = X_test.shape
        self.model_forward(X)

        # Calculate gradients
        self.dZ1 = (-Y + self.A1)          # 10 m
        self.dW1 = (1/samples) * np.dot(self.dZ1, X.T)  # 10, 784  
        self.dB1 = (1/samples) * np.reshape(np.sum(self.dZ1,1),(10,1)) # 10, 1

        
    #Update the weight and bias with the pre-calciulated gradients
    def update_parameters(self):
        self.weights1 -= self.learningRate * self.dW1
        self.bias1 -= self.learningRate * self.dB1


    

    #Train the network
    def train_linear_model(self, X, Y, iterations):
        self.model_backward(X, Y)
        self.update_parameters()


    def print_parameters(self):
        print(self.weights1.shape)
        print(self.bias1)
    
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
        self.model_forward(X)
        percent = self.compare(self.A1, Y)
        return percent

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



# Create neural network
nn = NeuralNetwork(784, 1e-2, 1)
nn.initiliaze_parameters()


epochs = 200
history = []
history_test = []
accuracy = []
test_accuracy = []
xline = []

# Train the network for number of epochs
for y in range(epochs):
    print(str(y+1) + ' out of ' + str(epochs) + ' epochs')
    for x in range(len(mini_batches)):
        nn.train_linear_model(mini_batches[x][0].T, mini_batches[x][1].T,  1)
        
        if x % 25 == 0:
            history.append(nn.cross_entropy(mini_batches[x][0].T, mini_batches[x][1].T))
            history_test.append(nn.cross_entropy(shuffled_testdata.T,shuffled_testlabels.T))
            accuracy.append(nn.predict(mini_batches[x][0].T, mini_batches[x][1].T))
            test_accuracy.append(nn.predict(shuffled_testdata.T,shuffled_testlabels.T))
            xline.append((y * 150) + x)


# plot the weights as images
nn.weights_as_image()

#Plot training cost and accuarcy history after training
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_figwidth(10)

ax1.set_title('Cost')
ax1.plot(xline, history, label='training cost')
ax1.plot(xline, history_test, label='test cost')
ax2.set_xlabel('Iterations')
ax1.grid()
ax1.legend()

ax2.set_title('Accuracy')
ax2.set_xlabel('Iterations')
ax2.plot(xline, accuracy, label= 'accuracy')
ax2.plot(xline, test_accuracy, label= 'test accuracy')
ax2.grid()
ax2.legend()

plt.show()





correct = nn.predict(shuffled_testdata.T, shuffled_testlabels.T)


print("Percentage of correct matches:", correct)