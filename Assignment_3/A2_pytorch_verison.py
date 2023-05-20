import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import load_mnist as lm
import matplotlib.pyplot as plt
import time



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        U1 = 128 
        U2 = 64


        self.W1 = nn.Parameter(0.01 * torch.randn(784, U1))
        self.b1 = nn.Parameter(torch.ones(U1)/U1)
        self.W2 = nn.Parameter(0.01 * torch.randn(U1, U2))
        self.b2 = nn.Parameter(torch.ones(U2)/U2)
        self.W3 = nn.Parameter(0.01 * torch.randn(U2, 10))
        self.b3 = nn.Parameter(torch.ones(10)/10)


    def forward(self, X):
        X = X.view(X.size(0), -1)
        Q1 = F.relu(X.mm(self.W1) + self.b1)
        Q2 = F.relu(Q1.mm(self.W2) + self.b2)
        Z = Q2.mm(self.W3) + self.b3
        return Z
    

def accuracy(G, Y):
    return (G.argmax(dim=1) == Y.argmax(dim=1)).float().mean()


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


#device = torch.device("mps")
device = torch.device("cpu")

X_train, Y_train, X_test, Y_test = lm.load_mnist()


# Shuffle the data and labels
train_X, train_Y = shuffle_data_and_labels(X_train, Y_train)
test_X, test_Y = shuffle_data_and_labels(X_test, Y_test)

# Create mini-batches
number_of_batches = 2000
mini_batches = create_mini_batches(train_X, train_Y, number_of_batches)

test_X = torch.tensor(test_X, dtype=torch.float)
test_Y = torch.tensor(test_Y, dtype=torch.float)
test_X = test_X.to(device)
test_Y = test_Y.to(device)

train_X = torch.tensor(train_X, dtype=torch.float)
train_Y = torch.tensor(train_Y, dtype=torch.float)
train_X = train_X.to(device).unsqueeze(1)
train_Y = train_Y.to(device)


# initialize the test and training error statistics
test_accuracy = []
test_crossentropy = []
test_iter = []
train_accuracy = []
train_crossentropy = []
train_iter = []

# initialize the neural network and move it to the GPU
net = Net()
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total number of learnable weights:", total_params)

net = net.to(torch.float)
net.to(device)

# define the optimization algorithm
learningrate = 0.01
optimizer = optim.SGD(net.parameters(), lr=learningrate)

epochs = 50
subset_size = 6000

start_time = time.time()
k = 1000
iter = 0
for y in range(epochs):
    print(str(y+1) + ' out of ' + str(epochs) + ' epochs')
    for x in range(len(mini_batches)):
        minibatch_X = torch.tensor(mini_batches[x][0], dtype=torch.float)
        minibatch_Y = torch.tensor(mini_batches[x][1], dtype=torch.float)
        minibatch_X = minibatch_X.to(device)
        minibatch_Y = minibatch_Y.to(device)
        
        optimizer.zero_grad()
        X_forward = net(minibatch_X)
        loss = F.cross_entropy(X_forward, minibatch_Y)
        loss.backward()
        optimizer.step()
        if (x + y * number_of_batches) % k == 0:
            train_indices = np.random.choice(train_X.shape[0], subset_size, replace=False)
            train_X_subset = train_X[train_indices]
            train_Y_subset = train_Y[train_indices]

            X_forward = net(train_X_subset)
            train_accuracy.append(accuracy(X_forward, train_Y_subset).item())
            train_crossentropy.append(loss.item())
            
            X_forward = net(test_X)
            test_cost = F.cross_entropy(X_forward, test_Y)
            test_crossentropy.append(test_cost.item())
            test_accuracy.append(accuracy(X_forward, test_Y).item())
            iter += k
            test_iter.append(iter)
            
    X_forward = net(test_X)
    print(accuracy(X_forward, test_Y))
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

fig1, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
fig1.set_figwidth(10)

ax1.set_title('Cost')
ax1.set_xlabel('Iterations')
ax1.plot(test_iter, train_crossentropy, label='training cost')
ax1.plot(test_iter, test_crossentropy, label='test cost')
ax1.grid()
ax1.legend()

ax2.set_title('Accuracy')
ax2.set_xlabel('Iterations')
ax2.plot(test_iter, train_accuracy, label= 'accuracy')
ax2.plot(test_iter, test_accuracy, label= 'test accuracy')
ax2.grid()
ax2.legend()
#plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/A2_torchversion.png', dpi = 200)
plt.show()



