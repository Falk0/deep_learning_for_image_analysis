import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import load_mnist as lm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        U1 = 8 # number of hidden units
        U2 = 16
        U3 = 32
        U3flat = 1568
        U4 = 200
     
        self.W1 = nn.Parameter(0.1 * torch.randn(U1, 1, 3, 3))
        self.b1 = nn.Parameter(torch.ones(U1)/10)
        #self.bn1 = nn.BatchNorm2d(U1)

        self.W2 = nn.Parameter(0.1 * torch.randn(U2, U1, 3, 3))
        self.b2 = nn.Parameter(torch.ones(U2)/10)
        #self.bn2 = nn.BatchNorm2d(U2) 

        self.W3 = nn.Parameter(0.1 * torch.randn(U3, U2, 3, 3))
        self.b3 = nn.Parameter(torch.ones(U3)/10)
        #self.bn3 = nn.BatchNorm1d(U3flat)


        self.W4 = nn.Parameter(0.1 * torch.randn(U3flat, U4))
        self.b4 = nn.Parameter(torch.ones(U4)/10)
        #self.bn4 = nn.BatchNorm1d(U4)
        self.dropout = nn.Dropout(p=0.25)
        
        self.W5 = nn.Parameter(0.1 * torch.randn(U4, 10))
        self.b5 = nn.Parameter(torch.ones(10)/10)

#best 99.07% with BatchNormalization, instruction architecture, lr = 0.003
#best 99.00% with DropOut(p=0.3) in fc layers, instruction architecture, lr = 0.003

    def forward(self, X):
        Q1 = F.relu(F.conv2d(X, self.W1, bias=self.b1,stride=1, padding=1))
        M1 = F.max_pool2d(Q1, kernel_size=2, stride=2)
        Q2 = F.relu(F.conv2d(M1, self.W2, bias=self.b2,stride=1, padding=1))
        M2 = F.max_pool2d(Q2, kernel_size=2, stride=2)
        Q3 = F.relu(F.conv2d(M2, self.W3, bias=self.b3,stride=1, padding=1))
        
        Q3flat = Q3.view(-1, 1568)
        Q4 = F.relu(Q3flat.mm(self.W4) + self.b4)
        Q4dropout = self.dropout(Q4)
        Z = Q4dropout.mm(self.W5) + self.b5
        return Z


def crossentropy(G, Y):
    return -(Y * G.log()).sum(dim = 1).mean()


def accuracy(G, Y):
    return (G.argmax(dim=1) == Y.argmax(dim=1)).float().mean()

def predict(G):
    return G.argmax(dim=1).float()


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


def gamma_update(t, gamma_max, gamma_min):
    new_gamma = gamma_min + (gamma_max-gamma_min)*np.exp(-t/2000)
    return new_gamma

device = torch.device("mps")
#device = torch.device("cpu")

X_train, Y_train, X_test, Y_test = lm.load_mnist()


# Shuffle the data and labels
train_X, train_Y = shuffle_data_and_labels(X_train, Y_train)
test_X, test_Y = shuffle_data_and_labels(X_test, Y_test)

# Create mini-batches
number_of_batches = 600
mini_batches = create_mini_batches(train_X, train_Y, number_of_batches)

test_X = torch.tensor(test_X, dtype=torch.float)
test_Y = torch.tensor(test_Y, dtype=torch.float)
test_X = test_X.to(device).unsqueeze(1)
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
net = net.to(torch.float)
net.to(device)

# define the optimization algorithm
learningrate = 0.003
optimizer = optim.Adam(net.parameters(), lr=learningrate)

epochs = 10
subset_size = 6000

start_time = time.time()
k = 100
iter = 0
for y in range(epochs):
    print(str(y+1) + ' out of ' + str(epochs) + ' epochs')
    for x in range(len(mini_batches)):
        
        #update learning rate       
        learningrate = gamma_update(iter, 0.003, 0.0001)
        for p in optimizer.param_groups:
            p['lr'] = learningrate

        minibatch_X = torch.tensor(mini_batches[x][0], dtype=torch.float).unsqueeze(1)
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
            net.eval() 
            X_forward = net(train_X_subset)
            train_accuracy.append(accuracy(X_forward, train_Y_subset).item())
            train_crossentropy.append(loss.item())
            
            X_forward = net(test_X)
            test_cost = F.cross_entropy(X_forward, test_Y)
            test_crossentropy.append(test_cost.item())
            test_accuracy.append(accuracy(X_forward, test_Y).item())
            iter += k
            test_iter.append(iter)

            net.train()
    net.eval()      
    X_forward = net(test_X)
    print(accuracy(X_forward, test_Y))
end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

fig1, (ax1, ax2) = plt.subplots(1, 2)
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
#plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/improved_torch4.png', dpi = 200)
plt.show()


Yhat = X_forward
Yhat = Yhat.cpu()
test_Y = test_Y.cpu()
Yhat_labels = Yhat.argmax(dim=1)
test_Y_labels = test_Y.argmax(dim=1)
cf_matrix = confusion_matrix(test_Y_labels, Yhat_labels)

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
#plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/improved_torch4_CM.png', dpi = 200)
plt.show()

# inspiration https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7