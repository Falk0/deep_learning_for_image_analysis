import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import load_mnist as lm
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from scipy import ndimage

import imageio
import glob


train_folder_path = '/Users/falk/Documents/python_projects/Deep_learning_for_image_analysis/Assignment_3/WARWICK/Train'
test_folder_path = '/Users/falk/Documents/python_projects/Deep_learning_for_image_analysis/Assignment_3/WARWICK/Test'

def load_images(folder_path):
    filenames = sorted(os.listdir(folder_path))
    images = []
    for filename in filenames:
        if filename.endswith('.png') and filename.startswith('image') :
            image_path = os.path.join(folder_path, filename)
            try:
               image = imageio.imread(image_path)
               #image_rotated90 = ndimage.rotate(image, -90, reshape=False, mode='reflect')
               #image_rotated180 = ndimage.rotate(image, 180, reshape=False, mode='reflect')
               images.append(image)
               #images.append(image_rotated90)
               #images.append(image_rotated180)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
            
    return np.array(images)/255.0


def load_labels(folder_path):       
    images = []
    filenames = sorted(os.listdir(folder_path))
    for filename in filenames:
        if filename.endswith('.png') and filename.startswith('label') :
            image_path = os.path.join(folder_path, filename)
            try:
               image = imageio.imread(image_path)
               #image_rotated90 = ndimage.rotate(image, -90, reshape=False, mode='reflect')
               #image_rotated180 = ndimage.rotate(image, 180, reshape=False, mode='reflect')
               images.append(image)
               #images.append(image_rotated90)
               #images.append(image_rotated180)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    return np.array(images)/255.0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        U1 = 8 # number of hidden units
        U2 = 16
        U3 = 32

        self.W1 = nn.Parameter(0.1 * torch.randn(8, 2, 3, 3))
        self.b1 = nn.Parameter(torch.ones(U1)/10)
        self.t1 = nn.ConvTranspose2d(8, 8, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(U1)
        
        self.W2 = nn.Parameter(0.1 * torch.randn(16, 8, 3, 3))
        self.b2 = nn.Parameter(torch.ones(U2)/10)
        self.t2 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(U2)

        self.W3 = nn.Parameter(0.1 * torch.randn(32, 16, 3, 3))
        self.b3 = nn.Parameter(torch.ones(U3)/10)
        self.t3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(U3)
  
        self.W4 = nn.Parameter(0.1 * torch.randn(2, 32, 1, 1))
        self.b4 = nn.Parameter(torch.ones(2)/10)

    
    def forward(self, X):
        X = X.permute(0, 3, 1, 2)

        Q1 = F.relu(F.conv2d(X, self.W1, bias=self.b1,stride=1, padding=1))
        Q1_normalized = F.relu(self.bn1(Q1))   
        M1 = F.max_pool2d(Q1_normalized, kernel_size=2, stride=2)
        T1 = self.t1(M1, output_size=Q1.size())

        Q2 = F.relu(F.conv2d(T1, self.W2, bias=self.b2,stride=1, padding=1))
        Q2_normalized = F.relu(self.bn2(Q2))
        M2 = F.max_pool2d(Q2_normalized, kernel_size=2, stride=2)
        T2 = self.t2(M2, output_size=Q2.size())

        Q3 = F.relu(F.conv2d(T2, self.W3, bias=self.b3,stride=1, padding=1))
        Q3_normalized = F.relu(self.bn3(Q3))
        M3 = F.max_pool2d(Q3_normalized, kernel_size=2, stride=2)
        T3 = self.t3(M3, output_size=Q3.size())

        Z = F.conv2d(T3, self.W4, bias=self.b4, stride=1, padding=0)
        return Z
    

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coefficient(y_true, y_pred, smooth)


def crossentropy(G, Y):
    samples = Y.shape[0]
    G1 = G.permute(0, 2, 3, 1)
    Y1 = one_hot_encode(Y)
    G2 = F.softmax(G1, dim=-1)

    G3 = G2.reshape(samples, 128, 128, 2)
    Y2 = Y1.reshape(samples, 128, 128, 2)

    return F.cross_entropy(G3,Y2)


def one_hot_encode(labels, num_classes=2):
    # Get the shape of the input labels
    labels = labels.long()
    shape = list(labels.shape)

    # Add the number of classes as the last dimension
    shape.append(num_classes)

    # Create a zero-filled tensor with the new shape
    one_hot = torch.zeros(shape, dtype=torch.float)
    #one_hot = one_hot.to(device)

    # Set the values at the corresponding positions to 1
    one_hot.scatter_(-1, labels.unsqueeze(-1), 1)

    return one_hot


def gamma_update(t, gamma_max, gamma_min):
    new_gamma = gamma_min + (gamma_max-gamma_min)*np.exp(-t/100)
    return new_gamma



device = torch.device("cpu")


train_X = load_images(train_folder_path)
train_Y = load_labels(train_folder_path)
test_X = load_images(test_folder_path)
test_Y = load_labels(test_folder_path)


train_X = train_X[:, :, :, :2]
test_X = test_X[:, :, :, :2]

print(train_X.shape)
# Split the training set into a new training set and a validation set (80% training, 20% validation)
train_X, val_X = train_test_split(train_X, test_size=0.2, random_state=32)
train_Y, val_Y = train_test_split(train_Y, test_size=0.2, random_state=32)
print(train_X.shape)
print(val_X.shape)

print(f"Loaded {len(train_X)} training images, and {len(test_X)} test images from the dataset.")
print(f"Loaded {len(train_Y)} training labels, and {len(test_Y)} test labels from the dataset.")


train_X = torch.tensor(train_X, dtype=torch.float)
train_Y = torch.tensor(train_Y, dtype=torch.long)
val_X = torch.tensor(train_X, dtype=torch.float)
val_Y = torch.tensor(train_Y, dtype=torch.long)
test_X = torch.tensor(test_X, dtype=torch.float)
test_Y = torch.tensor(test_Y, dtype=torch.long)

'''
test_X = test_X.to(device)
test_Y = test_Y.to(device)
train_X = train_X.to(device)
train_Y = train_Y.to(device)
'''

# initialize the neural network and move it to the GPU
net = Net()
net = net.to(torch.float)
#net.to(device)

# define the optimization algorithm
learningrate = 0.003
optimizer = optim.Adam(net.parameters(), lr=learningrate)
epochs = 100

training_loss = []
val_loss = []

for y in range(epochs):
    print(y)
    
    #learningrate = gamma_update(y, 0.003, 0.0001)
    #for p in optimizer.param_groups:
    #     p['lr'] = learningrate

    optimizer.zero_grad()
    X_forward = net(train_X)
    
    loss = crossentropy(X_forward, train_Y)

    training_loss.append(loss.detach().numpy())

    loss.backward()
    optimizer.step()
    
    X_forward_test = net(test_X)
    loss_test = crossentropy(X_forward_test, test_Y)
    val_loss.append(loss_test.detach().numpy())


X_forward = net(test_X)
X_forward1 = X_forward.permute(0, 2, 3, 1)
X_forward3 = X_forward1
X_forward4 = X_forward3.detach().numpy()

result = np.argmax(X_forward4, axis=-1)

plt.plot(training_loss, label = 'training loss')
plt.plot(val_loss, label='test loss')
plt.legend()
plt.grid()
plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/segmentation_model_best.png', dpi = 200)
plt.show()

test_Y = test_Y.detach().numpy()
print(dice_coefficient(test_Y, result))



fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig1.set_figwidth(10)
ax1.imshow(imageio.imread('/Users/falk/Documents/python_projects/Deep_learning_for_image_analysis/Assignment_3/WARWICK/Test/image_05.png'))
ax2.imshow(result[4])
ax3.imshow(test_Y[4,:,:])
ax1.set_title('Input')
ax2.set_title('Prediction')
ax3.set_title('Label')
plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/segmentation_test_best.png', dpi = 200)
plt.show()

index = np.zeros(len(test_Y))

for i in range(len(test_Y)):
    index[i] = dice_coefficient(test_Y[i], result[i])

print(index.argmin())

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig2.set_figwidth(10)
ax1.imshow(imageio.imread('/Users/falk/Documents/python_projects/Deep_learning_for_image_analysis/Assignment_3/WARWICK/Test/image_11.png'))
ax2.imshow(result[index.argmin()])
ax3.imshow(test_Y[index.argmin(),:,:])
ax1.set_title('Input')
ax2.set_title('Prediction')
ax3.set_title('Label')
plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/segmentation_worse_best.png', dpi = 200)
plt.show()
