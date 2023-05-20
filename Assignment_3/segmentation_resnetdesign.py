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

train_folder_path = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/Assignment_3/WARWICK/Train'
test_folder_path = 'C:/Users/linus/python_projects/deep_learning_for_image_analysis/Assignment_3/WARWICK/Test'

def load_images(folder_path):
    filenames = sorted(os.listdir(folder_path))
    images = []
    for filename in filenames:
        if filename.endswith('.png') and filename.startswith('image') :
            image_path = os.path.join(folder_path, filename)
            try:
               image = imageio.imread(image_path)
               images.append(image)
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
               images.append(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    return np.array(images)/255.0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        U1 = 32
        U2 = 64
        U3 = 128
        ###
        self.W1 = nn.Parameter(0.1 * torch.randn(U1, 2, 3, 3))
        self.b1 = nn.Parameter(torch.ones(U1)/10) 
        self.bn1 = nn.BatchNorm2d(U1)

        self.W1a = nn.Parameter(0.1 * torch.randn(U1, U1, 3, 3))
        self.b1a = nn.Parameter(torch.ones(U1)/10) 
        self.bn1a = nn.BatchNorm2d(U1)

        self.W1b = nn.Parameter(0.1 * torch.randn(U1, U1, 3, 3))
        self.b1b = nn.Parameter(torch.ones(U1)/10) 
        self.bn1b = nn.BatchNorm2d(U1)
        ####
        self.W2 = nn.Parameter(0.1 * torch.randn(U2, U1, 3, 3))
        self.b2 = nn.Parameter(torch.ones(U2)/10)
        self.bn2 = nn.BatchNorm2d(U2)

        self.W2a = nn.Parameter(0.1 * torch.randn(U2, U2, 3, 3))
        self.b2a = nn.Parameter(torch.ones(U2)/10)
        self.bn2a = nn.BatchNorm2d(U2)

        self.W2b = nn.Parameter(0.1 * torch.randn(U2, U2, 3, 3))
        self.b2b = nn.Parameter(torch.ones(U2)/10)
        self.bn2b = nn.BatchNorm2d(U2)

        ####

        self.W3 = nn.Parameter(0.1 * torch.randn(U3, U2, 3, 3))
        self.b3 = nn.Parameter(torch.ones(U3)/10)
        self.bn3 = nn.BatchNorm2d(U3)

        self.W3a = nn.Parameter(0.1 * torch.randn(U3, U3, 3, 3))
        self.b3a = nn.Parameter(torch.ones(U3)/10)
        self.bn3a = nn.BatchNorm2d(U3)

        self.W3b = nn.Parameter(0.1 * torch.randn(U3, U3, 3, 3))
        self.b3b = nn.Parameter(torch.ones(U3)/10)
        self.bn3b = nn.BatchNorm2d(U3)

        ####

        self.t1 = nn.ConvTranspose2d(U3, U2, 4, stride=2, padding=1)

        self.W5 = nn.Parameter(0.1 * torch.randn(U2, U2, 3, 3))
        self.b5 = nn.Parameter(torch.ones(U2)/10)
        self.bn5 = nn.BatchNorm2d(U2)

        self.W5a = nn.Parameter(0.1 * torch.randn(U2, U2, 3, 3))
        self.b5a = nn.Parameter(torch.ones(U2)/10)
        self.bn5a = nn.BatchNorm2d(U2)

        self.W5b = nn.Parameter(0.1 * torch.randn(U2, U2, 3, 3))
        self.b5b = nn.Parameter(torch.ones(U2)/10)
        self.bn5b = nn.BatchNorm2d(U2)

        self.t2 = nn.ConvTranspose2d(U2, U1, 4, stride=2, padding=1)

        self.W6 = nn.Parameter(0.1 * torch.randn(U1, U1, 3, 3))
        self.b6 = nn.Parameter(torch.ones(U1)/10)
        self.bn6 = nn.BatchNorm2d(U1)

        self.W6a = nn.Parameter(0.1 * torch.randn(U1, U1, 3, 3))
        self.b6a = nn.Parameter(torch.ones(U1)/10)
        self.bn6a = nn.BatchNorm2d(U1)

        self.W6b = nn.Parameter(0.1 * torch.randn(U1, U1, 3, 3))
        self.b6b = nn.Parameter(torch.ones(U1)/10)
        self.bn6b = nn.BatchNorm2d(U1)

        self.W4 = nn.Parameter(0.1 * torch.randn(2, 32, 1, 1))
        self.b4 = nn.Parameter(torch.ones(2)/10)




    def forward(self, X):
        X = X.permute(0, 3, 1, 2)

        Q1 = (F.conv2d(X, self.W1, bias=self.b1,stride=1, padding=1)) #[85, 32, 128, 128]
        Q1_normalized = F.relu(self.bn1(Q1)) 
        Q1a = (F.conv2d(Q1_normalized, self.W1a, bias=self.b1a,stride=1, padding=1)) #[85, 32, 128, 128]
        Q1a_normalized = F.relu(self.bn1a(Q1a))
        Q1b = (F.conv2d(Q1a_normalized, self.W1b, bias=self.b1b,stride=1, padding=1)) #[85, 32, 128, 128]
        #Q1b = (F.conv2d(Q1a_normalized + Q1_normalized, self.W1b, bias=self.b1b,stride=1, padding=1)) #[85, 32, 128, 128]
        Q1b_normalized = F.relu(self.bn1b(Q1b))
        
        M1 = F.max_pool2d(Q1b_normalized, kernel_size=2, stride=2) #[85, 32, 64, 64]
        
        Q2 = (F.conv2d(M1, self.W2, bias=self.b2,stride=1, padding=1)) #[85, 32, 64, 64] - > [85, 64, 64, 64]
        Q2_normalized = F.relu(self.bn2(Q2))
        Q2a = (F.conv2d(Q2_normalized, self.W2a, bias=self.b2a,stride=1, padding=1)) #[85, 32, 128, 128]
        Q2a_normalized = F.relu(self.bn2a(Q2a))
        Q2b = (F.conv2d(Q2a_normalized, self.W2b, bias=self.b2b,stride=1, padding=1)) #[85, 32, 128, 128]
        #Q2b = (F.conv2d(Q2a_normalized + Q2_normalized, self.W2b, bias=self.b2b,stride=1, padding=1)) #[85, 32, 128, 128]
        Q2b_normalized = F.relu(self.bn2b(Q2b))
        
        M2 = F.max_pool2d(Q2b_normalized, kernel_size=2, stride=2) #[85, 64, 64, 64] -> [85, 64, 32, 32]

        Q3 = (F.conv2d(M2, self.W3, bias=self.b3,stride=1, padding=1)) #[85, 64, 32, 32] -> [85, 128, 32, 32]
        Q3_normalized = F.relu(self.bn3(Q3))
        Q3a = (F.conv2d(Q3_normalized, self.W3a, bias=self.b3a,stride=1, padding=1)) #[85, 32, 128, 128]
        Q3a_normalized = F.relu(self.bn3a(Q3a))
        Q3b = (F.conv2d(Q3a_normalized, self.W3b, bias=self.b3b,stride=1, padding=1)) #[85, 32, 128, 128]
        #Q3b = (F.conv2d(Q3a_normalized + Q3_normalized, self.W3b, bias=self.b3b,stride=1, padding=1)) #[85, 32, 128, 128]
        Q3b_normalized = F.relu(self.bn3b(Q3b))

        T1 = self.t1(Q3b_normalized, output_size=(X.shape[0], 64, 64, 64)) #[85, 128, 32, 32] -> [85, 64, 64, 64] 
        T1_normalized = self.bn5(T1)
        Q4 = F.relu(F.conv2d(T1_normalized, self.W5, bias=self.b5, stride=1, padding=1))
        T1a_normalized = self.bn5a(Q4)
        Q4a = F.relu(F.conv2d(T1a_normalized, self.W5a, bias=self.b5a, stride=1, padding=1)) 
        T1b_normalized = self.bn5b(Q4a)
        Q4b = F.relu(F.conv2d(T1b_normalized, self.W5b, bias=self.b5b, stride=1, padding=1)) 
        #Q4b = F.relu(F.conv2d(T1b_normalized + T1_normalized, self.W5b, bias=self.b5b, stride=1, padding=1)) 

        T2 = self.t2(Q4b, output_size=(X.shape[0], 32, 128, 128 )) # [85, 64, 64, 64] -> [85, 128, 32, 32]
        T2_normalized = self.bn6(T2)
        Q5 = F.relu(F.conv2d(T2_normalized, self.W6, bias=self.b6, stride=1, padding=1))
        T2a_normalized = self.bn6a(Q5)
        Q5a = F.relu(F.conv2d(T2a_normalized, self.W6a, bias=self.b6a, stride=1, padding=1)) 
        T2b_normalized = self.bn6b(Q5a)
        Q5b = F.relu(F.conv2d(T2b_normalized, self.W6b, bias=self.b6b, stride=1, padding=1)) 
        #Q5b = F.relu(F.conv2d(T2b_normalized + T2_normalized, self.W6b, bias=self.b6b, stride=1, padding=1)) 

        Z = F.conv2d(Q5b, self.W4, bias=self.b4, stride=1, padding=0)

 
        return Z
    

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    intersection = np.sum(y_true_flat * y_pred_flat)
    union = np.sum(y_true_flat) + np.sum(y_pred_flat)

    dice = (2 * intersection + smooth) / (union + smooth)
    return dice


def crossentropy(G, Y):
    samples = Y.shape[0]
    G1 = G.permute(0, 2, 3, 1)
    Y1 = one_hot_encode(Y)
    G2 = F.softmax(G1, dim=-1)

    G3 = G2.reshape(samples, 128, 128, 2)
    Y2 = Y1.reshape(samples, 128, 128, 2)

    return F.cross_entropy(G3,Y2)


def one_hot_encode(labels, num_classes=2):
    labels = labels.long()
    shape = list(labels.shape)

    # Add the number of classes as the last dimension
    shape.append(num_classes)

    one_hot = torch.zeros(shape, dtype=torch.float)
    one_hot = one_hot.to(device)

    # Set the values at the corresponding positions to 1
    one_hot.scatter_(-1, labels.unsqueeze(-1), 1)

    return one_hot


def gamma_update(t, gamma_max, gamma_min):
    new_gamma = gamma_min + (gamma_max-gamma_min)*np.exp(-t/100)
    return new_gamma


device = torch.device("cuda")


train_X = load_images(train_folder_path)
train_Y = load_labels(train_folder_path)
test_X = load_images(test_folder_path)
test_Y = load_labels(test_folder_path)


train_X = train_X[:, :, :, :2]
test_X = test_X[:, :, :, :2]


# Split the training set into a new training set and a validation set (80% training, 20% validation)
#train_X, val_X = train_test_split(train_X, test_size=0.2, random_state=42)
#train_Y, val_Y = train_test_split(train_Y, test_size=0.2, random_state=42)


print(f"Loaded {len(train_X)} training images, and {len(test_X)} test images from the dataset.")
print(f"Loaded {len(train_Y)} training labels, and {len(test_Y)} test labels from the dataset.")


train_X = torch.tensor(train_X, dtype=torch.float)
train_Y = torch.tensor(train_Y, dtype=torch.long)
#val_X = torch.tensor(train_X, dtype=torch.float)
#val_Y = torch.tensor(train_Y, dtype=torch.long)
test_X = torch.tensor(test_X, dtype=torch.float)
test_Y = torch.tensor(test_Y, dtype=torch.long)


test_X = test_X.to(device)
test_Y = test_Y.to(device)
train_X = train_X.to(device)
train_Y = train_Y.to(device)


# initialize the neural network and move it to the GPU
net = Net()
net = net.to(torch.float)
net = net.to(device)

# define the optimization algorithm
learningrate = 0.003
optimizer = optim.Adam(net.parameters(), lr=learningrate)
epochs = 100

training_loss = []
val_loss = []

for y in range(epochs):
    print(y)
    
    learningrate = gamma_update(y, 0.003, 0.0001)
    for p in optimizer.param_groups:
         p['lr'] = learningrate

    optimizer.zero_grad()
    X_forward = net(train_X)
    
    loss = crossentropy(X_forward, train_Y)
    loss.backward()
    optimizer.step()
    loss = loss.cpu()
    training_loss.append(loss.detach().numpy())
    net.eval()

    X_forward_test = net(test_X)
    loss_test = crossentropy(X_forward_test, test_Y)
    loss_test = loss_test.cpu()
    val_loss.append(loss_test.detach().numpy())
    print(loss_test.detach().numpy())
    net.train()

net.eval()
test_X = test_X.cpu()
test_Y = test_Y.cpu()
net = net.cpu()

X_forward = net(test_X)
X_forward1 = X_forward.permute(0, 2, 3, 1)
X_forward3 = X_forward1
X_forward4 = X_forward3.detach().numpy()


result = np.argmax(X_forward4, axis=-1)

plt.plot(training_loss, label = 'training loss')
plt.plot(val_loss, label='test loss')
plt.legend()
plt.grid()
#plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/segmentation_unet2.png', dpi = 200)
plt.show()

test_Y = test_Y.detach().numpy()
print(dice_coefficient(test_Y, result))




fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig1.set_figwidth(10)
ax1.imshow(imageio.imread('C:/Users/linus/python_projects/deep_learning_for_image_analysis/Assignment_3/WARWICK/Test/image_05.png'))
ax2.imshow(result[4])
ax3.imshow(test_Y[4,:,:])
ax1.set_title('Input')
ax2.set_title('Prediction')
ax3.set_title('Label')
#plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/segmentation_im5_unet2.png', dpi = 200)
plt.show()

index = np.zeros(len(test_Y))

for i in range(len(test_Y)):
    index[i] = dice_coefficient(test_Y[i], result[i])

print(index.argmin())

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig2.set_figwidth(10)
ax1.imshow(imageio.imread('C:/Users/linus/python_projects/deep_learning_for_image_analysis/Assignment_3/WARWICK/Test/image_11.png'))
ax2.imshow(result[10])
ax3.imshow(test_Y[10,:,:])
ax1.set_title('Input')
ax2.set_title('Prediction')
ax3.set_title('Label')
#plt.savefig('/Users/falk/Documents/latex_documents/latex_master1_semester2/deep_learning_for_image_analysis/figures/assignment_3/segmentation_im11_unet2.png', dpi = 200)
plt.show()

#0.83
#0.82