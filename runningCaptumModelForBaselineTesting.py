#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 14:16:35 2021

@author: rodrigosandon
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F 
from torch import optim 
from time import time

mini_batch_size = 32

# Define transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define train and validation data sets 
train_set = datasets.ImageFolder("/Volumes/Passport/ResearchDataChen/Code/InputData/shuffled_official_all_regions_input/train/", transform=transform)
val_set = datasets.ImageFolder("/Volumes/Passport/ResearchDataChen/Code/InputData/shuffled_official_all_regions_input/test/", transform=transform)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=mini_batch_size, shuffle=False)
valloader = torch.utils.data.DataLoader(val_set, batch_size=val_set.__len__(), shuffle=False)

classes = ("visal", "visam", "visl","visp","vispm", "visrl")


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #why is it these dimensions?
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()


    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

time0 = time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)


def train(epochs, model):
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
    
            # Training pass
            optimizer.zero_grad()
    
            output = model(images).to(device)
            loss = criterion(output, labels)
    
            # backpropagation: calculate the gradient of the loss function w.r.t model parameters
            loss.backward()
    
            # And optimizes its weights here
            optimizer.step()
    
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
    
    print("\nTraining Time (in minutes) =", (time()-time0)/60)
    #print('Finished Training')
    
    
train(150, model)


PATH = './conv_net_trained_resnet18.pth'
torch.save(Net().state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in valloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network test images: %d %%' % (
    100.00 * correct / total))


#How each classes accuracy is?

