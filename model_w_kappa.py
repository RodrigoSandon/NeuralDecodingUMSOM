#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 00:05:41 2021

@author: rodrigosandon
"""
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Plotting

# Import Image Libraries - Pillow and OpenCV
from PIL import Image

# Import PyTorch and useful fuctions
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim
import torchvision.models as models # Pre-Trained models

# Import useful sklearn functions
import sklearn
from sklearn.metrics import cohen_kappa_score, accuracy_score

from time import time
from tqdm import tqdm_notebook

import os


mini_batch_size = 32

# Define transform
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     
     ])

# Define train and validation data sets 
train_set = datasets.ImageFolder("/Volumes/Passport/ResearchDataChen/Code/InputData/shuffled_official_all_regions_input/train/", transform=transform)
val_set = datasets.ImageFolder("/Volumes/Passport/ResearchDataChen/Code/InputData/shuffled_official_all_regions_input/test/", transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=mini_batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_set.__len__(), shuffle=False)

classes = ("visal", "visam", "visl","visp","vispm", "visrl")


#Model

model_resnet18 = models.resnet18(pretrained=True)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    model_resnet18.cuda()

#replace last layer
for param in model_resnet18.parameters():
    param.requires_grad = False
    
# Replace the last fully-connected layer
model_resnet18.fc = nn.Linear(512, 6)

time0 = time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify loss function (categorical cross-entropy loss)
criterion = nn.MSELoss()

# specify optimizer
optimizer = optim.Adam(model_resnet18.parameters(), lr=0.00015)
model_resnet18.to(device)

# number of epochs to train the model
n_epochs = 15

valid_loss_min = np.Inf

# keeping track of losses as it happen
train_losses = []
valid_losses = []
val_kappa = []
test_accuracies = []
valid_accuracies = []
kappa_epoch = []
batch = 0

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    ###################
    # train the model #
    ###################
    model_resnet18.train()
    
    for data, target in tqdm_notebook(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float() #need to change target to float()
        target = target.view(-1, 1)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_resnet18(data.float()) #needed to change input to float
            # calculate the batch loss
            loss = criterion(output, target.float()) #needed to change target data to float
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
        # Update Train loss and accuracies
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model_resnet18.eval()
    for data, target in tqdm_notebook(val_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda().float()
        # forward pass: compute predicted outputs by passing inputs to the model
        target = target.view(-1, 1)
        with torch.set_grad_enabled(True):
            output = model_resnet18(data)
            # calculate the batch loss
            loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        #output = output.cohen_kappa_score_kappa_score)
        y_actual = target.data.cpu().numpy()
        y_pred = output[:,-1].detach().cpu().numpy()
        val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))        
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(val_loader.sampler)
    valid_kappa = np.mean(val_kappa)
    kappa_epoch.append(np.mean(val_kappa))
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
        
    # print training/validation statistics 
    print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f}'.format(
        epoch, train_loss, valid_loss, valid_kappa))
    
    ##################
    # Early Stopping #
    ##################
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model_resnet18.state_dict(), 'resnet18_w_kappa.pt')
        valid_loss_min = valid_loss
        

#Plot training loss and valid loss

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)


#plot kappa on every epoch

plt.plot(kappa_epoch, label='Val Kappa Score/Epochs')
plt.legend("")
plt.xlabel("Epochs")
plt.ylabel("Kappa Score")
plt.legend(frameon=False)

model_resnet18.load_state_dict(torch.load('resnet18_w_kappa.pt'))

#Test

correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_resnet18(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network test images: %d %%' % (
    100.00 * correct / total))