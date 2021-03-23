#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:41:57 2021
@author: Matthew Chen
"""

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import optim 
from time import time
import torchvision.models as models
import resNetCifar10Model

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr import Occlusion
import os

import sklearn
from sklearn.metrics import cohen_kappa_score, accuracy_score


import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

mini_batch_size = 32

# Define transform
transform_train = transforms.Compose([
     transforms.ToTensor(),
     ])
# Define train and validation data sets 

transform_val = transforms.Compose([transforms.ToTensor()])

train_set = datasets.ImageFolder("/Volumes/Passport/ResearchDataChen/Code/InputData/official_all_regions_input/train/", transform=transform_train)
val_set = datasets.ImageFolder("/Volumes/Passport/ResearchDataChen/Code/InputData/official_all_regions_input/test/", transform=transform_val)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=mini_batch_size, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=val_set.__len__(), shuffle=True)

classes = ("visal", "visam", "visl", "visp", "vispm", "visrl")


# keeping track of losses as it happen
train_losses = []
valid_losses = []
val_kappa = []
test_accuracies = []
valid_accuracies = []
kappa_epoch = []
batch = 0
time0 = time()

def train(epochs, model):
    valid_loss_min = np.Inf
    train_loss = 0.0
    valid_loss = 0.0
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
            
            train_loss += loss.item()*images.size(0)
            valid_loss += loss.item()*images.size(0)
            
            y_actual = labels.data.cpu().numpy()
            y_pred = output[:,-1].detach().cpu().numpy()
            val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))  
        else:
            #print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
                # calculate average losses
            train_loss = train_loss/len(trainloader.sampler)
            valid_loss = valid_loss/len(valloader.sampler)
            valid_kappa = np.mean(val_kappa)
            kappa_epoch.append(np.mean(val_kappa))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
                
            # print training/validation statistics 
            print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f}'.format(
                e, train_loss, valid_loss, valid_kappa))
            
            ##################
            # Early Stopping #
            ##################
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), 'resnet18_w_kappa.pt')
                valid_loss_min = valid_loss
    
    print("\nTraining Time (in minutes) =", (time()-time0)/60)


# GPU time!
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resNetCifar10Model.ResNet18()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)

train(10, model)

#Plot training loss and valid loss

plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)


#plot kappa on every epoch

# plt.plot(kappa_epoch, label='Val Kappa Score/Epochs')
# plt.legend("")
# plt.xlabel("Epochs")
# plt.ylabel("Kappa Score")
# plt.legend(frameon=False)

# model.load_state_dict(torch.load('resnet18_w_kappa.pt'))

correct_count, all_count = 0, 0

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
    100 * correct / total))

#torch.save(model.state_dict(), "trained_model")

images, labels = next(iter(valloader))

for ind in range(len(images)):
    img = images[ind].to(device)
    img = img[None]
    labels = labels.to(device)
    
    
    input = img
    input.requires_grad = True
    input = input.to(device)
    
    
    output = model(img)
        
    _, predicted = torch.max(output, 1)
        
        
    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                     target=labels[ind],
                                                     **kwargs
                                                    )
            
        return tensor_attributions     
        
    saliency = Saliency(model)
    grads = saliency.attribute(input, target=labels[ind].item())
        
    grads = grads.view(3, 32, 32)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        
        
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
        
  
    attr_ig = attr_ig.view(3, 32, 32)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))
        
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_ig_nt = attribute_image_features(nt, input, baselines=input * 0, nt_type='smoothgrad_sq',
                                              nt_samples=100, stdevs=0.2)
    attr_ig_nt = attr_ig_nt.view(3, 32, 32)
    attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
    dl = DeepLift(model)
    attr_dl = attribute_image_features(dl, input, baselines=input * 0)
    attr_dl = attr_dl.view(3, 32, 32)
    attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(input,
                                               strides = 2,
                                               target=labels[ind].item(),
                                               sliding_window_shapes= (3, 10, 10),
                                               baselines=0)
    attributions_occ = attributions_occ.view(3, 32, 32)
    attributions_occ = np.transpose(attributions_occ.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        
        
    print('Original Image')
    print('Predicted:', classes[predicted[0]], 'Actual:', labels[ind].cpu(),
          ' Probability:', torch.max(F.softmax(output, 1)).item())
        
    original_image = np.transpose((images[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
        
    fig1, _ = viz.visualize_image_attr(None, original_image, 
                              method="original_image", title="Original Image, Actual: " + str(labels[ind].cpu()) + " Predicted: " + str(classes[predicted[0]]))
    
        
    fig2, _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                  show_colorbar=True, title="Overlayed Gradient Magnitudes (Saliency)")
        
    fig3, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                                  show_colorbar=True, title="Overlayed Integrated Gradients")
        
    fig4, _ = viz.visualize_image_attr(attr_ig_nt, original_image, method="blended_heat_map", sign="absolute_value", 
                                     outlier_perc=10, show_colorbar=True, 
                                     title="Overlayed Integrated Gradients \n with SmoothGrad Squared")
        
    fig5, _ = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map",sign="all",show_colorbar=True, 
                                  title="Overlayed DeepLift")
        
    fig6, _  = viz.visualize_image_attr(attributions_occ,
                                              original_image,
                                              method="blended_heat_map",
                                              title="occlusion",
                                              sign="positive",
                                              show_colorbar=True,
                                              outlier_perc=2,
                                             )
    
    
    
    path = "/Volumes/Passport/ResearchDataChen/Code/analysis2/" + str(ind)
    if not os.path.exists(path):
        os.makedirs(path)
    
    fig1.savefig(path + "/OriginalImage.png")
    fig2.savefig(path + "/OverlayedGradientMagnitudes.png")
    fig3.savefig(path + "/OverlayedIntegratedGradients.png")
    fig4.savefig(path + "/OverlayedIntegratedGradientsWithSmoothGradSquared.png")
    fig5.savefig(path + "/OverlayedDeepLift.png")
    fig6.savefig(path + "/Occlusion.png")