#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:00:00 2021

@author: rodrigosandon
"""

import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

import resNetCifar10Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_classes():
    classes = [
        "visp",
        "visal",
        "visrl",
        "vispm",
        "visam",
        "visl",
    ]
    return classes


def get_pretrained_model():
    model = resNetCifar10Model.ResNet18()
    model.load_state_dict(torch.load("resnet18_w_kappa.pt"))
    model.eval()
    model.to(device)
    return model


def baseline_func(input):
    return input * 0


def formatted_data_iter():
    dataset = torchvision.datasets.CIFAR10(
        root="data/test", train=False, download=True, transform=transforms.ToTensor()
    )
    dataloader = iter(
        torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    )
    while True:
        images, labels = next(dataloader)
        yield Batch(inputs=images, labels=labels)


normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
model = get_pretrained_model()
visualizer = AttributionVisualizer(
    models=[model],
    score_func=lambda o: torch.nn.functional.softmax(o, 1),
    classes=get_classes(),
    features=[
        ImageFeature(
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[normalize],
        )
    ],
    dataset=formatted_data_iter(),
)

visualizer.render()

# show a screenshot if using notebook non-interactively
from IPython.display import Image
Image(filename='img/captum_insights.png')