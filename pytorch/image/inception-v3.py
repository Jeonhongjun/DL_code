%pylab inline
import torch
import numpy as np
import torchvision
import os

from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import torch.functional as F
import torch.nn as nn

# Data import

USE_CUDA = torch.cuda.is_available()
USE_CUDA
BATCH_SIZE = 6
FINE_TUNE = False

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
    transforms.RandomResizedCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])

train_data = datasets.ImageFolder('image/hymenoptera_data/train/', train_transform)
test_data = datasets.ImageFolder('image/hymenoptera_data/val/', test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

inp, classes = next(iter(train_loader))
title = [train_data.classes[i] for i in classes]
# Make a grid from batch
inp = torchvision.utils.make_grid(inp, nrow=8)

# device = torch.device('cuda' if USE_CUDA else "cpu")

# Visualize image

def sample_show():
    # Get a batch of training data
    inp, classes = next(iter(train_loader))
    title = [train_data.classes[i] for i in classes]
    # Make a grid from batch
    inp = torchvision.utils.make_grid(inp, nrow=8)

    figsize(10, 10)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

sample_show()

inception = models.inception_v3(pretrained=True)

# auxiliary classfier 의 사용 유무, 사용하지않으면 v2 와 같다.
inception.aux_logits = False

# requires_grad 를 false 로 둠으로써 학습이 안되게 한다.
if not FINE_TUNE:
    for parameter in inception.parameters():
        parameter.requires_grad = False

# 새로운 Fully connected layer 를 따야한다.
in_features = inception.fc.in_features
inception.fc = nn.Linear(in_features, 2)

if USE_CUDA:
    inception = inception.cuda()

# inception = inception.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer
# requires_grad 가 트루인 parameter 들이 들어갈 수 있다.
optimizer = optim.RMSprop(filter(lambda p : p.requires_grad, inception.parameters()), lr=0.001)

# Train

def train_model(model, criterion, optimizer, epochs = 30):
    for epoch in range(epochs):
        epoch_loss = 0
        for step, (inputs, y_true) in enumerate(train_loader):
            if USE_CUDA:
                x_sample = inputs.cuda()
                y_true = y_true.cuda()
                # x_sample = inputs.to(device)
                # y_true = y_true.to(device)

                x_sample, y_true = Variable(x_sample), Variable(y_true)

                # parameter 초기화
                optimizer.zero_grad()

                y_pred = inception(x_sample)
                loss = criterion(y_pred, y_true)
                loss.backward()

                optimizer.step()

                #print(loss.shape())
                _loss = loss.data[0]

                epoch_loss += _loss

        print(f'{epoch+1} epoch loss: {epoch_loss/step:.4}')

train_model(inception, criterion, optimizer)

def validate(model, epochs=1):
    model.train(False)
    n_total_correct = 0
    for step, (inputs, y_true) in enumerate(test_loader):
        if USE_CUDA:
            x_sample, y_true = inputs.cuda(), y_true.cuda()
        x_sample, y_true = Variable(x_sample), Variable(y_true)

        y_pred = model(x_sample)
        _, y_pred = torch.max(y_pred.data, 1)

        n_correct = torch.sum(y_pred == y_true.data)
        n_total_correct += n_correct

    print('accuracy:', np.array(n_total_correct)/len(test_loader.dataset))

validate(inception)
