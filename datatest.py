#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 20:10:23 2022

@author: pauloguedes
"""

import os
import pandas as pd
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.io import read_image
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import cv2
from time import time
from torch.utils.data import DataLoader




class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=6, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 3840, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=8)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#########DATA#############
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        out = self.img_labels.iloc[idx, 1:9]
        out = torch.tensor([out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, out






"""
training_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train')
test_data = CustomImageDataset(annotations_file='dataset/test/test.csv',img_dir = 'dataset/test')
trainloader = DataLoader(training_data, batch_size=8, shuffle=True)
testloader = DataLoader(test_data, batch_size=1, shuffle=False)


criterion = nn.L1Loss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet().to(device=device)
print(device)
optimizer = optim.Adam(model.parameters(), lr= 100e-8)
time0 = time()
epochs = 5000
for e in range(epochs):
    running_loss = 0
    for batch_idx, (images, labels) in enumerate(trainloader):
        images = images.float()
        images = images.to(device=device)
        labels = labels.to(device=device)
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()

    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
        
        if e == 50:
            torch.save(model,"model/test_train-50.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        elif e == 100:
            torch.save(model,"model/test_train-100.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        
        elif e == 250:
        
            torch.save(model,"model/test_train-250.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))

        elif e == 500:
            torch.save(model,"model/test_train-500.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
            
        elif e == 1000:
            torch.save(model,"model/test_train-1000.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        elif e == 2000:
            lr= 10e-8
            torch.save(model,"model/test_train-2000.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        elif e == 3000:
            torch.save(model,"model/test_train-3000.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        elif e == 4000:
            lr= 1e-8
            torch.save(model,"model/test_train-4000.pth")
    
print("\nTraining Time (in minutes) =",(time()-time0)/60)

for batch_idx, (data,targets) in enumerate(testloader):
        data = data.float()
        data = data.to(device=device)
        targets = targets.to(device=device)
        loss = criterion(output, labels)
        running_loss += loss.item()
        print("TEST LOSS:")
        print(running_loss/len(testloader))
        
#torch.save(model, "model/teste1.pth")
torch.save(model,"model/test_train-5000.pth")
"""