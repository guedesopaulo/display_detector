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
import torchvision.transforms.functional as fn
from torchvision import transforms
from torchvision.utils import save_image
import torch.optim as optim
import cv2
from time import time
from torch.utils.data import DataLoader


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.batch2d_1 = nn.BatchNorm2d(96)
        self.batch2d_2 = nn.BatchNorm2d(256)
        self.batch2d_3 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 10752, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=8)



    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.batch2d_1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.batch2d_2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = self.batch2d_3(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        #x = F.dropout(x, 0.5)
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
        image = fn.resize(image, size=[227,277])
        label = self.img_labels.iloc[idx, 1]
        out = self.img_labels.iloc[idx, 1:9]
        out = torch.tensor([out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]])
        #print (a)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, out




"""

PARAMS = {'batch_size_train': 8,
         'batch_size_test': 50,
         'learning_rate': 100e-8,
         'optimizer': 'Adam'}

#run['parameters'] = PARAMS






test_data = CustomImageDataset(annotations_file='dataset/test/test.csv',img_dir = 'dataset/test')


training_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train')
trainloader = DataLoader(training_data, batch_size= PARAMS['batch_size_train'], shuffle=True)


def mean_std(loader):
  sum, squared_sum, num_batches = 0,0,0
  for data,_ in loader:
    data = data.float()
    sum += torch.mean(data,dim=[0,2,3])
    squared_sum += torch.mean(data**2,dim=[0,2,3])
    num_batches += 1
  mean = sum/num_batches
  std = (squared_sum/num_batches - mean**2)**0.5
  return mean, std

mean,std = mean_std(trainloader)
transform1 = transforms.Compose([
          transforms.Normalize((mean),(std))
          ])
training_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train', transform=transform1)
trainloader = DataLoader(training_data, batch_size= PARAMS['batch_size_train'], shuffle=True)
training_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train',transform = transform1)


testloader = DataLoader(test_data, batch_size=PARAMS['batch_size_test'], shuffle=False)
criterion = nn.L1Loss()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet().to(device=device)
print(device)
optimizer = optim.Adam(model.parameters(), lr= 100e-8)
time0 = time()
epochs = 3000




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
        #run['metrics/train_loss'].log(running_loss/len(trainloader))
        
        if e == 25:
            torch.save(model,"model/alexnet-25.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        elif e == 50:
            torch.save(model,"model/alexnet-50.pth")
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
        
            torch.save(model,"model/alexnet-100.pth")
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
            torch.save(model,"model/alexnet-250.pth")
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
            torch.save(model,"model/alexnet-500.pth")
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
            lr= 10e-8
            torch.save(model,"model/alexnet-1000.pth")
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
            lr= 1e-8
            torch.save(model,"model/alexnet-2000.pth")
            for batch_idx, (data,targets) in enumerate(testloader):
                    data = data.float()
                    data = data.to(device=device)
                    targets = targets.to(device=device)
                    loss = criterion(output, labels)
                    running_loss += loss.item()
            else:
                
                print("TEST LOSS:")
                print(running_loss/len(testloader))
        
    #num_correct = 0
    #num_samples = 0
    
print("\nTraining Time (in minutes) =",(time()-time0)/60)

for batch_idx, (data,targets) in enumerate(testloader):
        data = data.float()
        data = data.to(device=device)
        targets = targets.to(device=device)
        loss = criterion(output, labels)
        running_loss += loss.item()
else:
    
    print("TEST LOSS:")
    print(running_loss/len(testloader))
        

torch.save(model,"model/alexnet-3000.pth")

#run.stop()
"""