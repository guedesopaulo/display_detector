#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:12:12 2022

@author: pauloguedes
"""
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import torchvision.transforms.functional as fn
#from alexnet import *
from datatest import *



"""
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
model = torch.load("model/run1-2000.pth")
#model.eval()

test_data = CustomImageDataset(annotations_file='dataset/test/test.csv',img_dir = 'dataset/test')
testloader = DataLoader(test_data, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('dataset/test/test.csv')
acc_t = 0
for batch_idx, (data,targets) in enumerate(testloader):
    with torch.no_grad():
        data = data.float()
        data = data.to(device=device)
        targets = targets.to(device=device)
        points = model(data)
        acc = torch.pow(abs(targets-points),2)
        acc_1 =  torch.sqrt(acc[0][0] + acc[0][1])
        acc_2 =  torch.sqrt(acc[0][2] + acc[0][3])
        acc_3 =  torch.sqrt(acc[0][4] + acc[0][5])
        acc_4 =  torch.sqrt(acc[0][6] + acc[0][7])
        acc = (acc_1 + acc_2 + acc_3 + acc_4)/4
        print(acc)
        acc_t += acc
        points = points.to(device=("cpu"))
        points = points.numpy()
        points = points.astype(int)
        img = cv2.imread("dataset/test/" + df["0"][batch_idx])
        
        V1 = (points[0][0],points[0][1])
        V2 = (points[0][2],points[0][3])
        V3 = (points[0][4],points[0][5])
        V4 = (points[0][6],points[0][7])
        radius = 10
        color = (0,0,255)
        thickness = -1
        img = cv2.circle(img, V1, radius, color, thickness)
        img = cv2.circle(img, V2, radius, color, thickness)
        img = cv2.circle(img, V3, radius, color, thickness)
        img = cv2.circle(img, V4, radius, color, thickness)
        b,g,r = cv2.split(img)
        img = cv2.merge((b,g,r))
        #plt.imshow(img)
        cv2.imwrite(str(batch_idx) + ".jpg",img)
        

acc_t = acc_t.item()
print(acc_t/len(testloader))



        
        


#v2.imshow('a', img)
        

        
        