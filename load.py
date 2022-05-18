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
#from datatest import *
from resnet18 import *




model = torch.load("model/run3-100.pth")
#model.eval()

test_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train')
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
        """
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
        """        

acc_t = acc_t.item()
print(acc_t/len(testloader))



        
        


#v2.imshow('a', img)
        

        
        