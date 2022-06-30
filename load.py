#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:12:12 2022

@author: pauloguedes
"""


# USE JUST ONE
#from alexnet import *
#from displaynet import *
from resnet18 import *



model = torch.load("models/resnet50.pth")
#model.eval()

test_data = CustomImageDataset(annotations_file='dataset/test/test.csv',img_dir = 'dataset/test')
testloader = DataLoader(test_data, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('dataset/test/test.csv')
acc_t = 0
acc_list = []

# USE JUST ONE
resize_vector =torch.Tensor([8.57,4.82,8.57,4.82,8.57,4.82,8.57,4.82]).to(device=device) #resize vector for resnet
#resize_vector = torch.Tensor([8.458,4.758,8.458,4.758,8.458,4.758,8.458,4.758]).to(device=device) #resize vector for alexnet
#resize_vector = torch.Tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]).to(device=device) #resize vector for displaynet

for batch_idx, (data,targets) in enumerate(testloader):
    with torch.no_grad():
        data = data.float()
        data = data.to(device=device)
        targets = targets.to(device=device)
        points = model(data)
        points = torch.mul(resize_vector,points)
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
              


#df_acc.hist(column=0,bins=50, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

        

        
        