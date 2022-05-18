from datatest import *
from torchsummary import summary



PARAMS = {'batch_size_train': 16,
         'batch_size_test': 1,
         'learning_rate': 100e-8,
         'epochs': 100,
         'optimizer': 'Adam',
         'network':'my_alexnet'}


training_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train')
trainloader = DataLoader(training_data, batch_size= PARAMS['batch_size_train'], shuffle=True)


criterion = nn.L1Loss() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AlexNet().to(device=device)
summary(model, (3, 1920, 1080))
print(device)
optimizer = optim.Adam(model.parameters(), lr= PARAMS["learning_rate"])
time0 = time()

for e in range(PARAMS["epochs"]):
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
        
print("\nTraining Time (in minutes) =",(time()-time0)/60)

        