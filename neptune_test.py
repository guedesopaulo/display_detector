import neptune.new as neptune

#Import just one network
#from datatest import *
#from alexnet import *
from resnet18 import *
from torchsummary import summary

run = neptune.init(project = "pog/TCC",
                   api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZGMwODI3OC0yYWI3LTQ5ZmYtOGM5MS1hY2ZkYmE0ZDNhMWUifQ==",
                   name= "resnet",
                   source_files = ['*.py'])


PARAMS = {'batch_size_train': 16,
         'batch_size_test': 1,
         'learning_rate': 1e-3,
         'epochs': 2000,
         'optimizer': 'Adam',
         'network':'resnet18'}

run['parameters'] = PARAMS

training_data = CustomImageDataset(annotations_file='dataset/train/train.csv',img_dir = 'dataset/train')
test_data = CustomImageDataset(annotations_file='dataset/test/test.csv',img_dir = 'dataset/test')
trainloader = DataLoader(training_data, batch_size= PARAMS['batch_size_train'], shuffle=True)
testloader = DataLoader(test_data, batch_size=PARAMS['batch_size_test'], shuffle=False)

criterion = nn.L1Loss() 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Use just one netwoek

#Resnet
model = ResNet(3, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=8).to(device=device)
summary(model, (3, 224, 224))

#Alexnet
#model = AlexNet().to(device=device)

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
        run['train_loss_epoch'].log(running_loss/len(trainloader))

        if e%10 == 0:
            acc_t = 0
            for batch_idx, (data,targets) in enumerate(testloader):
                data = data.float()
                data = data.to(device=device)
                targets = targets.to(device=device)
                loss = criterion(output, labels)
                running_loss += loss.item()
                points = model(data)
                acc = torch.sum(abs(targets-points))
                acc_t += acc
            else:
                    
                print("TEST LOSS:")
                print(running_loss/len(testloader))
                run['test_loss_10epochs'].log(running_loss/len(testloader))
                print("Pixel Acurracy:")   
                acc_t = acc_t.item()
                print(acc_t/(len(testloader)*4))
                run['acc_test_10epochs'].log(acc_t/(len(testloader)*4))
            
        if e == 25:
            torch.save(model,"model/run3-25.pth")


        elif e == 50:
            torch.save(model,"model/run3-50.pth")

        
        elif e == 100:
            torch.save(model,"model/run3-100.pth")
            lr = 1e-4


        elif e == 250:
            torch.save(model,"model/run3-250.pth")


        elif e == 500:
            torch.save(model,"model/run3-500.pth")

        
        elif e == 1000:
            lr= 1e-5
            torch.save(model,"model/run3-1000.pth")

        elif e == 1500:
            lr = 1e-6
print("\nTraining Time (in minutes) =",(time()-time0)/60)

        

torch.save(model,"model/run3-2000.pth")

run.stop()