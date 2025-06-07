#Importing ofc
import torch
import torch.utils
import torch.utils.data
import torchvision 
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.datasets as datasets
from torch.optim import SGD
batch_size = 64
num_classes = 10
device = torch.device('cuda')

#Loading inbuilt dataset CIFAR10 , defining our training and testing data
all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])])

#Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

train = torchvision.datasets.CIFAR10(root= './data',
                                     train= True,
                                     transform= train_transform,
                                     download= True)


test = torchvision.datasets.CIFAR10(root = './data',
                                    train = False,
                                    transform= all_transforms,
                                    download= True)

train_loader = torch.utils.data.DataLoader(dataset= train, 
                                           batch_size = batch_size,
                                           shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset= test,
                                          batch_size= batch_size,
                                          shuffle= True)
#OUR CNN MODEL 
class CNN(nn.Module):
    
    def __init__(self, num_classes = 10):
        super(CNN, self).__init__()

        self.clayer_1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= 3, padding = 1) 
        #self.clayer_2 = nn.Conv2d(in_channels= 32, out_channels= 32, kernel_size= 3,padding = 1)
        self.max_pool1= nn.MaxPool2d(kernel_size= 2,stride = 1)

        self.clayer_3 = nn.Conv2d( in_channels= 32, out_channels= 64,kernel_size= 3,padding= 1)
        #self.clayer_4 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= 3, padding= 1)
        self.max_pool2 = nn.MaxPool2d(kernel_size= 2, stride = 2)
        
        self.fc1 = nn.Linear(14400, 128)
        self.relu = nn.ReLU()
        self.fc2 =nn.Linear(128,num_classes)

    def forward(self, x):
        out = self.clayer_1(x)
        #out = self.clayer_2(out)
        out = self.max_pool1(out)

        out = self.clayer_3(out)
        #out = self.clayer_4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


model = CNN(num_classes = 10)
model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = SGD(model.parameters(),lr= .001, weight_decay= .005, momentum= .9)

#momentum = Momentum helps smooth out the updates by adding a fraction of the previous update to the current one.
#weight_decay=  regularization technique, specifically L2 regularization. It penalizes large weight values during training by slightly shrinking them on every update.
total_step = len(train)

#Training

for epoch in range(64):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        output = model(images)
        loss = criterion(output, labels)

        #Back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(epoch, loss)
#Testing
# Evaluate on test set
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f} %'.format(100 * correct / total))

    print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))

torch.save(model.state_dict(), 'cnn_model.pth')
