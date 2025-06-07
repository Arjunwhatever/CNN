import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# Redefine the model architecture (same class as before)
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

# Create an instance and load weights
model = CNN(num_classes=10)
model.load_state_dict(torch.load('cnn_model.pth'))
model.eval()

classes = ['Airplane', 'Car', 'Bus', 'Truck', 'Motorcycle', 'Ship', 'Bird', 'Cat', 'Dog', 'Horse']



# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
img = Image.open("A:\whatevervsc\\birb.jpg")
img_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, 32, 32]

with torch.no_grad():
    output = model(img_tensor)
    predicted = output.argmax(1).item()
    print("Predicted class:", classes[predicted])
