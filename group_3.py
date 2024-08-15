import torch
import torch.nn  as  nn  
import torch.optim as  optim
from    torchvision import datasets,  transforms
import seaborn as  sns   
import matplotlib.pyplot as  plt   # Download the MNIST dataset
from    torch.utils.data import DataLoader
import os
import PIL as Image
import torch.nn.functional as F
# Size 50x50 for images
input_dim = (400,400)
#Grayscaled images have 1 channel, which is fine for the purpose of deteting pneumonia in the x-ray images
channels = 1
class group_3(nn.Module):
    def __init__(self):
        super(group_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 10 * 10, 128)
        self.fc2 = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 10 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = "cpu" if not torch.cuda.is_available() else "cuda"

class CNN(nn.Module): #nn.Module base class for all neural network modules
    def __init__(self): # initialize the weights, bias, etc.
        super(CNN, self).__init__() # initialize the parent class (nn.Module)
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3)
        self.maxpool2=nn.MaxPool2d(kernel_size=2) # using ReLU activation function
        self.conv3=nn.Conv2d(in_channels=32,out_channels=10,kernel_size=3)
        #self.softmax=nn.Softmax(dim=1)

        self.fc1=nn.Linear(10*3*3,128)
        self.dropout=nn.Dropout(0.2)
        self.fc2=nn.Linear(128,10)
        self.relu=nn.ReLU()

    def forward(self,x): #passes the data through the network. Default built-in function
        x=self.relu(self.conv1(x))
        x=self.maxpool1(x)
        x=self.relu(self.conv2(x))
        x=self.maxpool2(x)
        x=self.relu(self.conv3(x))
        x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x) #soft max makes performance worse here 
        return x
class CNN_skip_connections(nn.Module): #nn.Module base class for all neural network modules
    def __init__(self): # initialize the weights, bias, etc.
        super(CNN, self).__init__() # initialize the parent class (nn.Module)
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1) # 3x3 is for the more complex features and edges
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2) #5x5 for the more simple features. Padding of 1 and 2 used respectively as 
        self.maxpool=nn.MaxPool2d(kernel_size=2) # maxpool pools the values in a 2x2 area of the image and takes the maximum value
        self.conv3=nn.Conv2d(in_channels=64+128,out_channels=16,kernel_size=3,padding=1)
        #self.softmax=nn.Softmax(dim=1)

        self.fc1=nn.Linear((16)*14*14,128)
        self.dropout=nn.Dropout(0.2)
        self.fc2=nn.Linear(128,10)
        self.relu=nn.ReLU()

    def forward(self,x): #passes the data through the network. Default built-in function
        x=self.relu(self.conv1(x))
        x1=self.relu(self.conv2(x))
        #print(x1.shape)
        x=torch.cat((x1, x), dim=1)
        #print(x.shape)
        x=self.maxpool(x)
        #print(x.shape)
        x=self.relu(self.conv3(x))
        #print(x.shape)
        x=x.view(x.size(0),-1)
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x) #soft max makes performance worse here 
        return x
    
