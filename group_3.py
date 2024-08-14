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
input_dim = (50,50)
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

