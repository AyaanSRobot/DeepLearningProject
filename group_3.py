import torch
import torch.nn  as  nn  
import torch.optim as  optim
from    torchvision import datasets,  transforms
import seaborn as  sns   
import matplotlib.pyplot as  plt   # Download the MNIST dataset
from    torch.utils.data import DataLoader
# Size 50x50 for images
input_dim = (50,50)
#Grayscaled images have 1 channel, which is fine for the purpose of deteting pneumonia in the x-ray images
channels = 1
transform = 
class customDataSet(torch.utils.data.Dataset):
    def __init__(self, path,data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        transforms.Compose(
        [transforms.Resize(input_dim),transforms.Grayscale(num_output_channels=channels),transforms.ToTensor()
])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)