import torch
import torch.nn  as  nn  
import torch.optim as  optim
from    torchvision import datasets,  transforms
import seaborn as  sns   
import matplotlib.pyplot as  plt   # Download the MNIST dataset
from    torch.utils.data import DataLoader
import os
import PIL as Image
# Size 50x50 for images
input_dim = (50,50)
#Grayscaled images have 1 channel, which is fine for the purpose of deteting pneumonia in the x-ray images
channels = 1
# Custom dataset class
class customDataSet(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transforms.Compose(
        [transforms.Resize(input_dim),transforms.Grayscale(num_output_channels=channels),transforms.ToTensor()
])
        self.data = []
        self.labels = []
        for class_label in os.listdir(path):
            class_path = os.path.join(path, class_label)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    self.data.append(os.path.join(class_path, file))
                    self.target.append(class_label)
                    label = 0
                    if class_label == 'pneumonia':
                        label = 1
                    self.labels.append(label)

    def __getitem__(self, index):
        class_path = self.data[index]
        image = Image.open(class_path)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)
def dataLoading():
    data_path = ['data/normal','data/training, data/validation']
    # Load the data
    training_data_loader = DataLoader(customDataSet(data_path[1],batch_size=32, shuffle=True))
    validation_data_loader = DataLoader(customDataSet(data_path[2],batch_size=32, shuffle=True))
    return training_data_loader, validation_data_loader
