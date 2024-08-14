import torch
import torch.nn  as  nn  
import torch.optim as  optim
from    torchvision import datasets,  transforms
import seaborn as  sns   
import matplotlib.pyplot as  plt
from    torch.utils.data import DataLoader
import os
import PIL as Image
import torch.nn.functional as F
import group_3 as g3
# Custom dataset class
class customDataSet(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transforms.Compose([
         transforms.ToTensor()
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
    data_path = ['data/normal','data/training', 'data/validation']
    # Load the data
    training_data_loader = DataLoader(customDataSet(data_path[1],batch_size=32, shuffle=True))
    validation_data_loader = DataLoader(customDataSet(data_path[2],batch_size=32, shuffle=True))
    return training_data_loader, validation_data_loader
model = g3.group_3()
def train(model, train_loader, val_loader, num_epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)   
    train_loss = []
    val_loss = []
    best_loss = float('inf')
    epochs_WI = 0
    patience = 3
    for epoch in range(num_epochs):
        if epochs_WI >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))
        print(f'Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}')
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
            val_loss.append(running_loss / len(val_loader))
            
            print(f'Epoch {epoch + 1}, Validation Loss: {running_loss / len(val_loader)}')
            if running_loss < best_loss:
                best_loss = running_loss
                epochs_WI = 0
            else:
                epochs_WI += 1
    return train_loss, val_loss
def saveModel(model,train_loss,val_loss):
    accuracy = 100 * train_loss/val_loss
    if(accuracy < 90):
        print(f'Accuracy is less than 90%: {accuracy}')
        return
    torch.save(model.state_dict(), f'model_{accuracy}.pth')
def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
def plotLossCurve(train_loss, val_loss):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.xlabel('Epoch')
    ax.ylabel('Loss')
    ax.set_title('Loss Curve')
    ax.legend()
    ax.show()