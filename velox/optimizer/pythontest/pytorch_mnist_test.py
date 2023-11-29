import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import time
from torch.utils.data import Subset

device = 'cpu'

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

model = DNN().to(device)
model.load_state_dict(torch.load('mnist_dnn_model.pth'))
#model = torch.compile(model)

start = time.time()
batch_size = 200
transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x / 255.0)])
train_dataset = MNIST(root='./data', train=False, transform=transform)
train_subset_indices = range(1000)
train_dataset = Subset(train_dataset, train_subset_indices)
print(len(train_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)


with torch.no_grad():
    for images, labels in train_loader:
        None
        # images = images.to(device)
        # outputs = model(images)
    
print("--- %s seconds ---" % (time.time() - start))