import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

input_dimension = 597540
batch_size = 1000

class Amazon14K(torch.nn.Module):
    def __init__(self):
            
            # calling constructor of parent class
            super().__init__()
            
            # defining the inputs to the first hidden layer
            self.hid1 = nn.Linear(597540, 1024, dtype=torch.double) 
            nn.init.normal_(self.hid1.weight, mean = 0, std = 2)
            nn.init.normal_(self.hid1.bias, mean = 0, std = 2)
            self.act1 = nn.ReLU()
            
            # defining the inputs to the third hidden layer
            self.hid2 = nn.Linear(1024, 14588, dtype=torch.double)
            nn.init.normal_(self.hid2.weight, mean = 0, std = 2)
            nn.init.normal_(self.hid2.bias, mean = 0, std = 2)
            self.act2 = nn.Softmax()

    def forward(self, X):
            #input and act for layer 1
            X = self.hid1(X)
            X = self.act1(X)
            
            #input and act for layer 2
            X = self.hid2(X)
            X = self.act2(X)
            
            return X

class TextLineDataset(Dataset):
    def __init__(self, file_path):
        self.lines = []
        with open(file_path, 'r') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # Convert the text line to a tensor (you may need to preprocess the line)
        line = self.lines[idx]
        tensor_line = torch.tensor(list(map(float, line.strip().split())))
        return tensor_line
    
class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
# load the model
totalStart = time.time()
modelStart = time.time()
model = torch.load('Amazon14K.pth')
modelEnd = time.time()

# Load input from txt
dataStart = time.time()
# file_path = 'random_data.txt'
# custom_dataset = TextLineDataset(file_path)
# data_loader = DataLoader(dataset=custom_dataset, batch_size=batch, num_workers=8)
# file_path = 'random_data.txt'
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # Convert text lines to a NumPy array
# data = []
# for line in lines:
#     values = list(map(float, line.strip().split()))
#     data.append(values)
# data = np.array(data, dtype=np.float32)
file_path = 'random_data.npy'
data = np.load(file_path)
data_tensor = torch.tensor(data)
custom_dataset = CustomDataset(data_tensor)

data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
dataEnd = time.time()
inferStart = time.time()
with torch.no_grad():
    for batch in data_loader:
        # None
        output = model(batch)
inferEnd = time.time()


print ("Data Load Time: ", dataEnd - dataStart)
print ("Model Load Time:", modelEnd - modelStart)
print ("Inference Time:", inferEnd - inferStart)