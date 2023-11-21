import torch
import torch.nn as nn
import multiprocessing
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
# Define a custom neural network model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(597540, 1024)  # First layer: 597540 input features, 1024 output features
        self.bias1 = nn.Parameter(torch.zeros(1024))  # Bias for the first layer
        self.fc2 = nn.Linear(1024, 14588)  # Second layer: 1024 input features, 14588 output features
        self.bias2 = nn.Parameter(torch.zeros(14588))  # Bias for the second layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for classification

    def forward(self, x):
        x = self.fc1(x)
        x = x + self.bias1
        x = torch.relu(x)  # You can use another activation function if needed
        x = self.fc2(x)
        x = x + self.bias2
        x = self.softmax(x)
        return x

# Define a function for batch inference
def batch_inference_wrapper(args):
    model, input_batch = args
    with torch.no_grad():
        output_batch = model(input_batch)
    return output_batch


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx])
        return sample


if __name__ == '__main__':
    # Load the model from a file
    loaded_model = CustomModel()  # Create an instance of the custom model
    loaded_model.eval()  # Set the model to evaluation mode

    # Create random input data of shape 1000x597540 (you should replace this with your actual input data)
    # input_data = torch.randn(1000, 597540)
    data = np.random.rand(1000, 597540).astype(np.float32)
    # Save the data to a binary file
    np.save('custom_data.npy', data)
    batch_size = 1000
    data_path = 'custom_data.npy'
    start = time.time()
    custom_dataset = CustomDataset(data_path)
    data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, num_workers=1)
    
    with torch.no_grad():
        for images in data_loader:
            None
            # outputs = loaded_model(images)


    # Split the input data into batches (adjust the batch size as needed)
    # batch_size = 500
    # input_batches = torch.split(input_data, batch_size)
    # args_list = [(loaded_model, batch) for batch in input_batches]

    # Create a multiprocessing pool to parallelize inference
    # num_processes = multiprocessing.cpu_count()
    # num_processes = 1
    # pool = multiprocessing.Pool(processes=num_processes)

    # Perform batch inference in parallel

    # results = pool.map(batch_inference_wrapper, args_list)

    # Combine the results from each batch
    # output = torch.cat(results)
    print("--- %s seconds ---" % (time.time() - start))

    # Print the predictions
    # print(output)
