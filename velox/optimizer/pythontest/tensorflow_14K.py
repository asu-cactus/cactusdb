import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import time

class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.fc1 = layers.Dense(1024, activation='relu', input_shape=(597540,))
        self.fc2 = layers.Dense(14588, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Step 1: Generate and save random data
data = np.random.rand(1000, 597540).astype(np.float32)
np.save('custom_data.npy', data)
model = CustomModel()
# Step 2: Create a custom dataset function
def load_custom_dataset(data_path):
    data = np.load(data_path)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset

# Step 3: Use the custom dataset function to load the data
data_path = 'custom_data.npy'
batch_size = 64
start_time = time.time()
custom_dataset = load_custom_dataset(data_path)
custom_dataset = custom_dataset.batch(batch_size)


# Step 6: Iterate over the dataset to access batches and use the loaded model
for batch in custom_dataset:
    # batch is a tensor of shape (batch_size, 597540)
    # Perform any preprocessing if needed
    
    # Forward pass through the loaded TensorFlow model
    predictions = model(batch)
    
    # Perform your operations on the predictions here
print("--- %s seconds ---" % (time.time() - start_time))
