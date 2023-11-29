import time
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import layers


input_dimension = 597540
batch = 1000

def parse_line(line):
    # Split the line into space-separated values and convert them to float32
    values = tf.strings.split(line, ' ')
    values = tf.strings.to_number(values, tf.float32)
    return values


# load the model
modelStart = time.time()
loaded_model = keras.models.load_model('amazon_14K.h5')
modelEnd = time.time()

# Load input from txt
dataStart = time.time()
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

# Create TensorFlow Dataset
batch_size = 1000
dataset = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
dataEnd = time.time()

inferStart= time.time()
for item in dataset:
    predictions = loaded_model(item)
inferEnd = time.time()

print ("Data Load Time: ", dataEnd - dataStart)
print ("Model Load Time:", modelEnd - modelStart)
print ("Inference Time:", inferEnd  - inferStart)