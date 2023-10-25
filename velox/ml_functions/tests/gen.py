import numpy as np
# Define the dimensions
height = 768
width = 1
channels = 500
# Create a random 4D NumPy array with the specified dimensions
image = np.random.rand(height, width, channels)
# Flatten each channel and write to a text file
with open('bert_input.txt', 'a') as file:
    for channel in range(channels):
        flattened_channel = image[:, :, channel].flatten()
        channel_data = ','.join(map(str, flattened_channel))
        file.write(channel_data + '\n')
print("Channels written to 'bert_input.txt'")