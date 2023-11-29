import numpy as np

# Step 1: Generate random data
data = np.random.rand(1000, 597540).astype(np.float32)

# Step 2: Save the data to a text file
# np.savetxt('random_data.txt', data, delimiter=' ')

# Step 3: Save the data to a npy file
np.save('random_data.npy', data)

