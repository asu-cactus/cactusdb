import numpy as np
import torch
# Step 1: Generate random data
# data = np.random.rand(1000, 597540).astype(np.float32)

# Step 2: Save the data to a text file
# np.savetxt('random_data.txt', data, delimiter=' ')

# Step 3: Save the data to a npy file
# np.save('random_data.npy', data)
# Step 4: Save the data to a dwrf file
data = torch.randn(1000, 597540)
script_module = torch.jit.script(data)
file_path = 'random_data.dwrf'
torch.jit.save(script_module, file_path)

