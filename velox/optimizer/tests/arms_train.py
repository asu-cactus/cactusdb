import numpy as np
import pandas as pd

# Read the CSV file
data = pd.read_csv("./result_benchmark_arms_h_12_test.csv")
data2 = pd.read_csv("./result_benchmark_arms_v_12_test.csv")
data3 = pd.read_csv("./result_benchmark_arms_100_test.csv")
data4 = pd.read_csv("./result_benchmark_arms_10_test.csv")

# Define the choices for each feature
num_driver_choices = [1, 2, 4, 8]
num_threads_choices = [1, 2, 4, 8]
num_blocks_choices = [4, 8, 16]
batch_size_choices = [100, 1000, 10000, 100000]


# Determine the number of arms (combinations of feature choices)
num_arms = len(num_driver_choices) * len(num_threads_choices) * len(num_blocks_choices) * len(batch_size_choices)

# Initialize prior distributions for each arm
alpha = np.ones(num_arms) # Prior distribution parameters
beta = np.ones(num_arms)  # Prior distribution parameters

# Function to convert feature choices to arm index
def feature_choices_to_arm_index(num_driver, num_threads, num_blocks, batch_size):
    num_driver_index = num_driver_choices.index(num_driver)
    num_threads_index = num_threads_choices.index(num_threads)
    num_blocks_index = num_blocks_choices.index(num_blocks)
    batch_size_index = batch_size_choices.index(batch_size)
    return num_driver_index * len(num_threads_choices) * len(num_blocks_choices) * len(batch_size_choices) + \
           num_threads_index * len(num_blocks_choices) * len(batch_size_choices) + \
           num_blocks_index * len(batch_size_choices) + \
           batch_size_index

# Function to convert arm index to feature choices
def arm_index_to_feature_choices(arm_index):
    num_driver_index = arm_index // (len(num_threads_choices) * len(num_blocks_choices) * len(batch_size_choices))
    arm_index %= (len(num_threads_choices) * len(num_blocks_choices) * len(batch_size_choices))
    num_threads_index = arm_index // (len(num_blocks_choices) * len(batch_size_choices))
    arm_index %= (len(num_blocks_choices) * len(batch_size_choices))
    num_blocks_index = arm_index // len(batch_size_choices)
    batch_size_index = arm_index % len(batch_size_choices)
    return num_driver_choices[num_driver_index], num_threads_choices[num_threads_index], \
           num_blocks_choices[num_blocks_index], batch_size_choices[batch_size_index]

# Function to update the chosen arm's distribution based on observed latency
def update(arm, latency):
    if latency is not None and not str:
        reward = 1 / latency  # Convert latency to reward, lower latency -> higher reward
        alpha[arm] += reward
        beta[arm] += 1

# Function to choose an arm (sample from the updated prior distributions)
def choose_arm():
    # Sample from Beta distribution for each arm
    samples = np.random.beta(alpha, beta)
    return np.argmax(samples)

# Example function to apply the configuration in your system
def apply_configuration(num_driver, num_threads, num_blocks, batch_size):
    # Apply the configuration to your system
    # This could involve setting parameters or configurations
    pass

# Example function to measure the performance (latency) with the chosen configuration
def measure_latency(num_driver, num_threads, num_blocks, batch_size):
    # Measure the performance (e.g., latency) with the configuration
    # This could involve running some workload and measuring latency
    # Return the measured latency
    pass

# Function to perform Thompson Sampling for a new workload
def thompson_sampling_for_new_workload():
    # Step 1: Choose an arm (sample configuration)
    chosen_arm = choose_arm()
    
    # Step 2: Apply the configuration
    num_driver, num_threads, num_blocks, batch_size = arm_index_to_feature_choices(chosen_arm)
    apply_configuration(num_driver, num_threads, num_blocks, batch_size)
    
    # Step 3: Measure performance (latency)
    latency = measure_latency(num_driver, num_threads, num_blocks, batch_size)
    
    # Step 4: Update distributions based on observed latency
    update(chosen_arm, latency)

    return num_driver, num_threads, num_blocks, batch_size, latency


# training prior distribution
for index, row in data.iterrows():
    if not row.dropna().empty:  # Check if the row contains valid data
        num_driver = row['num_driver']
        num_threads = row['num_threads']
        num_blocks = row['num_blocks']
        batch_size = row['batch_size']
        latency = row['latency']
        arm = feature_choices_to_arm_index(num_driver, num_threads, num_blocks, batch_size)
        update(arm, latency)
print("data1")
for index, row in data2.iterrows():
    if not row.dropna().empty:  # Check if the row contains valid data
        num_driver = row['num_driver']
        num_threads = row['num_threads']
        num_blocks = row['num_blocks']
        batch_size = row['batch_size']
        latency = row['latency']
        arm = feature_choices_to_arm_index(num_driver, num_threads, num_blocks, batch_size)
        update(arm, latency)
print("data2")
for index, row in data3.iterrows():
    if not row.dropna().empty:  # Check if the row contains valid data
        num_driver = row['num_driver']
        num_threads = row['num_threads']
        num_blocks = row['num_blocks']
        batch_size = row['batch_size']
        latency = row['latency']
        arm = feature_choices_to_arm_index(num_driver, num_threads, num_blocks, batch_size)
        update(arm, latency)
print("data3")
for index, row in data4.iterrows():
    if not row.dropna().empty:  # Check if the row contains valid data
        num_driver = row['num_driver']
        num_threads = row['num_threads']
        num_blocks = row['num_blocks']
        batch_size = row['batch_size']
        latency = row['latency']
        arm = feature_choices_to_arm_index(num_driver, num_threads, num_blocks, batch_size)
        update(arm, latency)
print("data4")


# Example usage
num_driver, num_threads, num_blocks, batch_size, latency = thompson_sampling_for_new_workload()
print("Chosen configuration:", num_driver, num_threads, num_blocks, batch_size)
print("Latency:", latency)
