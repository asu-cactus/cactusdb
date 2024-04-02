# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the data from the CSV file
# data = pd.read_csv("result_benchmark_arms_weightsfile_h_2048_test.csv")

# # Assuming 'feature_size', 'num_sample', 'output_size', 'num_blocks', and 'latency' are the relevant columns
# X = data[['feature_size', 'num_sample', 'output_size', 'num_blocks']]
# y = data['latency']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a decision tree regression model
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Predict latency for the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")

# # Define the range of num_blocks you want to test
# num_blocks_range = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # Example range, adjust as needed

# # Create a DataFrame to store the predicted latencies for different num_blocks
# predictions = pd.DataFrame(columns=['num_blocks', 'predicted_latency'])

# for num_blocks in num_blocks_range:
#     # Create a new DataFrame with the current num_blocks value and specified conditions
#     input_data = pd.DataFrame({
#         'feature_size': [10000],
#         'num_sample': [1000],
#         'output_size': [2048],
#         'num_blocks': [num_blocks]
#     })

#     # Predict latency for the current num_blocks value
#     predicted_latency = model.predict(input_data)
    
#     # Append the result to the predictions DataFrame
#     print(f"Num Blocks: {num_blocks}, Predicted Latency: {predicted_latency[0]}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the CSV file
data = pd.read_csv("result_benchmark_arms_weightsfile_h_2048_test.csv")

# Assuming 'feature_size', 'num_sample', 'output_size', 'num_blocks', and 'latency' are the relevant columns
X = data[['feature_size', 'num_sample', 'output_size', 'num_blocks']]
y = data['latency']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a decision tree regression model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

# Predict latency for the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Define the range of num_blocks you want to test
num_blocks_range = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # Example range, adjust as needed

for num_blocks in num_blocks_range:
    # Create a new DataFrame with the current num_blocks value and specified conditions
    input_data = pd.DataFrame({
        'feature_size': [10000],
        'num_sample': [1000],
        'output_size': [2048],
        'num_blocks': [num_blocks]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict latency for the current num_blocks value
    predicted_latency = model.predict(input_data_scaled)
    
    # Print the predictions directly
    print(f"Num Blocks: {num_blocks}, Predicted Latency: {predicted_latency[0]}")
