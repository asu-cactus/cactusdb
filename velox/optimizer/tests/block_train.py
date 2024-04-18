# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

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

# plt.figure(figsize=(20, 10))  # Adjust the size of the plot
# plot_tree(model, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=3)  # Limit the depth of the tree
# plt.savefig('decision_tree.jpg')  # Save the figure as a JPG file
# plt.show()


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

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# # Load the data from the CSV file
# data = pd.read_csv("result_benchmark_arms_weightsfile_h_2048_test.csv")

# # Assuming 'feature_size', 'num_sample', 'output_size', 'num_blocks', and 'latency' are the relevant columns
# X = data[['feature_size', 'num_sample', 'output_size', 'num_blocks']]
# y = data['latency']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the input features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Train a decision tree regression model
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train_scaled, y_train)

# # Predict latency for the test set
# y_pred = model.predict(X_test_scaled)

# plt.figure(figsize=(30, 20))  # Increase the size of the figure
# plot_tree(model, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10)  # Adjust the font size
# plt.savefig('decision_tree.jpg')  # Save the figure as a JPG file
# plt.show()
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")

# # Define the range of num_blocks you want to test
# num_blocks_range = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # Example range, adjust as needed

# for num_blocks in num_blocks_range:
#     # Create a new DataFrame with the current num_blocks value and specified conditions
#     input_data = pd.DataFrame({
#         'feature_size': [10000],
#         'num_sample': [1000],
#         'output_size': [2048],
#         'num_blocks': [num_blocks]
#     })

#     # Scale the input data
#     input_data_scaled = scaler.transform(input_data)

#     # Predict latency for the current num_blocks value
#     predicted_latency = model.predict(input_data_scaled)
    
#     # Print the predictions directly
#     print(f"Num Blocks: {num_blocks}, Predicted Latency: {predicted_latency[0]}")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt

# # Load the data from the CSV file
# data = pd.read_csv("test.csv")

# # Assuming 'feature_size', 'num_sample', 'output_size', 'num_blocks', and 'latency' are the relevant columns
# X = data[['feature_size', 'output_size']]
# y = data['num_blocks']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a decision tree regression model
# model = DecisionTreeRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Predict latency for the test set
# y_pred = model.predict(X_test)

# # plt.figure(figsize=(20, 10))  # Adjust the size of the plot
# # plot_tree(model, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10, max_depth=3)  # Limit the depth of the tree
# # plt.savefig('decision_tree.jpg')  # Save the figure as a JPG file
# # plt.show()


# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")

# Define the range of num_blocks you want to test

# Create a DataFrame to store the predicted latencies for different num_blocks


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import BayesianRidge  # Import BayesianRidge
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the data from the CSV file
# data = pd.read_csv("test.csv")

# # Assuming 'feature_size', 'num_sample', 'output_size', 'num_blocks', and 'latency' are the relevant columns
# X = data[['feature_size', 'output_size']]
# y = data['num_blocks']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Bayesian Ridge regression model
# model = BayesianRidge()  # No hyperparameters to tune in this case
# model.fit(X_train, y_train)

# # Predict latency for the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load the data from the CSV file
# data = pd.read_csv("test.csv")

# # Assuming 'feature_size', 'output_size', and 'num_blocks' are all categorical variables
# X = data[['feature_size', 'num_sample','output_size']]
# y = data['num_blocks']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest Classifier model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy}")




# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# # Load the data from the CSV file
# data = pd.read_csv("test.csv")

# # Assuming 'feature_size', 'num_sample', 'output_size', and 'num_blocks' are all categorical variables
# X = data[['feature_size', 'output_size']]
# y = data['num_blocks']

# # Encoding categorical features in X
# label_encoder_X = LabelEncoder()
# X_encoded = X.apply(label_encoder_X.fit_transform)

# # Encoding target variable y
# label_encoder_y = LabelEncoder()
# y_encoded = label_encoder_y.fit_transform(y)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# # Train a Decision Tree Classifier model
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy}")


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor  # Change to Gradient Boosting Regressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler

# # Load the data from the CSV file
# data = pd.read_csv("test6.csv")

# # Assuming 'feature_size', 'output_size', and 'num_blocks' are all categorical variables
# X = data[['feature_size', 'output_size', 'num_sample']]
# y = data['num_blocks']

# # Create a new feature 'feature_output_product' which is the product of 'feature_size' and 'output_size'
# X['feature_output_product'] = X['feature_size'] * X['output_size']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Gradient Boosting Regressor model
# model = GradientBoostingRegressor(random_state=42)  # Change to Gradient Boosting Regressor
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {r2}")


#contains 'num_sample' acc = 0.86
#no acc = 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data from the CSV file
data = pd.read_csv("test6.csv")

# Assuming 'feature_size', 'output_size', and 'num_blocks' are all categorical variables
X = data[['feature_size', 'output_size', 'num_sample']]
# X = data[['feature_size', 'output_size']]
y = data['num_blocks']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.metrics import accuracy_score

# # Load the data from the CSV file
# data = pd.read_csv("test6.csv")

# # Assuming 'feature_size', 'output_size', and 'num_blocks' are all categorical variables
# X = data[['feature_size', 'output_size', 'num_sample']]
# y = data['num_blocks']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train an ExtraTreesClassifier model
# model = ExtraTreesClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy}")


























