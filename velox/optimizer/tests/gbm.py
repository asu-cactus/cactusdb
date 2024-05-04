import pandas as pd
import numpy as np
import copy
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/train.csv')

# Separate numerical and categorical columns
numerical_cols = ['feature_size', 'num_sample', 'output_size', 'eigen_cost', 'torch_cost']
categorical_cols = ['num_driver', 'eigen_threads', 'torch_threads', 'batch_size']

# One-hot encode the categorical columns
data = pd.get_dummies(data, columns=categorical_cols)

# Split the data into features and target variable
X = data.drop('latency', axis=1)  # Drop the target column
y = data['latency']  # Extract the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.head(1))
# gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
# # Train the model
# gbm.fit(X_train, y_train)

# # Make predictions on the test set
# predictions = gbm.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, predictions)
# print("Mean Squared Error:", mse)
# joblib.dump(gbm, 'gradient_boosting_model.pkl')
gbm = joblib.load('gradient_boosting_model.pkl')
# Function to generate random combinations
def generate_random_combinations(X_test, num_samples):
    possible_categories = {
        'num_driver': [1, 2, 3, 4, 5, 6, 7, 8],
        'eigen_threads': [1, 2, 3, 4, 5, 6, 7, 8],
        'torch_threads': [1, 2, 3, 4, 5, 6, 7, 8],
        'batch_size': [100, 1000, 10000, 100000]
    }

    combinations =[]
    random_row_index = np.random.randint(len(X_test))
    for _ in range(num_samples):
        # Create a deep copy of the random_row for each iteration
        random_row = copy.deepcopy(X_test.iloc[random_row_index])
        random_row.iloc[5:] = False

        num_driver = np.random.choice(possible_categories['num_driver'], size=1)[0]
        eigen_threads = np.random.choice(possible_categories['eigen_threads'], size=1)[0]
        torch_threads = np.random.choice(possible_categories['torch_threads'], size=1)[0]
        batch_size = np.random.choice(possible_categories['batch_size'], size=1)[0]

        random_row.iloc[4+num_driver] = True
        random_row.iloc[12+eigen_threads] = True
        random_row.iloc[20+torch_threads] = True
        if batch_size == 100:
            random_row.iloc[29] = True
        elif batch_size == 1000:
            random_row.iloc[30] = True
        elif batch_size == 10000:
            random_row.iloc[31] = True
        else:
            random_row.iloc[32] = True

        combinations.append(random_row)

    return combinations

def setConfig(X_test, numbers):
    combinations =[]
    random_row_index = np.random.randint(len(X_test))
        # Create a deep copy of the random_row for each iteration
    random_row = copy.deepcopy(X_test.iloc[random_row_index])
    random_row.iloc[5:] = False
    random_row.iloc[0] = numbers[0]
    random_row.iloc[1] = numbers[1]
    random_row.iloc[2] = numbers[2]
    random_row.iloc[3] = numbers[3]
    random_row.iloc[4] = numbers[4]

    num_driver = numbers[5]
    eigen_threads = numbers[6]
    torch_threads = numbers[7]
    batch_size = numbers[8]

    random_row.iloc[4+num_driver] = True
    random_row.iloc[12+eigen_threads] = True
    random_row.iloc[20+torch_threads] = True
    if batch_size == 100:
        random_row.iloc[29] = True
    elif batch_size == 1000:
        random_row.iloc[30] = True
    elif batch_size == 10000:
        random_row.iloc[31] = True
    else:
        random_row.iloc[32] = True

    combinations.append(random_row)

    return combinations

# Generate random combinations
num_samples = 200  # Number of random combinations to generate
# for _ in range(num_samples):
random_combinations = generate_random_combinations(X_test, num_samples)

# Predict latency for each combination
latencies = []

for combination in random_combinations:
    # Reshape the combination for prediction
    combination_reshaped = combination.values.reshape(1, -1)
    # Predict latency using the trained GBM model
    latency = gbm.predict(combination_reshaped)[0]
    print(latency)
    latencies.append(latency)

# Find the combination with the lowest latency
best_index = np.argmin(latencies)
best_combination = random_combinations[best_index]
best_latency = latencies[best_index]

print("Best Combination:", best_combination)
# print("Best Latency:", best_latency)

data2 = pd.read_csv('/home/ubuntu/velox/velox/optimizer/tests/train.csv')

n1 = best_combination['feature_size']
n2 = best_combination['num_sample']
n3 = best_combination['output_size']
n4 = best_combination['eigen_cost']
n5 = best_combination['torch_cost']

filtered_rows = data2[(data2['feature_size'] == n1) & (data2['num_sample'] == n2) & (data2['output_size'] == n3)]

sorted_rows = filtered_rows.sort_values(by='latency')
print("top:")
print(sorted_rows.head(20))

print("bottom:")
print(sorted_rows.tail(20))

num=[]
for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    if best_combination['num_driver_' + str(i)] == True:
        num.append(i)
        break

for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    if best_combination['eigen_threads_' + str(i)] == True:
        num.append(i)
        break

for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    if best_combination['torch_threads_' + str(i)] == True:
        num.append(i)
        break

for i in [100, 1000, 10000, 100000]:
    if best_combination['batch_size_' + str(i)] == True:
        num.append(i)
        break

true_latency = data2[(data2['feature_size'] == n1) & (data2['num_sample'] == n2) & (data2['output_size'] == n3) & (data2['num_driver'] == num[0]) & (data2['eigen_threads'] == num[1]) & (data2['torch_threads'] == num[2]) & (data2['batch_size'] == num[3])]

print("we select:")
print(true_latency)


print("heuristic configs[8,4,4,100000]:")
true_latency = data2[(data2['feature_size'] == n1) & (data2['num_sample'] == n2) & (data2['output_size'] == n3) & (data2['num_driver'] == 8) & (data2['eigen_threads'] == 4) & (data2['torch_threads'] == 4) & (data2['batch_size'] == 100000)]
print(true_latency)

print("heuristic configs[4,4,4,10000]:")
true_latency = data2[(data2['feature_size'] == n1) & (data2['num_sample'] == n2) & (data2['output_size'] == n3) & (data2['num_driver'] == 4) & (data2['eigen_threads'] == 4) & (data2['torch_threads'] == 4) & (data2['batch_size'] == 10000)]
print(true_latency)

print("heuristic configs[1,4,1,10000]:")
true_latency = data2[(data2['feature_size'] == n1) & (data2['num_sample'] == n2) & (data2['output_size'] == n3) & (data2['num_driver'] == 1) & (data2['eigen_threads'] == 4) & (data2['torch_threads'] == 1) & (data2['batch_size'] == 10000)]
print(true_latency)


# configNum = [n1, n2, n3, n4, n5, num[0], num[1], num[2], num[3]]

# cn = setConfig(X_test, configNum)

# for c1 in cn:
#     # Reshape the combination for prediction
#     combination_reshaped = combination.values.reshape(1, -1)
#     # Predict latency using the trained GBM model
#     latency = gbm.predict(combination_reshaped)[0]
#     print(latency)