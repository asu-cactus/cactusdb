#!/usr/bin/env python
# coding: utf-8


import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx
import ast



class SimpleNN(nn.Module):
    def __init__(self, num_unique_customer, embedding_dim, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(num_unique_customer, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + input_size, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, output_size)  # Output size 2 for binary classification
        #self._initialize_weights()

    def forward(self, x_customer, x_other):
        embedded_customer = self.embedding(x_customer)
        x = torch.cat((embedded_customer, x_other), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)



# Load DNN model
def load_dnn_model(num_unique_customer, embedding_dim, input_size, output_size, model_path):
    model = SimpleNN(num_unique_customer, embedding_dim, input_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model



# Define your PostgreSQL connection details
username = 'postgres'       # Replace with your username
password = 'postgres'               # Replace with your password (if any)
hostname = 'localhost'      # Replace with your hostname
port = '5432'               # Replace with your port
database_name = 'postgres'  # Replace with your database name



daysOfWeek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
secondsInADay = 86400
maxDayValue = 15340.0

# Define the function to get order features
def get_order_feature(row):
    date_str = row['o_date']
    weekday = row['weekday']
    
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    dtDays = float(int(dt.timestamp()) // secondsInADay) / maxDayValue
    outputVec = [dtDays]

    for day in daysOfWeek:
        outputVec.append(1.0 if weekday == day else 0.0)
    
    return outputVec



def concatenate_features(array1, array2):
    return array1 + array2



def convert_string_to_float_list(x):
    try:
        # Split the string by comma and convert each value to float
        return [float(i.strip()) for i in x.strip('{}').split(',')]
    except ValueError:
        return []



# Connect to the PostgreSQL database
try:
    connection = psycopg2.connect(
        user=username,
        password=password,
        host=hostname,
        port=port,
        database=database_name
    )
    cursor = connection.cursor()

    cursor.execute("DROP FUNCTION IF EXISTS is_popular_store(FLOAT[]);")

    # Define the SQL for creating the function
    create_function_sql = """
    CREATE OR REPLACE FUNCTION is_popular_store(store_feature FLOAT[])
RETURNS INTEGER AS $$
DECLARE
    popularity FLOAT;
BEGIN
    IF array_length(store_feature, 1) <= 1 THEN
        RETURN 0;  -- Handle edge case for arrays with 1 or 0 elements
    END IF;

    SELECT (SUM(value) - store_feature[1]) / (array_length(store_feature, 1) - 1) INTO popularity
    FROM unnest(store_feature) AS value;

    IF popularity >= 0.4 THEN
        RETURN 1;
    ELSE
        RETURN 0;
    END IF;
END;
$$ LANGUAGE plpgsql;

    """

    # Execute the SQL command to create the function
    cursor.execute(create_function_sql)
    connection.commit()  # Commit the changes to the database

    print("Function 'is_popular_store' created successfully.")

except Exception as e:
    print(f"Error creating function: {e}")

finally:
    # Close the database connection
    if connection:
        cursor.close()
        connection.close()



# Create the connection URL
connect_url = f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}"



tStart = time.time()


table_name1 = "order_data"
table_name2 = "store_data"
join_query = f"SELECT * FROM {table_name1} JOIN {table_name2} ON o_store = s_store WHERE weekday != 'Sunday' AND is_popular_store(store_feature::FLOAT[]) = 1"
joinDf = cx.read_sql(connect_url, join_query)
joinDf.head()



joinDf['order_feature'] = joinDf.apply(get_order_feature, axis=1)
#joinDf['store_feature'] = joinDf['store_feature'].apply(lambda x: [float(i) for i in x] if isinstance(x, (list, set)) else [])
joinDf['store_feature'] = joinDf['store_feature'].apply(convert_string_to_float_list)
joinDf['all_feature'] = joinDf.apply(lambda row: concatenate_features(row['order_feature'], row['store_feature']), axis=1)
joinDf = joinDf.drop(columns=['weekday', 'o_date', 'o_store', 's_store', 'order_feature', 'store_feature'])
joinDf.head()



model = load_dnn_model(70710, 16, 77, 1000, "resources/model/trip_type_classify.pth")
model.eval()



X_embed = joinDf['o_customer_sk'].values
#X = joinDf['all_feature'].values
X = np.array(joinDf['all_feature'].tolist(), dtype=np.float32)

X_embed_tensor = torch.tensor(X_embed, dtype=torch.long)  # For embedding
X_tensor = torch.tensor(X, dtype=torch.float32)

batch_size = 32  # or use 8
test_data = TensorDataset(X_embed_tensor, X_tensor)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)



predictions = []
for X_embed_batch, X_batch in test_loader:
    outputs = model(X_embed_batch, X_batch)
    _, predicted = outputs.max(1)  # Convert probabilities to binary predictions
    predictions.extend(predicted.tolist())
joinDf['predicted_trip_type'] = predictions
joinDf = joinDf.drop(columns=['o_customer_sk', 'all_feature'])
print(joinDf.head())
print(joinDf.count())


tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")

