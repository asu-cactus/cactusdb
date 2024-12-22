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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx
import ast
import xgboost as xgb


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)



class FraudDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)



# Load DNN model
def load_dnn_model(num_feature, model_path):
    model = SimpleNN(num_feature)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model



# Load XGBoost model
def load_xgboost_model(model_path):
    model = xgb.Booster()
    model.load_model(model_path)
    return model


# Define your PostgreSQL connection details
username = 'postgres'       # Replace with your username
password = 'postgres'               # Replace with your password (if any)
hostname = 'localhost'      # Replace with your hostname
port = '5432'               # Replace with your port
database_name = 'postgres'  # Replace with your database name



def datetime_to_timestamp(date_str):
    # Parse the date string to a datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
    # Return the timestamp in milliseconds
    return int(dt.timestamp())
    #return int(dt.timestamp()) + 25200



def is_working_day(timestamp):
    # Convert timestamp to datetime object
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    weekday = (dt.weekday() + 1) % 7  # Monday is 1 and Sunday is 0
    if weekday == 0 or weekday == 6:
        return 0
    else:
        return 1



def age_during_transaction(timestamp, birth_year):
    year = datetime.datetime.fromtimestamp(timestamp).year  # Convert milliseconds to seconds
    return year - birth_year


def get_customer_features(address_num, cust_flag, birth_day, birth_month, birth_year, birth_country, trans_limit):
    fAddressNum = float(address_num) / 35352.0
    fCustFlag = float(cust_flag)
    fBirthDay = float(birth_day) / 31.0
    fBirthMonth = float(birth_month) / 12.0
    fBirthYear = float(birth_year) / 2002.0
    fBirthCountry = float(birth_country) / 212.0
    fTransLimit = float(trans_limit) / 16116.0
    return [fAddressNum, fCustFlag, fBirthDay, fBirthMonth, fBirthYear, fBirthCountry, fTransLimit]


def get_transaction_features(amount, timestamp):   
    fAmount = amount / 16048.0
    
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    fDay = float(dt.day) / 31.0
    fMonth = float(dt.month) / 12.0
    fYear = float(dt.year) / 2011.0
    fDayOfWeek = float((dt.weekday() + 1) % 7) / 6.0
    
    return [fAmount, fDay, fMonth, fYear, fDayOfWeek]



def concatenate_features(array1, array2):
    return array1 + array2



def predict_xgb_trans(features):
    # Convert input features to DMatrix
    dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1))
    # Perform prediction
    preds = xgb_trans.predict(dmatrix)
    return float(preds[0])



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

except Exception as e:
    print(f"Error creating function: {e}")



# Connect to the PostgreSQL database
try:

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
engine = create_engine(connect_url)
#connection = engine.connect()


xgb_trans = load_xgboost_model("resources/model/fraud_xgboost_trans_no_f_5_32.json")
model = load_dnn_model(12, "resources/model/fraud_detection.pth")
model.eval()


tStart = time.time()


table_name_account = "account_data"
table_name_customer = "customer_data"
table_name_transaction = "transaction_data"
join_query = f"SELECT c_customer_sk, c_current_addr_sk, c_preferred_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, transaction_limit, transaction_id, sender_id, amount, t_time FROM {table_name_account} INNER JOIN {table_name_customer} ON fa_customer_sk = c_customer_sk INNER JOIN {table_name_transaction} ON c_customer_sk = sender_id"
#joinDf = cx.read_sql(connect_url, join_query)
joinDf = pd.read_sql_query(join_query, engine)
#joinDf.head()


joinDf['t_timestamp'] = joinDf.apply(lambda row: datetime_to_timestamp(row['t_time']), axis=1)
joinDf['is_working_day'] = joinDf.apply(lambda row: is_working_day(row['t_timestamp']), axis=1)
joinDf = joinDf[joinDf['is_working_day'] == 1]
joinDf['age_during_transaction'] = joinDf.apply(lambda row: age_during_transaction(row['t_timestamp'], row['c_birth_year']), axis=1)
joinDf = joinDf[joinDf['age_during_transaction'] >= 18]
joinDf['is_working_day'] = joinDf.apply(lambda row: is_working_day(row['t_timestamp']), axis=1)
joinDf['transaction_feature'] = joinDf.apply(lambda row: get_transaction_features(row['amount'], row['t_timestamp']), axis=1)
joinDf = joinDf.drop(columns=['amount', 'sender_id', 't_time', 't_timestamp', 'is_working_day', 'age_during_transaction'])
#joinDf.head()


joinDf['xgb_predict'] = joinDf.apply(lambda row: predict_xgb_trans(row['transaction_feature']), axis=1)
#joinDf = joinDf[joinDf['xgb_predict'] >= 0.5]
joinDf['customer_feature'] = joinDf.apply(lambda row: get_customer_features(row['c_current_addr_sk'], row['c_preferred_cust_flag'], row['c_birth_day'], row['c_birth_month'], row['c_birth_year'], row['c_birth_country'], row['transaction_limit']), axis=1)
joinDf['all_feature'] = joinDf.apply(lambda row: concatenate_features(row['customer_feature'], row['transaction_feature']), axis=1)
joinDf = joinDf.drop(columns=['c_customer_sk', 'c_current_addr_sk', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'transaction_limit', 'customer_feature', 'transaction_feature'])
#joinDf.head()


X_data = np.array(joinDf['all_feature'].tolist(), dtype=np.float32)
#X = joinDf['all_feature'].values
#X_tensor = torch.tensor(X, dtype=torch.float32)

batch_size = 32  # or use 8
test_data = FraudDataset(X_data)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)



predictions = []
for X_batch in test_loader:
    #X_batch = torch.tensor(X_batch, dtype=torch.float32)
    #print(type(X_batch))
    outputs = model(X_batch)
    _, predicted = outputs.max(1)  # Convert probabilities to binary predictions
    predictions.extend(predicted.tolist())
joinDf['is_fraudulant'] = predictions
joinDf = joinDf[joinDf['xgb_predict'] >= 0.5]
joinDf = joinDf.drop(columns=['all_feature', 'xgb_predict'])
print(joinDf.head())
print(joinDf.count())



tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")





