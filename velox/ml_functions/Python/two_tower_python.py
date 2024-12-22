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
#from pyspark.sql.functions import pandas_udf
#from pyspark.sql.types import ArrayType, FloatType
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx
import ast
import xgboost as xgb



class CustomerEncoder(nn.Module):
    def __init__(self, input_size, num_customer, num_addr, num_age, num_country, customer_emb_dim, addr_emb_dim,
                 age_emb_dim, country_emb_dim):
        super(CustomerEncoder, self).__init__()
        self.emb_customer = nn.Embedding(num_customer, customer_emb_dim)
        self.emb_addr = nn.Embedding(num_addr, addr_emb_dim)
        self.emb_age = nn.Embedding(num_age, age_emb_dim)
        self.emb_country = nn.Embedding(num_country, country_emb_dim)
        self.fc1 = nn.Linear(input_size + customer_emb_dim + addr_emb_dim + age_emb_dim + country_emb_dim, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x_customer, x_addr, x_age, x_country, x_other):
        e_customer = self.emb_customer(x_customer)
        e_addr = self.emb_addr(x_addr)
        e_age = self.emb_age(x_age)
        e_country = self.emb_country(x_country)
        x = torch.cat((e_customer, e_addr, e_age, e_country, x_other), dim=1)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.relu(self.batch_norm3(self.fc3(x)))
        return x


class ProductEncoder(nn.Module):
    def __init__(self, input_size, num_product, num_dept, product_emb_dim, dept_emb_dim):
        super(ProductEncoder, self).__init__()
        self.emb_product = nn.Embedding(num_product, product_emb_dim)
        self.emb_dept = nn.Embedding(num_dept, dept_emb_dim)
        self.fc1 = nn.Linear(input_size + product_emb_dim + dept_emb_dim, 48)
        self.batch_norm1 = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 32)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x_product, x_dept, x_other):
        e_product = self.emb_product(x_product)
        e_dept = self.emb_dept(x_dept)
        x = torch.cat((e_product, e_dept, x_other), dim=1)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.relu(self.batch_norm3(self.fc3(x)))
        return x



class FraudDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)




# Load DNN model 1
def load_customer_model(input_size, num_customer, num_addr, num_age, num_country, customer_emb_dim, addr_emb_dim,
                 age_emb_dim, country_emb_dim, model_path):
    model = CustomerEncoder(input_size, num_customer, num_addr, num_age, num_country, customer_emb_dim, addr_emb_dim,
                 age_emb_dim, country_emb_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    return model



# Load DNN model 2
def load_product_model(input_size, num_product, num_dept, product_emb_dim, dept_emb_dim, model_path):
    model = ProductEncoder(input_size, num_product, num_dept, product_emb_dim, dept_emb_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
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

def get_age(birth_year):
    age = 2024 - birth_year
    if age < 0 or age > 94:
        age = 0
    return age



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

def vector_add(array1, array2):
    result = [a + b for a, b in zip(array1, array2)]
    return result



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



# Create the connection URL
connect_url = f"postgresql://{username}:{password}@{hostname}:{port}/{database_name}"
engine = create_engine(connect_url)
#connection = engine.connect()



customer_model = load_customer_model(2, 70710, 35355, 95, 213, 16, 16, 8, 8, "resources/model/customer_encoder.pth")
product_model = load_product_model(1, 708, 46, 16, 8, "resources/model/product_encoder.pth")
customer_model.eval()
product_model.eval()


tStart = time.time()


table_name_customer = "customer_data"
table_name_product = "product_data"
table_name_rating = "rating_data"
join_query1 = f"SELECT c_customer_sk, c_current_addr_sk, c_preferred_cust_flag, c_birth_year, c_birth_country, avg_customer_rating FROM {table_name_customer} INNER JOIN (SELECT r_user_id, AVG(r_rating) AS avg_customer_rating FROM {table_name_rating} GROUP BY r_user_id HAVING AVG(r_rating) >= 4.0) AS user_rating ON {table_name_customer}.c_customer_sk = user_rating.r_user_id"
join_query2 = f"SELECT p_product_id, p_department, avg_product_rating FROM {table_name_product} INNER JOIN (SELECT r_product_id, AVG(r_rating) AS avg_product_rating FROM {table_name_rating} GROUP BY r_product_id) AS product_rating ON {table_name_product}.p_product_id = product_rating.r_product_id"
join_query = f"SELECT c_customer_sk, c_current_addr_sk, c_preferred_cust_flag, c_birth_year, c_birth_country, avg_customer_rating, p_product_id, p_department, avg_product_rating FROM ({join_query1}) AS customer_details CROSS JOIN ({join_query2}) AS product_details"
#joinDf = cx.read_sql(connect_url, join_query)
joinDf = pd.read_sql_query(join_query, engine)
#joinDf.head()



joinDf['c_age'] = joinDf.apply(lambda row: get_age(row['c_birth_year']), axis=1)
joinDf['avg_customer_rating'] = joinDf['avg_customer_rating']/5.0
joinDf['avg_product_rating'] = joinDf['avg_product_rating']/5.0
joinDf = joinDf.drop(columns=['c_birth_year'])
#joinDf.head()


X_emb_customer = joinDf['c_customer_sk'].values
X_emb_addr = joinDf['c_current_addr_sk'].values
X_emb_age = joinDf['c_age'].values
X_emb_country = joinDf['c_birth_country'].values
X1 = joinDf[['c_preferred_cust_flag', 'avg_customer_rating']].values

X_emb_customer_tensor = torch.tensor(X_emb_customer, dtype=torch.long)  # For embedding
X_emb_addr_tensor = torch.tensor(X_emb_addr, dtype=torch.long)  # For embedding
X_emb_age_tensor = torch.tensor(X_emb_age, dtype=torch.long)  # For embedding
X_emb_country_tensor = torch.tensor(X_emb_country, dtype=torch.long)  # For embedding
X1_tensor = torch.tensor(X1, dtype=torch.float32)

# Create DataLoader for batching
batch_size = 32  # or use 8
test_data = TensorDataset(X_emb_customer_tensor, X_emb_addr_tensor, X_emb_age_tensor, X_emb_country_tensor, X1_tensor)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

customer_encode = []
for X_emb_customer_batch, X_emb_addr_batch, X_emb_age_batch, X_emb_country_batch, X1_batch in test_loader:
    output = customer_model(X_emb_customer_batch, X_emb_addr_batch, X_emb_age_batch, X_emb_country_batch, X1_batch)
    customer_encode.extend(output.tolist())

joinDf['customer_encode'] = customer_encode
joinDf = joinDf.drop(columns=['c_current_addr_sk', 'c_preferred_cust_flag', 'c_age', 'c_birth_country', 'avg_customer_rating'])


X_emb_product = joinDf['p_product_id'].values
X_emb_dept = joinDf['p_department'].values
X2 = joinDf[['avg_product_rating']].values

X_emb_product_tensor = torch.tensor(X_emb_product, dtype=torch.long)  # For embedding
X_emb_dept_tensor = torch.tensor(X_emb_dept, dtype=torch.long)  # For embedding
X2_tensor = torch.tensor(X2, dtype=torch.float32)

# Create DataLoader for batching
batch_size = 32  # or use 8
test_data = TensorDataset(X_emb_product_tensor, X_emb_dept_tensor, X2_tensor)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

product_encode = []
for X_emb_product_batch, X_emb_dept_batch, X2_batch in test_loader:
    output = product_model(X_emb_product_batch, X_emb_dept_batch, X2_batch)
    product_encode.extend(output.tolist())

joinDf['product_encode'] = product_encode
joinDf = joinDf.drop(columns=['p_department', 'avg_product_rating'])

joinDf['final_encoding'] = joinDf.apply(lambda row: vector_add(row['customer_encode'], row['product_encode']), axis=1)
print(joinDf.head())
print(joinDf.count())

tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")





