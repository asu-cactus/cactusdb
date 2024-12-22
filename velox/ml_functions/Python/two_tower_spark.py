#!/usr/bin/env python
# coding: utf-8

# # This Example shows the Prediction of Bike Flow in the NYC City using the deep learning model ST-ResNet.

# Find the details of the ST-ResNet model in the <a href="https://dl.acm.org/doi/10.5555/3298239.3298479">corresponding paper</a>

# Details of the dataset can be found <a href="https://github.com/FIBLAB/DeepSTN">here</a>.

# ### Import Modules and Define Parameters



import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, when, udf, max, count, trim
from pyspark.sql.types import IntegerType, ArrayType, FloatType, StringType, LongType
import datetime
import time
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
        x = torch.cat((e_customer, e_addr, e_age, e_country, x_other), dim=0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
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
        x = torch.cat((e_product, e_dept, x_other), dim=0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.relu(self.batch_norm3(self.fc3(x)))
        return x



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



spark = SparkSession.builder.appName("FraudDetectionSpark").config("spark.driver.memory", "16g").config("spark.dynamicAllocation.enabled", "true").getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "209715200")



countryDict = {}
with open('resources/data/country_mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(",")
        countryDict[key] = int(value)

deptDict = {}
with open('resources/data/department_mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split("=")
        deptDict[key] = int(value)



def mapCountry(countrStr):
    return countryDict.get(countrStr, 212)  # Returns original value if not found

# Register the UDF (specify the return type)
map_country_udf = udf(mapCountry, IntegerType())


def mapDepartment(deptStr):
    return deptDict.get(deptStr, 0)  # Returns original value if not found

# Register the UDF (specify the return type)
map_dept_udf = udf(mapDepartment, IntegerType())



def remove_quotes(numberStr):
    if len(numberStr) >= 2 and numberStr[0] == '"' and numberStr[-1] == '"':
        return numberStr[1:-1]
    return numberStr

remove_quotes_udf = udf(remove_quotes, StringType())



dfCustomer = spark.read.csv("resources/data/customer.csv", header=True, inferSchema=True)
dfCustomer = dfCustomer.drop('c_customer_id').drop('c_first_name').drop('c_last_name').drop('c_login').drop('c_email_address')
#dfCustomer = dfCustomer.dropna()

column_names = dfCustomer.columns
trimmed_columns = [trim(col(c)).alias(c) for c in column_names]
dfCustomer = dfCustomer.select(*trimmed_columns)

dfCustomer = dfCustomer.dropna(subset=['c_birth_year', 'c_birth_country'])
dfCustomer = dfCustomer.withColumn("c_customer_sk", col("c_customer_sk").cast(IntegerType())).withColumn("c_current_addr_sk", col("c_current_addr_sk").cast(IntegerType())).withColumn("c_birth_day", col("c_birth_day").cast(IntegerType())).withColumn("c_birth_month", col("c_birth_month").cast(IntegerType())).withColumn("c_birth_year", col("c_birth_year").cast(IntegerType()))
dfCustomer = dfCustomer.withColumn("c_preferred_cust_flag", when(col("c_preferred_cust_flag") == 'N', 0).otherwise(1))
dfCustomer = dfCustomer.withColumn("c_birth_country", map_country_udf(col("c_birth_country")))
dfCustomer = dfCustomer.fillna(0)
dfCustomer = dfCustomer.repartition("c_customer_sk")
dfCustomer.collect()
print("Customer table size: ", dfCustomer.count())
dfCustomer.cache()
#dfCustomer.show()


dfProduct = spark.read.csv("resources/data/product.csv", header=True, inferSchema=True)
#dfCustomer = dfCustomer.dropna()

column_names = dfProduct.columns
trimmed_columns = [trim(col(c)).alias(c) for c in column_names]
dfProduct = dfProduct.select(*trimmed_columns)

dfProduct = dfProduct.withColumn("p_product_id", col("p_product_id").cast(IntegerType()))
dfProduct = dfProduct.withColumn("p_department", map_dept_udf(col("department")))
dfProduct = dfProduct.drop('name').drop('department')
dfProduct = dfProduct.fillna(0)
dfProduct = dfProduct.repartition("p_product_id")
dfProduct.collect()
print("Product table size: ", dfProduct.count())
dfProduct.cache()


dfRating = spark.read.csv("resources/data/ProductRating.csv", header=True, inferSchema=True)
#dfTransaction = dfTransaction.dropna()

str_cols = [field.name for field in dfRating.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfRating = dfRating.withColumn(c, remove_quotes_udf(col(c)))
    
dfRating = dfRating.withColumn("r_user_id", col("userID").cast(IntegerType())).withColumn("r_product_id", col("productID").cast(IntegerType())).withColumn("r_rating", col("rating").cast(IntegerType()))
dfRating = dfRating.drop('userID').drop('productID').drop('rating')
#dfRating = dfRating.repartition("r_user_id")
dfRating.collect()
print("Rating table size: ", dfRating.count())
dfRating.cache()
#dfTransaction.show()



def get_age(birth_year):
    age = 2024 - birth_year
    if age < 0 or age > 94:
        age = 0
    return age

# Register the UDF
#get_age_udf = udf(get_age, IntegerType())
spark.udf.register("get_age_udf", get_age, IntegerType())


def vector_add(array1, array2):
    result = [a + b for a, b in zip(array1, array2)]
    return result

# Register the UDF
#vector_add_udf = udf(vector_add, LongType())
spark.udf.register("vector_add_udf", vector_add, ArrayType(FloatType()))




def concatenate_features(array1, array2):
    return array1 + array2

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("concatenate_features_udf", concatenate_features, ArrayType(FloatType()))



# Load and register DNN the model
customer_model = load_customer_model(2, 70710, 35355, 95, 213, 16, 16, 8, 8, "resources/model/customer_encoder.pth")
product_model = load_product_model(1, 708, 46, 16, 8, "resources/model/product_encoder.pth")
customer_model.eval()
product_model.eval()

# Define UDF function for customer encoding
def encode_customer(X_customer, X_addr, X_age, X_country, X_cust_flag, X_avg_rating):
    # Convert input features to tensor
    embed_tensor_customer = torch.tensor(X_customer, dtype=torch.long)
    embed_tensor_addr = torch.tensor(X_addr, dtype=torch.long)
    embed_tensor_age = torch.tensor(X_age, dtype=torch.long)
    embed_tensor_country = torch.tensor(X_country, dtype=torch.long)
    feature_tensor = torch.tensor([X_cust_flag, X_avg_rating/5.0], dtype=torch.float32)
    # Perform prediction
    result = customer_model(embed_tensor_customer, embed_tensor_addr, embed_tensor_age, embed_tensor_country, feature_tensor)
    return result[0].tolist()

# Register the UDF
#encode_customer_udf = udf(encode_customer, ArrayType(FloatType()))
spark.udf.register("encode_customer_udf", encode_customer, ArrayType(FloatType()))



# Define UDF function for customer encoding
def encode_product(X_product, X_dept, X_avg_rating):
    # Convert input features to tensor
    embed_tensor_product = torch.tensor(X_product, dtype=torch.long)
    embed_tensor_dept = torch.tensor(X_dept, dtype=torch.long)
    feature_tensor = torch.tensor([X_avg_rating/5.0], dtype=torch.float32)
    # Perform prediction
    result = product_model(embed_tensor_product, embed_tensor_dept, feature_tensor)
    return result[0].tolist()

# Register the UDF
#encode_product_udf = udf(encode_product, ArrayType(FloatType()))
spark.udf.register("encode_product_udf", encode_product, ArrayType(FloatType()))


dfCustomer.createOrReplaceTempView("df_customer")
dfProduct.createOrReplaceTempView("df_product")
dfRating.createOrReplaceTempView("df_rating")

table_name_customer = "df_customer"
table_name_product = "df_product"
table_name_rating = "df_rating"



join_query1 = f"SELECT c_customer_sk, c_current_addr_sk, get_age_udf(c_birth_year) AS c_age, c_birth_country, c_preferred_cust_flag, avg_customer_rating FROM {table_name_customer} INNER JOIN (SELECT r_user_id, AVG(r_rating) AS avg_customer_rating FROM {table_name_rating} GROUP BY r_user_id HAVING AVG(r_rating) >= 4.0) AS user_rating ON {table_name_customer}.c_customer_sk = user_rating.r_user_id"
join_query2 = f"SELECT p_product_id, p_department, avg_product_rating FROM {table_name_product} INNER JOIN (SELECT r_product_id, AVG(r_rating) AS avg_product_rating FROM {table_name_rating} GROUP BY r_product_id) AS product_rating ON {table_name_product}.p_product_id = product_rating.r_product_id"
join_query = f"SELECT c_customer_sk, p_product_id, encode_customer_udf(c_customer_sk, c_current_addr_sk, c_age, c_birth_country, c_preferred_cust_flag, avg_customer_rating) AS customer_encoding, encode_product_udf(p_product_id, p_department, avg_product_rating) AS product_encoding FROM ({join_query1}) AS customer_details CROSS JOIN ({join_query2}) AS product_details"

dfJoin = spark.sql(join_query)
dfJoin.createOrReplaceTempView("dfJoin")
dfJoin = spark.sql("SELECT c_customer_sk, p_product_id, vector_add_udf(customer_encoding, product_encoding) AS result_vec FROM dfJoin")


tStart = time.time()
dfJoin.collect()
tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")


print(dfJoin.count())
print(dfJoin.show(5))





