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


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=0)


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



spark = SparkSession.builder.appName("FraudDetectionSpark").config("spark.driver.memory", "16g").config("spark.dynamicAllocation.enabled", "true").getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "209715200")



countryDict = {}
with open('resources/model/country_mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(",")
        countryDict[key] = int(value)
#print(countryDict)



def mapCountry(countrStr):
    return countryDict.get(countrStr, -1)  # Returns original value if not found

# Register the UDF (specify the return type)
map_country_udf = udf(mapCountry, IntegerType())



def remove_quotes(numberStr):
    if len(numberStr) >= 2 and numberStr[0] == '"' and numberStr[-1] == '"':
        return numberStr[1:-1]
    return numberStr

remove_quotes_udf = udf(remove_quotes, StringType())



dfAccount = spark.read.csv("resources/data/financial_account.csv", header=True, inferSchema=True)

str_cols = [field.name for field in dfAccount.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfAccount = dfAccount.withColumn(c, remove_quotes_udf(col(c)))

dfAccount = dfAccount.withColumn("fa_customer_sk", col("fa_customer_sk").cast(IntegerType())).withColumn("transaction_limit", col("transaction_limit").cast(FloatType()))
dfAccount = dfAccount.repartition("fa_customer_sk")
dfAccount.collect()
print("Account table size: ", dfAccount.count())
dfAccount.cache()
#dfAccount.show()



dfTransaction = spark.read.csv("resources/data/financial_transactions.csv", header=True, inferSchema=True)
dfTransaction = dfTransaction.drop('receiverID')
#dfTransaction = dfTransaction.dropna()

str_cols = [field.name for field in dfTransaction.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfTransaction = dfTransaction.withColumn(c, remove_quotes_udf(col(c)))
    
dfTransaction = dfTransaction.withColumn("amount", col("amount").cast(FloatType())).withColumn("senderID", col("senderID").cast(IntegerType())).withColumn("transactionID", col("transactionID").cast(IntegerType())).withColumn("t_time", col("time").cast(StringType()))
dfTransaction = dfTransaction.drop('time')
dfTransaction = dfTransaction.repartition("senderID")
dfTransaction.collect()
print("Transaction table size: ", dfTransaction.count())
dfTransaction.cache()
#dfTransaction.show()



dfCustomer = spark.read.csv("resources/data/customer.csv", header=True, inferSchema=True)
dfCustomer = dfCustomer.drop('c_customer_id').drop('c_first_name').drop('c_last_name').drop('c_login').drop('c_email_address')
#dfCustomer = dfCustomer.dropna()


column_names = dfCustomer.columns
trimmed_columns = [trim(col(c)).alias(c) for c in column_names]
dfCustomer = dfCustomer.select(*trimmed_columns)

dfCustomer = dfCustomer.withColumn("c_customer_sk", col("c_customer_sk").cast(IntegerType())).withColumn("c_current_addr_sk", col("c_current_addr_sk").cast(IntegerType())).withColumn("c_birth_day", col("c_birth_day").cast(IntegerType())).withColumn("c_birth_month", col("c_birth_month").cast(IntegerType())).withColumn("c_birth_year", col("c_birth_year").cast(IntegerType())) 
dfCustomer = dfCustomer.withColumn("c_preferred_cust_flag", when(col("c_preferred_cust_flag") == 'N', 0).otherwise(1))
dfCustomer = dfCustomer.withColumn("c_birth_country", map_country_udf(col("c_birth_country")))
dfCustomer = dfCustomer.fillna(0)
dfCustomer = dfCustomer.repartition("c_customer_sk")
dfCustomer.collect()
print("Customer table size: ", dfCustomer.count())
dfCustomer.cache()
#dfCustomer.show()



def datetime_to_timestamp(date_str):
    # Parse the date string to a datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    # Return the timestamp in milliseconds
    #return int(dt.timestamp()) - 25200
    return int(dt.timestamp())

# Register the UDF (specifying the return type as LongType)
#date_to_timestamp_udf = udf(date_to_timestamp, LongType())
spark.udf.register("datetime_to_timestamp_udf", datetime_to_timestamp, LongType())


# In[13]:


def is_working_day(timestamp):
    # Convert timestamp to datetime object
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    weekday = (dt.weekday() + 1) % 7  # Monday is 1 and Sunday is 0
    if weekday == 0 or weekday == 6:
        return 0
    else:
        return 1

# Register the UDF with Spark
#is_working_day_udf = udf(is_working_day, IntegerType())
spark.udf.register("is_working_day_udf", is_working_day, IntegerType())



def age_during_transaction(timestamp, birth_year):
    year = datetime.datetime.fromtimestamp(timestamp).year  # Convert milliseconds to seconds
    return year - birth_year

# Register the UDF (specify return type as IntegerType)
#age_during_transaction_udf = udf(age_during_transaction, IntegerType())
spark.udf.register("age_during_transaction_udf", age_during_transaction, IntegerType())



def get_customer_features(address_num, cust_flag, birth_day, birth_month, birth_year, birth_country, trans_limit):
    fAddressNum = float(address_num) / 35352.0
    fCustFlag = float(cust_flag)
    fBirthDay = float(birth_day) / 31.0
    fBirthMonth = float(birth_month) / 12.0
    fBirthYear = float(birth_year) / 2002.0
    fBirthCountry = float(birth_country) / 211.0
    fTransLimit = float(trans_limit) / 16116.0
    return [fAddressNum, fCustFlag, fBirthDay, fBirthMonth, fBirthYear, fBirthCountry, fTransLimit]

# Register the UDF with Spark
#get_customer_features_udf = udf(get_customer_features, ArrayType(FloatType()))
spark.udf.register("get_customer_features_udf", get_customer_features, ArrayType(FloatType()))



def get_transaction_features(amount, timestamp):   
    fAmount = amount / 16048.0
    
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    fDay = float(dt.day) / 31.0
    fMonth = float(dt.month) / 12.0
    fYear = float(dt.year) / 2011.0
    fDayOfWeek = float((dt.weekday() + 1) % 7) / 6.0
    
    return [fAmount, fDay, fMonth, fYear, fDayOfWeek]

# Register the UDF with Spark
#get_transaction_features_udf = udf(get_transaction_features, ArrayType(FloatType()))
spark.udf.register("get_transaction_features_udf", get_transaction_features, ArrayType(FloatType()))


def concatenate_features(array1, array2):
    return array1 + array2

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("concatenate_features_udf", concatenate_features, ArrayType(FloatType()))



# Load and register DNN the model
model = load_dnn_model(12, "resources/model/fraud_detection.pth")
model.eval()

# Define UDF function for prediction
def predict_dnn(features):
    # Convert input features to tensor
    input_tensor = torch.tensor(np.array(features), dtype=torch.float32)
    # Perform prediction
    result = model(input_tensor)
    result = np.argmax(result.detach().numpy(), axis=0)
    return int(result)

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("predict_dnn_udf", predict_dnn, IntegerType())



# Load and register small xgboost model
xgb_trans = load_xgboost_model("resources/model/fraud_xgboost_trans_no_f_5_32.json")

def predict_xgb_trans(features):
    # Convert input features to DMatrix
    dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1))
    # Perform prediction
    preds = xgb_trans.predict(dmatrix)
    return float(preds[0])

# Register the UDF
#predict_xgb_trans_udf = udf(predict_xgb_trans, FloatType())
spark.udf.register("predict_xgb_trans_udf", predict_xgb_trans, FloatType())



dfAccount.createOrReplaceTempView("df_account")
dfTransaction.createOrReplaceTempView("df_transaction")
dfCustomer.createOrReplaceTempView("df_customer")



dfTemp = spark.sql("SELECT c_customer_sk, c_birth_year, get_customer_features_udf(c_current_addr_sk, c_preferred_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, transaction_limit) as customer_feature FROM df_customer INNER JOIN df_account ON c_customer_sk = fa_customer_sk")
dfTemp.createOrReplaceTempView("df_temp")
#dfTemp.show()



dfTemp2 = spark.sql("SELECT transactionID, senderID, amount, datetime_to_timestamp_udf(t_time) AS t_timestamp FROM df_transaction")
dfTemp2.createOrReplaceTempView("df_temp2")
dfTemp2 = spark.sql("SELECT transactionID, senderID, t_timestamp, get_transaction_features_udf(amount, t_timestamp) AS transaction_feature FROM df_temp2")
dfTemp2.createOrReplaceTempView("df_temp2")
#dfTemp2.show()



dfJoin = spark.sql("SELECT transactionID, concatenate_features_udf(customer_feature, transaction_feature) AS all_feature FROM df_temp INNER JOIN df_temp2 ON c_customer_sk = senderID WHERE is_working_day_udf(t_timestamp) = 1 AND age_during_transaction_udf(t_timestamp, c_birth_year) >= 18 AND predict_xgb_trans_udf(transaction_feature) >= 0.5")
dfJoin.createOrReplaceTempView("dfJoin")
dfJoin = spark.sql("SELECT transactionID, predict_dnn_udf(all_feature) as fraud_type FROM dfJoin")



tStart = time.time()
dfJoin.collect()
tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")


print(dfJoin.count())





