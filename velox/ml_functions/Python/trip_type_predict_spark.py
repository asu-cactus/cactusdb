#!/usr/bin/env python
# coding: utf-8

# # This Example shows the Prediction of Bike Flow in the NYC City using the deep learning model ST-ResNet.

# Find the details of the ST-ResNet model in the <a href="https://dl.acm.org/doi/10.5555/3298239.3298479">corresponding paper</a>

# Details of the dataset can be found <a href="https://github.com/FIBLAB/DeepSTN">here</a>.

# ### Import Modules and Define Parameters


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
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, when, udf, max, count, trim
from pyspark.sql.types import IntegerType, ArrayType, FloatType, StringType, LongType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
#import xgboost as xgb


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
        x = torch.cat((embedded_customer, x_other), dim=0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=0)



# Load DNN model
def load_dnn_model(num_unique_customer, embedding_dim, input_size, output_size, model_path):
    model = SimpleNN(num_unique_customer, embedding_dim, input_size, output_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
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


# Define the UDF to convert row values to a floating-point array
def row_to_float_array(*args):
    # Convert all inputs to float and return as a list
    return [float(x) for x in args]

# Register the UDF
row_to_float_array_udf = udf(row_to_float_array, ArrayType(FloatType()))



dfOrder = spark.read.csv("resources/data/order.csv", header=True, inferSchema=True)

str_cols = [field.name for field in dfOrder.schema.fields if isinstance(field.dataType, StringType)]
for c in str_cols:
    dfOrder = dfOrder.withColumn(c, remove_quotes_udf(col(c)))

dfOrder = dfOrder.withColumn("o_order_id", col("o_order_id").cast(IntegerType())).withColumn("o_customer_sk", col("o_customer_sk").cast(IntegerType())).withColumn("o_date", col("date").cast(StringType())).withColumn("o_store", col("store").cast(IntegerType()))
dfOrder = dfOrder.drop('date').drop('store')
dfOrder = dfOrder.repartition("o_store")
dfOrder.collect()
print("Order table size: ", dfOrder.count())
dfOrder.cache()
#dfOrder.show()


dfStore = spark.read.csv("resources/data/store_dept.csv", header=True, inferSchema=True)
dfStore = dfStore.withColumnRenamed("store", "s_store")
dfStore = dfStore.withColumn("s_store", col("s_store").cast(IntegerType()))

str_cols = [field.name for field in dfStore.schema.fields if field.name != "s_store"]
for c in str_cols:
    dfStore = dfStore.withColumn(c, col(c).cast(IntegerType()))

dfStore = dfStore.withColumn("store_feature", row_to_float_array_udf(*dfStore.columns))
for c in str_cols:
    dfStore = dfStore.drop(c)

dfStore = dfStore.repartition("s_store")
dfStore.collect()
print("store_dept table size: ", dfStore.count())
dfStore.cache()
#dfStore.show()


daysOfWeek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
secondsInADay = 86400
maxDayValue = 15340.0
def get_order_feature(date_str, weekday):
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    # Return the timestamp in milliseconds
    dtDays = float(int(dt.timestamp())//secondsInADay)/maxDayValue
    outputVec = [dtDays]

    for i in range(7):
        if weekday == daysOfWeek[i]:
            outputVec.append(1.0)
        else:
            outputVec.append(0.0)
            
    return outputVec

# Register the UDF with Spark
#get_order_feature_udf = udf(get_order_feature, ArrayType(FloatType()))
spark.udf.register("get_order_feature_udf", get_order_feature, ArrayType(FloatType()))



def concatenate_features(array1, array2):
    return array1 + array2

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("concatenate_features_udf", concatenate_features, ArrayType(FloatType()))


def is_popular_store(store_features):
    popularity = (sum(store_features) - store_features[0])/(len(store_features) - 1)
    if popularity >= 0.4:
        return 1
    else:
        return 0

# Register the UDF with Spark
#is_popular_store_udf = udf(is_popular_store, ArrayType(FloatType()))
spark.udf.register("is_popular_store_udf", is_popular_store, IntegerType())



# Load and register DNN the model
model = load_dnn_model(70710, 16, 77, 1000, "resources/model/trip_type_classify.pth")
model.eval()

# Define UDF function for prediction
def predict_dnn(customer_id, features):
    # Convert input features to tensor
    embed_tensor = torch.tensor(customer_id, dtype=torch.long) 
    input_tensor = torch.tensor(features, dtype=torch.float32)
    # Perform prediction
    with torch.no_grad():
        result = model(embed_tensor, input_tensor)
    result = np.argmax(result.detach().numpy(), axis=0)
    return int(result)

# Register the UDF
#concatenate_features_udf = udf(concatenate_features, ArrayType(FloatType()))
spark.udf.register("predict_dnn_udf", predict_dnn, IntegerType())


dfOrder.createOrReplaceTempView("df_order")
dfStore.createOrReplaceTempView("df_store")


dfJoin = spark.sql("SELECT o_order_id, o_customer_sk, get_order_feature_udf(o_date, weekday) AS order_feature, store_feature FROM df_order INNER JOIN df_store ON o_store = s_store WHERE weekday != 'Sunday' AND is_popular_store_udf(store_feature) = 1")
dfJoin.createOrReplaceTempView("df_join")
dfJoin = spark.sql("SELECT o_order_id, o_customer_sk, concatenate_features_udf(order_feature, store_feature) AS all_feature from df_join")
dfJoin.createOrReplaceTempView("df_join")
dfJoin = spark.sql("SELECT o_order_id, predict_dnn_udf(o_customer_sk, all_feature) AS predicted_trip_type from df_join")
dfJoin.createOrReplaceTempView("df_join")
#dfTemp.show()


tStart = time.time()
dfJoin.collect()
tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")
#dfOutput.show()


print(dfJoin.count())





