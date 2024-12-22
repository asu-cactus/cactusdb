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
import evadb



connection = evadb.connect()
cursor = connection.cursor()



sql_connect_db = """
CREATE DATABASE IF NOT EXISTS postgres_data WITH ENGINE = 'postgres', PARAMETERS = {
     "user": "postgres",
     "password": "postgres",
     "host": "localhost",
     "port": "5432",
     "database": "postgres"
};
"""

cursor.query(sql_connect_db).df()


# deregister function
cursor.query("DROP FUNCTION IF EXISTS TWO_TOWER_EVADB;").df()
# register function
cursor.query(
    """
    CREATE FUNCTION
    IF NOT EXISTS TWO_TOWER_EVADB
    IMPL './function_two_tower_evadb.py';
    """
).df()



tStart = time.time()
connector_name = "postgres_data"
table_name_customer = "customer_data"
table_name_product = "product_data"
table_name_rating = "rating_data"



cursor.query(
    """
    USE postgres_data {
    CREATE OR REPLACE VIEW customer_outer AS
    SELECT c_customer_sk, c_current_addr_sk, c_preferred_cust_flag, c_birth_year, c_birth_country, avg_customer_rating FROM customer_data INNER JOIN (SELECT r_user_id, AVG(r_rating) AS avg_customer_rating FROM rating_data GROUP BY r_user_id HAVING AVG(r_rating) >= 4.0) AS user_rating ON customer_data.c_customer_sk = user_rating.r_user_id
    };
"""
).df()


cursor.query(
    """
    USE postgres_data {
    CREATE OR REPLACE VIEW prod_outer AS
    SELECT p_product_id, p_department, avg_product_rating FROM product_data INNER JOIN (SELECT r_product_id, AVG(r_rating) AS avg_product_rating FROM rating_data GROUP BY r_product_id) AS product_rating ON product_data.p_product_id = product_rating.r_product_id
    };
"""
).df()




join_query = "SELECT c_customer_sk, p_product_id, TWO_TOWER_EVADB(c_customer_sk, c_current_addr_sk, c_preferred_cust_flag, c_birth_year, c_birth_country, avg_customer_rating, p_product_id, p_department, avg_product_rating).result_vec FROM postgres_data.customer_outer JOIN postgres_data.prod_outer ON true = true"
#join_query = "SELECT c_customer_sk, p_product_id FROM postgres_data.customer_outer JOIN postgres_data.prod_outer ON true = true"
#join_query = f"SELECT transaction_id, DNN_FRAUD_DETECT_EVADB(c_current_addr_sk, c_preferred_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, transaction_limit, t_time, amount) FROM postgres_data.customer_full_data JOIN postgres_data.transaction_data ON c_customer_sk = sender_id WHERE IS_WORKING_DAY_EVADB(t_time).label = 1 AND IS_ADULT_DURING_TRANSACTION_EVADB(t_time, c_birth_year).label = 1"
joinDf = cursor.query(join_query).df()
print(joinDf.head())
print(joinDf.count())



tEnd = time.time()
print("Query execution completed")
print("Execution Time: " + str(tEnd - tStart) + " Seconds")





