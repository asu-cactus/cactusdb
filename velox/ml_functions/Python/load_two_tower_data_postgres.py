#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx



countryDict = {}
with open('resources/data/country_mapping.txt', 'r') as file:
    for line in file:
        key, valueStr = line.strip().split(",")
        value = int(valueStr)
        if value < 0:
            value = 212
        countryDict[key] = value

deptDict = {}
with open('resources/data/department_mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split("=")
        deptDict[key] = int(value)



# Function to remove quotes from string columns
def remove_quotes(text):
    if isinstance(text, str):
        return text.replace('"', '').replace("'", '')
    return text



def map_country(key):
    return countryDict.get(key, 212)

def map_department(key):
    return deptDict.get(key, 0)



df_customer = pd.read_csv("resources/data/customer.csv")
str_cols = df_customer.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_customer[c] = df_customer[c].apply(remove_quotes)

df_customer = df_customer.dropna(subset=['c_birth_year', 'c_birth_country'])
df_customer['c_customer_sk'] = df_customer['c_customer_sk'].astype('Int32')  # Using 'Int64' for nullable integers
df_customer['c_current_addr_sk'] = df_customer['c_current_addr_sk'].astype('Int32')
df_customer['c_birth_day'] = df_customer['c_birth_day'].astype('Int32')
df_customer['c_birth_month'] = df_customer['c_birth_month'].astype('Int32')
df_customer['c_birth_year'] = df_customer['c_birth_year'].astype('Int32')
df_customer['c_preferred_cust_flag'] = df_customer['c_preferred_cust_flag'].replace({'N': 0, 'Y': 1}).astype('Int32')
df_customer['c_birth_country'] = df_customer['c_birth_country'].apply(map_country).astype('Int32')
df_customer = df_customer.drop(columns=['c_customer_id', 'c_first_name', 'c_last_name', 'c_login', 'c_email_address'])
#df_customer.head()


df_product = pd.read_csv("resources/data/product.csv")
str_cols = df_product.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_product[c] = df_product[c].apply(remove_quotes)

df_product['p_product_id'] = df_product['p_product_id'].astype('Int32')  # Using 'Int64' for nullable integers
df_product['p_department'] = df_product['department'].apply(map_department).astype('Int32')
df_product = df_product.drop(columns=['name', 'department'])
#df_transaction.head()


df_rating = pd.read_csv("resources/data/ProductRating.csv")
str_cols = df_rating.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_rating[c] = df_rating[c].apply(remove_quotes)

df_rating['r_user_id'] = df_rating['userID'].astype('Int32')  # Using 'Int64' for nullable integers
df_rating['r_product_id'] = df_rating['productID'].astype('Int32')
df_rating['r_rating'] = df_rating['rating'].astype('Int32')
df_rating = df_rating.drop(columns=['userID', 'productID', 'rating'])


#database_url = "postgresql://username:password@hostname:port/database_name"
database_url = "postgresql://postgres:postgres@localhost:5432/postgres"

# Create a SQLAlchemy engine
engine = create_engine(database_url)



with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS customer_data"))
    connection.execute(text("DROP TABLE IF EXISTS product_data"))
    connection.execute(text("DROP TABLE IF EXISTS rating_data"))



# Insert the DataFrame into the PostgreSQL table
df_customer.to_sql('customer_data', engine, if_exists='replace', index=False, chunksize=1000)
df_product.to_sql('product_data', engine, if_exists='replace', index=False, chunksize=1000)
df_rating.to_sql('rating_data', engine, if_exists='replace', index=False, chunksize=1000)



# Query to check row count
row_count_query1 = 'SELECT COUNT(*) FROM customer_data'
row_count_query2 = 'SELECT COUNT(*) FROM product_data'
row_count_query3 = 'SELECT COUNT(*) FROM rating_data'
with engine.connect() as connection:
    result = connection.execute(text(row_count_query1))
    row_count = result.fetchone()[0]
    print(f"Row count in the customer_data table: {row_count}")

    result = connection.execute(text(row_count_query2))
    row_count = result.fetchone()[0]
    print(f"Row count in the product_data table: {row_count}")

    result = connection.execute(text(row_count_query3))
    row_count = result.fetchone()[0]
    print(f"Row count in the rating_data table: {row_count}")






