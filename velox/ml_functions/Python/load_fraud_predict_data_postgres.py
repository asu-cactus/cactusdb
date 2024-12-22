#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx



countryDict = {}
with open('resources/model/country_mapping.txt', 'r') as file:
    for line in file:
        key, value = line.strip().split(",")
        countryDict[key] = int(value)



# Function to remove quotes from string columns
def remove_quotes(text):
    if isinstance(text, str):
        return text.replace('"', '').replace("'", '')
    return text



def replace_with_default(key):
    return countryDict.get(key, 212)



df_account = pd.read_csv("resources/data/financial_account.csv")
str_cols = df_account.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_account[c] = df_account[c].apply(remove_quotes)

df_account['fa_customer_sk'] = df_account['fa_customer_sk'].astype('Int64')  # Using 'Int64' for nullable integers
df_account['transaction_limit'] = df_account['transaction_limit'].astype('float')
#df_account.head()



df_customer = pd.read_csv("resources/data/customer.csv")
str_cols = df_customer.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_customer[c] = df_customer[c].apply(remove_quotes)

df_customer['c_customer_sk'] = df_customer['c_customer_sk'].astype('Int32')  # Using 'Int64' for nullable integers
df_customer['c_current_addr_sk'] = df_customer['c_current_addr_sk'].astype('Int32')
df_customer['c_birth_day'] = df_customer['c_birth_day'].astype('Int32')
df_customer['c_birth_month'] = df_customer['c_birth_month'].astype('Int32')
df_customer['c_birth_year'] = df_customer['c_birth_year'].astype('Int32')
df_customer['c_preferred_cust_flag'] = df_customer['c_preferred_cust_flag'].replace({'N': 0, 'Y': 1}).astype('Int32')
df_customer['c_birth_country'] = df_customer['c_birth_country'].apply(replace_with_default).astype('Int32')
df_customer = df_customer.drop(columns=['c_customer_id', 'c_first_name', 'c_last_name', 'c_login', 'c_email_address'])
#df_customer.head()



df_transaction = pd.read_csv("resources/data/financial_transactions.csv")
str_cols = df_transaction.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_transaction[c] = df_transaction[c].apply(remove_quotes)

df_transaction['amount'] = df_transaction['amount'].astype('float')  # Using 'Int64' for nullable integers
df_transaction['sender_id'] = df_transaction['senderID'].astype('Int32')
df_transaction['transaction_id'] = df_transaction['transactionID'].astype('Int64')
df_transaction['t_time'] = df_transaction['time'].astype(str)
df_transaction = df_transaction.drop(columns=['time', 'receiverID', 'senderID', 'transactionID'])
#df_transaction.head()


#database_url = "postgresql://username:password@hostname:port/database_name"
database_url = "postgresql://postgres:postgres@localhost:5432/postgres"

# Create a SQLAlchemy engine
engine = create_engine(database_url)



with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS account_data"))
    connection.execute(text("DROP TABLE IF EXISTS customer_data"))
    connection.execute(text("DROP TABLE IF EXISTS customer_data"))



# Insert the DataFrame into the PostgreSQL table
df_account.to_sql('account_data', engine, if_exists='replace', index=False, chunksize=1000)
df_customer.to_sql('customer_data', engine, if_exists='replace', index=False, chunksize=1000)
df_transaction.to_sql('transaction_data', engine, if_exists='replace', index=False, chunksize=1000)


# Query to check row count
row_count_query1 = 'SELECT COUNT(*) FROM account_data'
row_count_query2 = 'SELECT COUNT(*) FROM customer_data'
row_count_query3 = 'SELECT COUNT(*) FROM transaction_data'
with engine.connect() as connection:
    result = connection.execute(text(row_count_query1))
    row_count = result.fetchone()[0]
    print(f"Row count in the account_data table: {row_count}")

    result = connection.execute(text(row_count_query2))
    row_count = result.fetchone()[0]
    print(f"Row count in the customer_data table: {row_count}")

    result = connection.execute(text(row_count_query3))
    row_count = result.fetchone()[0]
    print(f"Row count in the transaction_data table: {row_count}")






