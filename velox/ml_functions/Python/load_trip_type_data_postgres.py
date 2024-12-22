#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import connectorx as cx



# Function to remove quotes from string columns
def remove_quotes(text):
    if isinstance(text, str):
        return text.replace('"', '').replace("'", '')
    return text



df_order = pd.read_csv("resources/data/order.csv")
str_cols = df_order.select_dtypes(include=['object']).columns.tolist()  # Get string columns
for c in str_cols:
    df_order[c] = df_order[c].apply(remove_quotes)

df_order['o_order_id'] = df_order['o_order_id'].astype('Int64')  # Using 'Int64' for nullable integers
df_order['o_customer_sk'] = df_order['o_customer_sk'].astype('Int64')
df_order['o_date'] = df_order['date'].astype(str)
df_order['o_store'] = df_order['store'].astype('Int64')

# Step 4: Drop unnecessary columns
df_order = df_order.drop(columns=['date', 'store'])



df_store = pd.read_csv("resources/data/store_dept.csv")

# Step 2: Rename the column
df_store.rename(columns={"store": "s_store"}, inplace=True)

# Step 3: Convert 's_store' to integer type
df_store['s_store'] = df_store['s_store'].astype(int)

# Identify the columns to convert to integers (excluding 's_store')
str_cols = df_store.columns[df_store.columns != 's_store'].tolist()

# Step 4: Convert the relevant columns to integers
for c in str_cols:
    df_store[c] = df_store[c].astype(int)

# Step 5: Create a new column 'store_feature' that contains the row values as a floating-point array
df_store['store_feature'] = df_store.apply(lambda row: [float(row[c]) for c in df_store.columns], axis=1)

# Step 6: Drop the original columns (excluding 'store_feature' and 's_store')
df_store.drop(columns=str_cols, inplace=True)



#database_url = "postgresql://username:password@hostname:port/database_name"
database_url = "postgresql://postgres:postgres@localhost:5432/postgres"

# Create a SQLAlchemy engine
engine = create_engine(database_url)


# In[64]:


# Insert the DataFrame into the PostgreSQL table
df_order.to_sql('order_data', engine, if_exists='replace', index=False, chunksize=1000)
df_store.to_sql('store_data', engine, if_exists='replace', index=False, chunksize=1000)


# In[65]:


# Query to check row count
row_count_query1 = 'SELECT COUNT(*) FROM order_data'
row_count_query2 = 'SELECT COUNT(*) FROM store_data'
with engine.connect() as connection:
    result = connection.execute(text(row_count_query1))
    row_count = result.fetchone()[0]
    print(f"Row count in the order table: {row_count}")

    result = connection.execute(text(row_count_query2))
    row_count = result.fetchone()[0]
    print(f"Row count in the store table: {row_count}")

