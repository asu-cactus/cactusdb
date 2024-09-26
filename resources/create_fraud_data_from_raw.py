import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq
import numpy as np
import os
import math
import shutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import pickle


def remove_all_in_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        try:
            # Check if it's a file or directory and remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f'Removed file: {file_path}')
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f'Removed directory and its contents: {file_path}')
        except Exception as e:
            print(f'Failed to remove {file_path}. Reason: {e}')

def write_parquet(df, batch_size, dir_path):
    for start in range(0, len(df), batch_size):
        path = os.path.join(dir_path, f"part_{start // batch_size}.parquet")
        end = min(start + batch_size, len(df))
        df[start:end].to_parquet(path)

def process_df(df):
    if "o_order_id" in df.columns:
      # order table
      df = df.drop(columns=["store"])
      df.columns = ["o_order_id", "o_customer_sk", "o_weekday", "o_date"]
      df["o_order_id"] = df["o_order_id"].astype(np.int32)
      df["o_customer_sk"] = df["o_customer_sk"].astype(np.int32)
      df["o_weekday"] = df["o_weekday"].astype(str)
      df["o_date"] = df["o_date"].astype(str)
    elif "amount" in df.columns:
      # transaction table
      df.columns = ["t_amount", "t_sender", "t_receiver", "transaction_id", "t_time"]
      df["t_amount"] = df["t_amount"].astype(np.float32)
      df["t_sender"] = df["t_sender"].astype(np.int32)
      df["t_receiver"] = df["t_receiver"].astype(str)
      df["transaction_id"] = df["transaction_id"].astype(np.int64)
      df["t_time"] = df["t_time"].astype(str)
    elif "c_customer_sk" in df.columns:
      # customer table 
      df = df.drop(columns=["c_customer_id", "c_first_name","c_last_name", "c_birth_day","c_birth_month","c_login","c_email_address"])
      df.columns = ["c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_year", "c_birth_country"]
      df = df.dropna(subset=["c_address_num", "c_birth_year"])
      df["c_cust_flag"] = df["c_cust_flag"].apply(lambda x : 0 if x == "N" else 1)
      label_encoded = LabelEncoder().fit_transform(df["c_birth_country"])
      df["c_birth_country"] = label_encoded
      df["c_customer_sk"] = df["c_customer_sk"].astype(np.int32)
      df["c_address_num"] = df["c_address_num"].astype(np.int32)
      df["c_cust_flag"] = df["c_cust_flag"].astype(np.int32)
      df["c_birth_year"] = df["c_birth_year"].astype(np.int32)
      df["c_birth_country"] = df["c_birth_country"].astype(np.int32)
    return df

def create_fraud_data(data_dir):
    NUM_SPLIT = 4
    df_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            dataset_name = file.split('.')[0]
            df = pd.read_csv(os.path.join(root, file), encoding_errors='ignore', on_bad_lines='skip')
            df = process_df(df)
            df_dict[dataset_name] = df
            batch_size = math.ceil(len(df) / NUM_SPLIT)
            target_dir_path = os.path.join(root.replace("data", "./data/parquet"), dataset_name)
            remove_all_in_directory(target_dir_path)
            print("[INFO] Writing parquet files to: ", target_dir_path)
            write_parquet(df, batch_size, target_dir_path)
            num_rows, num_cols = df.shape
            with open(os.path.join(root.replace("data", "./data/parquet"), dataset_name+"_stats.txt"), "w") as f:
                f.write(f'{num_rows}\n')
                f.write(f'{num_cols}\n')

create_fraud_data('data/fraud/50_mb')
create_fraud_data('data/fraud/500_mb')
