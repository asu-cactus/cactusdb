import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq
import numpy as np
import os
import math
import shutil


# Data needs to be downloaded from here: https://drive.google.com/drive/folders/14yde-Fc1xrz2EId6xYsF6QhmVNgLmqj6?usp=sharing
# TODO: need to migrate all data to a S3 bucket and write a script to sync the data.

def change_df_dtypes(df):
  df_dtypes = df.dtypes
  df_columns = df.columns
  for i in range(len(df.columns)):
    if df_dtypes.iloc[i] == np.int64:
      df[df_columns[i]] = df[df_columns[i]].astype(np.int32)
    elif df_dtypes.iloc[i] == np.float64:
      df[df_columns[i]] = df[df_columns[i]].astype(np.float32)
  return df
  

def remove_all_in_directory(directory):
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


def write_orc(df, batch_size, dir_path):
    for start in range(0, len(df), batch_size):
        path = os.path.join(dir_path, f"part_{start // batch_size}.parquet")
        end = min(start + batch_size, len(df))
        df[start:end].to_parquet(path)

NUM_USER_DATA = 40
NUM_MOVIE_DATA = 200
NUM_SPLIT = 4

remove_all_in_directory('movie_recommendation/movie')
df1 = pd.read_csv('mr_movie_metadata.csv')
df1 = change_df_dtypes(df1).iloc[:NUM_MOVIE_DATA]
batch_size = math.ceil(len(df1) / NUM_SPLIT)
write_orc(df1, batch_size, 'movie_recommendation/movie')

remove_all_in_directory('movie_recommendation/user')
df2 = pd.read_csv('mr_user_genre_ratings.csv')
df2 = change_df_dtypes(df2).iloc[:NUM_USER_DATA]
batch_size = math.ceil(len(df2) / NUM_SPLIT)
write_orc(df2, batch_size, 'movie_recommendation/user')