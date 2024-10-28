import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq
import numpy as np
import os
import math
import shutil
from sklearn.preprocessing import MinMaxScaler
import pickle

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


def write_orc(df, batch_size, dir_path):
    for start in range(0, len(df), batch_size):
        path = os.path.join(dir_path, f"part_{start // batch_size}.parquet")
        end = min(start + batch_size, len(df))
        df[start:end].to_parquet(path)

NUM_USER_DATA = 5
NUM_MOVIE_DATA = 30
NUM_SPLIT = 4

remove_all_in_directory('./data/parquet/llm_mr/movie')
df1 = pd.read_csv('./data/llm/mr_movie_metadata.csv')

movie_scaler = MinMaxScaler()
movie_scaler.fit(df1[['popularity', 'vote_average', 'vote_count']])
# movie_scaler.fit(df1[['popularity', 'vote_average']])
with open('./model/llm_mr/velox/llm_mr_minmax_scaler.txt', 'w') as f:
  f.write(' '.join(map(str, movie_scaler.data_min_)) + '\n')
  f.write(' '.join(map(str, movie_scaler.data_max_)) + '\n')

with open('./model/llm_mr/tf/llm_mr_minmax_scaler_py.pkl', 'wb') as f:
  pickle.dump(movie_scaler, f)

df1 = change_df_dtypes(df1).iloc[:NUM_MOVIE_DATA]
batch_size = math.ceil(len(df1) / NUM_SPLIT)
write_orc(df1, batch_size, './data/parquet/llm_mr/movie')

remove_all_in_directory('./data/parquet/llm_mr/user')
df2 = pd.read_csv('./data/llm/mr_user_genre_ratings.csv')
df2 = change_df_dtypes(df2).iloc[:NUM_USER_DATA]
batch_size = math.ceil(len(df2) / NUM_SPLIT)
write_orc(df2, batch_size, './data/parquet/llm_mr/user')

# write num_user and num_movie to a file 
with open('./data/parquet/llm_mr/llm_mr_statistics.txt', 'w') as f:
    f.write(f'{NUM_USER_DATA}\n')
    f.write(f'{NUM_MOVIE_DATA}\n')
