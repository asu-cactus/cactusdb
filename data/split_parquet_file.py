import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
import pyarrow.parquet as pq
import os
import math
import shutil

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


remove_all_in_directory('movielens/rating')
df1 = pq.read_table('movielens_rating_s_8192.parquet').to_pandas()
batch_size = math.ceil(len(df1) / 8)
write_orc(df1, batch_size, 'movielens/rating')

remove_all_in_directory('movielens/movie')
df2 = pq.read_table('movielens_movie_s_8192.parquet').to_pandas()
batch_size = math.ceil(len(df2) / 8)
write_orc(df2, batch_size, 'movielens/movie')

remove_all_in_directory('movielens/user')
df3 = pq.read_table('movielens_user_s_8192.parquet').to_pandas()
batch_size = math.ceil(len(df3) / 8)
write_orc(df3, batch_size, 'movielens/user')