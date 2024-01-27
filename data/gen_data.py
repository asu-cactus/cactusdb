import pandas as pd
import numpy as np
import argparse

def convert_df_dtype(df):
    int64_columns = df.select_dtypes(include=['int64']).columns
    df[int64_columns] = df[int64_columns].astype(np.int32)
    return df


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='A simple command-line argument parser')

# Add arguments to the parser
parser.add_argument('-n', '--number', type=int, help='Specify a number')
args = parser.parse_args()

RANDOM_SEED = 0
np.random.seed(0)

numSamples = args.number
sampledUserId = np.random.randint(1, 6041, numSamples)
sampledMovieId = np.random.randint(1, 3707, numSamples)
query_df = pd.DataFrame({"q_user_id": sampledUserId, "q_movie_id": sampledMovieId})
query_df = convert_df_dtype(query_df)
query_df.to_parquet(path="/root/velox_latest/data/query_data.parquet", row_group_size=1000)