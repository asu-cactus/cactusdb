import sys
import os

# Add resources directory to sys.path so we can import resources_utils
sys.path.append(os.path.join(os.path.dirname(__file__), "resources"))

import resources_utils as ru
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import math

def csv_to_partitioned_parquet(data_dir):
    NUM_SPLIT = 8
    for file in os.listdir(data_dir):
        if not file.endswith(".csv"):
            print(f"[INFO] Skipping non-CSV file: {file}")
            continue

        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)

        # 🔽 Convert all column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Convert datetime columns to string
        for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        dataset_name = file.replace(".csv", "")
        dataset_name = dataset_name.replace("S_listings_10times","S_listings_extension")
        dataset_name = dataset_name.replace("S_routes_10times_first4","S_routes_100G")
        batch_size = math.ceil(len(df) / NUM_SPLIT)
        target_dir_path = "/home/cactusdb/resources/data/parquet/expedia/" + dataset_name

        ru.remove_all_in_directory(target_dir_path)
        print(f"[INFO] Writing parquet files for {file} to: {target_dir_path}")
        ru.write_parquet(df, batch_size, target_dir_path)

        # Save shape metadata
        num_rows, num_cols = df.shape
        with open(os.path.join("/home/cactusdb/resources/data/parquet/expedia/", dataset_name + "_stats.txt"), "w") as f:
            f.write(f"{num_rows}\n{num_cols}\n")

csv_to_partitioned_parquet("/home/cactusdb/resources/data/imbridge/expedia")
