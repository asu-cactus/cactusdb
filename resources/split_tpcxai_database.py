import resources_utils as ru
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import math
import os


def create_tpcxai_dataset(data_dir):
    NUM_SPLIT = 4
    df_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".parquet"):
                print("[INFO] Skipping file: ", file)
                continue
            dataset_name = file.split(".")[0]
            df = pq.read_table(os.path.join(root, file)).to_pandas()
            df_dict[dataset_name] = df
            batch_size = math.ceil(len(df) / NUM_SPLIT)
            target_dir_path = os.path.join(
                root.replace("data", "./data/parquet"), dataset_name
            )
            ru.remove_all_in_directory(target_dir_path)
            print("[INFO] Writing parquet files to: ", target_dir_path)
            ru.write_parquet(df, batch_size, target_dir_path)
            num_rows, num_cols = df.shape
            with open(
                os.path.join(
                    root.replace("data", "./data/parquet"), dataset_name + "_stats.txt"
                ),
                "w",
            ) as f:
                f.write(f"{num_rows}\n")
                f.write(f"{num_cols}\n")


create_tpcxai_dataset("data/tpcxai_sf1/final/training")
create_tpcxai_dataset("data/tpcxai_sf1/final/serving")
