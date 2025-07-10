import os
import shutil
import math
import numpy as np
import pyarrow.parquet as pq


def count_sparsity_over_data(data_dir, column_sparsity_map):
    df_dict = {}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.endswith(".parquet"):
                print("[INFO] Skipping file: ", file)
                continue
            dataset_name = file.split(".")[0]
            df = pq.read_table(os.path.join(root, file)).to_pandas()
            df_dict[dataset_name] = df
    column_sparsity_map = {}

    for table, df in df_dict.items():
        sparsity = df.isnull().mean()
        for col, sparsity_value in sparsity.items():
            if col == "mt_relevance_score":
                sparsity_value = np.mean(np.stack(df[col].values) == 0)
            if col not in column_sparsity_map:
                column_sparsity_map[col] = sparsity_value
            else:
                if math.isclose(column_sparsity_map[col], sparsity_value, rel_tol=1e-5):
                    continue
                else:
                    raise Warning(
                        f"Column '{col}' found in multiple tables with different sparsity values: {column_sparsity_map[col]} vs {sparsity_value}"
                    )

    return column_sparsity_map


def remove_all_in_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if it's a file or directory and remove it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Removed directory and its contents: {file_path}")
        except Exception as e:
            print(f"Failed to remove {file_path}. Reason: {e}")


def write_parquet(df, batch_size, dir_path):
    for start in range(0, len(df), batch_size):
        path = os.path.join(dir_path, f"part_{start // batch_size}.parquet")
        end = min(start + batch_size, len(df))
        df.iloc[start:end, :].to_parquet(path)
