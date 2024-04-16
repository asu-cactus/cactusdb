import numpy as np
import pandas as pd
import psycopg2
import pandas as pd
import utils
import argparse
import os
import gc
import math
from tqdm.auto import tqdm
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder


def load_movielens_to_postgres():
    conn_string = utils.get_postgres_connection_config()
    db = create_engine(conn_string)
    conn = db.connect()

    data = pd.read_csv("data/movielens_processed.csv")
    a = LabelEncoder().fit(data["movie_id"])
    b = a.transform(data["movie_id"])
    data["movie_id"] = b + 1
    # change data type

    data = utils.convert_df_int64_to_int32(data)

    data.to_sql("movielens_ori", db, index=False, if_exists="replace")

    user_data = data[["user_id", "gender", "age", "occupation", "zipcode"]].copy()
    movie_data = data[["movie_id", "title", "genres"]].copy()
    rating_data = data[["user_id", "movie_id", "timestamp", "rating"]].copy()

    user_data.drop_duplicates(inplace=True)
    user_data.reset_index(inplace=True, drop=True)
    movie_data.drop_duplicates(inplace=True)
    movie_data.reset_index(inplace=True, drop=True)
    movie_data.sort_values(by="movie_id", inplace=True)
    rating_data.drop_duplicates(inplace=True)
    rating_data.reset_index(inplace=True, drop=True)
    rating_data.sort_values(by="timestamp", inplace=True)

    # load data to db
    user_data.to_sql("movielens_user", db, index=False, if_exists="replace")
    movie_data.to_sql("movielens_movie", db, index=False, if_exists="replace")
    rating_data.to_sql("movielens_rating", db, index=False, if_exists="replace")

    print("[INFO] load movielens to postgres success!")


def load_ffnn_data_to_postgres(num_generated_data=50, num_features=597540, table_name = "ffnn_data", overwrite=False):
    import evadb

    # Register postgres in evadb
    # TODO need to move this line of code to somewhere else
    utils.setup_postgres_for_evadb()

    db_connection = utils.get_psycopg2_connection()
    cursor = db_connection.cursor()
    if utils.check_table_exist(cursor, table_name) and not overwrite:
        print("[INFO] table: {} already exists, no need to reload again.".format(table_name))
        return
    else:
        if overwrite:
            cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))

        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS {} (
        index INTEGER,
        val REAL[]
        )""".format(table_name))
        db_connection.commit()

    csv_path = "./data/ffnn_data_{}_{}.csv".format(num_generated_data, num_features)
    if not os.path.exists(csv_path):

        size_per_tuple = num_features * 4
        SIZE_PER_GENERATION = 1*256*1024*1024
        num_tuple_per_generation = math.ceil(SIZE_PER_GENERATION / size_per_tuple)
        num_generations = math.ceil(num_generated_data / num_tuple_per_generation)
        print("[INFO] generate data: num_generations: {}, num_tuple_per_generation: {}".format(num_generations, num_tuple_per_generation))
        for gen_idx in tqdm(range(num_generations)):
            idx_start = gen_idx*num_tuple_per_generation
            idx_end = (gen_idx+1)*num_tuple_per_generation
            idx_end = idx_end if idx_end < num_generated_data else num_generated_data
            x = np.random.rand(idx_end - idx_start, num_features).astype(np.float32)
            x_df = pd.DataFrame(x)
            x_df_new = x_df.copy()
            x_df_new["val"] = None
            def create_feature_vec(row):
                return "{" + ",".join(row.values.astype(str)) + "}"
            
            x_df_new["val"] = x_df.apply(create_feature_vec, axis=1)
            x_df_new = x_df_new[["val"]]
            x_df_new.reset_index(inplace=True)
            x_df_new['index'] = range(idx_start, idx_end)
            
            x_df_new.to_csv(csv_path, index=False, mode='a')
            del x_df_new
            del x_df
            gc.collect()
    
    data_file_abs_path = os.path.abspath(csv_path)
    print("[INFO] temp data is saved to ", data_file_abs_path)
    # Load data into postgres

    

    cursor.execute("""    
        COPY {}(index,val)
        FROM '{}'
        DELIMITER ',' CSV HEADER
        """.format(table_name, data_file_abs_path))

    db_connection.commit()
    db_connection.close()

    # cursor.query(
    #     """
    #     USE postgres_data {{
    #     COPY {}(index,val)
    #     FROM '{}'
    #     DELIMITER ',' CSV HEADER
    #     }}
    #     """.format(table_name, data_file_abs_path)
    # ).df()
    print("[INFO] load FFNN data to postgres success!")


def main():
    parser = argparse.ArgumentParser(description="Argument parser")

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset name, available options: all, movielens",
        required=False,
    )

    args = parser.parse_args()
    dataset = args.dataset

    if dataset == "all":
        load_movielens_to_postgres()
        load_ffnn_data_to_postgres()
    elif dataset == "movielens":
        load_movielens_to_postgres()
    elif dataset == "ffnn":
        load_ffnn_data_to_postgres()


if __name__ == "__main__":
    main()
