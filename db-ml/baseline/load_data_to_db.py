import numpy as np
import pandas as pd
import psycopg2
import pandas as pd
import utils
import argparse
import os
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


def load_ffnn_data_to_postgres(num_generated_data=50, num_features=597540, table_name = "ffnn_data"):
    import evadb

    # Register postgres in evadb
    utils.setup_postgres_for_evadb()

    cursor = evadb.connect().cursor()
    cursor.query(
        """
        USE postgres_data {{
            DROP TABLE IF EXISTS {}
        }}
        """.format(table_name)
    ).df()
    cursor.query(
        """
        USE postgres_data {{
        CREATE TABLE IF NOT EXISTS {} (
        index INTEGER,
        val REAL[]
        )
        }}
        """.format(table_name)
    ).df()
    x = np.random.rand(num_generated_data, num_features).astype(np.float32)
    x_df = pd.DataFrame(x)
    x_df_new = x_df.copy()
    x_df_new["val"] = None
    for idx, row in tqdm(x_df.iterrows(), total=len(x_df)):
        x_df_new.loc[idx, "val"] = ",".join(x_df.loc[idx].values.astype(str))
    x_df_new["val"] = "{" + x_df_new["val"] + "}"
    x_df_new = x_df_new[["val"]]
    x_df_new.reset_index(inplace=True)
    x_df_new.to_csv("./data/ffnn_data.csv", index=False)
    data_file_abs_path = os.path.abspath('./data/ffnn_data.csv')
    print("[INFO] temp data is saved to ", data_file_abs_path)
    # Load data into postgres
    cursor.query(
        """
        USE postgres_data {{
        COPY {}(index,val)
        FROM '{}'
        DELIMITER ',' CSV HEADER
        }}
        """.format(table_name, data_file_abs_path)
    ).df()
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
