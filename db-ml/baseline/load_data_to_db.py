import numpy as np
import pandas as pd
import psycopg2
import pandas as pd
import utils
import argparse
import os
import gc
import math
import fcntl
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
import concurrent.futures
import multiprocessing


def load_movielens_final_to_datastore():
    conn_string = utils.get_postgres_connection_config()
    db = create_engine(conn_string)
    conn = db.connect()

    data_dir = "../../resources/data/movielens/final"
    df_movie = pq.read_table(os.path.join(data_dir, "movie.parquet")).to_pandas()
    a = LabelEncoder().fit(df_movie["m_movie_id"])
    b = a.transform(df_movie["m_movie_id"])
    df_movie["m_movie_id"] = b + 1

    df_rating = pq.read_table(os.path.join(data_dir, "rating.parquet")).to_pandas()
    df_user = pq.read_table(os.path.join(data_dir, "user.parquet")).to_pandas()
    df_movie_tag = pq.read_table(
        os.path.join(data_dir, "movie_tag_relevance.parquet")
    ).to_pandas()

    df_user.to_sql("movielens_user", db, index=False, if_exists="replace")
    df_movie.to_sql("movielens_movie", db, index=False, if_exists="replace")
    df_rating.to_sql("movielens_rating", db, index=False, if_exists="replace")

    for idx, row in tqdm(df_movie_tag.iterrows(), total=len(df_movie_tag)):
        df_movie_tag.at[idx, "mt_relevance_score"] = (
            str(row["mt_relevance_score"].tolist()).replace("[", "{").replace("]", "}")
        )

    df_movie_tag_name = "movielens_movie_tag"
    db_connection = utils.get_psycopg2_connection()
    cursor = db_connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS {}".format(df_movie_tag_name))
    cursor.execute(
        """
                CREATE TABLE IF NOT EXISTS {} (
    mt_movie_id	 INTEGER,
    mt_relevance_score REAL[]
    )""".format(
            df_movie_tag_name
        )
    )
    db_connection.commit()

    df_movie_tag.to_csv("./cache/ml-movie-tag.csv", index=False, header=True)
    data_file_abs_path = os.path.abspath("./cache/ml-movie-tag.csv")

    cursor.execute(
        """    
    COPY {}(mt_movie_id,mt_relevance_score)
    FROM '{}'
    DELIMITER ',' CSV HEADER
    """.format(
            df_movie_tag_name, data_file_abs_path
        )
    )

    db_connection.commit()
    db_connection.close()

    df_user.columns = ["user_id", "gender", "age", "occupation", "zipcode"]
    df_movie.columns = [
        "movie_id",
        "title",
        "genres",
        "spoken_languages",
        "popularity",
        "vote_average",
        "vote_count",
        "overview",
    ]
    df_rating.columns = ["user_id", "movie_id", "timestamp", "rating"]

    df_user.to_sql("movielens_user1", db, index=False, if_exists="replace")
    df_movie.to_sql("movielens_movie1", db, index=False, if_exists="replace")
    df_rating.to_sql("movielens_rating1", db, index=False, if_exists="replace")

    print("[INFO] load movielens dataset to postgres success!")

    # check hdfs path exist
    data_path = "/user/velox/data/movielens"
    if not utils.check_hdfs_dir_exist(data_path):
        utils.create_hdfs_dir(data_path)

    movie_path_in_hdfs = os.path.join(data_path, "movie")
    utils.load_csv_to_hdfs(
        "../../resources/data/movielens/final/movie.parquet",
        movie_path_in_hdfs,
        overwrite=True,
    )
    user_path_in_hdfs = os.path.join(data_path, "user")
    utils.load_csv_to_hdfs(
        "../../resources/data/movielens/final/user.parquet",
        user_path_in_hdfs,
        overwrite=True,
    )
    rating_path_in_hdfs = os.path.join(data_path, "rating")
    utils.load_csv_to_hdfs(
        "../../resources/data/movielens/final/rating.parquet",
        rating_path_in_hdfs,
        overwrite=True,
    )

    movie_tag_path_in_hdfs = os.path.join(data_path, "movie_tag")
    utils.load_csv_to_hdfs(
        "../../resources/data/movielens/final/movie_tag_relevance.parquet",
        movie_tag_path_in_hdfs,
        overwrite=True,
    )

    print("[INFO] load movielens dataset to hadoop success!")


def load_tpcxai_final_to_datastore():
    # table needs to be loaded into the database
    # order, lineitem, product, financial_account, financial_transactions
    conn_string = utils.get_postgres_connection_config()
    db = create_engine(conn_string)
    conn = db.connect()

    data_dir = "../../resources/data/tpcxai_sf1/final"
    parquet_files_to_load = [
        "order",
        "lineitem",
        "product",
        "financial_account",
        "financial_transactions",
        "store_dept",
        "product_rating",
    ]

    conn_params = utils.get_connectorx_configuration()

    for file in parquet_files_to_load:
        for dataset in ["training", "serving"]:
            if dataset == "training" and file == "store_dept":
                continue
            df_parquet_path = os.path.join(data_dir, dataset, "{}.parquet".format(file))
            table_name = "tpcxai_{}_{}".format(file, dataset)
            utils.load_parquet_to_postgres(df_parquet_path, table_name, conn_params)

    print("[INFO] load movielens dataset to postgres success!")

    # check hdfs path exist
    data_path = "/user/velox/data/tpcxai"
    if not utils.check_hdfs_dir_exist(data_path):
        utils.create_hdfs_dir(data_path)

    for file in parquet_files_to_load:
        for dataset in ["training", "serving"]:
            path_in_hdfs = os.path.join(data_path, "{}_{}".format(file, dataset))
            utils.load_csv_to_hdfs(
                "../../resources/data/tpcxai_sf1/final/{}/{}.parquet".format(
                    dataset, file
                ),
                path_in_hdfs,
                overwrite=True,
            )

    print("[INFO] load movielens dataset to hadoop success!")


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


def generate_data_to_disk(
    start_gen_idx,
    end_gen_idx,
    num_tuple_per_generation,
    num_total_data,
    num_features,
    file_path,
    lock,
):
    enable_progress_bar = end_gen_idx * num_tuple_per_generation >= num_total_data
    if enable_progress_bar:
        progress_bar = tqdm(range(start_gen_idx, end_gen_idx), desc="Progress")

    for gen_idx in range(start_gen_idx, end_gen_idx):
        # print("thread: ", start_gen_idx, enable_progress_bar)
        idx_start = gen_idx * num_tuple_per_generation
        if idx_start >= num_total_data:
            return
        idx_end = (gen_idx + 1) * num_tuple_per_generation
        idx_end = idx_end if idx_end <= num_total_data else num_total_data
        x = np.random.rand(idx_end - idx_start, num_features).astype(np.float32)
        x_df = pd.DataFrame(x)
        x_df_new = x_df.copy()
        x_df_new["val"] = None

        def create_feature_vec(row):
            return "{" + ",".join(row.values.astype(str)) + "}"

        x_df_new["val"] = x_df.apply(create_feature_vec, axis=1)
        x_df_new = x_df_new[["val"]]
        x_df_new.reset_index(inplace=True)
        x_df_new["index"] = range(idx_start, idx_end)

        # with open(file_path, "a") as g:
        #     print(1)
        #     fcntl.flock(g, fcntl.LOCK_EX)
        #     print("write to csv: ", file_path)
        #     x_df_new.to_csv(file_path, index=False, header=False, mode="a")
        #     fcntl.flock(g, fcntl.LOCK_UN)

        with lock:
            print("write to csv: ", file_path)
            x_df_new.to_csv(file_path, index=False, header=False, mode="a")
        del x_df_new
        del x_df
        gc.collect()
        if enable_progress_bar:
            progress_bar.update(1)

    return "DONE"


def load_ffnn_data_to_postgres(
    num_generated_data=50, num_features=597540, table_name="ffnn_data", overwrite=False
):
    import evadb

    # Register postgres in evadb
    # TODO need to move this line of code to somewhere else
    utils.setup_postgres_for_evadb()

    db_connection = utils.get_psycopg2_connection()
    cursor = db_connection.cursor()

    csv_path = "./data/ffnn_data_{}_{}.csv".format(num_generated_data, num_features)

    # return

    if utils.check_table_exist(cursor, table_name) and not overwrite:
        print(
            "[INFO] table: {} already exists, no need to reload again.".format(
                table_name
            )
        )
        return
    else:
        cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
        cursor.execute(
            """
                    CREATE TABLE IF NOT EXISTS {} (
        index INTEGER,
        val REAL[]
        )""".format(
                table_name
            )
        )
        db_connection.commit()

        # if os.path.exists(csv_path):
        #     os.remove(csv_path)

    if not os.path.exists(csv_path) or overwrite:
        pd.DataFrame({"index": [], "val": []}).to_csv(csv_path, index=False)
        size_per_tuple = num_features * 4
        SIZE_PER_GENERATION = 1 * 256 * 1024 * 1024
        num_tuple_per_generation = math.ceil(SIZE_PER_GENERATION / size_per_tuple)
        num_generations = math.ceil(num_generated_data / num_tuple_per_generation)
        print(
            "[INFO] generate data: num_generations: {}, num_tuple_per_generation: {}".format(
                num_generations, num_tuple_per_generation
            )
        )
        num_threads_to_generate_file = utils.get_sys_num_threads()
        num_threads_to_generate_file = (
            num_threads_to_generate_file
            if num_threads_to_generate_file <= num_generations
            else num_generations
        )

        num_generation_per_core = int(
            np.ceil(num_generations / num_threads_to_generate_file)
        )
        m = multiprocessing.Manager()
        lock = m.Lock()
        print(
            "[INFO] DATA GEN INFO \n \t size_per_generation: {}, \n\t num_tuple_per_generation: {}, \n\t num_generations: {}, \n\t num_threads_to_generate_file: {}, \n\t num_generation_per_core: {}".format(
                SIZE_PER_GENERATION,
                num_tuple_per_generation,
                num_generations,
                num_threads_to_generate_file,
                num_generation_per_core,
            )
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_threads_to_generate_file
        ) as executor:
            futures = [
                executor.submit(
                    generate_data_to_disk,
                    i * num_generation_per_core,
                    (i + 1) * num_generation_per_core,
                    num_tuple_per_generation,
                    num_generated_data,
                    num_features,
                    csv_path,
                    lock,
                )
                for i in range(num_threads_to_generate_file)
            ]
            # Retrieve the results from each thread
            for future in concurrent.futures.as_completed(futures):
                print(future.result())

    data_file_abs_path = os.path.abspath(csv_path)
    print("[INFO] temp data is saved to ", data_file_abs_path)
    # Load data into postgres

    cursor.execute(
        """    
        COPY {}(index,val)
        FROM '{}'
        DELIMITER ',' CSV HEADER
        """.format(
            table_name, data_file_abs_path
        )
    )

    db_connection.commit()
    db_connection.close()

    print("[INFO] load FFNN data to postgres success!")

    # check hdfs path exist
    if not utils.check_hdfs_dir_exist("/user/velox/data/"):
        utils.create_hdfs_dir("/user/velox/data/")

    path_in_hdfs = "/user/velox/data/{}".format(table_name)
    utils.load_csv_to_hdfs(data_file_abs_path, path_in_hdfs)

    print("[INFO] load FFNN data to HDFS success!")


def load_llm_recommendation_data_to_postgres(num_user_data, num_movie_data):
    conn_string = utils.get_postgres_connection_config()
    db = create_engine(conn_string)
    conn = db.connect()

    movie_data = pd.read_csv("/home/velox/resources/data/llm/mr_movie_metadata.csv")
    movie_data = utils.change_df_dtypes(movie_data).iloc[:num_movie_data]

    user_data = pd.read_csv("/home/velox/resources/data/llm/mr_user_genre_ratings.csv")
    user_data = utils.change_df_dtypes(user_data).iloc[:num_user_data]

    user_data.to_sql("llm_recommend_user", db, index=False, if_exists="replace")
    movie_data.to_sql("llm_recommend_movie", db, index=False, if_exists="replace")

    # data = utils.convert_df_int64_to_int32(data)

    print("[INFO] load llm recommendation data to postgres success!")


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
    utils.setup_postgres_for_evadb()

    if dataset == "all":
        load_movielens_to_postgres()
        load_ffnn_data_to_postgres()
    elif dataset == "movielens":
        load_movielens_to_postgres()
    elif dataset == "ffnn":
        load_ffnn_data_to_postgres()
    elif dataset == "movielens_final":
        load_movielens_final_to_datastore()
    elif dataset == "tpcxai":
        load_tpcxai_final_to_datastore()


if __name__ == "__main__":
    main()
