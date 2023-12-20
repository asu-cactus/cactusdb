import numpy as np
import pandas as pd
import psycopg2
import pandas as pd
import utils
from sqlalchemy import create_engine
import argparse
from sklearn.preprocessing import LabelEncoder

def load_movielens_to_postgres():
    conn_string = utils.get_postgres_connection_config()
    db = create_engine(conn_string)
    conn = db.connect()
    
    data = pd.read_csv('data/movielens.txt')
    a = LabelEncoder().fit(data['movie_id'])
    b = a.transform(data['movie_id'])
    data['movie_id'] = b+1
    # change data type

    data = utils.convert_df_int64_to_int32(data)

    data.to_sql('movielens_ori', db, index=False, if_exists='replace')

    user_data = data[['user_id', 'gender', 'age', 'occupation', 'zipcode']].copy()
    movie_data = data[['movie_id', 'title', 'genres']].copy()
    rating_data = data[['user_id', 'movie_id', 'timestamp', 'rating']].copy()

    user_data.drop_duplicates(inplace=True)
    user_data.reset_index(inplace=True, drop=True)
    movie_data.drop_duplicates(inplace=True)
    movie_data.reset_index(inplace=True, drop=True)
    movie_data.sort_values(by='movie_id', inplace=True)
    rating_data.drop_duplicates(inplace=True)
    rating_data.reset_index(inplace=True, drop=True)
    rating_data.sort_values(by='timestamp', inplace=True)

    # load data to db
    user_data.to_sql('movielens_user', db, index=False, if_exists='replace')
    movie_data.to_sql('movielens_movie', db, index=False, if_exists='replace')
    rating_data.to_sql('movielens_rating', db, index=False, if_exists='replace')

    print("[INFO] load movielens to postgres success!")


def main():
    parser = argparse.ArgumentParser(description="Argument parser")

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="dataset name, available options: all, movielens",
        required=False
    )

    args = parser.parse_args()
    dataset = args.dataset

    if dataset == 'all' or dataset == 'movielens':
        load_movielens_to_postgres()


if __name__ == "__main__":
    main()
