import connectorx as cx
import evadb
import ffnn
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import utils
import load_data_to_db
import collections
import os
import psycopg2
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from models.preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from models.dssm import DSSM_Torch, DSSM_TF, get_var_feature, get_test_var_feature
from models.dlrm import DLRM
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, when, from_unixtime
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
from dssm_evadb import DSSM_Moel_Wrapper
import pickle
import multiprocessing as mp
from sentence_transformers import SentenceTransformer
import psycopg2


def get_batch_sizes(num_samples, batch_size):
    """
    Returns a list of batch sizes that fit the given number of samples.

    Args:
        num_samples (int): The total number of samples.
        batch_size (int): The desired batch size.

    Returns:
        list: A list of batch sizes that fit the number of samples.
    """
    batch_sizes = []
    remaining_samples = num_samples

    while remaining_samples > 0:
        current_batch_size = min(remaining_samples, batch_size)
        batch_sizes.append(current_batch_size)
        remaining_samples -= current_batch_size

    return batch_sizes


class Pipeline(object):
    """A convenient class to measure the running time of a program"""

    def __init__(self, name, num_sample=500, num_loop=10):
        self.name = name
        self.num_loop = num_loop
        self.num_sample = num_sample
        self.meta = dict()
        self.list_batches = [num_sample]
        if hasattr(self, "batch_size"):
            self.list_batches = get_batch_sizes(self.num_sample, self.batch_size)
        print("[INFO] Batches: ", self.list_batches)
        self.metrics_additional = collections.defaultdict(int)

    @abstractmethod
    def loading_meta_impl(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def data_loading_impl(self, batch_size):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def data_processing_impl(self, data):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def model_inference_impl(self, data):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def run_customized_pipeline(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def clean_up(self, data):
        pass

    def run_pipeline(self):
        self.loading_meta_impl()
        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0
        self.metrics_additional = collections.defaultdict(int)

        for _ in tqdm(range(self.num_loop)):
            return_data = []
            timer_end_end.tic()
            for batch_size in self.list_batches:
                data = None
                timer_data_loading.tic()
                data = self.data_loading_impl(batch_size)
                t_data_loading += timer_data_loading.toc()

                timer_data_processing.tic()
                data = self.data_processing_impl(data)
                t_data_processing += timer_data_processing.toc()

                timer_model_inference.tic()
                data = self.model_inference_impl(data)
                t_model_inference += timer_model_inference.toc()
                return_data.append(data)

            t_end_end += timer_end_end.toc()
            self.clean_up(data)

        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop
        t_end_end /= self.num_loop
        t_additional = ""
        for metric_name, metric_value in self.metrics_additional.items():
            t_additional += f"{metric_name}: {metric_value/self.num_loop}, "

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "num_sample": self.num_sample,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
                "t_additonal": t_additional,
            },
            index=[0],
        )
        return result_df


sql_movielens_integrated_result = """
WITH t_changed_rating AS (
SELECT *,
       CASE
           WHEN rating > 3 THEN 1
           ELSE 0
       END AS changed_rating
FROM movielens_rating
),
t_user_rating AS (
SELECT mu.user_id, gender, age, occupation, avg(tcr.changed_rating) AS user_mean_rating
FROM movielens_user mu, t_changed_rating tcr
WHERE mu.user_id = tcr.user_id
GROUP BY mu.user_id, gender, age, occupation
),
t_movie_rating AS (
SELECT mm.movie_id, genres, avg(tcr.changed_rating) AS movie_mean_rating
FROM movielens_movie mm , t_changed_rating tcr
WHERE mm.movie_id = tcr.movie_id
GROUP BY mm.movie_id, genres
)
select user_id, gender, age, occupation, user_mean_rating, movie_id, genres, movie_mean_rating
from movielens_q_temp mqt, t_user_rating tur, t_movie_rating tmr
where mqt.q_user_id = tur.user_id AND mqt.q_movie_id = tmr.movie_id;
"""


class TwoTowerModelPipelinePyTorch(Pipeline):
    def __init__(self, num_sample=500, num_loop=10):
        self.postgres_conn = utils.get_postgres_connection_config()
        super(TwoTowerModelPipelinePyTorch, self).__init__(
            "two-tower-model-pytorch", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        embedding_dim = 32
        epoch = 15
        batch_size = 2048
        lr = 0.001
        seed = 1023
        dropout = 0.3

        ori_data = pd.read_csv("data/movielens_processed.csv")

        sparse_features = ["user_id", "movie_id", "gender", "age", "occupation"]
        dense_features = ["user_mean_rating", "movie_mean_rating"]
        target = ["rating"]
        device = "cpu"
        user_sparse_features, user_dense_features = [
            "user_id",
            "gender",
            "age",
            "occupation",
        ], ["user_mean_rating"]
        item_sparse_features, item_dense_features = [
            "movie_id",
        ], ["movie_mean_rating"]
        dict_encoder = dict()
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(ori_data[feat])
            dict_encoder[feat] = lbe

        genres_key2index, train_genres_list, genres_maxlen = get_var_feature(
            ori_data, "genres"
        )

        user_feature_columns = [
            SparseFeat(feat, ori_data[feat].nunique(), embedding_dim=embedding_dim)
            for i, feat in enumerate(user_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in user_dense_features
        ]
        item_feature_columns = [
            SparseFeat(feat, ori_data[feat].nunique(), embedding_dim=embedding_dim)
            for i, feat in enumerate(item_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in item_dense_features
        ]
        model = DSSM_Torch(
            user_feature_columns, item_feature_columns, task="binary", device=device
        )
        model.compile(
            "adam", "binary_crossentropy", metrics=["auc", "accuracy", "logloss"], lr=lr
        )
        self.model = model
        self.meta["model"] = model
        self.meta["sparse_features"] = sparse_features
        self.meta["dense_features"] = dense_features
        self.meta["dict_encoder"] = dict_encoder
        self.meta["genres_key2index"] = genres_key2index
        self.meta["genres_maxlen"] = genres_maxlen

    def data_loading_impl(self, batch_size):
        sampledUserId = np.random.randint(1, 6041, batch_size)
        sampledMovieId = np.random.randint(1, 3707, batch_size)
        query_df = pd.DataFrame(
            {"q_user_id": sampledUserId, "q_movie_id": sampledMovieId}
        )
        query_df.to_sql(
            "movielens_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        data = utils.fetch_data_from_postgres_via_psycopg2(
            sql_movielens_integrated_result
        )
        return data

    def data_processing_impl(self, data):
        sparse_features = self.meta["sparse_features"]
        dense_features = self.meta["dense_features"]
        dict_encoder = self.meta["dict_encoder"]
        genres_key2index = self.meta["genres_key2index"]
        genres_maxlen = self.meta["genres_maxlen"]

        data["user_mean_rating"] = data["user_mean_rating"].astype(np.float64)
        data["movie_mean_rating"] = data["movie_mean_rating"].astype(np.float64)

        for feat in sparse_features:
            lbe = dict_encoder[feat]
            data[feat] = lbe.transform(data[feat])

        test_genres_list = get_test_var_feature(
            data, "genres", genres_key2index, genres_maxlen
        )
        test_model_input = {
            name: data[name] for name in sparse_features + dense_features
        }
        test_model_input["genres"] = test_genres_list
        # data['COL1'] = label_encoder.transform(data['COL1'])
        return test_model_input

    def model_inference_impl(self, data):
        y_preds = self.model.predict(data)
        return y_preds


sql_movielens_final_fetch_query = """
WITH t_changed_rating AS (
SELECT *,
       CASE
           WHEN r_rating > 3 THEN 1
           ELSE 0
       END AS changed_rating
FROM movielens_rating
),
t_user_rating AS (
SELECT mu.u_user_id, u_gender, u_age, u_occupation, avg(tcr.changed_rating) AS u_user_mean_rating
FROM movielens_user mu, t_changed_rating tcr
WHERE mu.u_user_id = tcr.r_user_id
GROUP BY mu.u_user_id, u_gender, u_age, u_occupation
),
t_movie_rating AS (
SELECT mm.m_movie_id, m_genres, m_popularity, m_vote_average, m_vote_count, avg(tcr.changed_rating) AS m_movie_mean_rating
FROM movielens_movie mm , t_changed_rating tcr
WHERE mm.m_movie_id = tcr.r_movie_id AND m_genres like '%Action%'
GROUP BY mm.m_movie_id, m_genres, m_popularity, m_vote_average, m_vote_count
)
select *
from t_user_rating tur cross join t_movie_rating tmr;
"""


class MovielensQ1PipelineDLCentric(Pipeline):
    def __init__(self, num_sample=-1, num_loop=4):
        self.postgres_conn = utils.get_postgres_connection_config()
        super(MovielensQ1PipelineDLCentric, self).__init__(
            "movielens-q1-pipeline-dl-centric", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        embedding_dim = 32
        epoch = 15
        batch_size = 2048
        lr = 0.001
        seed = 1023
        dropout = 0.3

        ori_data = pd.read_csv(
            "/home/velox/resources/data/movielens/final/movielens_processed.csv"
        )

        sparse_features = ["user_id", "movie_id", "gender", "age", "occupation"]
        dense_features = ["user_mean_rating", "movie_mean_rating"]
        target = ["rating"]
        device = "cpu"
        user_sparse_features, user_dense_features = [
            "user_id",
            "gender",
            "age",
            "occupation",
        ], ["user_mean_rating"]
        item_sparse_features, item_dense_features = [
            "movie_id",
        ], ["movie_mean_rating"]
        dict_encoder = dict()
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(ori_data[feat])
            dict_encoder[feat] = lbe

        genres_key2index, train_genres_list, genres_maxlen = get_var_feature(
            ori_data, "genres"
        )

        user_feature_columns = [
            SparseFeat(feat, ori_data[feat].nunique(), embedding_dim=embedding_dim)
            for i, feat in enumerate(user_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in user_dense_features
        ]
        item_feature_columns = [
            SparseFeat(feat, ori_data[feat].nunique(), embedding_dim=embedding_dim)
            for i, feat in enumerate(item_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in item_dense_features
        ]
        model = DSSM_Torch(
            user_feature_columns, item_feature_columns, task="binary", device=device
        )
        model.compile(
            "adam", "binary_crossentropy", metrics=["auc", "accuracy", "logloss"], lr=lr
        )
        self.model = model
        self.meta["model"] = model
        self.meta["sparse_features"] = sparse_features
        self.meta["dense_features"] = dense_features
        self.meta["dict_encoder"] = dict_encoder
        self.meta["genres_key2index"] = genres_key2index
        self.meta["genres_maxlen"] = genres_maxlen
        self.ffnn_model = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/q1_ffnn_tf.h5", compile=False
        )
        self.min_max_scaler = pickle.load(
            open(
                "../../resources/model/movielens/final/tf/q1_ffnn_minmax_scaler_py.pkl",
                "rb",
            )
        )

    def data_loading_impl(self, batch_size):
        data = utils.fetch_data_from_postgres_via_psycopg2(
            sql_movielens_final_fetch_query
        )
        return data

    def data_processing_impl(self, data):
        # first stage filtering
        X_for_ffnn = self.min_max_scaler.transform(
            data[["m_popularity", "m_vote_average", "m_vote_count"]].values
        )
        y = np.argmax(self.ffnn_model(X_for_ffnn), axis=1)
        data = data[y == 1]

        # rename column names
        data.columns = [
            "user_id",
            "gender",
            "age",
            "occupation",
            "user_mean_rating",
            "movie_id",
            "genres",
            "popularity",
            "vote_average",
            "vote_count",
            "movie_mean_rating",
        ]

        sparse_features = self.meta["sparse_features"]
        dense_features = self.meta["dense_features"]
        dict_encoder = self.meta["dict_encoder"]
        genres_key2index = self.meta["genres_key2index"]
        genres_maxlen = self.meta["genres_maxlen"]

        data["user_mean_rating"] = data["user_mean_rating"].astype(np.float64)
        data["movie_mean_rating"] = data["movie_mean_rating"].astype(np.float64)

        for feat in sparse_features:
            lbe = dict_encoder[feat]
            data[feat] = lbe.transform(data[feat])

        test_genres_list = get_test_var_feature(
            data, "genres", genres_key2index, genres_maxlen
        )
        test_model_input = {
            name: data[name] for name in sparse_features + dense_features
        }
        test_model_input["genres"] = test_genres_list
        # data['COL1'] = label_encoder.transform(data['COL1'])
        return test_model_input

    def model_inference_impl(self, data):
        y_preds = self.model.predict(data)
        return y_preds


class MovielensQ2PipelineDLCentric(Pipeline):
    def __init__(self, num_sample=-1, num_loop=4):
        self.postgres_conn = utils.get_postgres_connection_config()
        super(MovielensQ2PipelineDLCentric, self).__init__(
            "movielens-q2-pipeline-dl-centric", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        embedding_dim = 32

        embedding_dim = 128
        num_numerical_features = 256
        categorical_feature_sizes = [7, 21, 2]  # Example sizes
        bottom_mlp_sizes = [128]
        top_mlp_sizes = [256, 128]

        self.dlrm_model = DLRM(
            embedding_dim,
            num_numerical_features,
            categorical_feature_sizes,
            bottom_mlp_sizes,
            top_mlp_sizes,
        )
        # self.dlrm_model = self.dlrm_model.to(torch.device("cuda:0"))
        self.gender_encoder = {"M": 1, "F": 0}
        self.age_encoder = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

        self.trending_ffnn_model = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/q1_ffnn_tf.h5", compile=False
        )
        self.trending_min_max_scaler = pickle.load(
            open(
                "../../resources/model/movielens/final/tf/q1_ffnn_minmax_scaler_py.pkl",
                "rb",
            )
        )
        self.encoder = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/movie_tag_standalone_encoder.h5",
            compile=False,
        )
        self.interest_min_max_scaler = pickle.load(
            open(
                "../../resources/model/movielens/final/tf/q1_ffnn_interest_scaler_py.pkl",
                "rb",
            )
        )
        self.interest_ffnn_model = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/interest_ffnn_model.h5",
            compile=False,
        )

    def data_loading_impl(self, batch_size):
        sql_q2_fetch_user_movie_data = """
          with temp_user as(
           select * from movielens_user
           limit 500
             )
          select m_movie_id, u_user_id, m_popularity, m_vote_average, m_vote_count, u_age, u_gender, u_occupation 
          from
          movielens_movie 
          cross join temp_user
          """

        data = utils.fetch_data_from_postgres_via_connectorx(
            sql_q2_fetch_user_movie_data
        )

        sql_q2_fetch_movie_tag_data = """
          select * from movielens_movie_tag
        """

        data_tag = utils.fetch_data_from_postgres_via_connectorx(
            sql_q2_fetch_movie_tag_data
        )
        return data, data_tag

    def data_processing_impl(self, data):
        data, data_tag = data
        data["u_gender"] = data["u_gender"].apply(lambda x: self.gender_encoder[x])
        X_trending_features = data[
            ["m_popularity", "m_vote_average", "m_vote_count"]
        ].values
        X_trending_features = self.trending_min_max_scaler.transform(
            X_trending_features
        )
        trending_y = np.argmax(self.trending_ffnn_model(X_trending_features), axis=1)
        data_sub = data[trending_y == 1]

        X_relevance_score = np.stack(data_tag["mt_relevance_score"].values)
        X_relevance_score_lr = self.encoder(X_relevance_score).numpy()

        data_tag["mt_relevance_score_lr"] = 0
        data_tag["mt_relevance_score_lr"] = X_relevance_score_lr.tolist()

        data_sub = data_sub.merge(
            data_tag, left_on="m_movie_id", right_on="mt_movie_id", how="inner"
        )

        X_interest_features = data_sub[["u_age", "u_occupation"]].values
        X_interest_features = self.interest_min_max_scaler.transform(
            X_interest_features
        )
        X_relevance_score_lr = np.stack(data_sub["mt_relevance_score_lr"].values)
        X_interest_features = np.hstack(
            [
                data_sub["u_gender"].values.reshape(-1, 1),
                X_interest_features,
                X_relevance_score_lr,
            ]
        )

        y_is_interested = np.argmax(
            self.interest_ffnn_model(X_interest_features), axis=1
        )
        data_sub2 = data_sub[y_is_interested == 1]

        dlrm_numerical_features = X_relevance_score_lr[y_is_interested == 1]
        dlrm_categorical_features = np.hstack(
            [
                data_sub2["u_age"]
                .apply(lambda x: self.age_encoder[x])
                .values.reshape(-1, 1),
                data_sub2["u_occupation"].values.reshape(-1, 1),
                data_sub2["u_gender"].values.reshape(-1, 1),
            ]
        )

        dlrm_numerical_features = torch.Tensor(dlrm_numerical_features)
        dlrm_categorical_features = torch.Tensor(
            dlrm_categorical_features.astype(np.int32)
        )
        dlrm_categorical_features = dlrm_categorical_features.to(torch.int32)

        return dlrm_numerical_features, dlrm_categorical_features

    def model_inference_impl(self, data):
        dlrm_numerical_features, dlrm_categorical_features = data
        # dlrm_numerical_features = dlrm_numerical_features.to(torch.device("cuda:0"))
        # dlrm_categorical_features = dlrm_categorical_features.to(torch.device("cuda:0"))
        y_preds = self.dlrm_model(dlrm_numerical_features, dlrm_categorical_features)
        return y_preds


def row_wise_cosine_similarity(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    # Ensure both arrays have the same shape
    assert arr1.shape == arr2.shape, "Both arrays must have the same shape"

    # Compute the dot product row-wise
    dot_product = np.sum(arr1 * arr2, axis=1)

    # Compute the norm (magnitude) of each row for both arrays
    norm_arr1 = np.linalg.norm(arr1, axis=1)
    norm_arr2 = np.linalg.norm(arr2, axis=1)

    # Compute cosine similarity for each row
    cosine_similarity = dot_product / (norm_arr1 * norm_arr2)

    return cosine_similarity


class MovielensQ3PipelineDLCentric(Pipeline):
    def __init__(self, num_sample=-1, num_loop=4):
        self.postgres_conn = utils.get_postgres_connection_config()
        super(MovielensQ3PipelineDLCentric, self).__init__(
            "movielens-q3-pipeline-dl-centric", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):

        self.encoder = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/movie_tag_standalone_encoder.h5",
            compile=False,
        )
        self.user_movie_interest_model = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/q3_user_movie_interest_ffnn.h5",
            compile=False,
        )
        self.user_movie_rating_model = tf.keras.models.load_model(
            "../../resources/model/movielens/final/tf/q3_user_movie_rating_ffnn.h5",
            compile=False,
        )

    def data_loading_impl(self, batch_size):
        sql_q3_fetch_user_movie_data = """
          with temp_movie as (
          select *
          from
          movielens_movie limit 1000) 
          select m_movie_id, u_user_id, m_genres, m_popularity, m_vote_average, u_gender, u_age, u_occupation
          from temp_movie
          cross join 
          (select * from movielens_user limit 50) as temp_user
          where m_genres like '%Adventure%'
          """

        data = utils.fetch_data_from_postgres_via_connectorx(
            sql_q3_fetch_user_movie_data
        )

        sql_q3_fetch_movie_tag_data = """
          select * from movielens_movie_tag
        """

        data_tag = utils.fetch_data_from_postgres_via_connectorx(
            sql_q3_fetch_movie_tag_data
        )
        return data, data_tag

    def data_processing_impl(self, data):
        data, data_tag = data
        data["u_gender"] = data["u_gender"].apply(lambda x: 1 if x == "M" else 0)
        X = (
            data[
                ["u_age", "u_gender", "u_occupation", "m_popularity", "m_vote_average"]
            ]
            .to_numpy()
            .astype(np.float32)
        )

        pred_is_interest = self.user_movie_interest_model.predict(X, batch_size=2048)
        X_sub = X[np.argmax(pred_is_interest, axis=1) == 1]
        pred_rating = self.user_movie_rating_model.predict(X_sub, batch_size=2048)
        data_sub2 = data[np.argmax(pred_is_interest, axis=1) == 1][
            np.argmax(pred_rating, axis=1) == 5
        ]

        mt_relevance_ir = self.encoder.predict(
            np.stack(data_tag["mt_relevance_score"].values), batch_size=2048
        )
        data_tag["mt_relevance_score_ir"] = mt_relevance_ir.tolist()

        df_stage1 = pd.merge(
            data_sub2,
            data_tag,
            left_on="m_movie_id",
            right_on="mt_movie_id",
            how="inner",
        )[["m_movie_id", "u_user_id", "mt_relevance_score_ir"]]
        return df_stage1, data_tag

    def model_inference_impl(self, data):
        df_stage1, data_tag = data
        data_tag2 = data_tag.copy()
        data_tag2 = data_tag2.drop(columns=["mt_relevance_score"])
        data_tag2.columns = ["mt_movie_id1", "mt_relevance_score_ir1"]

        df_stage2 = pd.merge(df_stage1, data_tag2, how="cross")

        cosin_sim = row_wise_cosine_similarity(
            np.stack(df_stage2["mt_relevance_score_ir"].values),
            np.stack(df_stage2["mt_relevance_score_ir1"].values),
        )
        return cosin_sim


class MovielensQ1PipelineEvaDB(Pipeline):

    def __del__(self):
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_v_user_rating};"
        ).df()
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_v_movie_rating};"
        ).df()
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_v_changed_rating};"
        ).df()

    def __init__(self, num_sample=-1, num_loop=10):
        self.cursor = evadb.connect().cursor()
        # create changed_rating_view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_v_changed_rating AS
            SELECT *,
                CASE
                    WHEN rating > 3 THEN 1
                    ELSE 0
                END AS changed_rating
            FROM movielens_rating1
            };
        """
        ).df()
        # create user_rating_view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_v_user_rating AS
            SELECT mu.user_id, gender, age, occupation, avg(tcr.changed_rating) AS user_mean_rating
                FROM movielens_user1 mu, evadb_v_changed_rating tcr
                WHERE mu.user_id = tcr.user_id
                GROUP BY mu.user_id, gender, age, occupation
            };
        """
        ).df()
        # create movie_rating_view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_v_movie_rating AS
            SELECT mm.movie_id, genres, avg(tcr.changed_rating) AS movie_mean_rating, popularity, vote_average, vote_count, genres LIKE '%Action%' AS is_action
                FROM movielens_movie1 mm , evadb_v_changed_rating tcr
                WHERE mm.movie_id = tcr.movie_id
                GROUP BY mm.movie_id, genres, popularity, vote_average, vote_count
            };
        """
        ).df()
        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS DSSM_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS DSSM_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ1FFNN_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ1FFNN_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        self.postgres_conn = utils.get_postgres_connection_config()
        super(MovielensQ1PipelineEvaDB, self).__init__(
            "movielens-q1-pipeline-evadb", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):

        return None

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        return data

    def run_pipeline(self):
        self.loading_meta_impl()

        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0

        for _ in tqdm(range(self.num_loop)):
            timer_end_end.tic()
            return_data = []
            for batch_size in self.list_batches:
                data = None
                timer_data_loading.tic()
                data = self.data_loading_impl(batch_size)
                t_data_loading += timer_data_loading.toc()

                timer_data_processing.tic()
                data = self.data_processing_impl(data)
                t_data_processing += timer_data_processing.toc()

                timer_model_inference.tic()
                data = self.model_inference_impl(data)
                t_model_inference += timer_model_inference.toc()

                result_df = self.cursor.query(
                    """
                select DSSM_EVADB(user_id, gender, age, occupation, user_mean_rating, movie_id, genres, movie_mean_rating)
                from postgres_data.evadb_v_user_rating join postgres_data.evadb_v_movie_rating on true = true
                where MLQ1FFNN_EVADB(popularity, vote_average, vote_count).label = 1 AND is_action = 1
                """
                ).df()

                t_data_processing += result_df["t_process"].values[-1]
                t_model_inference += result_df["t_model_inference"].values[-1]
                return_data.append(result_df["label"])

            t_end_end += timer_end_end.toc()
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop
        t_end_end /= self.num_loop

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "num_sample": self.num_sample,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
            },
            index=[0],
        )
        return result_df


class MovielensQ2PipelineEvaDB(Pipeline):

    def __del__(self):
        pass
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_q2_temp_view};"
        ).df()
        self.cursor.query("USE postgres_data{DROP VIEW IF EXISTS evadb_q2_user};").df()
        self.cursor.query("USE postgres_data{DROP VIEW IF EXISTS evadb_q2_movie};").df()

    def __init__(self, num_sample=-1, num_loop=10):
        self.cursor = evadb.connect().cursor()
        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q2_user AS
            SELECT *
            FROM movielens_user
            limit 500
            };
        """
        ).df()
        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q2_movie AS
            SELECT * from movielens_movie
            };
        """
        ).df()

        self.cursor.query(
            """
          USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q2_temp_view AS
            SELECT u_user_id, m_movie_id, m_popularity, m_vote_average, m_vote_count, u_age, u_gender, u_occupation, mt_relevance_score
            from evadb_q2_movie m join movielens_movie_tag mt
            on mt.mt_movie_id = m.m_movie_id
            cross join evadb_q2_user u
          };
          """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ2MovieTagEncoder_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ2MovieTagEncoder_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ2FFNN_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ2FFNN_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ2InterestModel_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ2InterestModel_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ2DLRMModel_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ2DLRMModel_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        self.postgres_conn = utils.get_postgres_connection_config()
        super(MovielensQ2PipelineEvaDB, self).__init__(
            "movielens-q2-pipeline-evadb", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):

        return None

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        return data

    def run_pipeline(self):
        self.loading_meta_impl()

        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0

        for _ in tqdm(range(self.num_loop)):
            timer_end_end.tic()
            return_data = []
            for batch_size in self.list_batches:
                data = None
                timer_data_loading.tic()
                data = self.data_loading_impl(batch_size)
                t_data_loading += timer_data_loading.toc()

                timer_data_processing.tic()
                data = self.data_processing_impl(data)
                t_data_processing += timer_data_processing.toc()

                timer_model_inference.tic()
                data = self.model_inference_impl(data)
                t_model_inference += timer_model_inference.toc()

                result_df = self.cursor.query(
                    """
                select MLQ2DLRMModel_EVADB(u_gender, u_age, u_occupation, MLQ2MovieTagEncoder_EVADB(mt_relevance_score))
                from postgres_data.evadb_q2_temp_view 
                where MLQ2FFNN_EVADB(m_popularity, m_vote_average, m_vote_count).label = 1 and MLQ2InterestModel_EVADB(u_gender, u_age, u_occupation, MLQ2MovieTagEncoder_EVADB(mt_relevance_score)).label = 1
                """
                ).df()

                t_data_processing += result_df["t_process"].values[-1]
                t_model_inference += result_df["t_model_inference"].values[-1]
                return_data.append(result_df["label"])

            t_end_end += timer_end_end.toc()
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop
        t_end_end /= self.num_loop

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "num_sample": self.num_sample,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
            },
            index=[0],
        )
        return result_df


class MovielensQ3PipelineEvaDB(Pipeline):

    def __del__(self):
        pass
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_q3_temp_view};"
        ).df()
        self.cursor.query("USE postgres_data{DROP VIEW IF EXISTS evadb_q3_user};").df()
        self.cursor.query("USE postgres_data{DROP VIEW IF EXISTS evadb_q3_movie};").df()
        self.cursor.query("USE postgres_data{DROP VIEW IF EXISTS evadb_q3_tag1};").df()

    def __init__(self, num_sample=-1, num_loop=10):
        self.cursor = evadb.connect().cursor()
        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q3_user AS
            SELECT *
            FROM movielens_user
            limit 50
            };
        """
        ).df()
        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q3_movie AS
            SELECT * from movielens_movie
            limit 1000
            };
        """
        ).df()

        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q3_tag1 AS
            SELECT mt_movie_id AS mt_movie_id1, mt_relevance_score AS mt_relevance_score1
            from movielens_movie_tag
            };
        """
        ).df()

        self.cursor.query(
            """
          USE postgres_data {
            CREATE OR REPLACE VIEW evadb_q3_temp_view AS
            SELECT u_user_id, m_movie_id, m_popularity, m_vote_average, m_vote_count, u_age, u_gender, u_occupation, mt_relevance_score
            from evadb_q3_movie m join movielens_movie_tag mt
            on mt.mt_movie_id = m.m_movie_id
            cross join evadb_q3_user u
            where m_genres like '%Adventure%'
          };
          """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ2MovieTagEncoder_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ2MovieTagEncoder_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS MLQ3MovieTagEncoder_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ3MovieTagEncoder_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS CosinSimilarity_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS CosinSimilarity_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query(
            "DROP FUNCTION IF EXISTS MLQ3UserMovieRatingModel_EVADB;"
        ).df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ3UserMovieRatingModel_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        # deregister function
        self.cursor.query(
            "DROP FUNCTION IF EXISTS MLQ3UserMovieInterestModel_EVADB;"
        ).df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS MLQ3UserMovieInterestModel_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()

        self.postgres_conn = utils.get_postgres_connection_config()
        super(MovielensQ3PipelineEvaDB, self).__init__(
            "movielens-q3-pipeline-evadb", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):

        return None

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        return data

    def run_pipeline(self):
        self.loading_meta_impl()

        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0

        for _ in tqdm(range(self.num_loop)):
            timer_end_end.tic()
            return_data = []
            for batch_size in self.list_batches:
                data = None
                timer_data_loading.tic()
                data = self.data_loading_impl(batch_size)
                t_data_loading += timer_data_loading.toc()

                timer_data_processing.tic()
                data = self.data_processing_impl(data)
                t_data_processing += timer_data_processing.toc()

                timer_model_inference.tic()
                data = self.model_inference_impl(data)
                t_model_inference += timer_model_inference.toc()

                result_df = self.cursor.query(
                    """
                  select m_movie_id, mt_movie_id1, CosinSimilarity_EVADB(MLQ2MovieTagEncoder_EVADB(mt_relevance_score), MLQ3MovieTagEncoder_EVADB(mt_relevance_score1))
                  from postgres_data.evadb_q3_temp_view
                  join postgres_data.evadb_q3_tag1 on true=true
                  where MLQ3UserMovieInterestModel_EVADB(u_age, u_gender, u_occupation, m_popularity, m_vote_average).label = 1 and MLQ3UserMovieRatingModel_EVADB(u_age, u_gender, u_occupation, m_popularity, m_vote_average).label = 1
                """
                ).df()

                # t_data_processing += result_df["t_process"].values[-1]
                # t_model_inference += result_df["t_model_inference"].values[-1]
                # return_data.append(result_df["cosine_sim"])

            t_end_end += timer_end_end.toc()
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop
        t_end_end /= self.num_loop

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "num_sample": self.num_sample,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
            },
            index=[0],
        )
        return result_df


class TwoTowerModelPipelineTF(Pipeline):
    def __init__(self, num_sample=500, num_loop=10):
        self.postgres_conn = utils.get_postgres_connection_config()
        super(TwoTowerModelPipelineTF, self).__init__(
            "two-tower-model-tensorflow", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        embedding_dim = 32
        epoch = 15
        batch_size = 2048
        lr = 0.001
        seed = 1023
        dropout = 0.3

        ori_data = pd.read_csv("data/movielens_processed.csv")

        sparse_features = ["user_id", "movie_id", "gender", "age", "occupation"]
        dense_features = ["user_mean_rating", "movie_mean_rating"]
        target = ["rating"]
        device = "cpu"
        user_sparse_features, user_dense_features = [
            "user_id",
            "gender",
            "age",
            "occupation",
        ], ["user_mean_rating"]
        item_sparse_features, item_dense_features = [
            "movie_id",
        ], ["movie_mean_rating"]
        dict_encoder = dict()
        for feat in sparse_features:
            lbe = LabelEncoder()
            lbe.fit(ori_data[feat])
            dict_encoder[feat] = lbe

        genres_key2index, train_genres_list, genres_maxlen = get_var_feature(
            ori_data, "genres"
        )

        user_feature_columns = [
            SparseFeat(feat, ori_data[feat].nunique(), embedding_dim=embedding_dim)
            for i, feat in enumerate(user_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in user_dense_features
        ]
        item_feature_columns = [
            SparseFeat(feat, ori_data[feat].nunique(), embedding_dim=embedding_dim)
            for i, feat in enumerate(item_sparse_features)
        ] + [
            DenseFeat(
                feat,
                1,
            )
            for feat in item_dense_features
        ]
        # FIXME
        model = DSSM_TF(["user_id"], ["movie_id"])
        # model.compile(
        #     "adam", "binary_crossentropy", metrics=["auc", "accuracy", "logloss"], lr=lr
        # )
        self.model = model
        self.meta["model"] = model
        self.meta["sparse_features"] = sparse_features
        self.meta["dense_features"] = dense_features
        self.meta["dict_encoder"] = dict_encoder
        self.meta["genres_key2index"] = genres_key2index
        self.meta["genres_maxlen"] = genres_maxlen

    def data_loading_impl(self, batch_size):
        sampledUserId = np.random.randint(1, 6041, batch_size)
        sampledMovieId = np.random.randint(1, 3707, batch_size)
        query_df = pd.DataFrame(
            {"q_user_id": sampledUserId, "q_movie_id": sampledMovieId}
        )
        query_df.to_sql(
            "movielens_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        data = utils.fetch_data_from_postgres_via_psycopg2(
            sql_movielens_integrated_result
        )
        return data

    def data_processing_impl(self, data):
        sparse_features = self.meta["sparse_features"]
        dense_features = self.meta["dense_features"]
        dict_encoder = self.meta["dict_encoder"]
        genres_key2index = self.meta["genres_key2index"]
        genres_maxlen = self.meta["genres_maxlen"]

        data["user_mean_rating"] = data["user_mean_rating"].astype(np.float64)
        data["movie_mean_rating"] = data["movie_mean_rating"].astype(np.float64)

        for feat in sparse_features:
            lbe = dict_encoder[feat]
            data[feat] = lbe.transform(data[feat])

        test_genres_list = get_test_var_feature(
            data, "genres", genres_key2index, genres_maxlen
        )
        test_model_input = {
            name: data[name] for name in sparse_features + dense_features
        }
        test_model_input["genres"] = test_genres_list
        # data['COL1'] = label_encoder.transform(data['COL1'])

        test_data = dict()
        for k, v in test_model_input.items():
            if "rating" in k:
                test_data[k] = tf.convert_to_tensor(value=v, dtype="float64")
            else:
                test_data[k] = tf.convert_to_tensor(value=v, dtype="int64")

        return test_data

    def model_inference_impl(self, data):
        y_preds = self.model(data)
        return y_preds


class FFNNPipelineEvaDB(Pipeline):
    def __init__(
        self,
        list_hidden_layer_sizes,
        num_sample=500,
        num_total_record=10000,
        num_loop=10,
        ffnn_table_name="ffnn_data",
    ):
        self.num_total_record = num_total_record
        self.postgres_conn = utils.get_postgres_connection_config()
        self.cursor = evadb.connect().cursor()
        self.ffnn_table_name = ffnn_table_name
        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS FFNN_EVADB;").df()
        # there is a bug that evadb cannot pass the argument when registering the function
        np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        # register function
        sql_register_function = """
            CREATE FUNCTION IF NOT EXISTS FFNN_EVADB
            IMPL './ffnn.py';
            """
        self.cursor.query(sql_register_function).df()
        self.batch_size = 1000
        super(FFNNPipelineEvaDB, self).__init__(
            "ffnn-evadb", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        sampledIndex = np.random.randint(1, self.num_total_record, batch_size)
        query_df = pd.DataFrame({"q_index": sampledIndex})
        query_df.to_sql(
            "ffnn_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        return None

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        return data

    def run_pipeline(self):
        self.loading_meta_impl()

        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0

        for _ in tqdm(range(self.num_loop)):
            return_data = []
            timer_end_end.tic()
            for batch_size in self.list_batches:
                data = None
                timer_data_loading.tic()
                data = self.data_loading_impl(batch_size)
                t_data_loading += timer_data_loading.toc()

                timer_data_processing.tic()
                data = self.data_processing_impl(data)
                t_data_processing += timer_data_processing.toc()

                timer_model_inference.tic()
                data = self.model_inference_impl(data)
                t_model_inference += timer_model_inference.toc()

                result_df = self.cursor.query(
                    """
                SELECT FFNN_EVADB(val)
                FROM postgres_data.{} fd JOIN postgres_data.ffnn_q_temp fqt
                ON fd.index=fqt.q_index;
                """.format(
                        self.ffnn_table_name
                    )
                ).df()

                t_data_processing += result_df["t_process"].values[-1]
                t_model_inference += result_df["t_model_inference"].values[-1]
                return_data.append(result_df["label"])

            t_end_end += timer_end_end.toc()
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop
        t_data_loading = t_end_end - t_data_processing - t_model_inference
        # t_data_loading /= self.num_loop
        t_end_end /= self.num_loop

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "num_sample": self.num_sample,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
            },
            index=[0],
        )
        return result_df


class TwoTowerModelPipelineEvaDB(Pipeline):
    def __init__(self, num_sample=500, num_loop=10):
        self.cursor = evadb.connect().cursor()
        # create changed_rating_view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_v_changed_rating AS
            SELECT *,
                CASE
                    WHEN rating > 3 THEN 1
                    ELSE 0
                END AS changed_rating
            FROM movielens_rating
            };
        """
        ).df()
        # create user_rating_view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_v_user_rating AS
            SELECT mu.user_id, gender, age, occupation, avg(tcr.changed_rating) AS user_mean_rating
                FROM movielens_user mu, evadb_v_changed_rating tcr
                WHERE mu.user_id = tcr.user_id
                GROUP BY mu.user_id, gender, age, occupation
            };
        """
        ).df()
        # create movie_rating_view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_v_movie_rating AS
            SELECT mm.movie_id, genres, avg(tcr.changed_rating) AS movie_mean_rating
                FROM movielens_movie mm , evadb_v_changed_rating tcr
                WHERE mm.movie_id = tcr.movie_id
                GROUP BY mm.movie_id, genres
            };
        """
        ).df()
        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS DSSM_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS DSSM_EVADB
            IMPL './dssm_evadb.py';
            """
        ).df()
        self.postgres_conn = utils.get_postgres_connection_config()
        super(TwoTowerModelPipelineEvaDB, self).__init__(
            "two-tower-evadb", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        sampledUserId = np.random.randint(1, 6041, batch_size)
        sampledMovieId = np.random.randint(1, 3707, batch_size)
        query_df = pd.DataFrame(
            {"q_user_id": sampledUserId, "q_movie_id": sampledMovieId}
        )
        # Note: for EvaDB, here we only store the query data in db. We don't fetch data.
        query_df.to_sql(
            "movielens_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )

        return None

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        return data

    def run_pipeline(self):
        self.loading_meta_impl()

        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0

        for _ in tqdm(range(self.num_loop)):
            timer_end_end.tic()
            return_data = []
            for batch_size in self.list_batches:
                data = None
                timer_data_loading.tic()
                data = self.data_loading_impl(batch_size)
                t_data_loading += timer_data_loading.toc()

                timer_data_processing.tic()
                data = self.data_processing_impl(data)
                t_data_processing += timer_data_processing.toc()

                timer_model_inference.tic()
                data = self.model_inference_impl(data)
                t_model_inference += timer_model_inference.toc()

                result_df = self.cursor.query(
                    """
                select DSSM_EVADB(user_id, gender, age, occupation, user_mean_rating, movie_id, genres, movie_mean_rating)
                from postgres_data.movielens_q_temp mqt JOIN postgres_data.evadb_v_user_rating tur
                ON mqt.q_user_id=tur.user_id JOIN postgres_data.evadb_v_movie_rating tmr
                ON mqt.q_movie_id=tmr.movie_id;
                """.format(
                        self.num_sample
                    )
                ).df()

                t_data_processing += result_df["t_process"].values[-1]
                t_model_inference += result_df["t_model_inference"].values[-1]
                return_data.append(result_df["label"])

            t_end_end += timer_end_end.toc()
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop
        t_end_end /= self.num_loop

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "num_sample": self.num_sample,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
            },
            index=[0],
        )
        return result_df


class FFNNPipelineTF(Pipeline):
    def __init__(
        self,
        list_hidden_layer_sizes,
        num_sample=500,
        num_total_record=10000,
        num_loop=10,
        ffnn_table_name="ffnn_data",
    ):
        self.model = ffnn.FFNNTensorFlow(list_hidden_layer_sizes)
        self.num_total_record = num_total_record
        self.postgres_conn = utils.get_postgres_connection_config()
        self.ffnn_table_name = ffnn_table_name
        self.batch_size = 1000
        super(FFNNPipelineTF, self).__init__(
            "ffnn-tensorflow", num_sample=num_sample, num_loop=num_loop
        )
        print(self.list_batches)

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        sql_ffnn_query = """
        select * from {ffnn_table_name},ffnn_q_temp where {ffnn_table_name}.index=ffnn_q_temp.q_index
        """.format(
            ffnn_table_name=self.ffnn_table_name
        )
        sampledIndex = np.random.randint(1, self.num_total_record, batch_size)
        query_df = pd.DataFrame({"q_index": sampledIndex})
        query_df.to_sql(
            "ffnn_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        data = utils.fetch_data_from_postgres_via_connectorx(sql_ffnn_query)
        return data

    def data_processing_impl(self, data):
        features = np.stack(data["val"].apply(np.array)).astype(np.float32)
        data = features
        return data

    def model_inference_impl(self, data):
        # once the data can fit to memory, using model(data) will lead to best performance
        data = self.model(data)
        # batch_size = 1024*10
        # data = self.model.predict(data, batch_size=batch_size)
        return data


class FFNNPipelinePyTorch(Pipeline):
    def __init__(
        self,
        list_hidden_layer_sizes,
        num_sample=500,
        num_total_record=10000,
        num_loop=10,
        ffnn_table_name="ffnn_data",
    ):
        self.model = ffnn.FFNNPyTorch(list_hidden_layer_sizes)
        self.model.eval()
        self.num_total_record = num_total_record
        self.postgres_conn = utils.get_postgres_connection_config()
        self.ffnn_table_name = ffnn_table_name
        self.batch_size = 1000
        super(FFNNPipelinePyTorch, self).__init__(
            "ffnn-torch", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        sql_ffnn_query = """
        select * from {ffnn_table_name},ffnn_q_temp where {ffnn_table_name}.index=ffnn_q_temp.q_index
        """.format(
            ffnn_table_name=self.ffnn_table_name
        )
        sampledIndex = np.random.randint(1, self.num_total_record, batch_size)
        query_df = pd.DataFrame({"q_index": sampledIndex})
        query_df.to_sql(
            "ffnn_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        data = utils.fetch_data_from_postgres_via_connectorx(sql_ffnn_query)
        return data

    def data_processing_impl(self, data):
        features = np.stack(data["val"].apply(np.array)).astype(np.float32)
        data = features
        data = torch.from_numpy(data)
        return data

    def model_inference_impl(self, data):
        self.model.eval()
        result = self.model(data)
        # batch_size = 1024*10*5
        # dataloader = DataLoader(data, batch_size=batch_size)
        # result = None
        # for batch in dataloader:
        #     # Unpack the batch
        #     inputs = batch  # Assuming you're not using labels for prediction
        #     # Perform inference
        #     predictions = self.model(inputs)
        #     if result is None:
        #         result = predictions
        #     else:
        #         result = torch.cat((result, predictions), axis=0)
        return result


# @pandas_udf(FloatType())
@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf(features: pd.Series) -> pd.Series:
    # features = np.stack(features.apply(np.array)).astype(np.float32)
    features = np.stack(features).astype(np.float32)
    features = torch.Tensor(features)
    list_hidden_layer_sizes = np.load("evadb_ffnn_reg.npy")
    model = ffnn.FFNNPyTorch(list_hidden_layer_sizes)
    result = model(features)
    # result = np.argmax(result.detach().numpy(), axis=1)
    result = result.detach().numpy()
    result = [result[i, :] for i in range(result.shape[0])]
    return pd.Series(result)


class FFNNPipelineSparkSQL(Pipeline):
    def __init__(
        self,
        list_hidden_layer_sizes,
        num_sample=500,
        num_total_record=10000,
        num_loop=10,
        ffnn_table_name="ffnn_data",
    ):
        np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.1")
            .config("spark.driver.memory", "60g")
            .getOrCreate()
        )
        self.num_total_record = num_total_record
        self.postgres_conn = utils.get_postgres_connection_config()
        self.jdbc_url = utils.get_jdbc_postgres_connection_config()
        self.connection_properties = utils.get_sparksql_postgres_connection_properties()
        self.ffnn_table_name = ffnn_table_name
        self.batch_size = 1000
        super(FFNNPipelineSparkSQL, self).__init__(
            "ffnn-sparksql", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        sampledIndex = np.random.randint(1, self.num_total_record, batch_size)
        query_df = pd.DataFrame({"q_index": sampledIndex})
        query_df.to_sql(
            "ffnn_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )

        # df_q = self.spark.read.jdbc(
        #     url=self.jdbc_url, table="ffnn_q_temp", properties=self.connection_properties
        # )
        # df_ffnn = self.spark.read.jdbc(
        #     url=self.jdbc_url, table="ffnn_data", properties=self.connection_properties
        # )
        # joined_df = df_q.join(df_ffnn, df_q['q_index'] == df_ffnn['index'], 'inner')

        join_query = """
            select * from {ffnn_table_name},ffnn_q_temp where {ffnn_table_name}.index=ffnn_q_temp.q_index
        """.format(
            ffnn_table_name=self.ffnn_table_name
        )
        joined_df = self.spark.read.jdbc(
            url=self.jdbc_url,
            table="({0}) AS temp".format(join_query),
            properties=self.connection_properties,
        )
        # joined_df.collect()
        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        result_df = data.select(predict_batch_udf(col("val")).alias("prediction"))
        # result_df.count()
        # result_df.cache().count()
        # result_df.rdd.count()
        result_df.collect()
        # print("count:", result_df.count())
        # print(result_df.show())
        return result_df


@pandas_udf(ArrayType(FloatType()))
def predict_batch_udf_hadoop(features: pd.Series) -> pd.Series:
    features = np.stack(features.str.strip("{}").str.split(",")).astype(np.float32)
    features = torch.Tensor(features)
    list_hidden_layer_sizes = np.load("evadb_ffnn_reg.npy")
    model = ffnn.FFNNPyTorch(list_hidden_layer_sizes)
    result = model(features)
    # result = np.argmax(result.detach().numpy(), axis=1)
    result = result.detach().numpy()
    result = [result[i, :] for i in range(result.shape[0])]
    return pd.Series(result)


class FFNNPipelineSparkSQLHadoop(Pipeline):
    def __init__(
        self,
        list_hidden_layer_sizes,
        num_sample=500,
        num_total_record=10000,
        num_loop=10,
        ffnn_table_name="ffnn_data",
    ):
        np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .getOrCreate()
        )
        self.num_total_record = num_total_record
        self.ffnn_table_name = ffnn_table_name
        self.batch_size = 1000
        super(FFNNPipelineSparkSQLHadoop, self).__init__(
            "ffnn-sparksqlhadoop", num_sample=num_sample, num_loop=num_loop
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        sampledIndex = np.random.randint(1, self.num_total_record, batch_size)
        query_df = pd.DataFrame({"q_index": sampledIndex})
        spark_df = self.spark.createDataFrame(query_df)
        spark_df.createOrReplaceTempView("ffnn_q_temp")

        feature_hdfs_path = "hdfs://localhost:9900/user/velox/data/{}".format(
            self.ffnn_table_name
        )
        spark_feature_data = self.spark.read.csv(
            feature_hdfs_path, header=True, inferSchema=True
        )
        spark_feature_data.createOrReplaceTempView(self.ffnn_table_name)

        join_query = """
            select * from {ffnn_table_name},ffnn_q_temp where {ffnn_table_name}.index=ffnn_q_temp.q_index
        """.format(
            ffnn_table_name=self.ffnn_table_name
        )
        joined_df = self.spark.sql(join_query)

        # joined_df = self.spark.read.jdbc(
        #     url=self.jdbc_url,
        #     table="({0}) AS temp".format(join_query),
        #     properties=self.connection_properties,
        # )
        # joined_df.collect()
        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        result_df = data.select(
            predict_batch_udf_hadoop(col("val")).alias("prediction")
        )
        # result_df.count()
        # result_df.cache().count()
        # result_df.rdd.count()
        result_df.collect()
        # print("count:", result_df.count())
        # print(result_df.show())
        return result_df


movielens_q1_ffnn_spark_min_max_scaler = pickle.load(
    open("../../resources/model/movielens/final/tf/q1_ffnn_minmax_scaler_py.pkl", "rb")
)
movielens_q1_ffnn_spark_trendeng_model = tf.keras.models.load_model(
    "../../resources/model/movielens/final/tf/q1_ffnn_tf.h5", compile=False
)


@pandas_udf(IntegerType())
def movielens_q1_trending_ffnn(
    m_popularity: pd.Series, m_vote_average: pd.Series, m_vote_count: pd.Series
) -> pd.Series:
    X = np.array([m_popularity.values, m_vote_average.values, m_vote_count.values]).T
    X = movielens_q1_ffnn_spark_min_max_scaler.transform(X)
    y = np.argmax(movielens_q1_ffnn_spark_trendeng_model(X), axis=1)

    return pd.Series(y)


class MovielensQ1PipelineSparkHadoop(Pipeline):
    def __init__(
        self,
        num_sample=-1,
        num_loop=10,
    ):
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .getOrCreate()
        )

        super(MovielensQ1PipelineSparkHadoop, self).__init__(
            "movielens-q1-pipeline-sparkhadoop",
            num_sample=num_sample,
            num_loop=num_loop,
        )

        self.model = DSSM_Moel_Wrapper()
        self.data_path = "hdfs://localhost:9900/user/velox/data/movielens/"
        self.movie_path_in_hdfs = os.path.join(self.data_path, "movie")
        self.user_path_in_hdfs = os.path.join(self.data_path, "user")
        self.rating_path_in_hdfs = os.path.join(self.data_path, "rating")

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_movie = self.spark.read.parquet(self.movie_path_in_hdfs)
        df_movie.createOrReplaceTempView("movie")
        df_user = self.spark.read.parquet(self.user_path_in_hdfs)
        df_user.createOrReplaceTempView("user")
        df_rating = self.spark.read.parquet(self.rating_path_in_hdfs)

        df_rating = df_rating.withColumn(
            "new_rating", when(df_rating["r_rating"] > 3, 1).otherwise(0)
        )
        df_rating.createOrReplaceTempView("rating")

        movielens_q1_spark_sql = """
        WITH t_user_rating AS (
        SELECT mu.u_user_id, u_gender, u_age, u_occupation, avg(tcr.new_rating) AS u_user_mean_rating
        FROM user mu, rating tcr
        WHERE mu.u_user_id = tcr.r_user_id
        GROUP BY mu.u_user_id, u_gender, u_age, u_occupation
        ),
        t_movie_rating AS (
        SELECT mm.m_movie_id, m_genres, m_popularity, m_vote_average, m_vote_count, avg(tcr.new_rating) AS m_movie_mean_rating
        FROM movie mm , rating tcr
        WHERE mm.m_movie_id = tcr.r_movie_id
        GROUP BY mm.m_movie_id, m_genres, m_popularity, m_vote_average, m_vote_count
        )
        select *
        from t_user_rating tur cross join t_movie_rating tmr;
        """
        data = self.spark.sql(movielens_q1_spark_sql)

        return data

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        data = data.withColumn(
            "ffnn_pred",
            movielens_q1_trending_ffnn(
                F.col("m_popularity"), F.col("m_vote_average"), F.col("m_vote_count")
            ),
        ).filter((F.col("ffnn_pred") == 1) & (F.col("m_genres").contains("Action")))

        df_data = data.toPandas()
        df_data = df_data[
            [
                "u_user_id",
                "u_gender",
                "u_age",
                "u_occupation",
                "u_user_mean_rating",
                "m_movie_id",
                "m_genres",
                "m_movie_mean_rating",
            ]
        ]
        df_data.columns = [
            "user_id",
            "gender",
            "age",
            "occupation",
            "user_mean_rating",
            "movie_id",
            "genres",
            "movie_mean_rating",
        ]
        result_df = self.model.predict(df_data)
        return result_df


class MovielensQ2PipelineSparkHadoop(Pipeline):

    def __init__(
        self,
        num_sample=-1,
        num_loop=10,
    ):
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .getOrCreate()
        )

        super(MovielensQ2PipelineSparkHadoop, self).__init__(
            "MovielensQ2PipelineSparkHadoop", num_sample=num_sample, num_loop=num_loop
        )

        self.model = DSSM_Moel_Wrapper()
        self.data_path = "hdfs://localhost:9900/user/velox/data/movielens/"
        self.movie_path_in_hdfs = os.path.join(self.data_path, "movie")
        self.user_path_in_hdfs = os.path.join(self.data_path, "user")
        self.rating_path_in_hdfs = os.path.join(self.data_path, "rating")
        self.movie_tag_path_in_hdfs = os.path.join(self.data_path, "movie_tag")

        from register_q2_spark_func import (
            predict_trending_ffnn,
            relevance_encoder,
            predict_interest_ffnn,
            predict_q2_dlrm,
        )

        self.predict_trending_ffnn = predict_trending_ffnn
        self.relevance_encoder = relevance_encoder
        self.predict_interest_ffnn = predict_interest_ffnn
        self.predict_q2_dlrm = predict_q2_dlrm

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_movie = self.spark.read.parquet(self.movie_path_in_hdfs)
        df_movie.createOrReplaceTempView("movie")
        df_user = self.spark.read.parquet(self.user_path_in_hdfs)
        df_user.createOrReplaceTempView("user")
        # df_rating = self.spark.read.parquet(self.rating_path_in_hdfs)
        df_movie_tag = self.spark.read.parquet(self.movie_tag_path_in_hdfs)
        df_movie_tag.createOrReplaceTempView("movie_tag")

        movielens_q2_spark_sql = """
        SELECT u_user_id, m_movie_id, m_popularity, m_vote_average, m_vote_count, u_age, u_gender, u_occupation, mt_relevance_score
            from movie m join movie_tag mt
            on mt.mt_movie_id = m.m_movie_id
            cross join user u
        """
        data = self.spark.sql(movielens_q2_spark_sql)

        return data

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        # from register_q2_spark_func import predict_trending_ffnn, relevance_encoder, predict_interest_ffnn, predict_q2_dlrm

        data = (
            data.withColumn(
                "trending_prediction",
                self.predict_trending_ffnn(
                    "m_popularity", "m_vote_average", "m_vote_count"
                ),
            )
            .filter(F.col("trending_prediction") == 1)
            .withColumn("mt_relevance_ir", self.relevance_encoder("mt_relevance_score"))
            .withColumn(
                "interest_prediction",
                self.predict_interest_ffnn(
                    "u_gender", "u_age", "u_occupation", "mt_relevance_ir"
                ),
            )
            .filter(F.col("interest_prediction") == 1)
            .withColumn(
                "dlrm_prediction",
                self.predict_q2_dlrm(
                    "u_gender", "u_age", "u_occupation", "mt_relevance_ir"
                ),
            )
        )
        result = data.collect()
        print(result)
        return result


class MovielensQ3PipelineSparkHadoop(Pipeline):

    def __init__(
        self,
        num_sample=-1,
        num_loop=10,
    ):
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .getOrCreate()
        )

        super(MovielensQ3PipelineSparkHadoop, self).__init__(
            "MovielensQ3PipelineSparkHadoop", num_sample=num_sample, num_loop=num_loop
        )

        self.model = DSSM_Moel_Wrapper()
        self.data_path = "hdfs://localhost:9900/user/velox/data/movielens/"
        self.movie_path_in_hdfs = os.path.join(self.data_path, "movie")
        self.user_path_in_hdfs = os.path.join(self.data_path, "user")
        self.rating_path_in_hdfs = os.path.join(self.data_path, "rating")
        self.movie_tag_path_in_hdfs = os.path.join(self.data_path, "movie_tag")

        from register_q3_spark_func import (
            predict_user_movie_interest_ffnn,
            relevance_encoder,
            predict_user_movie_rating_ffnn,
            spark_cosine_similarity,
        )

        self.predict_user_movie_interest_ffnn = predict_user_movie_interest_ffnn
        self.relevance_encoder = relevance_encoder
        self.predict_user_movie_rating_ffnn = predict_user_movie_rating_ffnn
        self.spark_cosine_similarity = spark_cosine_similarity

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_movie = self.spark.read.parquet(self.movie_path_in_hdfs).limit(1000)
        df_movie.createOrReplaceTempView("movie")
        df_user = self.spark.read.parquet(self.user_path_in_hdfs).limit(50)
        df_user.createOrReplaceTempView("user")
        # df_rating = self.spark.read.parquet(self.rating_path_in_hdfs)
        df_movie_tag = self.spark.read.parquet(self.movie_tag_path_in_hdfs)
        df_movie_tag.createOrReplaceTempView("movie_tag")

        movielens_q3_spark_sql = """
        SELECT u_user_id, m_movie_id, m_popularity, m_vote_average, m_vote_count, u_age, u_gender, u_occupation, mt_relevance_score, m_genres
            from movie m join movie_tag mt
            on mt.mt_movie_id = m.m_movie_id
            cross join user u
            where m_genres LIKE '\%Adventure\%'
        """
        data = self.spark.sql(movielens_q3_spark_sql)

        tag_data = self.spark.sql(
            "SELECT mt_movie_id as mt_movie_id1, mt_relevance_score as mt_relevance_score1 FROM movie_tag"
        ).withColumn(
            "relevance_score_ir1", self.relevance_encoder(col("mt_relevance_score1"))
        )

        return data, tag_data

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        data, tag_data = data
        data = (
            data.withColumn(
                "is_interested",
                self.predict_user_movie_interest_ffnn(
                    "u_age",
                    "u_gender",
                    "u_occupation",
                    "m_popularity",
                    "m_vote_average",
                ),
            )
            .filter(F.col("is_interested") == 1)
            .withColumn(
                "predict_rating",
                self.predict_user_movie_rating_ffnn(
                    "u_age",
                    "u_gender",
                    "u_occupation",
                    "m_popularity",
                    "m_vote_average",
                ),
            )
            .filter(F.col("predict_rating") == 5)
            .withColumn(
                "relevance_score_ir", self.relevance_encoder("mt_relevance_score")
            )
            .crossJoin(tag_data)
            .withColumn(
                "cosine_sim",
                self.spark_cosine_similarity(
                    "relevance_score_ir", "relevance_score_ir1"
                ),
            )
        )

        result = data.collect()
        print(result)
        return result


def pd_func_summarize_description(df, column_name, prompt):
    df.loc[
        :,
        [
            "{}_summarized".format(column_name),
            "num_send_token",
            "num_receive_token",
            "num_failures",
        ],
    ] = df.apply(
        lambda x: utils.chatgpt_server_restfulAPI(prompt + x[column_name]),
        axis=1,
        result_type="expand",
    ).values
    return df


def pd_func_recommend_description(df, prompt):
    df.loc[:, ["result", "num_send_token", "num_receive_token", "num_failures"]] = (
        df.apply(
            lambda x: utils.chatgpt_server_restfulAPI(
                "Summarized user statistics data (preference): "
                + x["user_description_summarized"]
                + ". \n Summarized user movie metadata:  "
                + x["movie_description_summarized"]
                + prompt,
            ),
            axis=1,
            result_type="expand",
        ).values
    )
    return df


def parallelize_dataframe(
    df, column_name, prompt, is_recommend_task=False, num_cores=4
):
    df_split = np.array_split(df, num_cores)

    with mp.Pool(num_cores) as pool:
        if is_recommend_task:
            results = pool.starmap(
                pd_func_recommend_description,
                [(df_chunk, prompt) for df_chunk in df_split],
            )
        else:
            results = pool.starmap(
                pd_func_summarize_description,
                [(df_chunk, column_name, prompt) for df_chunk in df_split],
            )
    return pd.concat(results)


class LLMRecommendationPipelinePython(Pipeline):
    def __init__(
        self,
        num_user=5,
        num_movie=10,
        num_loop=10,
    ):

        super(LLMRecommendationPipelinePython, self).__init__(
            "llm-recommendation_python",
            num_sample=num_user * num_movie,
            num_loop=num_loop,
        )
        self.num_user = num_user
        self.num_movie = num_movie
        load_data_to_db.load_llm_recommendation_data_to_postgres(
            self.num_user, self.num_movie
        )
        self.postgres_conn = utils.get_postgres_connection_config()

        self.prompt1 = "Please summarize the users description. The following are the average ratings given by users to movies in each genre."
        self.prompt2 = "Please summarize the movies description. The following are the detailed information of the movie."
        self.prompt3 = "Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason."

        self.openAI_client = utils.get_openAI_client()
        self.llm_ffnn_model = tf.keras.models.load_model(
            "/home/velox/resources/model/llm_mr/tf/llm_ffnn.h5", compile=False
        )
        self.min_max_scaler = pickle.load(
            open(
                "/home/velox/resources/model/llm_mr/tf/llm_mr_minmax_scaler_py.pkl",
                "rb",
            )
        )
        self.timer = utils.Timer()
        self.num_thread = int(os.environ.get("NUM_THREADS", 8))

    def loading_meta_impl(self):
        self.metrics_additional["t_llm1"] = 0
        self.metrics_additional["t_llm2"] = 0
        self.metrics_additional["t_llm3"] = 0

    def data_loading_impl(self, batch_size):
        join_query = """
            select user_id, llm_u.description as user_description, id as movie_id, llm_m.description as movie_description, llm_m.popularity, llm_m.vote_average, llm_m.vote_count, llm_m.spoken_languages from llm_recommend_user llm_u cross join llm_recommend_movie  llm_m where llm_m.spoken_languages LIKE '%English%' limit {}
        """.format(
            batch_size
        )
        joined_df = utils.fetch_data_from_postgres_via_connectorx(join_query)
        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        # stage - 1 filtering
        # done in data loading stage
        # spoken_language_filter = data["spoken_languages"].str.contains("English")
        # data = data[data["spoken_languages"].str.contains("English")]

        # print("stage 1 filtering selectivity: ", len(data) / len(spoken_language_filter))

        # stage - 2 filtering
        data_features = data[["popularity", "vote_average", "vote_count"]]
        data_features = self.min_max_scaler.transform(data_features)
        trendening_label = np.argmax(self.llm_ffnn_model.predict(data_features), axis=1)
        trendening_label = trendening_label == True
        data = data[trendening_label]

        # print("stage 2 filtering selectivity ", len(data) / len(trendening_label))

        self.timer.tic()
        data = parallelize_dataframe(
            data, "user_description", self.prompt1, False, self.num_thread
        )
        # data.loc[:, "user_description_summarized"] = data.apply(lambda x: utils.chatgpt_server(self.openAI_client, self.prompt1 + x['user_description']), axis=1)
        self.metrics_additional["t_llm1"] += self.timer.toc()
        self.metrics_additional["num_send_tokens"] += np.sum(data["num_send_token"])
        self.metrics_additional["num_receive_tokens"] += np.sum(
            data["num_receive_token"]
        )
        self.metrics_additional["num_failures"] += np.sum(data["num_failures"])
        self.timer.tic()

        data = parallelize_dataframe(
            data, "movie_description", self.prompt2, False, self.num_thread
        )
        # data.loc[:, "movie_description_summarized"] = data.apply(lambda x: utils.chatgpt_server(self.openAI_client, self.prompt2 + x['movie_description']), axis=1)

        self.metrics_additional["t_llm2"] += self.timer.toc()
        self.metrics_additional["num_send_tokens"] += np.sum(data["num_send_token"])
        self.metrics_additional["num_receive_tokens"] += np.sum(
            data["num_receive_token"]
        )
        self.metrics_additional["num_failures"] += np.sum(data["num_failures"])
        self.timer.tic()
        data = parallelize_dataframe(data, None, self.prompt3, True, self.num_thread)
        # data.loc[:, "result"] = data.apply(lambda x: utils.chatgpt_server(self.openAI_client, "Summarized user statistics data (preference): " + x['user_description_summarized'] +". \n Summarized user movie metadata:  " + x['movie_description_summarized'] + self.prompt3), axis=1)

        self.metrics_additional["t_llm3"] += self.timer.toc()
        self.metrics_additional["num_send_tokens"] += np.sum(data["num_send_token"])
        self.metrics_additional["num_receive_tokens"] += np.sum(
            data["num_receive_token"]
        )
        self.metrics_additional["num_failures"] += np.sum(data["num_failures"])

        return data


def get_rag_data(model, index, metadata, query, k=1):

    query_embedding = model.encode(query)

    # Top-k results
    distances, indices = index.search(np.array([query_embedding]), k)

    # Create a response
    # retrieved_info = "\n".join(
    #     [f"{metadata.iloc[r]['Title']}: {metadata.iloc[r]['Plot']}" for r in indices[0]]
    # )
    retrieved_info = "\n".join([metadata.iloc[r]["augmented_text"] for r in indices[0]])
    return retrieved_info


def get_rag_with_embed(index, metadata, list_of_query_embedding, k=1):
    results = []

    for query_embedding in list_of_query_embedding:
        # Top-k results
        distances, indices = index.search(np.array([query_embedding]), k)

        # Create a response
        # retrieved_info = "\n".join(
        #     [f"{metadata.iloc[r]['Title']}: {metadata.iloc[r]['Plot']}" for r in indices[0]]
        # )
        retrieved_info = "\n".join(
            [metadata.iloc[r]["augmented_text"] for r in indices[0]]
        )
        results.append(retrieved_info)
    return results


class LLMRecommendationPipeline2Python(Pipeline):
    def __init__(
        self,
        num_user=5,
        num_movie=10,
        num_loop=10,
    ):

        super(LLMRecommendationPipeline2Python, self).__init__(
            "llm-recommendation2_python",
            num_sample=num_user * num_movie,
            num_loop=num_loop,
        )
        self.num_user = num_user
        self.num_movie = num_movie
        load_data_to_db.load_llm_recommendation_data_to_postgres(
            self.num_user, self.num_movie
        )
        self.postgres_conn = utils.get_postgres_connection_config()

        # load from pre-saved data
        with open(
            "/home/velox/resources/model/llm_mr/tf/llm2workload_data.pkl", "rb"
        ) as f:
            self.llm2_data = pickle.load(f)
        # self.rag_index = self.llm2_data["index"]
        self.rag_metadata = self.llm2_data["movie_data"]
        self.document_embedding = self.llm2_data["embeddings"]
        import faiss

        self.rag_index = faiss.IndexFlatL2(self.llm2_data["dimension"])
        self.rag_index.add(self.document_embedding)

        # self.model = self.llm2_data["model"]

        # self.model = SentenceTransformer("all-MiniLM-L6-v2")
        # self.rag_index, self.rag_metadata = utils.get_rag_reference(self.model)

        self.prompt1 = "Please summarize the users description. The following are the average ratings given by users to movies in each genre."
        self.prompt2 = "Please summarize the movies description. The following are the detailed information of the movie."
        self.prompt3 = "Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason."

        self.openAI_client = utils.get_openAI_client()
        self.llm_ffnn_model = tf.keras.models.load_model(
            "/home/velox/resources/model/llm_mr/tf/llm_ffnn.h5", compile=False
        )
        self.min_max_scaler = pickle.load(
            open(
                "/home/velox/resources/model/llm_mr/tf/llm_mr_minmax_scaler_py.pkl",
                "rb",
            )
        )
        self.timer = utils.Timer()
        self.num_thread = int(os.environ.get("NUM_THREADS", 8))

    def loading_meta_impl(self):
        self.metrics_additional["t_llm1"] = 0
        self.metrics_additional["t_llm2"] = 0
        self.metrics_additional["t_llm3"] = 0

    def data_loading_impl(self, batch_size):
        join_query = """
            select
            user_id,
            llm_u.description as user_description,
            id as movie_id,
            llm_m.description as movie_description,
            llm_m.popularity,
            llm_m.title,
            llm_m.vote_average,
            llm_m.vote_count,
            llm_m.spoken_languages
          from
            llm_recommend_user llm_u
            cross join llm_recommend_movie llm_m
          where
            llm_m.spoken_languages LIKE '%English%'
          limit
            {}
        """.format(
            batch_size
        )
        joined_df = utils.fetch_data_from_postgres_via_connectorx(join_query)
        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        # stage - 2 filtering
        data_features = data[["popularity", "vote_average", "vote_count"]]
        data_features = self.min_max_scaler.transform(data_features)
        trendening_label = np.argmax(self.llm_ffnn_model.predict(data_features), axis=1)
        trendening_label = trendening_label == True
        data = data[trendening_label]

        self.timer.tic()
        data = parallelize_dataframe(
            data, "user_description", self.prompt1, False, self.num_thread
        )
        self.metrics_additional["t_llm1"] += self.timer.toc()
        self.metrics_additional["num_send_tokens"] += np.sum(data["num_send_token"])
        self.metrics_additional["num_receive_tokens"] += np.sum(
            data["num_receive_token"]
        )
        self.metrics_additional["num_failures"] += np.sum(data["num_failures"])
        self.timer.tic()

        # data = parallelize_dataframe(
        #    data, "movie_description", self.prompt2, False, self.num_thread
        # )

        embeddings, num_input_token, num_output_token, count_failures = (
            utils.hf_MiniLM_model(data["title"].values.tolist())
        )

        # data.loc[:, "query_embedding"] = embeddings
        data.loc[:, "movie_description_summarized"] = get_rag_with_embed(
            self.rag_index, self.rag_metadata, embeddings, 1
        )
        # data.loc[:, "movie_description_summarized"] = data.apply(
        #     lambda x: get_rag_data(
        #         self.model, self.rag_index, self.rag_metadata, x["title"], 1
        #     ),
        #     axis=1,
        # )

        self.metrics_additional["t_llm2"] += self.timer.toc()
        self.metrics_additional["num_send_tokens"] += num_input_token
        self.metrics_additional["num_receive_tokens"] += num_output_token
        self.metrics_additional["num_failures"] += count_failures
        self.timer.tic()
        data = parallelize_dataframe(data, None, self.prompt3, True, self.num_thread)

        self.metrics_additional["t_llm3"] += self.timer.toc()
        self.metrics_additional["num_send_tokens"] += np.sum(data["num_send_token"])
        self.metrics_additional["num_receive_tokens"] += np.sum(
            data["num_receive_token"]
        )
        self.metrics_additional["num_failures"] += np.sum(data["num_failures"])

        return data


class TPCxAIUsecase03PipelineTF(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase03PipelineTF, self).__init__(
            "tpcxai-usecase03-tf", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase3.h5", compile=False
        )
        self.le_store = pickle.load(
            open(
                "../../resources/model/tpcxai_sf1/final/tf/usecase3_le_store.pkl", "rb"
            )
        )
        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase3_le_dept.pkl", "rb")
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_fetch_serving_data = """
        select store, department, num_of_week from tpcxai_store_dept_serving
        """

        data = utils.fetch_data_from_postgres_via_connectorx(
            query_to_fetch_serving_data
        )
        return data

    def data_processing_impl(self, data):
        max_num_of_week = 52 * 3
        data["store"] = self.le_store.transform(data["store"].values)
        data["department"] = self.le_dept.transform(data["department"].values)
        data["num_of_week"] = (data["num_of_week"] - 0) / max_num_of_week

        X_features = data[["store", "department", "num_of_week"]].values.astype(float)
        return X_features

    def model_inference_impl(self, data):
        return self.model(data)


class TPCxAIUsecase08PipelineTF(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase08PipelineTF, self).__init__(
            "tpcxai-usecase08-tf", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase8.h5", compile=False
        )
        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase8_le_dept.pkl", "rb")
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_fetch_serving_data = """
        SELECT 
          o_order_id,
          department,
          quantity,
          SUM(quantity) AS scan_count,                
          MIN(EXTRACT(DOW FROM date)) AS weekday     
        FROM tpcxai_order_serving 
        JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
        JOIN tpcxai_product_serving ON li_product_id = p_product_id
        GROUP BY o_order_id, date, department, quantity
        """

        data = utils.fetch_data_from_postgres_via_psycopg2(query_to_fetch_serving_data)
        return data

    def data_processing_impl(self, data):
        data["department_encoded"] = self.le_dept.transform(data[["department"]].values)
        data["scan_count"] = data["scan_count"].astype(int)
        data["weekday"] = data["weekday"].astype(int)

        X_features = data[
            ["quantity", "scan_count", "weekday", "department_encoded"]
        ].values.astype(float)
        return X_features

    def model_inference_impl(self, data):
        return self.model(data)


class TPCxAIUsecase08PipelineML(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase08PipelineML, self).__init__(
            "tpcxai-usecase08-ml", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()
        self.model = pickle.load(
            open(
                "../../resources/model/tpcxai_sf1/final/tf/usecase8_ml_xgboost.pkl",
                "rb",
            )
        )
        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase8_le_dept.pkl", "rb")
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_fetch_serving_data = """
        SELECT 
          o_order_id,
          department,
          quantity,
          SUM(quantity) AS scan_count,                
          MIN(EXTRACT(DOW FROM date)) AS weekday     
        FROM tpcxai_order_serving 
        JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
        JOIN tpcxai_product_serving ON li_product_id = p_product_id
        GROUP BY o_order_id, date, department, quantity
        """

        data = utils.fetch_data_from_postgres_via_psycopg2(query_to_fetch_serving_data)
        return data

    def data_processing_impl(self, data):
        data["department_encoded"] = self.le_dept.transform(data[["department"]].values)
        data["scan_count"] = data["scan_count"].astype(int)
        data["weekday"] = data["weekday"].astype(int)

        X_features = data[
            ["department_encoded", "quantity", "scan_count", "weekday"]
        ].values.astype(float)
        return X_features

    def model_inference_impl(self, data):
        return self.model.predict(data)


class TPCxAIUsecase10PipelineTF(Pipeline):

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10PipelineTF, self).__init__(
            "tpcxai-usecase10-tf", num_loop=num_loop
        )
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase10.h5", compile=False
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # Use case 10, trainig query
        query_to_fetch_serving_data = """
        select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm
        from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
        """

        data = utils.fetch_data_from_postgres_via_connectorx(
            query_to_fetch_serving_data
        )
        return data

    def data_processing_impl(self, data):
        X_features = data[["business_hour_norm", "amount_norm"]].values
        return X_features

    def model_inference_impl(self, data):
        data = self.model(data)
        return data


class TPCxAIUsecase10MLPipelineTF(Pipeline):

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10MLPipelineTF, self).__init__(
            "tpcxai-usecase10-ml-tf", num_loop=num_loop
        )
        with open(
            "../../resources/model/tpcxai_sf1/final/tf/usecase10_lr_model.h5", "rb"
        ) as f:
            self.model = pickle.load(f)

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # Use case 10, trainig query
        query_to_fetch_serving_data = """
        select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm
        from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
        """

        data = utils.fetch_data_from_postgres_via_connectorx(
            query_to_fetch_serving_data
        )
        return data

    def data_processing_impl(self, data):
        X_features = data[["business_hour_norm", "amount_norm"]].values
        return X_features

    def model_inference_impl(self, data):
        data = self.model.predict(data)
        return data


from systemds.context import SystemDSContext
from systemds.examples.tutorials.adult import DataManager
from systemds.operator.algorithm import multiLogReg
from systemds.operator.algorithm import multiLogRegPredict


class TPCxAIUsecase10MLPipelineSystemDS(Pipeline):

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10MLPipelineSystemDS, self).__init__(
            "tpcxai-usecase10-ml-systemds", num_loop=num_loop
        )

    def loading_meta_impl(self):
        # train a model
        # Use case 10, trainig query
        query_to_fetch_training_data = """
        select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm, is_fraud
        from tpcxai_financial_account_training join tpcxai_financial_transactions_training on fa_customer_sk=sender_id
        limit 100
        """
        df = utils.fetch_data_from_postgres_via_connectorx(query_to_fetch_training_data)

        X_features = df[["business_hour_norm", "amount_norm"]].values
        y = df["is_fraud"].values.astype(int)

        with SystemDSContext() as sds:
            X = sds.from_numpy(X_features)
            Y = sds.from_numpy(y)
            # Train model
            self.model = multiLogReg(X, Y, verbose=False)

    def data_loading_impl(self, batch_size):
        # Use case 10, trainig query
        query_to_fetch_serving_data = """
        select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm
        from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
        """

        data = utils.fetch_data_from_postgres_via_connectorx(
            query_to_fetch_serving_data
        )
        return data

    def data_processing_impl(self, data):
        X_features = data[["business_hour_norm", "amount_norm"]].values
        return X_features

    def model_inference_impl(self, data):

        with SystemDSContext() as sds:
            # X_test = sds.from_numpy(X_serve)

            # Apply model
            X = sds.from_numpy(data)

            [_, y_pred, acc] = multiLogRegPredict(X, self.model)

            # # Confusion Matrix
            # confusion_matrix_abs, _ = confusionMatrix(y_pred, Yt).compute()
            data = y_pred.compute()
        return data


class TPCxAIUsecase03PipelineEvaDB(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase03PipelineEvaDB, self).__init__(
            "tpcxai-usecase03-evadb", num_loop=num_loop
        )
        # self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        # utils.setup_postgres_for_evadb()

        self.cursor = evadb.connect().cursor()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS Model_UseCase3_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS Model_UseCase3_EVADB
            IMPL './function_tpcxai_evadb.py';
            """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_to_fetch_serving_data = "select Model_UseCase3_EVADB(store, department, num_of_week).predicted from postgres_data.tpcxai_store_dept_serving"

        result_df = self.cursor.query(query_to_fetch_serving_data).df()
        return result_df.values


class TPCxAIUsecase08PipelineEvaDB(Pipeline):

    def __del__(self):
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_tpcxai_uc8_view};"
        ).df()

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase08PipelineEvaDB, self).__init__(
            "tpcxai-usecase08-evadb", num_loop=num_loop
        )
        # self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        # utils.setup_postgres_for_evadb()

        self.cursor = evadb.connect().cursor()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS Model_UseCase8_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS Model_UseCase8_EVADB
            IMPL './function_tpcxai_evadb.py';
            """
        ).df()

        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_tpcxai_uc8_view AS
            SELECT 
              o_order_id,
              department,
              quantity,
              SUM(quantity) AS scan_count,                
              MIN(EXTRACT(DOW FROM date)) AS weekday    
            FROM tpcxai_order_serving 
            JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
            JOIN tpcxai_product_serving ON li_product_id = p_product_id
            GROUP BY o_order_id, date, department, quantity
            };
        """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_to_fetch_serving_data = "select o_order_id, Model_UseCase8_EVADB(quantity, scan_count, weekday, department).predicted from postgres_data.evadb_tpcxai_uc8_view"

        result_df = self.cursor.query(query_to_fetch_serving_data).df()
        return result_df.values


class TPCxAIUsecase08MLPipelineEvaDB(Pipeline):

    def __del__(self):
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_tpcxai_uc8_view};"
        ).df()

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase08MLPipelineEvaDB, self).__init__(
            "tpcxai-usecase08-ml-evadb", num_loop=num_loop
        )
        # self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        # utils.setup_postgres_for_evadb()

        self.cursor = evadb.connect().cursor()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS Model_UseCase8_ML_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS Model_UseCase8_ML_EVADB
            IMPL './function_tpcxai_evadb.py';
            """
        ).df()

        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_tpcxai_uc8_view AS
            SELECT 
              o_order_id,
              department,
              quantity,
              SUM(quantity) AS scan_count,                
              MIN(EXTRACT(DOW FROM date)) AS weekday    
            FROM tpcxai_order_serving 
            JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
            JOIN tpcxai_product_serving ON li_product_id = p_product_id
            GROUP BY o_order_id, date, department, quantity
            };
        """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_to_fetch_serving_data = "select o_order_id, Model_UseCase8_ML_EVADB(quantity, scan_count, weekday, department).predicted from postgres_data.evadb_tpcxai_uc8_view"

        result_df = self.cursor.query(query_to_fetch_serving_data).df()
        return result_df.values


class TPCxAIUsecase10PipelineEvaDB(Pipeline):

    def __del__(self):
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_tpcxai_uc10};"
        ).df()

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10PipelineEvaDB, self).__init__(
            "tpcxai-usecase10-evadb", num_loop=num_loop
        )
        # self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        utils.setup_postgres_for_evadb()
        self.cursor = evadb.connect().cursor()

        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_tpcxai_uc10 AS
            select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
            };
        """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS Model_UseCase10_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS Model_UseCase10_EVADB
            IMPL './function_tpcxai_evadb.py';
            """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_to_fetch_serving_data = "SELECT Model_UseCase10_EVADB(business_hour_norm, amount_norm).label FROM postgres_data.evadb_tpcxai_uc10"
        result_df = self.cursor.query(query_to_fetch_serving_data).df()
        return result_df.values


class TPCxAIUsecase10MLPipelineEvaDB(Pipeline):

    def __del__(self):
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_tpcxai_uc10};"
        ).df()

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10MLPipelineEvaDB, self).__init__(
            "tpcxai-usecase10-ml-evadb", num_loop=num_loop
        )
        # self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        utils.setup_postgres_for_evadb()
        self.cursor = evadb.connect().cursor()

        # create a view
        self.cursor.query(
            """
            USE postgres_data {
            CREATE OR REPLACE VIEW evadb_tpcxai_uc10 AS
            select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
            };
        """
        ).df()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS Model_UseCase10_ML_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS Model_UseCase10_ML_EVADB
            IMPL './function_tpcxai_evadb.py';
            """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_to_fetch_serving_data = "SELECT Model_UseCase10_ML_EVADB(business_hour_norm, amount_norm).label FROM postgres_data.evadb_tpcxai_uc10"
        result_df = self.cursor.query(query_to_fetch_serving_data).df()
        return result_df.values


class TPCxAIUsecase3PipelineSparkHadoop(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        # np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate()
        )
        super(TPCxAIUsecase3PipelineSparkHadoop, self).__init__(
            "tpcxai-usecase3-sparkhadoop", num_loop=num_loop
        )

        from register_tpcxai_spark_func import uc3_sales_predicator

        self.model_predictor = uc3_sales_predicator

        self.data_path = "hdfs://localhost:9900/user/velox/data/tpcxai/"
        self.store_depth_path_in_hdfs = os.path.join(
            self.data_path, "store_dept_serving"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df = self.spark.read.parquet(self.store_depth_path_in_hdfs)
        df.createOrReplaceTempView("tpcxai_store_dept_serving")
        uc3_sql = """
        select store, department, num_of_week from tpcxai_store_dept_serving
        """

        joined_df = self.spark.sql(uc3_sql)

        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        result_df = data.withColumn(
            "predicted", self.model_predictor("store", "department", "num_of_week")
        )
        result_df.collect()
        return result_df


class TPCxAIUsecase8PipelineSparkHadoop(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        # np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate()
        )
        super(TPCxAIUsecase8PipelineSparkHadoop, self).__init__(
            "tpcxai-usecase8-sparkhadoop", num_loop=num_loop
        )

        from register_tpcxai_spark_func import uc8_trip_classifier

        self.model_predictor = uc8_trip_classifier

        self.data_path = "hdfs://localhost:9900/user/velox/data/tpcxai/"
        self.order_path_in_hdfs = os.path.join(self.data_path, "order_serving")
        self.lineitem_path_in_hdfs = os.path.join(self.data_path, "lineitem_serving")
        self.product_path_in_hdfs = os.path.join(self.data_path, "product_serving")

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_order = self.spark.read.parquet(self.order_path_in_hdfs).withColumn(
            "date", from_unixtime(col("date"))
        )
        df_lineitem = self.spark.read.parquet(self.lineitem_path_in_hdfs)
        df_product = self.spark.read.parquet(self.product_path_in_hdfs)
        df_order.createOrReplaceTempView("tpcxai_order_serving")
        df_lineitem.createOrReplaceTempView("tpcxai_lineitem_serving")
        df_product.createOrReplaceTempView("tpcxai_product_serving")

        uc8_sql = """
        SELECT 
            o_order_id,
            department,
            quantity,
            SUM(quantity) AS scan_count,               
            MIN(EXTRACT(DOW FROM date)) AS weekday     
        FROM tpcxai_order_serving 
        JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
        JOIN tpcxai_product_serving ON li_product_id = p_product_id
        GROUP BY o_order_id, date, department, quantity
        """

        joined_df = self.spark.sql(uc8_sql)

        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        result_df = data.withColumn(
            "predicted",
            self.model_predictor("quantity", "scan_count", "weekday", "department"),
        )
        result_df.collect()
        return result_df


class TPCxAIUsecase8MLPipelineSparkHadoop(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        # np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate()
        )
        super(TPCxAIUsecase8MLPipelineSparkHadoop, self).__init__(
            "tpcxai-usecase8-ml-sparkhadoop", num_loop=num_loop
        )

        from register_tpcxai_spark_func import uc8_trip_ml_classifier

        self.model_predictor = uc8_trip_ml_classifier

        self.data_path = "hdfs://localhost:9900/user/velox/data/tpcxai/"
        self.order_path_in_hdfs = os.path.join(self.data_path, "order_serving")
        self.lineitem_path_in_hdfs = os.path.join(self.data_path, "lineitem_serving")
        self.product_path_in_hdfs = os.path.join(self.data_path, "product_serving")

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_order = self.spark.read.parquet(self.order_path_in_hdfs).withColumn(
            "date", from_unixtime(col("date"))
        )
        df_lineitem = self.spark.read.parquet(self.lineitem_path_in_hdfs)
        df_product = self.spark.read.parquet(self.product_path_in_hdfs)
        df_order.createOrReplaceTempView("tpcxai_order_serving")
        df_lineitem.createOrReplaceTempView("tpcxai_lineitem_serving")
        df_product.createOrReplaceTempView("tpcxai_product_serving")

        uc8_sql = """
        SELECT 
            o_order_id,
            department,
            quantity,
            SUM(quantity) AS scan_count,               
            MIN(EXTRACT(DOW FROM date)) AS weekday     
        FROM tpcxai_order_serving 
        JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
        JOIN tpcxai_product_serving ON li_product_id = p_product_id
        GROUP BY o_order_id, date, department, quantity
        """

        joined_df = self.spark.sql(uc8_sql)

        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        result_df = data.withColumn(
            "predicted",
            self.model_predictor("quantity", "scan_count", "weekday", "department"),
        )
        result_df.collect()
        return result_df


class TPCxAIUsecase10PipelineSparkHadoop(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        # np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate()
        )
        super(TPCxAIUsecase10PipelineSparkHadoop, self).__init__(
            "tpcxai-usecase10-sparkhadoop", num_loop=num_loop
        )

        from register_tpcxai_spark_func import uc10_fraud_spark_predicator

        self.model_predictor = uc10_fraud_spark_predicator

        self.data_path = "hdfs://localhost:9900/user/velox/data/tpcxai/"
        self.fa_path_in_hdfs = os.path.join(self.data_path, "financial_account_serving")
        self.ft_path_in_hdfs = os.path.join(
            self.data_path, "financial_transactions_serving"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_ft = self.spark.read.parquet(self.ft_path_in_hdfs).withColumn(
            "time", from_unixtime(col("time"))
        )
        df_fa = self.spark.read.parquet(self.fa_path_in_hdfs)
        df_ft.createOrReplaceTempView("tpcxai_financial_transactions_serving")
        df_fa.createOrReplaceTempView("tpcxai_financial_account_serving")

        uc10_sql = """
        select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm
        from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
        """

        joined_df = self.spark.sql(uc10_sql)

        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        result_df = data.withColumn(
            "predicted", self.model_predictor("business_hour_norm", "amount_norm")
        )
        result_df.collect()
        return result_df


class TPCxAIUsecase10PipelineSparkMLHadoop(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        # np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate()
        )
        super(TPCxAIUsecase10PipelineSparkMLHadoop, self).__init__(
            "tpcxai-usecase10-sparkhadoop-ml", num_loop=num_loop
        )

        from register_tpcxai_spark_func import uc10_fraud_ml_spark_predicator

        self.model_predictor = uc10_fraud_ml_spark_predicator

        self.data_path = "hdfs://localhost:9900/user/velox/data/tpcxai/"
        self.fa_path_in_hdfs = os.path.join(self.data_path, "financial_account_serving")
        self.ft_path_in_hdfs = os.path.join(
            self.data_path, "financial_transactions_serving"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_ft = self.spark.read.parquet(self.ft_path_in_hdfs).withColumn(
            "time", from_unixtime(col("time"))
        )
        df_fa = self.spark.read.parquet(self.fa_path_in_hdfs)
        df_ft.createOrReplaceTempView("tpcxai_financial_transactions_serving")
        df_fa.createOrReplaceTempView("tpcxai_financial_account_serving")

        uc10_sql = """
        select transaction_id, EXTRACT(HOUR FROM time) / 23 as business_hour_norm, amount / transaction_limit as amount_norm
        from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
        """

        joined_df = self.spark.sql(uc10_sql)

        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        result_df = data.withColumn(
            "predicted", self.model_predictor("business_hour_norm", "amount_norm")
        )
        result_df.collect()
        return result_df


class TPCxAIUsecase03PipelineMadlib(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase03PipelineMadlib, self).__init__(
            "tpcxai-usecase03-madlib", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase3.h5", compile=False
        )
        self.le_store = pickle.load(
            open(
                "../../resources/model/tpcxai_sf1/final/tf/usecase3_le_store.pkl", "rb"
            )
        )
        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase3_le_dept.pkl", "rb")
        )

        # register encoder
        sql_to_register_encoder = """
        CREATE OR REPLACE FUNCTION uc3_store_encoder(value BIGINT)
        RETURNS BIGINT AS $$
        BEGIN
            RETURN value - 1;
        END;
        $$ LANGUAGE plpgsql;


        CREATE OR REPLACE FUNCTION uc3_department_encoder(category TEXT)
        RETURNS INT AS $$
        DECLARE
            category_list TEXT[] := ARRAY[
                'AUTOMOTIVE', 'BATH AND SHOWER', 'BEAUTY', 'BEDDING', 'BOYS WEAR',
                'CANDY, TOBACCO, COOKIES', 'CELEBRATION', 'COMM BREAD',
                'COOK AND DINE', 'DAIRY', 'DSD GROCERY', 'ELECTRONICS',
                'FABRICS AND CRAFTS', 'FINANCIAL SERVICES', 'FROZEN FOODS',
                'GIRLS WEAR, 4-6X  AND 7-14', 'GROCERY DRY GOODS', 'HARDWARE',
                'HOME DECOR', 'HOME MANAGEMENT', 'HORTICULTURE AND ACCESS',
                'HOUSEHOLD CHEMICALS/SUPP', 'HOUSEHOLD PAPER GOODS',
                'IMPULSE MERCHANDISE', 'INFANT APPAREL',
                'INFANT CONSUMABLE HARDLINES', 'JEWELRY AND SUNGLASSES',
                'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',
                'MEAT - FRESH & FROZEN', 'MEDIA AND GAMING', 'MENS WEAR',
                'OFFICE SUPPLIES', 'PAINT AND ACCESSORIES', 'PERSONAL CARE',
                'PETS AND SUPPLIES', 'PHARMACY OTC', 'PHARMACY RX',
                'PLAYERS AND ELECTRONICS', 'PRODUCE', 'SERVICE DELI', 'SHOES',
                'SPORTING GOODS', 'TOYS', 'WIRELESS'
            ];
            index INT;
        BEGIN
            -- Find the index of the category in the list
            index := array_position(category_list, category);
            
            -- If not found, return -1
            IF index IS NULL THEN
                RETURN -1;
            ELSE
                RETURN index - 1; -- Convert 1-based index to 0-based index
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        """

        utils.execute_sql_query_via_psycopg2(sql_to_register_encoder)

        # register model through madlib
        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc3_predictor"
        )

        weights = self.model.get_weights()
        weights_flat = [w.flatten() for w in weights]
        weights1d = np.concatenate(weights_flat).ravel()
        weights_bytea = psycopg2.Binary(weights1d.tobytes())

        query = "SELECT madlib.load_keras_model('tpcxai_uc3_predictor', %s, %s, %s, %s)"
        conn = utils.get_psycopg2_connection()
        cur = conn.cursor()
        cur.execute(
            query,
            [self.model.to_json(), weights_bytea, "uc3 model", "tpcxai_uc3_model."],
        )
        conn.commit()

        sql_to_create_view_for_data_processing = """
          DROP VIEW IF EXISTS tpcxai_uc3_view;

          create view tpcxai_uc3_view as (
              SELECT
                  store,
                  department,
                  num_of_week,
                  ROW_NUMBER() OVER () AS ctid,
                  ROW_NUMBER() OVER () AS id,
                  ARRAY [
                  (store)::real, 
                  (department)::real,
                  (num_of_week)::real
              ] AS x
              FROM
                  (
                      select
                          uc3_store_encoder(store) AS store,
                          uc3_department_encoder(department) AS department,
                          num_of_week / 156 AS num_of_week
                      from
                          tpcxai_store_dept_serving
                  ) as t
          );
        """
        utils.execute_sql_query_via_psycopg2(sql_to_create_view_for_data_processing)

        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc3_predictions;"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_run_model_inference = """
          DROP TABLE IF EXISTS tpcxai_uc3_predictions;
          SELECT madlib.madlib_keras_predict_byom('tpcxai_uc3_predictor',  
                                          1,                           
                                          'tpcxai_uc3_view',         
                                          'id',                  
                                          'x',
                                          'tpcxai_uc3_predictions',      
                                          'response',
                                          FALSE,
                                          NULL,
                                          NULL
          );
        """

        data = utils.execute_sql_query_via_psycopg2(query_to_run_model_inference)
        return data

    def data_processing_impl(self, data):

        return data

    def model_inference_impl(self, data):
        query_to_load_data = """select * from tpcxai_uc3_predictions;"""
        return utils.fetch_data_from_postgres_via_psycopg2(query_to_load_data)


class TPCxAIUsecase08PipelineMadlib(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase08PipelineMadlib, self).__init__(
            "tpcxai-usecase08-madlib", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase8.h5", compile=False
        )

        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase8_le_dept.pkl", "rb")
        )

        # register encoder
        sql_to_register_encoder = """
          CREATE OR REPLACE FUNCTION uc8_department_encoder(category TEXT)
          RETURNS INT AS $$
          DECLARE
              category_list TEXT[] := ARRAY[
                  'AUTOMOTIVE', 'BATH AND SHOWER', 'BEAUTY', 'BEDDING', 'BOYS WEAR',
                  'CANDY, TOBACCO, COOKIES', 'CELEBRATION', 'COMM BREAD',
                  'COOK AND DINE', 'DAIRY', 'DSD GROCERY', 'ELECTRONICS',
                  'FABRICS AND CRAFTS', 'FINANCIAL SERVICES', 'FROZEN FOODS',
                  'GIRLS WEAR, 4-6X  AND 7-14', 'GROCERY DRY GOODS', 'HARDWARE',
                  'HOME DECOR', 'HOME MANAGEMENT', 'HORTICULTURE AND ACCESS',
                  'HOUSEHOLD CHEMICALS/SUPP', 'HOUSEHOLD PAPER GOODS',
                  'IMPULSE MERCHANDISE', 'INFANT APPAREL',
                  'INFANT CONSUMABLE HARDLINES', 'JEWELRY AND SUNGLASSES',
                  'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',
                  'MEAT - FRESH & FROZEN', 'MEDIA AND GAMING', 'MENS WEAR',
                  'OFFICE SUPPLIES', 'PAINT AND ACCESSORIES', 'PERSONAL CARE',
                  'PETS AND SUPPLIES', 'PHARMACY OTC', 'PHARMACY RX',
                  'PLAYERS AND ELECTRONICS', 'PRODUCE', 'SERVICE DELI', 'SHOES',
                  'SPORTING GOODS', 'TOYS', 'WIRELESS'
              ];
              index INT;
          BEGIN
              -- Find the index of the category in the list
              index := array_position(category_list, category);
              
              -- If not found, return -1
              IF index IS NULL THEN
                  RETURN -1;
              ELSE
                  RETURN index - 1; -- Convert 1-based index to 0-based index
              END IF;
          END;
          $$ LANGUAGE plpgsql;
        """

        utils.execute_sql_query_via_psycopg2(sql_to_register_encoder)

        # register model through madlib
        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc8_predictor"
        )

        weights = self.model.get_weights()
        weights_flat = [w.flatten() for w in weights]
        weights1d = np.concatenate(weights_flat).ravel()
        weights_bytea = psycopg2.Binary(weights1d.tobytes())

        query = "SELECT madlib.load_keras_model('tpcxai_uc8_predictor', %s, %s, %s, %s)"
        conn = utils.get_psycopg2_connection()
        cur = conn.cursor()
        cur.execute(
            query,
            [self.model.to_json(), weights_bytea, "uc8 model", "tpcxai_uc8_model."],
        )
        conn.commit()

        sql_to_create_view_for_data_processing = """
          DROP VIEW IF EXISTS tpcxai_uc8_view;

          create view tpcxai_uc8_view as (
              SELECT
                  o_order_id,
                  department,
                  quantity,
                  scan_count,
                  weekday,
                  ROW_NUMBER() OVER () AS ctid,
                  ROW_NUMBER() OVER () AS id,
                  ARRAY [
                  (quantity)::real, 
                  (scan_count)::real,
                  (weekday)::real,
                  (department)::real
              ] AS x
              FROM
                  (
                      SELECT 
                    o_order_id,
                    uc8_department_encoder(department) as department,
                    quantity,
                    SUM(quantity) AS scan_count,                
                    MIN(EXTRACT(DOW FROM date)) AS weekday     
                  FROM tpcxai_order_serving 
                  JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
                  JOIN tpcxai_product_serving ON li_product_id = p_product_id
                  GROUP BY o_order_id, date, department, quantity
                  ) as t
          );
        """
        utils.execute_sql_query_via_psycopg2(sql_to_create_view_for_data_processing)

        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc8_predictions;"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_run_model_inference = """
        DROP TABLE IF EXISTS tpcxai_uc8_predictions;
          SELECT madlib.madlib_keras_predict_byom('tpcxai_uc8_predictor',  
                                          1,                           
                                          'tpcxai_uc8_view',         
                                          'id',                  
                                          'x',
                                          'tpcxai_uc8_predictions',      
                                          'response',
                                          FALSE,
                                          NULL,
                                          NULL
          );
        """

        data = utils.execute_sql_query_via_psycopg2(query_to_run_model_inference)
        return data

    def data_processing_impl(self, data):

        return data

    def model_inference_impl(self, data):
        query_to_load_data = """select * from tpcxai_uc8_predictions;"""
        return utils.fetch_data_from_postgres_via_psycopg2(query_to_load_data)


class TPCxAIUsecase08PipelineMLMadlib(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase08PipelineMLMadlib, self).__init__(
            "tpcxai-usecase08-ml-madlib", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()

        utils.execute_sql_query_via_psycopg2(
            """
            DROP TABLE IF EXISTS public.uc8_xgboost;
            CREATE TABLE public.uc8_xgboost (
              model BYTEA,             -- Binary data for the model
              label_encoder BYTEA,     -- Binary data for the label encoder
              features TEXT[],         -- Array of text strings for features
              params_index INTEGER     -- Integer column for parameters index
          );
        """
        )

        self.model = pickle.load(
            open(
                "../../resources/model/tpcxai_sf1/final/tf/usecase8_ml_xgboost.pkl",
                "rb",
            )
        )
        self.xgboost_le = pickle.load(
            open(
                "../../resources/model/tpcxai_sf1/final/tf/usecase8_ml_xgboost_le.pkl",
                "rb",
            )
        )
        # load model into postgres

        params_index = 1

        # Serialize model and label_encoder using pickle
        serialized_model = pickle.dumps(self.model)
        serialized_label_encoder = pickle.dumps(self.xgboost_le)

        insert_query = """
        INSERT INTO public.uc8_xgboost (model, label_encoder, features, params_index)
        VALUES (%s, %s, %s, %s);
        """
        conn = utils.get_psycopg2_connection()
        cur = conn.cursor()
        # Execute the query with serialized data
        features = ["department", "quantity", "scan_count", "weekday"]
        cur.execute(
            insert_query,
            (serialized_model, serialized_label_encoder, features, params_index),
        )
        # Commit and close the connection
        conn.commit()
        cur.close()
        conn.close()

        # register encoder
        sql_to_register_encoder = """
          CREATE OR REPLACE FUNCTION uc8_department_encoder(category TEXT)
          RETURNS INT AS $$
          DECLARE
              category_list TEXT[] := ARRAY[
                  'AUTOMOTIVE', 'BATH AND SHOWER', 'BEAUTY', 'BEDDING', 'BOYS WEAR',
                  'CANDY, TOBACCO, COOKIES', 'CELEBRATION', 'COMM BREAD',
                  'COOK AND DINE', 'DAIRY', 'DSD GROCERY', 'ELECTRONICS',
                  'FABRICS AND CRAFTS', 'FINANCIAL SERVICES', 'FROZEN FOODS',
                  'GIRLS WEAR, 4-6X  AND 7-14', 'GROCERY DRY GOODS', 'HARDWARE',
                  'HOME DECOR', 'HOME MANAGEMENT', 'HORTICULTURE AND ACCESS',
                  'HOUSEHOLD CHEMICALS/SUPP', 'HOUSEHOLD PAPER GOODS',
                  'IMPULSE MERCHANDISE', 'INFANT APPAREL',
                  'INFANT CONSUMABLE HARDLINES', 'JEWELRY AND SUNGLASSES',
                  'LADIESWEAR', 'LAWN AND GARDEN', 'LIQUOR,WINE,BEER',
                  'MEAT - FRESH & FROZEN', 'MEDIA AND GAMING', 'MENS WEAR',
                  'OFFICE SUPPLIES', 'PAINT AND ACCESSORIES', 'PERSONAL CARE',
                  'PETS AND SUPPLIES', 'PHARMACY OTC', 'PHARMACY RX',
                  'PLAYERS AND ELECTRONICS', 'PRODUCE', 'SERVICE DELI', 'SHOES',
                  'SPORTING GOODS', 'TOYS', 'WIRELESS'
              ];
              index INT;
          BEGIN
              -- Find the index of the category in the list
              index := array_position(category_list, category);
              
              -- If not found, return -1
              IF index IS NULL THEN
                  RETURN -1;
              ELSE
                  RETURN index - 1; -- Convert 1-based index to 0-based index
              END IF;
          END;
          $$ LANGUAGE plpgsql;
        """

        utils.execute_sql_query_via_psycopg2(sql_to_register_encoder)

        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc8_predictions;"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_get_inference_data = """
        DROP TABLE IF EXISTS tpcxai_uc8_table_serving;

        CREATE TABLE tpcxai_uc8_table_serving as (
            SELECT
            ROW_NUMBER() OVER () AS id,
                department,
                quantity,
                scan_count,
                weekday
            FROM
                (
                    SELECT 
                  o_order_id,
                  uc8_department_encoder(department) as department,
                  quantity,
                  SUM(quantity) AS scan_count,                
                  MIN(EXTRACT(DOW FROM date)) AS weekday     
                FROM tpcxai_order_serving 
                JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id 
                JOIN tpcxai_product_serving ON li_product_id = p_product_id
                GROUP BY o_order_id, date, department, quantity
                ) as t
        );
        """

        utils.execute_sql_query_via_psycopg2(query_to_get_inference_data)

        data = None
        return data

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        query_to_run_model_inference = """
        DROP TABLE IF EXISTS xgb_single_score_out, xgb_single_score_out_metrics, xgb_single_score_out_roc_curve;

        SELECT madlib.xgboost_predict(
            'tpcxai_uc8_table_serving',          -- test_table
            'uc8_xgboost',   -- model_table
            'xgb_single_score_out',    -- predict_output_table
            'id'               -- id_column
        );

        """
        utils.execute_sql_query_via_psycopg2(query_to_run_model_inference)
        return None


class TPCxAIUsecase10PipelineMadlib(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10PipelineMadlib, self).__init__(
            "tpcxai-usecase10-madlib", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase10.h5", compile=False
        )

        # register model through madlib
        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc10_predictor"
        )

        weights = self.model.get_weights()
        weights_flat = [w.flatten() for w in weights]
        weights1d = np.concatenate(weights_flat).ravel()
        weights_bytea = psycopg2.Binary(weights1d.tobytes())

        query = (
            "SELECT madlib.load_keras_model('tpcxai_uc10_predictor', %s, %s, %s, %s)"
        )
        conn = utils.get_psycopg2_connection()
        cur = conn.cursor()
        cur.execute(
            query,
            [self.model.to_json(), weights_bytea, "uc10 model", "tpcxai_uc10_model."],
        )
        conn.commit()

        sql_to_create_view_for_data_processing = """
          DROP VIEW IF EXISTS tpcxai_uc10_view;
          create view tpcxai_uc10_view as (
          SELECT 
              transaction_id AS id, 
              ROW_NUMBER() OVER () AS ctid,
              ARRAY[
                  (EXTRACT(HOUR FROM time) / 23.0)::real, 
                  (amount / transaction_limit)::real
              ] AS x
          FROM tpcxai_financial_account_serving 
          JOIN tpcxai_financial_transactions_serving 
              ON fa_customer_sk = sender_id
          );
        """
        utils.execute_sql_query_via_psycopg2(sql_to_create_view_for_data_processing)

        utils.execute_sql_query_via_psycopg2(
            "DROP TABLE IF EXISTS tpcxai_uc10_predictions;"
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_run_model_inference = """
        DROP TABLE IF EXISTS tpcxai_uc10_predictions;
          SELECT madlib.madlib_keras_predict_byom('tpcxai_uc10_predictor',  
                                         1,                           
                                        'tpcxai_uc10_view',         
                                        'id',                  
                                        'x',
                                        'tpcxai_uc10_predictions',      
                                        'response',
                                        FALSE,
                                        NULL,
                                        NULL
            );
        """

        data = utils.execute_sql_query_via_psycopg2(query_to_run_model_inference)
        return data

    def data_processing_impl(self, data):

        return data

    def model_inference_impl(self, data):
        query_to_load_data = """select * from tpcxai_uc10_predictions;"""
        return utils.fetch_data_from_postgres_via_psycopg2(query_to_load_data)


class TPCxAIUsecase10MLPipelineMadlib(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10MLPipelineMadlib, self).__init__(
            "tpcxai-usecase10-ml-madlib", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()

        utils.execute_sql_query_via_psycopg2(
            """
        DROP TABLE IF EXISTS tpcxai_uc10_table_training;
        CREATE TABLE tpcxai_uc10_table_training as (
        SELECT 
            transaction_id AS id, 
            (is_fraud != 0) AS label,
            ARRAY[
                (EXTRACT(HOUR FROM time) / 23.0)::real, 
                (amount / transaction_limit)::real
            ] AS x
        FROM tpcxai_financial_account_training
        JOIN tpcxai_financial_transactions_training 
            ON fa_customer_sk = sender_id
        );

        DROP TABLE IF EXISTS uc10_logregr;
        DROP TABLE IF EXISTS uc10_logregr_summary;

        SELECT madlib.logregr_train(
            'tpcxai_uc10_table_training',
            'uc10_logregr',
            'label',  
            'x',  
            NULL,
            '100',
            'cg'
        );
        """
        )

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        query_to_gather_serving_data = """

          DROP TABLE IF EXISTS tpcxai_uc10_table_serving_table;
          CREATE TABLE tpcxai_uc10_table_serving_table as (
            SELECT
              transaction_id AS id,
              ARRAY [
                        (EXTRACT(HOUR FROM time) / 23.0)::real, 
                        (amount / transaction_limit)::real
                      ] AS x
            FROM
              tpcxai_financial_account_serving
              JOIN tpcxai_financial_transactions_serving ON fa_customer_sk = sender_id
          );
        """

        data = utils.execute_sql_query_via_psycopg2(query_to_gather_serving_data)
        return data

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        query_to_run_model_inference = """
        SELECT
          madlib.logregr_predict(coef, x)
        FROM
          uc10_logregr m,
          tpcxai_uc10_table_serving_table;
        """
        return utils.fetch_data_from_postgres_via_psycopg2(query_to_run_model_inference)

class TPCxAIUsecase07MLPipelineML(Pipeline):

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase07MLPipelineML, self).__init__(
            "tpcxai-usecase07-ml", num_loop=num_loop
        )
        with open(
            "../../resources/model/tpcxai_sf1/final/tf/usecase7_svd.pkl", "rb"
        ) as f:
            self.model = pickle.load(f)

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # Use case 10, trainig query
        query_to_fetch_serving_data = """
        select user_id, product_id
        from tpcxai_product_rating_serving
        """

        data = utils.fetch_data_from_postgres_via_connectorx(
            query_to_fetch_serving_data
        )
        return data

    def data_processing_impl(self, data):
        X_features = data[["user_id", "product_id"]].values
        return X_features

    def model_inference_impl(self, data):
        results = []
        for i in range(len(data)):
            user_id = data[i, 0]
            product_id = data[i, 1]
            results.append(self.model.predict(user_id, product_id).est)
        return data


class TPCxAIUsecase07MLPipelineMadlib(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase07MLPipelineMadlib, self).__init__(
            "tpcxai-usecase07-ml-madlib", num_loop=num_loop
        )
        self.postgres_conn_param = utils.get_connectorx_configuration()

        query_to_initialize = """
        CREATE OR REPLACE PROCEDURE uc07_preprocess(schema VARCHAR(100), output_table VARCHAR(200))
        LANGUAGE plpgsql
        AS $$
        declare
            has_rating_column boolean;
        BEGIN

            EXECUTE format('
            SELECT EXISTS (
                  SELECT 1
                  FROM information_schema.columns
                  WHERE table_name = ''tpcxai_product_rating_training''
                  AND table_schema = ''%I''
                  AND column_name = ''rating''
                  )', schema)
            INTO has_rating_column;

            EXECUTE FORMAT('DROP VIEW IF EXISTS %I', output_table);

            IF has_rating_column THEN
                EXECUTE FORMAT('CREATE OR REPLACE VIEW %I AS
                                SELECT user_id + 1 AS user_id,
                                      product_id,
                                      CAST(rating AS FLOAT) AS rating
                                FROM %I.tpcxai_product_rating_training', output_table, schema);
            ELSE
                EXECUTE FORMAT('CREATE OR REPLACE VIEW %I AS
                                SELECT user_id + 1 AS user_id,
                                      product_id
                                FROM %I.tpcxai_product_rating_serving', output_table, schema);
            END IF;
        END;
        $$;



        drop procedure if exists uc07_train(adjust_params boolean);
        CREATE OR REPLACE PROCEDURE uc07_train(input_table text, model text, adjust_params boolean DEFAULT true)
        AS $$
        DECLARE
            numRows INTEGER;
            numCols INTEGER;
        BEGIN
            execute format('Drop table if exists %I', model);
            -- Get the number of rows
            EXECUTE format('SELECT matrix_ndims[1] FROM (SELECT madlib.matrix_ndims(''%I'', ''row=user_id, col=product_id, val=rating'')) AS foo', input_table) INTO numRows;

            -- Get the number of columns
            EXECUTE format('SELECT matrix_ndims[2] FROM (SELECT madlib.matrix_ndims(''%I'', ''row=user_id, col=product_id, val=rating'')) AS foo', input_table) INTO numCols;

            -- Execute lmf with adjusted parameters if adjust_params is true
            IF adjust_params THEN
                EXECUTE format('SELECT madlib.lmf_igd_run(%L, %L, %L, %L, %L, %L, %L, %L, %L, %L, %L, %L)',
                              model, -- output
                              input_table, -- input
                              'user_id', -- rows
                              'product_id', -- cols
                              'rating', -- values
                              numRows, -- row dim
                              numCols, -- col dim
                              100, -- max rank (number of latent factors)
                              0.005, -- step size (learning rate)
                              0.1, -- scale_factor (initialization)
                              20, -- num_iterations
                              1e-4); -- tolerance
            ELSE
                EXECUTE format('SELECT madlib.lmf_igd_run(%L, %L, %L, %L, %L, %s, %s, %s)',
                              model, -- output
                              input_table, -- input
                              'user_id', -- rows
                              'product_id', -- cols
                              'rating', -- values
                              numRows, -- row dim
                              numCols,
                                20); -- col dim
            END IF;
        END;
        $$ LANGUAGE plpgsql;

        drop procedure if exists uc07_predict(input_table varchar, output_table varchar, model varchar);
        CREATE OR REPLACE PROCEDURE uc07_predict(
          input_table varchar,
          model VARCHAR,
          output_table VARCHAR
        )
        AS $$
        DECLARE
          avg NUMERIC;
          max_user_id int;
          max_product_id int;
          matrix_u DOUBLE PRECISION[];
          matrix_v DOUBLE PRECISION[];
        BEGIN
          -- Drop the output table if it exists
          EXECUTE format('DROP TABLE IF EXISTS %I', output_table);

          -- Create the output table
          EXECUTE format('
            CREATE TABLE %I (
              user_id INTEGER,
              product_id INTEGER,
              prediction DOUBLE PRECISION
            )',
            output_table);

          -- Retrieve the avg in case of missing product_id / user_id
          SELECT avg(rating) FROM public.tpcxai_product_rating_training INTO avg;
          SELECT max(user_id) FROM public.tpcxai_product_rating_training INTO max_user_id;
          SELECT max(product_id) FROM public.tpcxai_product_rating_training INTO max_product_id;

        -- Fetch the matrix_u and matrix_v from the model table
          EXECUTE format('
            SELECT matrix_u
            FROM %I
            WHERE id = 1
          ', model)
          INTO matrix_u;

          EXECUTE format('
            SELECT matrix_v
            FROM %I
            WHERE id = 1
          ', model)
          INTO matrix_v;

          -- Calculate the dot product for all rows in the productrating table
          EXECUTE format('
            INSERT INTO %I (user_id, product_id, prediction)
            SELECT
              pr.user_id,
              pr.product_id,
              CASE
                WHEN pr.user_id > %L OR pr.product_id > %L THEN %L
                ELSE COALESCE(
                  madlib.array_dot($1[pr.user_id:pr.user_id][1:100], $2[pr.product_id:pr.product_id][1:100]),
                  %L
                )
              END AS prediction
            FROM %I pr',
            output_table, max_user_id, max_product_id, avg, avg, input_table)
          USING matrix_u, matrix_v;

          -- Adjust the prediction values based on the conditions
          EXECUTE format('
            UPDATE %I
            SET prediction = CASE
              WHEN prediction > 5 THEN 5
              WHEN prediction < 1 THEN 1
              ELSE ROUND(prediction)
            END',
            output_table);
        END;
        $$ LANGUAGE plpgsql;

        create or replace procedure uc07_score(
            prediction_table varchar,
            output_table varchar)
        as $$
        declare
            mae_result float;
        begin
            EXECUTE format('DROP TABLE IF EXISTS %I', output_table);
            execute format('
                    create table %I as (
                    select l.user_id, l.product_id, l.rating as true_rating, p.prediction
                        from score.productrating_labels l join
                            %I p on l.user_id = p.user_id and l.product_id = p.product_id);
                    ', output_table, prediction_table);

            EXECUTE format('SELECT calculate_mae(%L, %L, %L)', output_table, 'true_rating', 'prediction') INTO mae_result;

            INSERT INTO public.evaluation_results (usecase, evaluation_score)
            VALUES ('07', mae_result);
        end;
        $$ language plpgsql;


        create or replace procedure uc07_serve(output_table varchar)
        language plpgsql
        as $$
        begin
            call uc07_preprocess('serve', 'uc07_serve_preprocessed');
            execute format('call uc07_predict(''uc07_serve_preprocessed'', ''uc07_model'', ''%I'');', output_table);
        end;
        $$;

        CREATE OR REPLACE PROCEDURE public.uc07_predict_numpy(model text, predictions text)
        LANGUAGE plpython3u
        AS $procedure$
          import numpy as np
          import gc
          
          # Execute the query and fetch the results
          result = plpy.execute("SELECT matrix_u, matrix_v FROM {model}".format(model=model))
          #plpy.notice(result[0]['matrix_u'])
          mat_u = np.array(result[0]['matrix_u'])
          mat_v = np.array(result[0]['matrix_v'])
          #plpy.notice((mat_u.shape))
          #plpy.notice((mat_v.shape))
          mat_prod = np.matmul(mat_u, mat_v.T)
          plpy.notice((mat_prod.shape))
          del mat_u
          del mat_v
          gc.collect()							# gc of matrices U and V
          mat_prod = np.round(mat_prod)			# round
          mat_prod = np.clip(mat_prod, 1, 5)	# clip to 1-5
          gc.collect()							# gc of temp intermediate matrices

          query = "DROP TABLE IF EXISTS public.{predictions}".format(predictions=predictions) #uc07_predictions_numpy
          result = plpy.execute(query)
          query = '''CREATE TABLE public.{predictions} (
          unnest_row_id int4 NULL,
          unnest_result _float8 NULL)'''.format(predictions=predictions)
          result = plpy.execute(query)
          query = "INSERT INTO public.{predictions} (unnest_row_id, unnest_result) VALUES ($1, $2)".format(predictions=predictions)
          plan = plpy.prepare(query, ["integer", "float8[]"])
          for i, row in enumerate(mat_prod):
            plpy.execute(plan, [i+1, mat_prod[i]])
          $procedure$;

          CREATE OR REPLACE PROCEDURE uc07_predict_with_matrix_mult(model text, predictions text)
        LANGUAGE plpgsql
        AS $$
        BEGIN
            -- Use CTEs to avoid creating intermediate tables
            drop table if exists lora_u;
            drop table if exists lora_v;
            EXECUTE format('Drop table if exists %I', predictions);
            EXECUTE format('
                create table lora_u AS (
                    SELECT (madlib.array_unnest_2d_to_1d(matrix_u)).*
                    FROM %I
                    WHERE id = 1
                );
                create table lora_v AS (
                    SELECT (madlib.array_unnest_2d_to_1d(matrix_v)).*
                    FROM %I
                    WHERE id = 1
                );
                SELECT madlib.matrix_mult(
                    ''lora_u'', ''row=unnest_row_id, val=unnest_result'',
                    ''lora_v'', ''row=unnest_row_id, val=unnest_result, trans=true'',
                    %L
                )', model, model, predictions);
        END;
        $$;
        """
        utils.execute_sql_query_via_psycopg2(query_to_initialize)

        query_to_train_svd = """
          CALL uc07_preprocess('public', 'uc07_train_preprocessed');
          CALL uc07_train('uc07_train_preprocessed', 'uc07_model', false);
        """

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):

        query_to_preprocess = """
          CALL uc07_preprocess('public', 'uc07_score_preprocessed');
        """

        data = utils.execute_sql_query_via_psycopg2(query_to_preprocess)
        return data

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):
        query_to_run_model_inference = """
          CALL uc07_predict('uc07_score_preprocessed', 'uc07_model', 'uc07_score_predictions');
        """
        return utils.execute_sql_query_via_psycopg2(query_to_run_model_inference)


class TPCxAIUsecase07MLPipelineEvaDB(Pipeline):

    def __del__(self):
        self.cursor.query(
            "USE postgres_data{DROP VIEW IF EXISTS evadb_tpcxai_uc7};"
        ).df()

    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase07MLPipelineEvaDB, self).__init__(
            "tpcxai-usecase07-ml-evadb", num_loop=num_loop
        )
        # self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        utils.setup_postgres_for_evadb()
        self.cursor = evadb.connect().cursor()

        # deregister function
        self.cursor.query("DROP FUNCTION IF EXISTS Model_UseCase07_ML_EVADB;").df()
        # register function
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS Model_UseCase07_ML_EVADB
            IMPL './function_tpcxai_evadb.py';
            """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_to_fetch_serving_data = "SELECT Model_UseCase07_ML_EVADB(user_id, product_id).label FROM postgres_data.tpcxai_product_rating_serving"
        result_df = self.cursor.query(query_to_fetch_serving_data).df()
        return result_df.values


class TPCxAIUsecase07PipelineSparkMLHadoop(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        # np.save("evadb_ffnn_reg.npy", list_hidden_layer_sizes)
        self.spark = (
            SparkSession.builder.appName("ModelInference")
            .config("spark.driver.memory", "60g")
            .config("spark.sql.legacy.parquet.nanosAsLong", "true")
            .getOrCreate()
        )
        super(TPCxAIUsecase07PipelineSparkMLHadoop, self).__init__(
            "tpcxai-usecase07-sparkhadoop-ml", num_loop=num_loop
        )

        from register_tpcxai_spark_func import uc07_svd_ml_spark_predicator

        self.model_predictor = uc07_svd_ml_spark_predicator

        self.data_path = "hdfs://localhost:9900/user/velox/data/tpcxai/"
        self.pr_path_in_hdfs = os.path.join(self.data_path, "product_rating_serving")

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        df_pr = self.spark.read.parquet(self.pr_path_in_hdfs)
        df_pr.createOrReplaceTempView("tpcxai_product_rating_serving")

        uc07_sql = """
        select user_id, product_id from tpcxai_product_rating_serving;
        """

        joined_df = self.spark.sql(uc07_sql)

        return joined_df

    def data_processing_impl(self, data):
        return data

    def model_inference_impl(self, data):

        result_df = data.withColumn(
            "predicted", self.model_predictor("user_id", "product_id")
        )
        result_df.collect()
        return result_df


class TPCxAIUsecase10PipelinePGML(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase10PipelinePGML, self).__init__(
            "tpcxai-usecase10-pgml", num_loop=num_loop
        )
        #self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        query_to_fetch_serving_data = """
        create or replace view uc10_serving_data as select transaction_id, ARRAY [(EXTRACT(HOUR FROM time) / 23)::real, (amount / transaction_limit)::real] AS features from tpcxai_financial_account_serving join tpcxai_financial_transactions_serving on fa_customer_sk=sender_id
        """
        
        # Get the connection
        conn = utils.get_psycopg2_connection()
        try:
            # Get the cursor
            cursor = conn.cursor()
            cursor.execute("DROP VIEW IF EXISTS uc10_serving_data;")
            cursor.execute(query_to_fetch_serving_data)
            
            # Close the cursor and connection
            cursor.close()
            conn.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
        

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_prediction = "SELECT transaction_id, pgml.predict('uc10_logistic_model', features) as prediction from uc10_serving_data;"
        result_df = utils.fetch_data_from_postgres_via_psycopg2(query_prediction)
        return result_df.values




class TPCxAIUsecase8PipelinePGML(Pipeline):
    def __init__(
        self,
        num_loop=10,
    ):
        super(TPCxAIUsecase8PipelinePGML, self).__init__(
            "tpcxai-usecase8-pgml", num_loop=num_loop
        )
        #self.postgres_conn_param = utils.get_connectorx_configuration()
        # TODO: init
        query_to_fetch_serving_data = """
        CREATE OR REPLACE VIEW uc8_serving_data as (
              SELECT o_order_id, ARRAY [(department)::real, (quantity)::real, (quantity)::real, (weekday)::real] AS features
              FROM
                  (
                    SELECT
                    o_order_id,
                    uc8_department_encoder(department) as department,
                    quantity,
                    SUM(quantity) AS scan_count,
                    MIN(EXTRACT(DOW FROM date)) AS weekday
                  FROM tpcxai_order_serving
                  JOIN tpcxai_lineitem_serving ON o_order_id = li_order_id
                  JOIN tpcxai_product_serving ON li_product_id = p_product_id
                  GROUP BY o_order_id, date, department, quantity
                  ) as t
          );
        """
        
        # Get the connection
        conn = utils.get_psycopg2_connection()
        try:
            # Get the cursor
            cursor = conn.cursor()
            cursor.execute("DROP VIEW IF EXISTS uc8_serving_data;")
            cursor.execute(query_to_fetch_serving_data)
            
            # Close the cursor and connection
            cursor.close()
            conn.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
        

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self, batch_size):
        # TODO: implement data loading
        return None

    def data_processing_impl(self, data):
        # TODO data processing
        return data

    def model_inference_impl(self, data):
        # TODO model inference
        query_prediction = "SELECT o_order_id, pgml.predict('uc8_xgboost_model', features) as prediction from uc8_serving_data;"
        result_df = utils.fetch_data_from_postgres_via_psycopg2(query_prediction)
        return result_df.values

        
