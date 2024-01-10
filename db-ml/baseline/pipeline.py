import utils
import pandas as pd
import numpy as np
import connectorx as cx
import tensorflow as tf
import evadb
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from models.dssm import DSSM_Torch, DSSM_TF


class Pipeline(object):
    """A convenient class to measure the running time of a program"""

    def __init__(self, name, num_sample=500, num_loop=10):
        self.name = name
        self.num_loop = num_loop
        self.num_sample = num_sample
        self.meta = dict()
        pass

    @abstractmethod
    def loading_meta_impl(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def data_loading_impl(self):
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

        data = None
        timer_end_end.tic()
        for _ in tqdm(range(self.num_loop)):
            timer_data_loading.tic()
            data = self.data_loading_impl()
            t_data_loading += timer_data_loading.toc()

            timer_data_processing.tic()
            data = self.data_processing_impl(data)
            t_data_processing += timer_data_processing.toc()

            timer_model_inference.tic()
            data = self.model_inference_impl(data)
            t_model_inference += timer_model_inference.toc()

        t_end_end += timer_end_end.toc() / self.num_loop
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop

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


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        key_ans = x.split("|")
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",\
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(
        var_feature,
        maxlen=max_len,
        padding="post",
    )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    # print("user_hist_list: \n")

    def split(x):
        key_ans = x.split("|")
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding="post")
    return test_hist


class TwoTowerModelPipelineTorch(Pipeline):
    def __init__(self, num_sample=500, num_loop=10):
        super(TwoTowerModelPipelineTorch, self).__init__(
            "two-tower-model-pytorch", num_loop, num_sample
        )
        self.postgres_conn = utils.get_postgres_connection_config()

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

    def data_loading_impl(self):
        sampledUserId = np.random.randint(1, 6041, self.num_sample)
        sampledMovieId = np.random.randint(1, 3707, self.num_sample)
        query_df = pd.DataFrame(
            {"q_user_id": sampledUserId, "q_movie_id": sampledMovieId}
        )
        query_df.to_sql(
            "movielens_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        data = utils.fetch_data_from_postgres_via_sql(sql_movielens_integrated_result)
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


class TwoTowerModelPipelineTF(Pipeline):
    def __init__(self, num_sample=500, num_loop=10):
        super(TwoTowerModelPipelineTF, self).__init__(
            "two-tower-model-tensorflow", num_loop, num_sample
        )
        # self.model = None  # TODO
        # self.model.eval()
        # self.num_sample = num_sample
        self.postgres_conn = utils.get_postgres_connection_config()

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

    def data_loading_impl(self):
        sampledUserId = np.random.randint(1, 6041, self.num_sample)
        sampledMovieId = np.random.randint(1, 3707, self.num_sample)
        query_df = pd.DataFrame(
            {"q_user_id": sampledUserId, "q_movie_id": sampledMovieId}
        )
        query_df.to_sql(
            "movielens_q_temp", self.postgres_conn, index=False, if_exists="replace"
        )
        data = utils.fetch_data_from_postgres_via_sql(sql_movielens_integrated_result)
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


class FFNNEvaDB(Pipeline):
    def __init__(self, num_sample=500, num_loop=10):
        super(FFNNEvaDB, self).__init__("ffnn-evadb", num_loop, num_sample)
        # deregister function
        # register function
        self.cursor = evadb.connect().cursor()
        self.cursor.query("DROP FUNCTION IF EXISTS FFNN_EVADB;").df()
        self.cursor.query(
            """
            CREATE FUNCTION
            IF NOT EXISTS FFNN_EVADB
            IMPL './ffnn_evadb.py';
            """
        ).df()

    def loading_meta_impl(self):
        pass

    def data_loading_impl(self):
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

        data = None
        timer_end_end.tic()
        for _ in tqdm(range(self.num_loop)):
            timer_data_loading.tic()
            data = self.data_loading_impl()
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
            FROM postgres_data.ffnn_data
            LIMIT {};
            """.format(
                    self.num_sample
                )
            ).df()

            t_data_processing += result_df['t_process'].values[-1]
            t_model_inference += result_df['t_model_inference'].values[-1]

        t_end_end += timer_end_end.toc() / self.num_loop
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop

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
