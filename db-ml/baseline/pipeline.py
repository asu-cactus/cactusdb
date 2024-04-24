import connectorx as cx
import evadb
import ffnn
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import utils
from abc import ABC, abstractmethod
from models.preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from models.dssm import DSSM_Torch, DSSM_TF, get_var_feature, get_test_var_feature
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StringType



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
        if hasattr(self, 'batch_size'):
            self.list_batches = get_batch_sizes(self.num_sample, self.batch_size)
        print("[INFO] Batches: ", self.list_batches)

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
        data = utils.fetch_data_from_postgres_via_psycopg2(sql_movielens_integrated_result)
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
        data = utils.fetch_data_from_postgres_via_psycopg2(sql_movielens_integrated_result)
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
        result  = self.model(data)
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
            .config("spark.driver.memory", "40g")
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
    features = np.stack(features.str.strip('{}').str.split(',')).astype(np.float32)
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
            .config("spark.driver.memory", "40g")
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

        feature_hdfs_path = "hdfs://localhost:9900/user/velox/data/{}".format(self.ffnn_table_name)
        spark_feature_data = self.spark.read.csv(feature_hdfs_path, header=True, inferSchema=True)
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
        result_df = data.select(predict_batch_udf_hadoop(col("val")).alias("prediction"))
        # result_df.count()
        # result_df.cache().count()
        # result_df.rdd.count()
        result_df.collect()
        # print("count:", result_df.count())
        # print(result_df.show())
        return result_df
