# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import utils
from evadb.functions.decorators.decorators import forward, setup
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision
from models.dssm import DSSM_Torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from models.dssm import DSSM_Torch, DSSM_TF, get_var_feature, get_test_var_feature


class DSSM_EVADB(AbstractFunction):
    def as_numpy(self, val) -> np.ndarray:
        """
        Given a tensor in GPU, detach and get the numpy output
        Arguments:
             val (Tensor): tensor to be converted
        Returns:
            np.ndarray: numpy array representation
        """
        return val.detach().cpu().numpy()

    @property
    def name(self) -> str:
        return "DSSM_EVADB"

    @setup(cacheable=True, function_type="classification", batchable=True)
    def setup(self):
        import torch.nn as nn

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

        item_varlen_feature_columns = [
            VarLenSparseFeat(
                SparseFeat("genres", vocabulary_size=1000, embedding_dim=embedding_dim),
                maxlen=genres_maxlen,
                combiner="mean",
                length_name=None,
            )
        ]

        item_feature_columns += item_varlen_feature_columns
        
        model = DSSM_Torch(
            user_feature_columns, item_feature_columns, task="binary", device=device
        )
        model.compile(
            "adam", "binary_crossentropy", metrics=["auc", "accuracy", "logloss"], lr=lr
        )
        self.model = model
        self.meta = dict()
        self.meta["model"] = model
        self.meta["sparse_features"] = sparse_features
        self.meta["dense_features"] = dense_features
        self.meta["dict_encoder"] = dict_encoder
        self.meta["genres_key2index"] = genres_key2index
        self.meta["genres_maxlen"] = genres_maxlen
        self.model.eval()
        self.timer_process = utils.Timer()
        self.timer_model_inference = utils.Timer()
        self.t_process = 0
        self.t_model_inference = 0

    @property
    def labels(self):
        return list([str(num) for num in range(10)])

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                    "user_id",
                    "gender",
                    "age",
                    "occupation",
                    "user_mean_rating",
                    "movie_id",
                    "genres",
                    "movie_mean_rating",
                ],
                column_types=[
                    NdArrayType.INT32,
                    NdArrayType.STR,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.INT32,
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[
                    (None,),
                    (None,),
                    (None,),
                    (None,),
                    (None,),
                    (None,),
                    (None,),
                    (None,),
                ],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label", "t_process", "t_model_inference"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        outcome = []
        self.timer_process.tic()

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
        self.t_process += self.timer_process.toc()
        self.timer_model_inference.tic()
        predictions = self.model.predict(test_model_input)
        self.t_model_inference += self.timer_model_inference.toc()
        result_df = pd.DataFrame(
            {
                "label": predictions,
                "t_process": self.t_process,
                "t_model_inference": self.t_model_inference,
            }
        )
        return result_df
