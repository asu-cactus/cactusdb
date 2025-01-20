from collections import OrderedDict
import os
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from evadb.functions.decorators.decorators import forward, setup
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision
import tensorflow as tf
import pickle


class Model_UseCase3_EVADB(AbstractFunction):

    def __del__(self):
        print("[INFO] Summarization of Model_UseCase3_EVADB: \n")

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
        return "Model_UseCase3_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        self.max_num_of_week = 52 * 3
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

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                    "store",
                    "department",
                    "num_of_week",
                ],
                column_types=[
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["predicted"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        data["store"] = self.le_store.transform(data["store"].values)
        data["department"] = self.le_dept.transform(data[["department"]].values)
        data["num_of_week"] = (data["num_of_week"] - 0) / self.max_num_of_week

        X_serve = data[["store", "department", "num_of_week"]].values.astype(float)
        y_pred = self.model.predict(X_serve)

        result_df = pd.DataFrame(
            {
                "predicted": y_pred.flatten(),
            }
        )
        return result_df


class Model_UseCase8_EVADB(AbstractFunction):

    def __del__(self):
        print("[INFO] Summarization of Model_UseCase8_EVADB: \n")

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
        return "Model_UseCase8_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        # self.max_num_of_week = 52*3
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase8.h5", compile=False
        )
        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase8_le_dept.pkl", "rb")
        )

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["quantity", "scan_count", "weekday", "department"],
                column_types=[
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.STR,
                ],
                column_shapes=[(None,), (None,), (None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["predicted"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        data["scan_count"] = data["scan_count"].astype(int)
        data["weekday"] = data["weekday"].astype(int)
        data["department_encoded"] = self.le_dept.transform(data[["department"]].values)

        X_serve = data[
            ["quantity", "scan_count", "weekday", "department_encoded"]
        ].values.astype(float)
        y_pred = np.argmax(self.model.predict(X_serve), axis=1)

        result_df = pd.DataFrame(
            {
                "predicted": y_pred.flatten(),
            }
        )
        return result_df


class Model_UseCase8_ML_EVADB(AbstractFunction):

    def __del__(self):
        print("[INFO] Summarization of Model_UseCase8_ML_EVADB: \n")

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
        return "Model_UseCase8_ML_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        # self.max_num_of_week = 52*3
        self.model = pickle.load(
            open(
                "../../resources/model/tpcxai_sf1/final/tf/usecase8_ml_xgboost.pkl",
                "rb",
            )
        )
        self.le_dept = pickle.load(
            open("../../resources/model/tpcxai_sf1/final/tf/usecase8_le_dept.pkl", "rb")
        )

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["department", "quantity", "scan_count", "weekday"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["predicted"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        data["scan_count"] = data["scan_count"].astype(int)
        data["weekday"] = data["weekday"].astype(int)
        data["department_encoded"] = self.le_dept.transform(data[["department"]].values)

        X_serve = data[
            ["department_encoded", "quantity", "scan_count", "weekday"]
        ].values.astype(float)
        y_pred = self.model.predict(X_serve)

        result_df = pd.DataFrame(
            {
                "predicted": y_pred.flatten(),
            }
        )
        return result_df


class Model_UseCase10_EVADB(AbstractFunction):

    def __del__(self):
        print("[INFO] Summarization of Model_UseCase10_EVADB: \n")

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
        return "Model_UseCase10_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        self.model = tf.keras.models.load_model(
            "../../resources/model/tpcxai_sf1/final/tf/usecase10.h5", compile=False
        )

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                    "business_hour_norm",
                    "amount_norm",
                ],
                column_types=[NdArrayType.FLOAT32, NdArrayType.FLOAT32],
                column_shapes=[(None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        X_serve = data[["business_hour_norm", "amount_norm"]].values.astype(float)
        y_pred = self.model.predict(X_serve)

        result_df = pd.DataFrame(
            {
                "label": y_pred.flatten(),
            }
        )
        return result_df


class Model_UseCase10_ML_EVADB(AbstractFunction):

    def __del__(self):
        print(
            "[INFO] Summarization of Model_UseCase10_ML_EVADB: # batch: {}, max_batch_size: {}".format(
                self.num_batches, self.max_batch_size
            )
        )

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
        return "Model_UseCase10_ML_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        with open(
            "../../resources/model/tpcxai_sf1/final/tf/usecase10_lr_model.h5", "rb"
        ) as f:
            self.model = pickle.load(f)
        self.num_batches = 0
        self.max_batch_size = 0

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                    "business_hour_norm",
                    "amount_norm",
                ],
                column_types=[NdArrayType.FLOAT32, NdArrayType.FLOAT32],
                column_shapes=[(None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        X_serve = data[["business_hour_norm", "amount_norm"]].values.astype(float)
        y_pred = self.model.predict(X_serve)

        result_df = pd.DataFrame(
            {
                "label": y_pred.flatten(),
            }
        )
        self.num_batches += 1
        self.max_batch_size = max(self.max_batch_size, len(X_serve))
        return result_df


class Model_UseCase07_ML_EVADB(AbstractFunction):

    def __del__(self):
        print(
            "[INFO] Summarization of Model_UseCase07_ML_EVADB: # batch: {}, max_batch_size: {}".format(
                self.num_batches, self.max_batch_size
            )
        )

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
        return "Model_UseCase07_ML_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        with open(
            "../../resources/model/tpcxai_sf1/final/tf/usecase7_svd.pkl", "rb"
        ) as f:
            self.model = pickle.load(f)
        self.num_batches = 0
        self.max_batch_size = 0

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                    "user_id",
                    "product_id",
                ],
                column_types=[NdArrayType.INT32, NdArrayType.INT32],
                column_shapes=[(None,), (None,)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        results = []
        for idx, row in data.iterrows():
            user_id = row["user_id"]
            product_id = row["product_id"]
            results.append(self.model.predict(user_id, product_id).est)

        result_df = pd.DataFrame(
            {
                "label": results,
            }
        )
        self.num_batches += 1
        self.max_batch_size = max(self.max_batch_size, len(data))
        return result_df
