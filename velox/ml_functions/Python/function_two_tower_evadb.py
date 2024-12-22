from collections import OrderedDict
import os
import time
import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
from evadb.functions.decorators.decorators import forward, setup
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe, NumpyArray
from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision
import xgboost as xgb


class CustomerEncoder(nn.Module):
    def __init__(self, input_size, num_customer, num_addr, num_age, num_country, customer_emb_dim, addr_emb_dim,
                 age_emb_dim, country_emb_dim):
        super(CustomerEncoder, self).__init__()
        self.emb_customer = nn.Embedding(num_customer, customer_emb_dim)
        self.emb_addr = nn.Embedding(num_addr, addr_emb_dim)
        self.emb_age = nn.Embedding(num_age, age_emb_dim)
        self.emb_country = nn.Embedding(num_country, country_emb_dim)
        self.fc1 = nn.Linear(input_size + customer_emb_dim + addr_emb_dim + age_emb_dim + country_emb_dim, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x_customer, x_addr, x_age, x_country, x_other):
        e_customer = self.emb_customer(x_customer)
        e_addr = self.emb_addr(x_addr)
        e_age = self.emb_age(x_age)
        e_country = self.emb_country(x_country)
        x = torch.cat((e_customer, e_addr, e_age, e_country, x_other), dim=1)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.relu(self.batch_norm3(self.fc3(x)))
        return x


class ProductEncoder(nn.Module):
    def __init__(self, input_size, num_product, num_dept, product_emb_dim, dept_emb_dim):
        super(ProductEncoder, self).__init__()
        self.emb_product = nn.Embedding(num_product, product_emb_dim)
        self.emb_dept = nn.Embedding(num_dept, dept_emb_dim)
        self.fc1 = nn.Linear(input_size + product_emb_dim + dept_emb_dim, 48)
        self.batch_norm1 = nn.BatchNorm1d(48)
        self.fc2 = nn.Linear(48, 32)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x_product, x_dept, x_other):
        e_product = self.emb_product(x_product)
        e_dept = self.emb_dept(x_dept)
        x = torch.cat((e_product, e_dept, x_other), dim=1)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.relu(self.batch_norm3(self.fc3(x)))
        return x


def convert_string_to_float_list(x):
    try:
        # Split the string by comma and convert each value to float
        return [float(i.strip()) for i in x.strip('{}').split(',')]
    except ValueError:
        return []


def get_age(birth_year):
    return 2024 - birth_year


def concatenate_features(array1, array2):
    return array1 + array2


def vector_add(array1, array2):
    result = [a + b for a, b in zip(array1, array2)]
    return result




class TWO_TOWER_EVADB(AbstractFunction):

    def __del__(self):
        print("[INFO] Summarization of TWO_TOWER_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "TWO_TOWER_EVADB"

    @setup(cacheable=True, batchable=True)
    def setup(self):
        self.customer_model = CustomerEncoder(2, 70710, 35355, 95, 213, 16, 16, 8, 8)
        self.customer_model.load_state_dict(
            torch.load("resources/model/customer_encoder.pth", map_location='cpu', weights_only=True))
        self.customer_model.eval()

        self.product_model = ProductEncoder(1, 708, 46, 16, 8)
        self.product_model.load_state_dict(
            torch.load("resources/model/product_encoder.pth", map_location='cpu', weights_only=True))
        self.product_model.eval()

        self.t_process = 0
        self.t_model_inference = 0
        self.count_inference = 0

    @property
    def labels(self):
        return []

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                      'c_customer_sk',
                      'c_current_addr_sk',
                      'c_preferred_cust_flag',
                      'c_birth_year',
                      'c_birth_country',
                      'avg_customer_rating',
                      'p_product_id',
                      'p_department',
                      'avg_product_rating',
                ],
                column_types=[
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.ANYTYPE,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.ANYTYPE
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
                    (None,)
                ],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["result_vec"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None, 16)],
            )
        ],
    )
    def forward(self, joinDf) -> pd.DataFrame:
        self.count_inference += len(joinDf)

        joinDf['c_age'] = joinDf.apply(lambda row: get_age(row['c_birth_year']), axis=1)
        joinDf['avg_customer_rating'] = joinDf['avg_customer_rating'].astype(float) / 5.0
        joinDf['avg_product_rating'] = joinDf['avg_product_rating'].astype(float) / 5.0

        X_emb_customer = joinDf['c_customer_sk'].values
        X_emb_addr = joinDf['c_current_addr_sk'].values
        X_emb_age = joinDf['c_age'].values
        X_emb_country = joinDf['c_birth_country'].values
        X1 = joinDf[['c_preferred_cust_flag', 'avg_customer_rating']].values

        X_emb_customer_tensor = torch.tensor(X_emb_customer, dtype=torch.long)  # For embedding
        X_emb_addr_tensor = torch.tensor(X_emb_addr, dtype=torch.long)  # For embedding
        X_emb_age_tensor = torch.tensor(X_emb_age, dtype=torch.long)  # For embedding
        X_emb_country_tensor = torch.tensor(X_emb_country, dtype=torch.long)  # For embedding
        X1_tensor = torch.tensor(X1, dtype=torch.float32)

        output = self.customer_model(X_emb_customer_tensor, X_emb_addr_tensor, X_emb_age_tensor, X_emb_country_tensor, X1_tensor)
        joinDf['customer_encode'] = output.tolist()


        X_emb_product = joinDf['p_product_id'].values
        X_emb_dept = joinDf['p_department'].values
        X2 = joinDf[['avg_product_rating']].values

        X_emb_product_tensor = torch.tensor(X_emb_product, dtype=torch.long)  # For embedding
        X_emb_dept_tensor = torch.tensor(X_emb_dept, dtype=torch.long)  # For embedding
        X2_tensor = torch.tensor(X2, dtype=torch.float32)

        output = self.product_model(X_emb_product_tensor, X_emb_dept_tensor, X2_tensor)
        joinDf['product_encode'] = output.tolist()

        joinDf['result_vec'] = joinDf.apply(lambda row: vector_add(row['customer_encode'], row['product_encode']), axis=1)

        result_df = pd.DataFrame(
            {
                "result_vec": joinDf['result_vec'].values,
            }
        )
        return result_df #[['result_vec']].values
