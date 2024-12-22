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


class SimpleNN(nn.Module):
    def __init__(self, num_unique_customer, embedding_dim, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(num_unique_customer, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim + input_size, 48)
        self.fc2 = nn.Linear(48, 24)
        self.fc3 = nn.Linear(24, output_size)  # Output size 2 for binary classification

    def forward(self, x_customer, x_other):
        embedded_customer = self.embedding(x_customer)
        x = torch.cat((embedded_customer, x_other), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)



def convert_string_to_float_list(x):
    try:
        # Split the string by comma and convert each value to float
        return [float(i.strip()) for i in x.strip('{}').split(',')]
    except ValueError:
        return []


def concatenate_features(array1, array2):
    return array1 + array2


daysOfWeek = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
secondsInADay = 86400
maxDayValue = 15340.0

# Define the function to get order features
def get_order_feature(row):
    date_str = row['o_date']
    weekday = row['weekday']
    
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    dtDays = float(int(dt.timestamp()) // secondsInADay) / maxDayValue
    outputVec = [dtDays]

    for day in daysOfWeek:
        outputVec.append(1.0 if weekday == day else 0.0)
    
    return outputVec




class IS_POPULAR_STORE_EVADB(AbstractFunction):
    

    def __del__(self):
        print("[INFO] Summarization of IS_POPULAR_STORE_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "IS_POPULAR_STORE_EVADB"

    @setup(cacheable=True, function_type="classification", batchable=True)
    def setup(self):
        self.t_process = 0
        self.t_model_inference = 0
        self.count_inference = 0

    @property
    def labels(self):
        return list([str(num) for num in range(2)])

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                      'store_feature',
                ],
                column_types=[
                    NdArrayType.ANYTYPE
                ],
                column_shapes=[
                    (None,)
                ],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label"],
                column_types=[
                    NdArrayType.INT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        self.count_inference += len(data)

        data['store_feature'] = data['store_feature'].apply(convert_string_to_float_list)
        X_vals = np.array(data['store_feature'].tolist(), dtype=np.float32)
        #X_vals = data[['store_feature']].values
        row_sums = np.sum(X_vals[:, 1:], axis=1)
        num_columns = X_vals.shape[1] - 1
        row_averages = row_sums / num_columns
        row_averages = np.where(row_averages >= 0.5, 1, 0)

        result_df = pd.DataFrame(
            {
                "label": row_averages,
            }
        )
        return result_df




class DNN_TRIP_TYPE_EVADB(AbstractFunction):
    

    def __del__(self):
        print("[INFO] Summarization of DNN_TRIP_TYPE_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "DNN_TRIP_TYPE_EVADB"

    @setup(cacheable=True, function_type="classification", batchable=True)
    def setup(self):
        self.model = SimpleNN(70710, 16, 77, 1000)
        self.model.load_state_dict(torch.load("resources/model/trip_type_classify.pth", map_location='cpu', weights_only=True))
        self.model.eval()
        self.t_process = 0
        self.t_model_inference = 0
        self.count_inference = 0

    @property
    def labels(self):
        return list([str(num) for num in range(1000)])

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                      'o_customer_sk',
                      'o_date',
                      'weekday',
                      'store_feature',
                ],
                column_types=[
                    NdArrayType.INT32,
                    NdArrayType.STR,
                    NdArrayType.STR,
                    NdArrayType.ANYTYPE
                ],
                column_shapes=[
                    (None,),
                    (None,),
                    (None,),
                    (None,)
                ],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label"],
                column_types=[
                    NdArrayType.INT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        self.count_inference += len(data)

        data['order_feature'] = data.apply(get_order_feature, axis=1)
        data['store_feature'] = data['store_feature'].apply(convert_string_to_float_list)
        data['all_feature'] = data.apply(lambda row: concatenate_features(row['order_feature'], row['store_feature']), axis=1)
        data = data.drop(columns=['weekday', 'o_date', 'order_feature', 'store_feature'])

        X_embed = data['o_customer_sk'].values
        X = np.array(data['all_feature'].tolist(), dtype=np.float32)
        X_embed_tensor = torch.tensor(X_embed, dtype=torch.long)  # For embedding
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = self.model(X_embed_tensor, X_tensor)
        _, predicted = outputs.max(1)

        result_df = pd.DataFrame(
            {
                "label": predicted.tolist(),
            }
        )
        return result_df
