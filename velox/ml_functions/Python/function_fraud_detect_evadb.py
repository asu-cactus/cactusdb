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
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision
import xgboost as xgb


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)



class FraudDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


xgb_model = xgb.Booster()
xgb_model.load_model("resources/model/fraud_xgboost_trans_no_f_5_32.json")

def predict_xgb_trans(features):
    # Convert input features to DMatrix
    dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1))
    # Perform prediction
    preds = xgb_model.predict(dmatrix)
    return float(preds[0])


def convert_string_to_float_list(x):
    try:
        # Split the string by comma and convert each value to float
        return [float(i.strip()) for i in x.strip('{}').split(',')]
    except ValueError:
        return []


def datetime_to_timestamp(date_str):
    # Parse the date string to a datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
    # Return the timestamp in milliseconds
    #return int(dt.timestamp()) - 25200
    return int(dt.timestamp()) + 25200


def is_working_day(date_str):
    # Convert timestamp to datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
    timestamp = int(dt.timestamp()) + 25200
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    weekday = (dt.weekday() + 1) % 7  # Monday is 1 and Sunday is 0
    if weekday == 0 or weekday == 6:
        return 0
    else:
        return 1


def is_adult_during_transaction(date_str, birth_year):
    # Convert timestamp to datetime object
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
    timestamp = int(dt.timestamp()) + 25200
    year = datetime.datetime.fromtimestamp(timestamp).year  # Convert milliseconds to seconds
    age = year - birth_year
    if age >= 18:
        return 1
    else:
        return 0


def get_transaction_features(date_str, amount):   
    fAmount = amount / 16048.0
    
    dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
    timestamp = int(dt.timestamp()) + 25200
    dt = datetime.datetime.fromtimestamp(timestamp)  # Convert milliseconds to seconds
    
    fDay = float(dt.day) / 31.0
    fMonth = float(dt.month) / 12.0
    fYear = float(dt.year) / 2011.0
    fDayOfWeek = float((dt.weekday() + 1) % 7) / 6.0
    
    return [fAmount, fDay, fMonth, fYear, fDayOfWeek]


def get_customer_features(address_num, cust_flag, birth_day, birth_month, birth_year, birth_country, trans_limit):
    fAddressNum = float(address_num) / 35352.0
    fCustFlag = float(cust_flag)
    fBirthDay = float(birth_day) / 31.0
    fBirthMonth = float(birth_month) / 12.0
    fBirthYear = float(birth_year) / 2002.0
    fBirthCountry = float(birth_country) / 212.0
    fTransLimit = float(trans_limit) / 16116.0
    return [fAddressNum, fCustFlag, fBirthDay, fBirthMonth, fBirthYear, fBirthCountry, fTransLimit]


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




class IS_WORKING_DAY_EVADB(AbstractFunction):
    

    def __del__(self):
        print("[INFO] Summarization of IS_WORKING_DAY_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "IS_WORKING_DAY_EVADB"

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
                      't_time',
                ],
                column_types=[
                    NdArrayType.STR
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

        data['label'] = data['t_time'].apply(is_working_day)
        data = data.drop(columns=['t_time'])

        return data




class IS_ADULT_DURING_TRANSACTION_EVADB(AbstractFunction):
    

    def __del__(self):
        print("[INFO] Summarization of IS_ADULT_DURING_TRANSACTION_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "IS_ADULT_DURING_TRANSACTION_EVADB"

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
                      't_time',
                      'c_birth_year',
                ],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.INT32
                ],
                column_shapes=[
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

        data['label'] = data.apply(lambda row: is_adult_during_transaction(row['t_time'], row['c_birth_year']), axis=1)
        data = data.drop(columns=['t_time', 'c_birth_year'])

        return data




class IS_FRAUD_XGB_TRANS_EVADB(AbstractFunction):
    

    def __del__(self):
        print("[INFO] Summarization of IS_FRAUD_XGB_TRANS_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "IS_FRAUD_XGB_TRANS_EVADB"

    @setup(cacheable=True, function_type="regression", batchable=True)
    def setup(self):
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model("resources/model/fraud_xgboost_trans_no_f_5_32.json")
        self.t_process = 0
        self.t_model_inference = 0
        self.count_inference = 0


    @forward(
        input_signatures=[
            PandasDataframe(
                columns=[
                      't_time',
                      'amount',
                ],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32
                ],
                column_shapes=[
                    (None,),
                    (None,)
                ],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["xgb_predict"],
                column_types=[
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        self.count_inference += len(data)

        data['transaction_feature'] = data.apply(lambda row: get_transaction_features(row['t_time'], row['amount']), axis=1)
        X_data = np.array(data['transaction_feature'].tolist(), dtype=np.float32)
        dmatrix = xgb.DMatrix(X_data)
        preds = self.xgb_model.predict(dmatrix)
        
        result_df = pd.DataFrame(
            {
                "xgb_predict": preds,
            }
        )
        return result_df
        




class DNN_FRAUD_DETECT_EVADB(AbstractFunction):
    

    def __del__(self):
        print("[INFO] Summarization of DNN_FRAUD_DETECT_EVADB: \n", "count_inference: ", self.count_inference)

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
        return "DNN_FRAUD_DETECT_EVADB"

    @setup(cacheable=True, function_type="classification", batchable=True)
    def setup(self):
        self.model = SimpleNN(12)
        self.model.load_state_dict(torch.load("resources/model/fraud_detection.pth", map_location='cpu', weights_only=True))     
        self.model.eval()
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
                      'c_current_addr_sk',
                      'c_preferred_cust_flag',
                      'c_birth_day',
                      'c_birth_month',
                      'c_birth_year',
                      'c_birth_country',
                      'transaction_limit',
                      't_time',
                      'amount',
                ],
                column_types=[
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.INT32,
                    NdArrayType.FLOAT32,
                    NdArrayType.STR,
                    NdArrayType.FLOAT32
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

        data['customer_feature'] = data.apply(lambda row: get_customer_features(row['c_current_addr_sk'], row['c_preferred_cust_flag'], row['c_birth_day'], row['c_birth_month'], row['c_birth_year'], row['c_birth_country'], row['transaction_limit']), axis=1)
        data['transaction_feature'] = data.apply(lambda row: get_transaction_features(row['t_time'], row['amount']), axis=1)
        data['all_feature'] = data.apply(lambda row: concatenate_features(row['customer_feature'], row['transaction_feature']), axis=1)
        #data = data.drop(columns=['c_customer_sk', 'c_current_addr_sk', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 't_time', 'amount', 'transaction_limit', 'customer_feature', 'transaction_feature'])

        X_data = np.array(data['all_feature'].tolist(), dtype=np.float32)
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
        outputs = self.model(X_tensor)
        _, predicted = outputs.max(1)

        result_df = pd.DataFrame(
            {
                "label": predicted.tolist(),
            }
        )
        return result_df
