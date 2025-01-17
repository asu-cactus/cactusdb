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
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from models.preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from models.dssm import DSSM_Torch, DSSM_TF, get_var_feature, get_test_var_feature
from models.dlrm import DLRM
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, when
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType, IntegerType
from dssm_evadb import DSSM_Moel_Wrapper
import pickle
import multiprocessing as mp

import pickle
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


le_uc3_store = pickle.load(
    open("../../resources/model/tpcxai_sf1/final/tf/usecase3_le_store.pkl", "rb")
)
le_uc3_dept = pickle.load(
    open("../../resources/model/tpcxai_sf1/final/tf/usecase3_le_dept.pkl", "rb")
)
uc3model = tf.keras.models.load_model(
    "../../resources/model/tpcxai_sf1/final/tf/usecase3.h5", compile=False
)


@pandas_udf(IntegerType())
def uc3_sales_predicator(
    store: pd.Series, department: pd.Series, num_of_week: pd.Series
) -> pd.Series:
    store = le_uc3_store.transform(store.values)
    department = le_uc3_dept.transform(department.values)
    num_of_week = num_of_week / 156
    X = np.array([store, department, num_of_week.values]).T
    y = np.argmax(uc3model(X), axis=1)

    return pd.Series(y)


uc8model = tf.keras.models.load_model(
    "../../resources/model/tpcxai_sf1/final/tf/usecase8.h5", compile=False
)
le_uc8_dept = pickle.load(
    open("../../resources/model/tpcxai_sf1/final/tf/usecase8_le_dept.pkl", "rb")
)


@pandas_udf(IntegerType())
def uc8_trip_classifier(
    quantity: pd.Series,
    scan_count: pd.Series,
    weekday: pd.Series,
    department: pd.Series,
) -> pd.Series:
    df_temp = pd.DataFrame(department.values)
    department = le_uc8_dept.transform(df_temp[[0]].values).reshape(-1)

    scan_count = scan_count.astype(int)
    weekday = weekday.astype(int)
    X = np.array([quantity.values, scan_count.values, weekday.values, department]).T
    y = np.argmax(uc8model(X), axis=1)
    return pd.Series(y)


uc10model = tf.keras.models.load_model(
    "../../resources/model/tpcxai_sf1/final/tf/usecase10.h5", compile=False
)


with open("../../resources/model/tpcxai_sf1/final/tf/usecase10_lr_model.h5", "rb") as f:
    uc10lr_model = pickle.load(f)


@pandas_udf(IntegerType())
def uc10_fraud_ml_spark_predicator(
    business_hour_norm: pd.Series, amount_norm: pd.Series
) -> pd.Series:
    X = np.array([business_hour_norm.values, amount_norm.values]).T
    y = uc10lr_model.predict(X)

    return pd.Series(y)
