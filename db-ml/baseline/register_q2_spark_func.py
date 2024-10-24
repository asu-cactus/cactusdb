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



min_max_scaler = pickle.load(open("../../resources/model/movielens/final/tf/q1_ffnn_minmax_scaler_py.pkl", "rb"))
trending_movie_model = tf.keras.models.load_model("../../resources/model/movielens/final/tf/q1_ffnn_tf.h5")

@pandas_udf(IntegerType())
def predict_trending_ffnn(m_popularity: pd.Series, m_vote_average: pd.Series, m_vote_count: pd.Series) -> pd.Series:
    X = np.array([m_popularity.values, m_vote_average.values, m_vote_count.values]).T
    X = min_max_scaler.transform(X)
    y = np.argmax(trending_movie_model(X), axis=1)
    
    return pd.Series(y)

encoder = tf.keras.models.load_model("../../resources/model/movielens/final/tf/movie_tag_standalone_encoder.h5", compile=False)
interest_min_max_scaler = pickle.load(open("../../resources/model/movielens/final/tf/q1_ffnn_interest_scaler_py.pkl", "rb"))
interest_ffnn_model = tf.keras.models.load_model("../../resources/model/movielens/final/tf/interest_ffnn_model.h5", compile=False)
gender_encoder = {
            'M': 1,
            'F': 0
        }
age_encoder = {
    1: 0,
    18: 1,
    25: 2,
    35: 3,
    45: 4,
    50: 5,
    56: 6
}

@pandas_udf(IntegerType())
def predict_interest_ffnn(u_gender: pd.Series, u_age: pd.Series, u_occupation: pd.Series, mt_relevance_ir: pd.Series) -> pd.Series:
    gender = u_gender.apply(lambda x: gender_encoder[x]).to_numpy()
    age_n_occupation = np.hstack([u_age.values.reshape(-1,1), u_occupation.values.reshape(-1,1)])
    age_n_occupation = interest_min_max_scaler.transform(age_n_occupation)
    X_relevance_score_lr = np.stack(mt_relevance_ir.values)
    X_interest_features = np.hstack([gender.reshape(-1,1), age_n_occupation, X_relevance_score_lr])
    predictions = np.argmax(interest_ffnn_model(X_interest_features), axis=1)
    return pd.Series(predictions)

@pandas_udf(ArrayType(FloatType()))
def relevance_encoder(mt_relevance_score: pd.Series) -> pd.Series:
    mt_relevance_score = np.stack(mt_relevance_score.values)
    output = encoder(mt_relevance_score).numpy()
    return pd.Series(output.tolist())

embedding_dim = 128
num_numerical_features = 256
categorical_feature_sizes = [7, 21, 2]  # Example sizes
bottom_mlp_sizes = [128]
top_mlp_sizes = [256, 128]

dlrm_model = DLRM(embedding_dim, num_numerical_features, categorical_feature_sizes, bottom_mlp_sizes, top_mlp_sizes)

@pandas_udf(FloatType())
def predict_q2_dlrm(u_gender: pd.Series, u_age: pd.Series, u_occupation: pd.Series, mt_relevance_ir: pd.Series) -> pd.Series:
    
    u_gender = u_gender.apply(lambda x: gender_encoder[x])
    u_age = u_age.apply(lambda x: age_encoder[x])


    dlrm_categorical_features = np.hstack([u_age.values.reshape(-1,1),
                                          u_occupation.values.reshape(-1,1),
                                          u_gender.values.reshape(-1,1)])
    dlrm_numerical_features = np.stack(mt_relevance_ir.values)

    dlrm_numerical_features = torch.Tensor(dlrm_numerical_features)
    dlrm_categorical_features = torch.Tensor(dlrm_categorical_features.astype(np.int32))
    dlrm_categorical_features = dlrm_categorical_features.to(torch.int32)
    
    predictions = dlrm_model(dlrm_numerical_features, dlrm_categorical_features).detach().cpu().numpy().reshape(-1)
    return pd.Series(predictions)