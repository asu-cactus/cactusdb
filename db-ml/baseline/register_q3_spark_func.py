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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
encoder = tf.keras.models.load_model("../../resources/model/movielens/final/tf/movie_tag_standalone_encoder.h5", compile=False)
user_movie_interest_model = tf.keras.models.load_model("../../resources/model/movielens/final/tf/q3_user_movie_interest_ffnn.h5", compile=False)
user_movie_rating_model = tf.keras.models.load_model("../../resources/model/movielens/final/tf/q3_user_movie_rating_ffnn.h5", compile=False)

gender_encoder = {
            'M': 1,
            'F': 0

}

@pandas_udf(IntegerType())
def predict_user_movie_interest_ffnn(u_age: pd.Series, u_gender: pd.Series, u_occupation: pd.Series,  m_popularity: pd.Series, m_vote_average: pd.Series) -> pd.Series:
    u_gender = u_gender.apply(lambda x: gender_encoder[x])
    X = np.array([u_age.values, u_gender.values, u_occupation.values, m_popularity.values, m_vote_average.values]).T
    y = np.argmax(user_movie_interest_model.predict(X, batch_size=2048), axis=1)
    
    return pd.Series(y)

@pandas_udf(IntegerType())
def predict_user_movie_rating_ffnn(u_age: pd.Series, u_gender: pd.Series, u_occupation: pd.Series,  m_popularity: pd.Series, m_vote_average: pd.Series) -> pd.Series:
    u_gender = u_gender.apply(lambda x: gender_encoder[x])
    X = np.array([u_age.values, u_gender.values, u_occupation.values, m_popularity.values, m_vote_average.values]).T
    y = np.argmax(user_movie_rating_model.predict(X, batch_size=2048), axis=1)
    
    return pd.Series(y)

@pandas_udf(ArrayType(FloatType()))
def relevance_encoder(mt_relevance_score: pd.Series) -> pd.Series:
    mt_relevance_score = np.stack(mt_relevance_score.values)
    output = encoder.predict(mt_relevance_score, batch_size=2048)
    return pd.Series(output.tolist())

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

@pandas_udf(FloatType())
def spark_cosine_similarity(mt_relevance_ir: pd.Series, mt_relevance_ir1: pd.Series) -> pd.Series:
    mt_relevance_score = np.stack(mt_relevance_ir.values)
    mt_relevance_score1 = np.stack(mt_relevance_ir1.values)
    
    return pd.Series(row_wise_cosine_similarity(mt_relevance_score, mt_relevance_score1))
