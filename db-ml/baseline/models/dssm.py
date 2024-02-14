"""
CREDIT: the following code implementation is original from the following link:
https://github.com/archersama/IntTower/tree/main
"""

from .base_tower import BaseTower
from .preprocessing.inputs import combined_dnn_input, compute_input_dim
from .layers.core import DNN
import torch
import tensorflow as tf
import numpy as np
from .preprocessing.utils import Cosine_Similarity
from .preprocessing.utils import col_score
from .preprocessing.utils import col_score_2
from .preprocessing.utils import single_score
from .preprocessing.utils import Timer
from .layers.interaction import SENETLayer
from .layers.interaction import LightSE
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

class DSSM_Torch(BaseTower):
    """DSSM Two Tower Model"""
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout = 0, init_std=0.0001, seed=124, task='binary', device='cpu', gpus=None):
        super(DSSM_Torch, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                    l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                    device=device, gpus=gpus)

        if len(user_dnn_feature_columns) > 0:
            self.user_dnn = DNN(compute_input_dim(user_dnn_feature_columns), dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            self.item_dnn = DNN(compute_input_dim(item_dnn_feature_columns), dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_dnn_embedding = None

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus

        self.user_filed_size = 4
        self.item_filed_size = 2

        self.user_timer = Timer()
        self.item_timer = Timer()
        self.rank_timer = Timer()

        self.t = [0, 0 , 0]


        # self.User_SE = SENETLayer(self.user_filed_size, 3, seed, device)
        # self.Item_SE = SENETLayer(self.item_filed_size, 3, seed, device)

        # self.dense = torch.nn.Linear(128*len(self.user_dnn_feature_columns),1).cuda()
        #
        # self.user_col_dense = torch.nn.Linear(128, 128*len(self.user_dnn_feature_columns)).cuda()
        # self.item_col_dense = torch.nn.Linear(128, 128*len(self.item_dnn_feature_columns)).cuda()
    
    def clear_time(self):
        self.t = [0, 0, 0]

    def compute_user_features(self, dnn_input, num_layers_to_run=0):
        for i in range(num_layers_to_run):
            fc = self.user_dnn.linears[i](dnn_input)

            if self.user_dnn.use_bn:
                fc = self.user_dnn.bn[i](fc)

            fc = self.user_dnn.activation_layers[i](fc)

            fc = self.user_dnn.dropout(fc)
            dnn_input = fc
        return dnn_input

    def compute_item_features(self, dnn_input, num_layers_to_run=0):
        for i in range(num_layers_to_run):
            fc = self.item_dnn.linears[i](dnn_input)

            if self.item_dnn.use_bn:
                fc = self.item_dnn.bn[i](fc)

            fc = self.item_dnn.activation_layers[i](fc)

            fc = self.item_dnn.dropout(fc)
            dnn_input = fc
        return dnn_input
    
    def get_user_dnn_input(self, inputs):
        user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
        return user_dnn_input
    
    def get_item_dnn_input(self, inputs):
        item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)


        item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
        return item_dnn_input

    def inner_product(self, user_dnn_embedding, item_dnn_embedding):
        score = Cosine_Similarity(user_dnn_embedding, item_dnn_embedding, gamma=self.gamma)
        output = self.out(score)
        return output

    def forward(self, inputs):
        if len(self.user_dnn_feature_columns) > 0:
            self.user_timer.tic()
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)
            
            # user_sparse_embedding = torch.cat(user_sparse_embedding_list, dim= 1)
            # # print(user_sparse_embedding.shape)
            # user_sim_embedding = self.User_SE(user_sparse_embedding)
            # sparse_dnn_input = torch.flatten(user_sim_embedding, start_dim=1)
            # if(len(user_dense_value_list)>0):
            #     dense_dnn_input = torch.flatten(torch.cat(user_dense_value_list, dim=-1), start_dim=1)
            #     user_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input],axis=-1)
            # else:
            #     user_dnn_input = sparse_dnn_input

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)


            self.user_dnn_embedding = self.user_dnn(user_dnn_input)

            t_user = self.user_timer.toc()
            self.t[0] += t_user


            # print(self.user_dnn_embedding.shape)

            # self.user_dnn_embedding = self.user_col_dense(self.user_dnn_embedding)
            # self.user_dnn_embedding = self.dense(self.user_dnn_embedding)

        if len(self.item_dnn_feature_columns) > 0:
            self.item_timer.tic()
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            # item_sparse_embedding = torch.cat(item_sparse_embedding_list, dim=1)
            # item_sim_embedding = self.Item_sim_non_local(item_sparse_embedding)
            # sparse_dnn_input = torch.flatten(item_sim_embedding, start_dim=1)
            # dense_dnn_input = torch.flatten(torch.cat(item_dense_value_list, dim=-1), start_dim=1)
            #
            # item_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], axis=-1)

            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)

            self.item_dnn_embedding = self.item_dnn(item_dnn_input)

            t_item = self.item_timer.toc()
            self.t[1] += t_item

            # self.item_dnn_embedding = self.item_col_dense(self.item_dnn_embedding)
            # self.item_dnn_embedding = self.dense(self.item_dnn_embedding)

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:
            self.rank_timer.tic()
            score = Cosine_Similarity(self.user_dnn_embedding, self.item_dnn_embedding, gamma=self.gamma)
            # print(score.shape)
            # score = col_score(self.user_dnn_embedding, self.item_dnn_embedding,len(self.user_dnn_feature_columns))
            # score = col_score_2(self.user_dnn_embedding, self.item_dnn_embedding, len(self.user_dnn_feature_columns),\
            #                   len(self.item_dnn_feature_columns),128)
            # score = single_score(self.item_dnn_embedding)
            # print(score.shape)
            output = self.out(score)
            t_rank = self.rank_timer.toc()
            self.t[2] += t_rank

            return output, self.user_dnn_embedding, self.item_dnn_embedding

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")


class CosineSimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CosineSimilarityLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x, y = inputs
        x_normalized = tf.nn.l2_normalize(x, axis=-1)
        y_normalized = tf.nn.l2_normalize(y, axis=-1)
        cosine_similarity = tf.reduce_sum(tf.multiply(x_normalized, y_normalized), axis=-1)
        return cosine_similarity

class ActivationLayer(tf.keras.layers.Layer):
    def __init__(self, activation, units, dice_dim):
        super(ActivationLayer, self).__init__()
        self.activation = tf.keras.layers.Activation(activation)
        self.dense = tf.keras.layers.Dense(units)
        self.dice_dim = dice_dim

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.activation(x)
        return x

class DNN_TF(tf.keras.Model):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=True,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN_TF, self).__init__()

        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        if inputs_dim > 0:
            hidden_units = [inputs_dim] + list(hidden_units)
        else:
            hidden_units = list(hidden_units)

        self.linears = [tf.keras.layers.Dense(hidden_units[i+1], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=init_std),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) for i in range(len(hidden_units) - 1)]

        if self.use_bn:
            self.bn = [tf.keras.layers.BatchNormalization() for _ in range(len(hidden_units) - 1)]

        self.activation_layers = [ActivationLayer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units) - 1)]

    def call(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

class SequencePoolingLayer(tf.keras.layers.Layer):
    def __init__(self, mode='mean', **kwargs):
        super(SequencePoolingLayer, self).__init__(**kwargs)
        if mode not in ['mean']:
            raise ValueError('SequencePoolingLayer mode should in [mean]')
        self.mode = mode
        self.eps = 1e-8

    def call(self, inputs):
        inputs, mask = inputs
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask1 = tf.expand_dims(mask, axis=-1)
            inputs *= mask1  # Apply mask to the inputs
        # Sum along the time dimension
        sum_pooling = tf.reduce_sum(inputs, axis=1, keepdims=False)
        # Count non-masked elements along the time dimension
        sequence_lengths = tf.reduce_sum(mask, axis=1, keepdims=True)
        # Avoid division by zero
        sequence_lengths = tf.where(tf.math.greater(sequence_lengths, 0), sequence_lengths, tf.ones_like(sequence_lengths))

        # Mean pooling
        mean_pooling = sum_pooling / (sequence_lengths+self.eps)

        return mean_pooling

class DSSM_TF(tf.keras.Model):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary'):
        super(DSSM_TF, self).__init__()
        self.user_dnn_feature_columns = user_dnn_feature_columns
        self.item_dnn_feature_columns = item_dnn_feature_columns

        self.user_dnn = DNN_TF(129, hidden_units=(300, 300, 128), activation='relu')
        self.item_dnn = DNN_TF(33, hidden_units=(300, 300, 128), activation='relu')
        self.embedding_dict = dict()
        self.embedding_dict['user_id'] = tf.keras.layers.Embedding(6040, 32)
        self.embedding_dict['gender'] = tf.keras.layers.Embedding(2, 32)
        self.embedding_dict['age'] = tf.keras.layers.Embedding(7, 32)
        self.embedding_dict['occupation'] = tf.keras.layers.Embedding(21, 32)
        self.embedding_dict['movie_id'] = tf.keras.layers.Embedding(3706, 32)
        self.embedding_dict['genres'] = tf.keras.layers.Embedding(1000, 32)
        
        self.sequence_pooling_layer = SequencePoolingLayer()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.cosine_similarity_layer = CosineSimilarityLayer()

    def call(self, inputs, training=None, mask=None):

        user_id_embed = self.embedding_dict['user_id'](inputs['user_id'])
        gender_embed = self.embedding_dict['gender'](inputs['gender'])
        age_embed = self.embedding_dict['age'](inputs['age'])
        occupation_embed = self.embedding_dict['occupation'](inputs['occupation'])
        movie_id_embed = self.embedding_dict['movie_id'](inputs['movie_id'])
        genres_embed = self.embedding_dict['genres'](inputs['genres'])
        genres_mask = inputs['genres'] != 0
        genres_embed = self.sequence_pooling_layer([genres_embed, genres_mask])

        user_dnn_input = self.concat_layer([user_id_embed,gender_embed,age_embed,occupation_embed, tf.expand_dims(inputs['user_mean_rating'], axis=-1)])
        item_dnn_input = self.concat_layer([movie_id_embed, genres_embed, tf.expand_dims(inputs['movie_mean_rating'], axis=-1)])
        # Tower 1 (User Tower)
        user_output = self.user_dnn(user_dnn_input)

        # Tower 2 (Item Tower)
        item_output = self.item_dnn(item_dnn_input)

        output = self.cosine_similarity_layer([user_output, item_output])

        return output