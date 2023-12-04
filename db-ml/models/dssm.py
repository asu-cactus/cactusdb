"""
CREDIT: the following code implementation is original from the following link:
https://github.com/archersama/IntTower/tree/main
"""

from model.base_tower import BaseTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim
from layers.core import DNN
import torch
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import col_score
from preprocessing.utils import col_score_2
from preprocessing.utils import single_score
from preprocessing.utils import Timer
from layers.interaction import SENETLayer
from layers.interaction import LightSE

class DSSM(BaseTower):
    """DSSM Two Tower Model"""
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout = 0, init_std=0.0001, seed=124, task='binary', device='cpu', gpus=None):
        super(DSSM, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
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


