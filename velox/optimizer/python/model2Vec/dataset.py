"""
CREDIT: The following code is adapted from the QueryFormer work: https://github.com/zhaoyue-ntu/QueryFormer
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
import ast
import warnings
from collections import deque
from .database_util import (
    format_filter,
    format_join,
    filterDict2Hist,
    ModelGraphEncoder,
)
from .database_util import *
from model_cactus.util import Normalizer


def get_numerical_min_max_mapping(df_stat):
    numerical_min_max_mapping = {}
    for idx, row in df_stat.iterrows():
        if row["type"] == "Numerical":
            numerical_min_max_mapping[row["column"]] = (row["bins"][0], row["bins"][-1])
    return numerical_min_max_mapping


def get_categorical_mapping(df_stat):
    categorical_mapping = {}
    for idx, row in df_stat.iterrows():
        if row["type"] == "Categorical":
            bins = row["bins"]
            # filter out None
            bins = [str(x) for x in bins if x != ""]
            column_mapping = {}
            for idx, val in enumerate(bins):
                column_mapping[val] = idx
            categorical_mapping[row["column"]] = column_mapping
    return categorical_mapping


def calculate_height(adj_list, tree_size):
    if tree_size == 1:
        return np.array([0])

    adj_list = np.array(adj_list)
    node_ids = np.arange(tree_size, dtype=int)
    node_order = np.zeros(tree_size, dtype=int)
    uneval_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adj_list[:, 0]
    child_nodes = adj_list[:, 1]

    n = 0
    while uneval_nodes.any():
        uneval_mask = uneval_nodes[child_nodes]
        unready_parents = parent_nodes[uneval_mask]

        node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
        node_order[node2eval] = n
        uneval_nodes[node2eval] = False
        n += 1
    return node_order


def node2dict(treeNode):

    adj_list, num_child, features = topo_sort(treeNode)
    heights = calculate_height(adj_list, len(features))

    return {
        "features": torch.FloatTensor(np.array(features)),
        "heights": torch.LongTensor(heights),
        "adjacency_list": torch.LongTensor(np.array(adj_list)),
    }


def topo_sort(root_node):
    #        nodes = []
    adj_list = []  # from parent to children
    num_child = []
    features = []

    toVisit = deque()
    toVisit.append((0, root_node))
    next_id = 1
    while toVisit:
        idx, node = toVisit.popleft()
        #            nodes.append(node)
        features.append(node.feature)
        num_child.append(len(node.children))
        for child in node.children:
            toVisit.append((next_id, child))
            adj_list.append((idx, next_id))
            next_id += 1

    return adj_list, num_child, features


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
    #    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype("long")
    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 60

    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])
    return M


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pre_collate(the_dict, max_node=50, rel_pos_max=20):

    x = pad_2d_unsqueeze(the_dict["features"], max_node)
    N = len(the_dict["features"])
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)

    edge_index = the_dict["adjacency_list"].t()
    if len(edge_index) == 0:
        shortest_path_result = np.array([[0]])
        path = np.array([[0]])
        adj = torch.tensor([[0]]).bool()
    else:
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        shortest_path_result = floyd_warshall_rewrite(adj.numpy())

    rel_pos = torch.from_numpy((shortest_path_result)).long()

    attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float("-inf")

    attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
    rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

    heights = pad_1d_unsqueeze(the_dict["heights"], max_node)

    return {"x": x, "attn_bias": attn_bias, "rel_pos": rel_pos, "heights": heights}


def read_and_process_histograms(path):
    df_stat_columns = ["table", "column", "table_column", "type", "freqs", "bins"]
    df_stat = pd.read_csv(path, sep="|", header=None, names=df_stat_columns)
    for idx, row in df_stat.iterrows():
        df_stat.loc[idx, "freqs"] = ast.literal_eval(row["freqs"])
        if row["type"] == "Numerical":
            df_stat.loc[idx, "bins"] = [float(x) for x in ast.literal_eval(row["bins"])]
        else:
            df_stat.loc[idx, "bins"] = ast.literal_eval(row["bins"])
    return df_stat


def remove_type_attribute(obj):
    if isinstance(obj, dict):
        # Use list() to avoid 'RuntimeError: dictionary changed size during iteration'
        for key in list(obj.keys()):
            if key == "type" or key == "outputType":
                del obj[key]
            elif key == "functionName" and obj[key] is None:
                del obj[key]
            else:
                # Recursively call the function on nested dictionaries
                remove_type_attribute(obj[key])
    elif isinstance(obj, list):
        # Recursively call the function on each element of the list
        for item in obj:
            remove_type_attribute(item)


def extract_dim_for_ml_op(ml_node):
    if "dims" in ml_node:
        return ml_node["dims"]
    else:
        # for the op like relu, sigmoid, argmax, etc. the dims is not specified
        # we give it a default value of -1, a future step is to infer the dims from the input
        return [-1]


def compute_flops_from_ml_dims(ml_op_dims):
    # compute the flops based on the dims
    if isinstance(ml_op_dims, list):
        if len(ml_op_dims) == 1:
            return ml_op_dims[0]  # for the op like relu, sigmoid, argmax, etc.
        else:
            return sum(a * b for a, b in zip(ml_op_dims, ml_op_dims[1:]))
            # return np.prod(ml_op_dims)
    else:
        raise ValueError(
            "ml_op_dims should be a list or numpy array, got {}".format(
                type(ml_op_dims)
            )
        )


import re

ml_kernel_pattern = r"(relu|mat_mul|mat_vector_add|softmax|argmax|batch_norm|torchdnn|svd|embedding|sigmoid)"


def find_ml_express_node_from_plan(node, search_through_source):
    if "projections" in node:
        for proj in node["projections"]:
            if "functionName" in proj and proj["functionName"] is not None:
                function_name = proj["functionName"]
                if re.search(ml_kernel_pattern, function_name, re.IGNORECASE):
                    return True, proj
    # if search_through_source is False, we only search the current node
    # otherwise, we search through the sources
    if search_through_source and "sources" in node:
        for subplan in node["sources"]:
            found, returned_node = find_ml_express_node_from_plan(
                subplan, search_through_source
            )
            if found:
                return True, returned_node
    return False, None


class ModelGraphTreeNode:
    def __init__(self, ml_op_type, ml_op_type_id, ml_op_dims, ml_op_flop):
        self.ml_op_type = ml_op_type
        self.ml_op_type_id = ml_op_type_id
        self.ml_op_dims = ml_op_dims
        self.ml_op_flop = ml_op_flop

        self.parent = None
        self.feature = None
        self.children = []

    def addChild(self, treeNode):
        self.children.append(treeNode)

    def __str__(self):
        #        return TreeNode.print_nested(self)
        return "{} with {}, {}, {}, {} children".format(
            self.ml_op_type,
            self.ml_op_type_id,
            self.ml_op_dims,
            self.ml_op_flop,
            len(self.children),
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        print(
            "--" * indent
            + "{} with {}, {}, {}, {} children".format(
                node.ml_op_type,
                node.ml_op_type_id,
                node.ml_op_dims,
                node.ml_op_flop,
                len(node.children),
            )
        )
        for k in node.children:
            ModelGraphTreeNode.print_nested(k, indent + 1)


def modelGraphNode2feature(node, encoder : ModelGraphEncoder, length : int = 50):
    ml_op_id = node.ml_op_type_id
    ml_op_dims_encoded = encoder.encode_ml_op_dims(node.ml_op_dims, length)
    ml_op_flop_encoded = encoder.encode_ml_op_flops(node.ml_op_flop)

    return np.concatenate(
        (np.array([ml_op_id]), ml_op_dims_encoded, np.array([ml_op_flop_encoded]))
    )


class ModelComputationGraphDataset(Dataset):
    def __init__(
        self,
        cost_normalizer: Normalizer,
        model_graph_encoder: ModelGraphEncoder,
        df: pd.DataFrame = None,
        query_plans: json = None,
        to_predict: str = "cost",
        dir_path: str = "/home/velox/velox/optimizer/tests/",
        mode: str = "train",
        ml_op_dim_length: int = 50,
        device: torch.device = "cpu",
    ):
        self.df = df
        self.query_plans = query_plans
        self.model_graph_encoder = model_graph_encoder

        self.dir_path = dir_path

        self.cost_normalizer = cost_normalizer
        self.mode = mode
        self.ml_op_dim_length = ml_op_dim_length
        self.device = device
        
        if df is not None:
            self.cards = np.zeros(len(df))
            self.costs = df["executionTime"].values
            self.card_labels = torch.from_numpy(self.cards)  # FIXME:
            self.cost_labels = torch.from_numpy(
                self.cost_normalizer.normalize_labels(self.costs)
            )
            self.query_plans = [
                read_json(os.path.join(dir_path, x)) for x in df["serializedPlanPath"]
            ]
        elif query_plans is not None:
            self.cards = np.zeros(len(query_plans))
            self.costs = np.zeros(len(query_plans))
            self.card_labels = torch.from_numpy(self.cards)
            self.cost_labels = torch.from_numpy(
                self.cost_normalizer.normalize_labels(self.costs)
            )

        self.length = len(self.query_plans)
        idxs = range(self.length)

        self.to_predict = to_predict
        if to_predict == "cost":
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise ValueError("Invalid to_predict value: {}".format(to_predict))

        self.treeNodes = []  ## for mem collection
        self.rootNodes = []
        self.collated_dicts = [
            self.js_node2dict(i, plan) for i, plan in zip(idxs, self.query_plans)
        ]

        self.similar_model_idx = {}
        self.dissimilar_model_idx = {}
        self.wl_features = []

    def js_node2dict(self, idx, plan):
        try:
            # if it is train mode, we search the plan through the sources
            search_through_source = self.mode == "train"
            find_ml_node, ml_node = find_ml_express_node_from_plan(
                plan, search_through_source
            )
            treeNode = self.traverse_model_graph(ml_node)
            self.rootNodes.append(treeNode)
            _dict = node2dict(treeNode)
            collated_dict = pre_collate(_dict)
        except Exception as e:
            print("Error in js_node2dict idx: ", idx)
            raise e

        # self.treeNodes.clear()
        # del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def traverse_model_graph(self, ml_node):
        # extract ml op name
        ml_op = format_ml_op_name(ml_node["functionName"])
        ml_type_id = self.model_graph_encoder.encode_ml_op(ml_op)
        ml_op_dims = extract_dim_for_ml_op(ml_node)
        ml_flops = compute_flops_from_ml_dims(ml_op_dims)
        root = ModelGraphTreeNode(ml_op, ml_type_id, ml_op_dims, ml_flops)
        try:
          root.feature = modelGraphNode2feature(root, self.model_graph_encoder, self.ml_op_dim_length)
        except Exception as e:
            print("Error in modelGraphNode2feature for node: ", ml_node)
            raise e
        self.treeNodes.append(root)
        if "inputs" in ml_node:
            for sub_op in ml_node["inputs"]:
                if "functionName" not in sub_op or sub_op["functionName"] is None:
                    # skip the node without functionName
                    continue
                # sub_op is a dict with functionName and dims
                sub_op["parent"] = ml_node
                node = self.traverse_model_graph(sub_op)
                node.parent = root
                root.addChild(node)
        return root
