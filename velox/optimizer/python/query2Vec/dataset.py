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
from .database_util import format_filter, format_join, TreeNode, filterDict2Hist
from .database_util import *
from query2Vec.util import Normalizer
from model2Vec.database_util import collator, ModelGraphEncoder, WeisfeilerLehmanEncoder
from model2Vec.dataset import (
    ModelGraphTreeNode,
    ModelComputationGraphDataset,
    find_ml_express_node_from_plan,
)
from model2Vec.model import Model2Vec


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


class PlanTreeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        encoder: Encoder,
        to_predict: str,
        table_sample: pd.DataFrame,
        cost_norm: Normalizer,
        dir_path: str = "/home/velox/velox/optimizer/tests/",
        query_process: bool = False,
        path_to_stats: str = None,
        model2vec: Model2Vec = None,
        model_graph_encoder: ModelGraphEncoder = None,
        model2vec_normalizer: Normalizer = None,
        device: torch.device = "cpu",
        pre_process_stats: bool = False,
        query_stats_df: pd.DataFrame = None,
    ):
        self.df = df
        self.encoder = encoder

        self.table_sample = table_sample
        self.length = len(df)
        self.dir_path = dir_path
        self.query_plans = [
            read_json(os.path.join(dir_path, x)) for x in df["serializedPlanPath"]
        ]
        if query_process:
            for plan in self.query_plans:
                remove_type_attribute(plan)
        self.model2vec = model2vec
        self.model_graph_encoder = model_graph_encoder
        self.model2vec_normalizer = model2vec_normalizer
        self.device = device

        # self.query_stats = [
        #     read_and_process_histograms(os.path.join(dir_path, x))
        #     for x in df["tableStatsPath"]
        # ]

        if not pre_process_stats:
            self.query_stats = read_and_process_histograms(path_to_stats)
            self.encoder_min_max_map = get_numerical_min_max_mapping(self.query_stats)
            self.encoder_cate_map = get_categorical_mapping(self.query_stats)
            self.encoder.set_column_normalizer(
                self.encoder_min_max_map, self.encoder_cate_map
            )
        else:
            self.query_stats = query_stats_df

        # self.encoder_min_max_map = get_numerical_min_max_mapping(self.query_stats)
        # self.encoder_cate_map = get_categorical_mapping(self.query_stats)
        # self.encoder_min_max_map = [
        #     get_numerical_min_max_mapping(df_state) for df_state in self.query_stats
        # ]
        # self.encoder_cate_map = [
        #     get_categorical_mapping(df_state) for df_state in self.query_stats
        # ]

        self.cost_norm = cost_norm

        self.cards = np.zeros(len(df))
        self.costs = df["executionTime"].values
        self.card_labels = torch.from_numpy(self.cards)  # FIXME:
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        self.to_predict = to_predict
        if to_predict == "cost":
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise ValueError("Invalid to_predict value: {}".format(to_predict))

        idxs = range(len(self.df))

        self.treeNodes = []  ## for mem collection
        self.rootNodes = []
        self.collated_dicts = [
            self.js_node2dict(i, plan) for i, plan in zip(idxs, self.query_plans)
        ]

        self.similar_query_idx = {}
        self.dissimilar_query_idx = {}
        self.wl_features = []

    def js_node2dict(self, idx, plan):
        try:
            # load the stats for each query
            # self.encoder.set_column_normalizer(
            #     self.encoder_min_max_map[idx], self.encoder_cate_map[idx]
            # )
            treeNode = self.traverse_query_plan(plan, idx, self.encoder)
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

    def get_node_ml_embed(self, node_plan):
        search_through_source = False
        found, returned_node = find_ml_express_node_from_plan(
            node_plan, search_through_source
        )
        if found:
            model2vec_ds = ModelComputationGraphDataset(
                cost_normalizer=self.model2vec_normalizer,
                model_graph_encoder=self.model_graph_encoder,
                df=None,
                query_plans=[node_plan],
                mode="eval",
            )
            model_embeds = (
                self.model2vec.get_embeddings(model2vec_ds, device=self.device)
                .detach()
                .cpu()
                .numpy()
            )
            assert len(model_embeds) == 1
            return model_embeds[0]
        else:
            return np.zeros(192)

    def traverse_query_plan(self, plan, idx, encoder):
        # traverse plan
        nodeType = plan["name"]
        nodeTypeId = encoder.encode_type(nodeType)
        card = None
        filters, alias = format_filter(plan)
        join = None
        joinType = encoder.encode_join(format_join(plan))
        filters_encoded = encoder.encode_filters(filters, alias)
        if not filters_encoded["colId"]:
            raise ValueError(
                "failed filters_encoded: ", idx, filters_encoded, filters, alias
            )
        # TODO: Incorporate the Model2Vec
        # model_embeds = np.random.random(192)

        ml_model_embeds = self.get_node_ml_embed(plan)
        # ml_ops, ml_op_dims, ml_nested_ops = format_ml_ops(plan)
        root = TreeNode(
            nodeType,
            nodeTypeId,
            filters,
            card,
            joinType,
            join,
            filters_encoded,
            ml_model_embeds=ml_model_embeds,
        )
        if root.nodeType == "AggregationNode":
            list_agg_keys = []
            for groupKey in plan.get("groupingKeys", []):
                list_agg_keys.append(groupKey["fieldName"])
            sorted(list_agg_keys)
            root.agg_keys = list_agg_keys
        self.treeNodes.append(root)
        if "tableName" in plan:
            root.table = plan["tableName"]
            root.table_id = encoder.encode_table(plan["tableName"])
            root.num_rows, root.num_cols = plan["tableStats"]

            # print("table: ", root.table, ", table_id: ", root.table_id)
        # root.ml_model_embeds = model_embeds
        # if ml_ops:
        #     root.ml_ops = ml_ops
        #     root.ml_op_dims = ml_op_dims
        #     root.ml_nested_ops = ml_nested_ops
        root.query_id = idx  # need to change
        root.feature = node2feature(
            root, encoder, self.query_stats, None
        )  # last is table_sample

        if "sources" in plan:
            for subplan in plan["sources"]:
                subplan["parent"] = plan
                node = self.traverse_query_plan(subplan, idx, encoder)
                node.parent = root
                root.addChild(node)
        return root


def node2feature(node, encoder, hist_df, table_sample):
    num_filter = len(node.filterDict["colId"])
    # number of filters is less than 3, pad with zeros
    pad = np.zeros((3, 3 - num_filter))
    filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    # use the history histrogram file and the query filter to get the histogram
    # print("node.filterDict: ", node.filterDict)
    hists = filterDict2Hist(hist_df, node.filterDict, encoder)

    # encode ml ops
    # version 1
    # num_ops, least_complicate_op, most_complicate_op, op_complixity_min, op_complixity_max  = encoder.encode_ml_ops(node.ml_ops, node.ml_op_dims)
    # ml_features = np.array([num_ops, least_complicate_op, most_complicate_op, list_op_complixity])

    # version 2
    # num_ops, least_complicate_op, most_complicate_op, list_op_complixity = (
    #     encoder.encode_ml_ops(node.ml_ops, node.ml_op_dims, node.ml_nested_ops)
    # )
    # ml_features = np.concatenate(
    #     ([num_ops, least_complicate_op, most_complicate_op], list_op_complixity)
    # )
    # print("most_complicate_op: ", most_complicate_op, " op_complixity_max: ", op_complixity_max, " least_complicate_op: ", least_complicate_op, " op_complixity_min: ", op_complixity_min)

    ml_features = node.ml_model_embeds
    # table stats
    num_rows, num_cols = encoder.normalize_table_stats(node.num_rows, node.num_cols)
    # print("num_rows: ", num_rows, " num_cols: ", num_cols, " node.num_rows: ", node.num_rows, " node.num_cols: ", node.num_cols)
    table_stat_features = np.array([num_rows, num_cols])

    # Retrive the sample from table_sample
    # The 1000 bits will only be set on the TableScan node
    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    # TODO
    # sample = np.zeros(1000)
    # if node.table_id == 0:
    #     sample = np.zeros(1000)
    # else:
    #     # node.table which is the tableName or Relation Name in query tree
    #     sample = table_sample[node.query_id][node.table]

    # return np.concatenate((type_join,filts,mask))
    # print(type_join.shape, filts.shape, mask.shape, ml_features.shape, hists.shape, table.shape, table_stat_features.shape, sample.shape)
    # return np.concatenate(
    #     (type_join, filts, mask, ml_features, hists, table, table_stat_features, sample)
    # )
    return np.concatenate(
        (type_join, filts, mask, ml_features, hists, table, table_stat_features)
    )
