"""
CREDIT: The following code is adapted from the QueryFormer work: https://github.com/zhaoyue-ntu/QueryFormer
"""
import numpy as np
import pandas as pd
import csv
import torch
import logging
import re
import json
import warnings
from tqdm import tqdm
import hashlib
import faiss
import warnings
import random
from collections import defaultdict


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


## bfs shld be enough
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


def get_job_table_sample(workload_file_name, num_materialized_samples=1000):

    tables = []
    samples = []

    # Load queries
    with open(workload_file_name + ".csv", "r") as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter="#"))
        for row in data_raw:
            tables.append(row[0].split(","))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))

    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", "rb") as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder="little")
            bitmaps = np.empty(
                (num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8
            )
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables, samples):
        d = {}
        for t, s in zip(ts, ss):
            tf = t.split(" ")[0]  # remove alias
            d[tf] = s
        table_sample.append(d)

    return table_sample


def get_hist_file(hist_path, bin_number=50):
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file["freq"][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=float)
        hist_file["freq"][i] = freq_np

    table_column = []
    for i in range(len(hist_file)):
        table = hist_file["table"][i]
        col = hist_file["column"][i]
        table_alias = "".join([tok[0] for tok in table.split("_")])
        if table == "movie_info_idx":
            table_alias = "mi_idx"
        combine = ".".join([table_alias, col])
        table_column.append(combine)
    hist_file["table_column"] = table_column

    for rid in range(len(hist_file)):
        hist_file["bins"][rid] = [
            int(i) for i in hist_file["bins"][rid][1:-1].split(" ") if len(i) > 0
        ]

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file


def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file["freq"][i]
        bins = freq2bin(freq, target_number)
        hist_file["bins"][i] = bins
    return hist_file


def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq) - 1

    step = 1.0 / target_number
    mini = 0
    while freq[mini + 1] == 0:
        mini += 1
    pointer = mini + 1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi + 1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1

    if len(res_pos) == target_number:
        res_pos.append(maxi)

    return res_pos


class Batch:
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos

    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(
            device
        )

        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


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


def collator(small_set):
    y = small_set[1]
    xs = [s["x"] for s in small_set[0]]

    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s["attn_bias"] for s in small_set[0]])
    rel_pos = torch.cat([s["rel_pos"] for s in small_set[0]])
    heights = torch.cat([s["heights"] for s in small_set[0]])

    return Batch(attn_bias, rel_pos, heights, x), y


def filterDict2Hist(hist_file, filterDict, encoder):
    buckets = len(hist_file["bins"][0])
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets - 1))
    # iterate over each filter
    for i in range(len(filterDict["colId"])):
        colId = filterDict["colId"][i]
        col = encoder.idx2col[colId]
        if col == "NA":
            ress[i] = empty
            continue
        bins = hist_file.loc[hist_file["column"] == col, "bins"].iloc[0]

        opId = filterDict["opId"][0]
        op = encoder.idx2op[opId]

        val = filterDict["val"][0]

        left = 0
        right = len(bins) - 1

        if col in encoder.categorical_vals_mapping:
            # categorical column
            # print("col: ", col, " val: ", val)
            left = right = val
            # print("left: ", left, " right: ", right)
        elif col in encoder.column_min_max_vals:
            # numerical column
            mini, maxi = encoder.column_min_max_vals[col]
            val_unnorm = val * (maxi - mini) + mini

            for j in range(len(bins)):
                if bins[j] < val_unnorm:
                    left = j
                if bins[j] > val_unnorm:
                    right = j
                    break
        else:
            logging.warning(f"Column {col} not found in encoder, current encoder map:", encoder.column_min_max_vals)
            ress = ress.flatten()
            return ress

        res = np.zeros(len(bins) - 1)

        if op == "eq":
            res[left:right] = 1
        elif op == "lt":
            res[:left] = 1
        elif op == "gt":
            res[right:] = 1
        elif op == "lte":
            res[: right + 1] = 1
        elif op == "gte":
            res[left:] = 1

        ress[i] = res

    ress = ress.flatten()
    return ress


def format_join(plan):
    if "joinType" in plan:
        return plan["joinType"]
    else:
        return None


def format_filter(plan):
    filters = []
    alias = None
    try:
        if "filter" in plan:
            if plan["filter"]["functionName"] in [
                "eq",
                "lt",
                "gt",
                "gte",
                "lte",
                "like",
            ]:
                if plan["filter"]["functionName"] == "like":
                    # remove the % in the value
                    plan["filter"]["inputs"][1]["value"]["value"] = plan["filter"][
                        "inputs"
                    ][1]["value"]["value"].strip("%")
                if "nullOnFailure" in plan["filter"]["inputs"][0]:
                    if plan["filter"]["inputs"][1]["name"] == "CastTypedExpr":
                      # there is a casting
                      filter = "{} {} {}".format(
                          plan["filter"]["inputs"][0]["inputs"][0]["fieldName"],
                          plan["filter"]["functionName"],
                          plan["filter"]["inputs"][1]["inputs"][0]["value"]["value"],
                      )
                    else:
                      filter = "{} {} {}".format(
                          plan["filter"]["inputs"][0]["inputs"][0]["fieldName"],
                          plan["filter"]["functionName"],
                          plan["filter"]["inputs"][1]["value"]["value"],
                      )
                else:
                  if plan["filter"]["inputs"][1]["name"] == "CastTypedExpr":
                    # there is a casting
                      filter = "{} {} {}".format(
                          plan["filter"]["inputs"][0]["fieldName"],
                          plan["filter"]["functionName"],
                          plan["filter"]["inputs"][1]["inputs"][0]["value"]["value"],
                      )
                  else:
                    filter = "{} {} {}".format(
                        plan["filter"]["inputs"][0]["fieldName"],
                        plan["filter"]["functionName"],
                        plan["filter"]["inputs"][1]["value"]["value"],
                    )
                filters.append(filter)
            else:
                raise ValueError(
                    "Unsupported filter function: {}".format(
                        plan["filter"]["functionName"]
                    )
                )

            # print("[INFO-format_filter] filter: ", plan["filter"], " filters: ", filters)
    except Exception as e:
        print("[ERROR-format_filter] plan: ", plan["filter"])
        raise e

    return filters, alias


def extract_ml_operators(node, ml_ops=None, ml_op_dims=None, ml_nested_kernels=None):
    if ml_ops is None:
        ml_ops = []
    if ml_op_dims is None:
        ml_op_dims = []
    if ml_nested_kernels is None:
        ml_nested_kernels = []

    # Define a regex pattern to match ML operators with suffixes
    ml_operator_pattern = re.compile(
        r"^(relu|mat_mul|mat_vector_add|softmax|argmax|batch_norm|torchdnn)"
    )

    # Check if the node contains a "functionName" and matches the pattern
    if "functionName" in node:
        function_name = node["functionName"]
        if ml_operator_pattern.match(function_name):
            ml_ops.append(function_name)
            ml_op_dims.append(node.get("dims", [0]))
            ml_nested_kernels.append(node.get("torchdnn_kernels", []))

        # Recursively check the "inputs" if they exist
        if "inputs" in node and isinstance(node["inputs"], list):
            for child in node["inputs"]:
                extract_ml_operators(child, ml_ops, ml_op_dims, ml_nested_kernels)
    # reverse the order: starting from the first layer
    return ml_ops[::-1], ml_op_dims[::-1], ml_nested_kernels[::-1]


def format_ml_ops(plan):
    list_ml_ops = []
    list_ml_op_dims = []
    list_ml_nested_kernels = []
    if "projections" in plan:
        for projection in plan["projections"]:
            # Check if the node contains a "functionName" and matches the pattern
            if "functionName" in projection:
                ml_ops, ml_op_dims, ml_nested_kernels = extract_ml_operators(projection)
                list_ml_ops.extend(ml_ops)
                list_ml_op_dims.extend(ml_op_dims)
                list_ml_nested_kernels.extend(ml_nested_kernels)

    return list_ml_ops, list_ml_op_dims, list_ml_nested_kernels


def format_ml_op_name(op_name):
    if "mat_mul" in op_name:
        return "mat_mul"
    elif "mat_vector_add" in op_name:
        return "mat_vector_add"
    elif "relu" in op_name:
        return "relu"
    elif "softmax" in op_name:
        return "softmax"
    elif "argmax" in op_name:
        return "argmax"
    elif "batch_norm" in op_name:
        return "batch_norm"
    elif "torchdnn" in op_name:
        return "torchdnn"
    else:
        logging.warning(f"[MLOP-Format] Unsupported ML operator: {op_name}")
        return "NA"


def compute_torchdnn_computation_complexity(ml_nested_op_dims, ml_nested_ops):
    complexity = 0
    for idx, op in enumerate(ml_nested_ops):
        if op == "MatMul":
            complexity += ml_nested_op_dims[idx * 2] * ml_nested_op_dims[idx * 2 + 1]
        elif op == "MatAdd":
            complexity += ml_nested_op_dims[idx * 2]
        elif op == "ReLU":
            complexity += ml_nested_op_dims[idx * 2]
        elif op == "Softmax":
            complexity += ml_nested_op_dims[idx * 2]
        elif op == "Argmax":
            complexity += ml_nested_op_dims[idx * 2]
        elif op == "BatchNorm":
            complexity += ml_nested_op_dims[idx * 2]
        else:
            logging.warning(
                f"[TorchDNN-Complex.-Comput.] Unsupported ML operator: {op}"
            )
    return complexity


def parse_predicate(predicate):
    # Define known operators (longer ones first to prevent partial match issues)
    operators = ['!=', '>=', '<=', '=', '>', '<', 'like', 'eq', 'gt', 'lt', 'gte', 'lte']
    # Sort operators by length to match longest operator first
    operators = sorted(operators, key=len, reverse=True)
    pattern = r'\s*(' + '|'.join(map(re.escape, operators)) + r')\s*'
    
    match = re.split(pattern, predicate, maxsplit=1)
    if len(match) == 3:
        column, operator, value = match
        column = column.strip()
        operator = operator.strip()
        value = value.strip()
        if column == "store_id":
          column = "store"
        return column, operator, value
    else:
        raise ValueError(f"Could not parse predicate: {predicate}")

class Encoder:
    def __init__(
        self,
        column_min_max_vals,
        categorical_vals_mapping,
        col2idx,
        op2idx={"NA": 0, "lt": 1, "eq": 2, "gt": 3, "lte": 4, "gte": 5, "like": 6},
        mlop2idx={
            "NA": 0,
            "mat_mul": 1,
            "mat_vector_add": 2,
            "relu": 3,
            "softmax": 4,
            "argmax": 5,
            "batch_norm": 6,
            "torchdnn": 7,
        },
    ):
        self.column_min_max_vals = column_min_max_vals
        self.categorical_vals_mapping = categorical_vals_mapping
        self.col2idx = col2idx
        self.op2idx = op2idx

        idx2col = {}
        for k, v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col
        self.idx2op = {v: k for k, v in op2idx.items()}
        self.mlop2idx = mlop2idx
        self.idx2mlop = {v: k for k, v in mlop2idx.items()}
        self.op_complexity_min_max = 1, 100000 * 2048

        # store the minmium value and maximum value of rows and cols
        self.table_rows_min_max = 10, 2500000
        self.table_cols_min_max = 1, 500

        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}

        self.table2idx = {"NA": 0}
        self.idx2table = {0: "NA"}

    def set_column_normalizer(self, column_min_max_vals, categorical_vals_mapping):
        self.column_min_max_vals = column_min_max_vals
        self.categorical_vals_mapping = categorical_vals_mapping

    def normalize_table_stats(self, num_rows, num_cols):
        num_rows_normalized = (num_rows - self.table_rows_min_max[0]) / (
            self.table_rows_min_max[1] - self.table_rows_min_max[0]
        )
        num_cols_normalized = (num_cols - self.table_cols_min_max[0]) / (
            self.table_cols_min_max[1] - self.table_cols_min_max[0]
        )
        if num_rows_normalized > 1:
            logging.warning(
                f"Table rows {num_rows} is greater than max value {self.table_rows_min_max[1]}"
            )
        if num_cols_normalized > 1:
            logging.warning(
                f"Table cols {num_cols} is greater than max value {self.table_cols_min_max[1]}"
            )
        return num_rows_normalized, num_cols_normalized

    def encode_ml_ops(self, ml_ops, ml_op_dims, ml_nested_ops):
        # #  version 1
        # most_complicate_op = "NA"
        # least_complicate_op = "NA"
        # op_complexity_min = self.op_complexity_min_max[1]
        # op_complexity_max = self.op_complexity_min_max[0]
        # num_ops = len(ml_ops)
        # if len(ml_ops) == 0:
        #     return 0, self.mlop2idx["NA"], self.mlop2idx["NA"], 0.0, 0.0,
        # for idx, op in enumerate(ml_ops):
        #     op_complexity = np.prod(ml_op_dims[idx])
        #     if op_complexity > op_complexity_max:
        #         op_complexity_max = op_complexity
        #         most_complicate_op = self.mlop2idx[format_ml_op_name(op)]
        #     if op_complexity < op_complexity_min:
        #         op_complexity_min = op_complexity
        #         least_complicate_op = self.mlop2idx[format_ml_op_name(op)]

        # # normalize op_complexity
        # op_complexity_min = (op_complexity_min - self.op_complexity_min_max[0]) / (self.op_complexity_min_max[1] - self.op_complexity_min_max[0])
        # op_complexity_max = (op_complexity_max - self.op_complexity_min_max[0]) / (self.op_complexity_min_max[1] - self.op_complexity_min_max[0])

        # return num_ops, least_complicate_op, most_complicate_op, op_complexity_min, op_complexity_max

        # version 2
        most_complicate_op = "NA"
        least_complicate_op = "NA"
        op_complexity_min = self.op_complexity_min_max[1]
        op_complexity_max = self.op_complexity_min_max[0]
        num_ops = len(ml_ops)
        list_op_complexity = np.zeros(50)  # use fixed lens
        if len(ml_ops) == 0:
            return 0, self.mlop2idx["NA"], self.mlop2idx["NA"], list_op_complexity
        for idx, op in enumerate(ml_ops):
            op_complexity = 0
            if "torchdnn" in op:
                # compute the complexity of the nested kernels if it is a torchdnn operation
                op_complexity = compute_torchdnn_computation_complexity(
                    ml_nested_op_dims=ml_op_dims[idx], ml_nested_ops=ml_nested_ops[idx]
                )
            else:
                op_complexity = np.prod(ml_op_dims[idx])
            if op_complexity == 0:
                # infer the complexity of the operation from last op
                # use the last dimension to estimate the complexity
                if idx > 0:
                    op_complexity = ml_op_dims[idx - 1][-1]

            if op_complexity > op_complexity_max:
                most_complicate_op = format_ml_op_name(op)
            if op_complexity < op_complexity_min:
                least_complicate_op = format_ml_op_name(op)
            list_op_complexity[idx] = op_complexity

        # normalize op_complexity
        list_op_complexity = (list_op_complexity - self.op_complexity_min_max[0]) / (
            self.op_complexity_min_max[1] - self.op_complexity_min_max[0]
        )

        most_complicate_op = self.mlop2idx[most_complicate_op]
        least_complicate_op = self.mlop2idx[least_complicate_op]

        return num_ops, least_complicate_op, most_complicate_op, list_op_complexity

    def normalize_val(self, column, val, log=False):
        # if column is categorical
        if column in self.categorical_vals_mapping:
            if val in self.categorical_vals_mapping[column]:
                return self.categorical_vals_mapping[column][val]
            else:
                logging.warning(f"[normalize_val] Value {val} not found in mapping for column {column}")
                return 0
        # if column is numerical
        else:
            mini, maxi = self.column_min_max_vals[column]

            val_norm = 0.0
            if maxi > mini:
                val_norm = (float(val) - mini) / (maxi - mini)
            else:
                raise ValueError(
                    f"[normalize_val] Min value {mini} is greater than max value {maxi} for column {column}"
                )
            return val_norm

    def encode_filters(self, filters=[], alias=None):
        ## filters: list of dict

        #        print(filt, alias)
        if len(filters) == 0:
            return {
                "colId": [self.col2idx["NA"]],
                "opId": [self.op2idx["NA"]],
                "val": [0.0],
            }
        res = {"colId": [], "opId": [], "val": []}
        for filt in filters:
            filt = "".join(c for c in filt if c not in "()")
            fs = filt.split(" AND ")
            for f in fs:
                #           print(filters)
                # col, op, val = f.split(" ")
                # improved version
                col, op, val = parse_predicate(f)
                if alias is not None:
                    column = alias + "." + col
                else:
                    column = col
                # column = alias + '.' + col
                #            print(f)
                if column not in self.col2idx:
                    # create a new label for it
                    self.col2idx[column] = len(self.col2idx)
                    self.idx2col[self.col2idx[column]] = column
                    # logging.warning(
                    #     f"[encode_filters] Column {column} not found in encoder.col2idx"
                    # )
                    # return {
                    #     "colId": [self.col2idx["NA"]],
                    #     "opId": [self.op2idx["NA"]],
                    #     "val": [0.0],
                    # }
                res["colId"].append(self.col2idx[column])
                res["opId"].append(self.op2idx[op])
                res["val"].append(self.normalize_val(column, val))
        return res

    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]

    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(
        self,
        nodeType,
        typeId,
        filt,
        card,
        join,
        join_str,
        filterDict,
        ml_model_embeds=None,
    ):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt

        self.table = "NA"
        self.table_id = 0
        self.query_id = None  ## so that sample bitmap can recognise

        self.join = join
        self.join_str = join_str
        self.card = card  #'Actual Rows'
        self.children = []
        self.rounds = 0
        self.agg_keys = []

        self.filterDict = filterDict

        # only be set at TableScan node
        self.num_rows = 0
        self.num_cols = 0

        # only be set when ML ops are invoked
        self.ml_model_embeds = ml_model_embeds
        # self.ml_ops = []
        # self.ml_op_dims = []
        # self.ml_nested_ops = []

        self.parent = None

        self.feature = None

    def addChild(self, treeNode):
        self.children.append(treeNode)

    def __str__(self):
        # TODO: add model computation graph here
        #        return TreeNode.print_nested(self)
        return "{} with table: {}, filter: {}, {}, {} children".format(
            self.nodeType,
            self.table,
            self.filter,
            # self.ml_ops,
            # self.ml_op_dims,
            self.join_str,
            len(self.children),
        )

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        print(
            "--" * indent
            + "{} with table: {}, filter: {}, {}, {} children".format(
                node.nodeType,
                node.table,
                node.filter,
                # node.ml_ops,
                # node.ml_op_dims,
                node.join_str,
                len(node.children),
            )
        )
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)

class WeisfeilerLehmanQueryEncoder:
    def __init__(self, range_percent=0.1, num_iterations=2, model_embed_dim=192, embed_sim_threshold=0.8):
        # Maintain groups of labels, where the key is the operator type
        # and the value is a list of unique node representations for different node type
        self.label_groups = defaultdict(set)
        self.range_percent = range_percent
        self.num_iterations = num_iterations
        self.embed_sim_threshold = embed_sim_threshold
        self.unique_labels = set()  # To track unique labels at final iteration
        self.model_embedding_index = faiss.index_factory(
              192, "Flat", faiss.METRIC_INNER_PRODUCT
          )
        self.model_embedding_idx = 0

    def inital_node_label(self, node):
        node_type = node.nodeType
        node_ml_model_embeds = node.ml_model_embeds
        node_contains_ml = not (node_ml_model_embeds == 0).all()

        node_groups = self.label_groups[node_type]
  
        if node_contains_ml:
            # search node
            query_embed = node_ml_model_embeds.reshape(1, -1).copy()
            faiss.normalize_L2(query_embed)
            D, I = self.model_embedding_index.search(query_embed, 5)
            if D[0][0] > self.embed_sim_threshold:
                label = f"{node_type}_{I[0][0]}"
            else:
                label = f"{node_type}_{self.model_embedding_idx}"
                self.model_embedding_idx += 1
                self.model_embedding_index.add(query_embed)
        elif node_type == "TableScanNode":
            # For TableScan nodes, we use the table name as the label
            table_id = node.table_id
            label = f"{node_type}_{table_id}"
        elif node_type == "FilterNode":
            # For Filter nodes, we use the filter condition as the label
            filter_col_id = node.filterDict['colId'][0]
            filter_op_id = node.filterDict['opId'][0]
            filter_op_val = node.filterDict['val'][0]
            label = f"{node_type}_{filter_col_id}_{filter_op_id}_{filter_op_val}"
        elif node_type == "ProjectNode":
            # For Project nodes, we use the projection columns as the label
            label = node_type
        elif node_type == "HashJoinNode" or node_type == "NestedLoopJoinNode":
            # For Join nodes, we use the join condition as the label
            label = f"{node_type}_{node.join}"
        elif node_type == "AggregationNode":
            # For Aggregation nodes, we use the aggregation columns as the label
            agg_cols = "_".join([str(col) for col in node.agg_keys])
            label = f"{node_type}_{agg_cols}"
        else:
            label = node_type
        
        node_groups.add(label)
        
        return label

    def hash_label(label):
        """Deterministically hash a string label into a fixed string."""
        return hashlib.md5(label.encode()).hexdigest()

    def get_wl_subtree_features(self, root):
        """
        Extract WL subtree features from a model tree.
        """
        label_dict = {}         # node -> current label
        label_history = {}      # node -> list of labels at each iteration
        node_list = []

        # Step 1: Traverse tree to initialize labels
        def dfs(node):
            label = self.inital_node_label(node)
            # label = str(node.ml_op_type)  # Initial version, using only the type
            label_dict[node] = label
            label_history[node] = [label]
            node_list.append(node)
            for child in node.children:
                dfs(child)
        dfs(root)

        # Step 2: WL iterations
        for i in range(self.num_iterations):
            new_labels = {}
            for node in node_list:
                child_labels = sorted([label_dict[child] for child in node.children])
                neighbor_str = label_dict[node] + "_" + "_".join(child_labels)
                # new_label = hash_label(neighbor_str)
                new_label = neighbor_str
                new_labels[node] = new_label
                label_history[node].append(new_label)
            label_dict = new_labels
        
        # Step 3: Count all labels across iterations and add labels to unique labels set
        feature_counter = defaultdict(int)
        for node in node_list:
            for lbl in label_history[node]:
                feature_counter[lbl] += 1
            self.unique_labels.update(label_history[node])

        return feature_counter

    def assign_init_label_for_dataset(self, dataset):
        """
        Encode all trees in the dataset and return a list of feature dictionaries.

        Args:
          dataset : ModelComputationGraphDataset
        """
        for i in range(len(dataset)):
          root_node = dataset.rootNodes[i]
          wl_feature = self.get_wl_subtree_features(root_node)
          dataset.wl_features.append(wl_feature)
    
    def obtain_wl_feature_for_dataset(self, dataset):
        self.assign_init_label_for_dataset(dataset)
        wl_sorted_labels = sorted(self.unique_labels)
        dataset.wl_feature_vectors = np.zeros((len(dataset), len(wl_sorted_labels)), dtype=np.float32)
        wl_labels2idx = {label: idx for idx, label in enumerate(wl_sorted_labels)}

        for i in tqdm(range(len(dataset))):
          wl_feature = dataset.wl_features[i]
          for label, count in wl_feature.items():
              if label in wl_labels2idx:
                  dataset.wl_feature_vectors[i, wl_labels2idx[label]] = count
    
    def construct_similar_dissimilar_pairs_for_dataset(self, dataset, sim_threshold=0.8, dissim_threshold=0.2):
        faiss_index = faiss.index_factory(
              dataset.wl_feature_vectors.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT
          )
        data_to_add = dataset.wl_feature_vectors.copy()
        faiss.normalize_L2(data_to_add)
        faiss_index.add(data_to_add)

        query_vectors = dataset.wl_feature_vectors[:].copy()
        faiss.normalize_L2(query_vectors)
        D, I = faiss_index.search(query_vectors, len(dataset))  
        for i in range(len(dataset)):
          max_sim = D[i][1]
          min_sim = D[i][-1]
          max_sim_idx = I[i][1]
          min_sim_idx = I[i][-1]
          if max_sim > sim_threshold:
              dataset.similar_query_idx[i] = max_sim_idx
          else:
              dataset.similar_query_idx[i] = i  
          if min_sim < dissim_threshold:
              dataset.dissimilar_query_idx[i] = min_sim_idx
          else:
              if i > 0:
                  dataset.dissimilar_query_idx[i] = dataset.similar_query_idx[i - 1]
              else:
                  dataset.dissimilar_query_idx[i] = random.randint(0, len(dataset) - 1)
              warnings.warn(f"[WL-Encoder] No dissimilar query found for model {i}, minimum similarity: {min_sim}")