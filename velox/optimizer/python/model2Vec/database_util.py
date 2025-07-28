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
import faiss
import random
from tqdm.auto import tqdm

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
        bins = hist_file.loc[hist_file["column"] == col, "bins"].item()

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
            logging.warning(f"Column {col} not found in encoder")
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
                    filter = "{} {} {}".format(
                        plan["filter"]["inputs"][0]["inputs"][0]["fieldName"],
                        plan["filter"]["functionName"],
                        plan["filter"]["inputs"][1]["value"]["value"],
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
    elif "embedding" in op_name:
        return "embedding"
    elif "sigmoid" in op_name:
        return "sigmoid"
    else:
        warnings.warn(f"[MLOP-Format] Unsupported ML operator: {op_name}")
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

class ModelGraphEncoder:
    def __init__(
        self,
        mlop2idx={
            "NA": 0,
            "mat_mul": 1,
            "mat_vector_add": 2,
            "relu": 3,
            "softmax": 4,
            "argmax": 5,
            "batch_norm": 6,
            "torchdnn": 7,
            "sigmoid": 8,
            "embedding": 9,
            "svd": 10,
        },
        ml_op_flops_min_max=(0, 100000 * 2048),
        ml_op_dims_min_max=(0, 100000) 
    ):
        self.mlop2idx = mlop2idx
        self.idx2mlop = {v: k for k, v in mlop2idx.items()}
        self.ml_op_flops_min_max = ml_op_flops_min_max
        self.ml_op_dims_min_max = ml_op_dims_min_max

    def set_ml_op_flops_min_max(self, ml_op_flops_min_max):
        self.ml_op_flops_min_max = ml_op_flops_min_max

    def set_ml_op_dims_min_max(self, ml_op_dims_min_max):
        self.ml_op_dims_min_max = ml_op_dims_min_max

    def encode_ml_op(self, ml_op):
        if ml_op not in self.mlop2idx:
            self.mlop2idx[ml_op] = len(self.mlop2idx)
            self.idx2mlop[self.mlop2idx[ml_op]] = ml_op
        return self.mlop2idx[ml_op]

    def encode_ml_op_dims(self, ml_op_dims, length=20):
        if isinstance(ml_op_dims, list):
            ml_op_dims = np.array(ml_op_dims)
            encoded_dims = (ml_op_dims - self.ml_op_dims_min_max[0]) / (
                self.ml_op_dims_min_max[1] - self.ml_op_dims_min_max[0]
            )
            encoded_dims = np.pad(
                encoded_dims, (0, length - len(encoded_dims)), "constant", constant_values=0
            )

            if (encoded_dims > 1).any():
                abnormal_idx = np.where(encoded_dims > 1)[0]
                warnings.warn(
                    "Encoded dimensions exceed 1.0, which may indicate an issue with the normalization. Original dimensions: {}, Encoded dimensions: {}".format(
                        ml_op_dims[abnormal_idx], encoded_dims[abnormal_idx]
                    )
                ) 
            return encoded_dims
        else:
            raise ValueError(
                "ml_op_dims should be a list or numpy array, got {}".format(
                    type(ml_op_dims)
                )
            )

    def encode_ml_op_flops(self, ml_flops):
        # Normalize the flops value to [0, 1] using min-max scaling
        min_flops, max_flops = self.ml_op_flops_min_max
        encoded_flops = (ml_flops - min_flops) / (max_flops - min_flops)
        if encoded_flops > 1:
            warnings.warn(
                "Encoded FLOPs value is out of bounds [0, 1]. "
                "This may indicate an issue with the normalization. Original FLOPs: {}, Encoded FLOPs: {}".format(
                    ml_flops, encoded_flops))
        return encoded_flops
    
from collections import defaultdict
import hashlib

def binary_search_with_range(arr, target, range_percent=0.1):
  """
  Perform a binary search to find the index of the value in a sorted array
  where the target is within a specified percentage range of the value.

  Args:
      arr: Sorted list or numpy array of numeric values.
      target: The value to search for.
      range_percent: Acceptable percentage difference (default 0.1 for 10%).

  Returns:
      The index of the value if found within the range, else -1.
  """
  low, high = 0, len(arr) - 1
  while low <= high:
    mid = (low + high) // 2
    val = arr[mid]
    if abs(val - target) <= abs(val) * range_percent:
      return mid
    elif val < target:
      low = mid + 1
    else:
      high = mid - 1
  return -1

class WeisfeilerLehmanEncoder:
    def __init__(self, range_percent=0.1, num_iterations=2):
        # Maintain groups of labels, where the key is the kernel type
        # and the value is a list of nodes with FLOPs. The final label
        # is constructed as kernel_type + "_" + flops. The values are 
        # sorted to ensure deterministic ordering.
        self.label_groups = defaultdict(list)
        self.range_percent = range_percent
        self.num_iterations = num_iterations
        self.unique_labels = set()  # To track unique labels at final iteration

    def inital_node_label(self, node):
        node_type = node.ml_op_type
        node_flop = node.ml_op_flop
        group_flops = self.label_groups[node_type]

        # Find the index of the node's FLOPs in the sorted list of FLOPs
        flops_idx = binary_search_with_range(group_flops, node_flop, self.range_percent)
        if flops_idx == -1:
            label = f"{node_type}_{node_flop}"
            group_flops.append(node_flop)
            group_flops.sort()  # Keep the list sorted for future searches
            self.label_groups[node_type] = group_flops
        else:
            label = f"{node_type}_{group_flops[flops_idx]}"
        
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
              dataset.similar_model_idx[i] = max_sim_idx
          else:
              dataset.similar_model_idx[i] = i  
          if min_sim < dissim_threshold:
              dataset.dissimilar_model_idx[i] = min_sim_idx
          else:
              if i > 0:
                  dataset.dissimilar_query_idx[i] = dataset.similar_query_idx[i - 1]
              else:
                  dataset.dissimilar_query_idx[i] = random.randint(0, len(dataset) - 1)
              warnings.warn(f"[WL-Encoder] No dissimilar query found for model {i}, minimum similarity: {min_sim}")