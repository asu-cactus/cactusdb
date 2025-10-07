import time 
class Timer(object):
    """A convenient class to measure the running time of a program"""

    def __init__(self):
        self.start = 0
        self.end = 0

    def tic(self):
        """Tic the start time"""
        self.start = time.perf_counter()

    def toc(self):
        """Toc the end time and return the running time

        Returns:
            float: running time (ms)
        """
        self.end = time.perf_counter()
        return (self.end - self.start) * 1000
    

import onnx
from onnx import numpy_helper
def get_list_node_names(node_list):
    return [node.name for node in node_list]
import onnx
from collections import deque, defaultdict
from onnx import numpy_helper
import h5py
import numpy as np
import os

model_dir = "/home/velox/resources/model/load_onnx_models/"
# model_path = os.path.join(model_dir, "two_input_model1.onnx")
model_path = os.path.join(model_dir, "two_input_model.onnx")
# model_path = os.path.join(model_dir, "ffnn.onnx")
model = onnx.load(model_path)
graph = model.graph
nodes = graph.node

timer = Timer()

timer.tic()
# build adjacency list and in-degree count 
adjacency_list = defaultdict(list)
in_degree = defaultdict(int)
node_map  = {}

for node in nodes:
  node_map[node.name] = node
  for output in node.output:
    for next_node in nodes:
      if output in next_node.input:
        adjacency_list[node.name].append(next_node.name)
        in_degree[next_node.name] += 1
    if node.name not in in_degree:
      in_degree[node.name] = 0

# Initialize partitioning
partitions = defaultdict(list)
queue = deque([node for node in graph.node if in_degree[node.name] == 0])
visited = set()   
node_to_partition = {}
dependency_graph = defaultdict(set)

group_id = 0
while queue:
  group = []
  group_node = queue.popleft()
  temp_queue = deque([group_node])
  while temp_queue:
      node = temp_queue.popleft()
      node_to_partition[node.name] = group_id
      for node_output in node.output:
        node_to_partition[node_output] = group_id
      if node.name in visited:
          continue
      group.append(node)
      visited.add(node.name)

      # add dependencies 
      for input in node.input:
         if input in node_to_partition:
            if group_id != node_to_partition[input]:
              # do not add self to dependency graph
              dependency_graph[group_id].add(node_to_partition[input])
      
      for next_node in adjacency_list[node.name]:
          # get node based on name
          next_node = node_map[next_node]
          if next_node.name in visited:
              continue
          if in_degree[next_node.name] == 1:
              temp_queue.append(next_node)
          else:
              # Defer processing if multiple inputs
              if next_node not in queue:
                queue.append(next_node)
  if len(group) > 1:
    partitions[group_id] = group
    group_id += 1

model_tensors = dict()
initializers = {init.name: init for init in graph.initializer}
for group_id, group in partitions.items():
  # iterate each node in the group
  for node in group:
    node_name = node.name
    if node.op_type == "Gemm":
      # process the node contains the parameters
      for input in node.input:
        if input in initializers:
          tensor = initializers[input]
          tensor_array = numpy_helper.to_array(tensor).astype(np.float32).T
          tensor_name = node.name
          # tensor_name = tensor_name.replace("/", "")
          if 'weight' in tensor.name:
            tensor_name = f"{tensor_name}_weight"
          elif 'bias' in tensor.name:
            tensor_name = f"{tensor_name}_bias"
          else:
            raise ValueError(f"Unknown tensor name: {tensor.name}")
          model_tensors[tensor_name] = tensor_array
          model_tensors[f"{tensor_name}_shape"] = tensor_array.shape
        else:
          pass

# prepare the group data
op_id = 0
serialized_model_graph = []
intermediate_group_output = set()
op_name2tensor_map = {}
for group_id, group in partitions.items():
  serialized_subgroup_left = ""
  serialized_subgroup_right = ""
  for i in range(len(group)):
    if i == 0:
      # add the intermediate group output to the input list
      intermedaite_results_list = [input for input in group[i].input if input in intermediate_group_output]
      serialized_subgroup_right = ",".join(intermedaite_results_list) + serialized_subgroup_right
    node = group[i]
    if node.op_type == "Gemm":
      mat_mul_op_name = "mat_mul" + str(op_id)
      op_name2tensor_map[node.name + "_weight"] = mat_mul_op_name + "_weight"
      op_name2tensor_map[node.name + "_weight_shape"] = mat_mul_op_name + "_shape"
      op_id += 1
      mat_add_op_name = "mat_add" + str(op_id)
      op_name2tensor_map[node.name + "_bias"] = mat_add_op_name + "_weight"
      op_name2tensor_map[node.name + "_bias_shape"] = mat_add_op_name + "_shape"
      op_id += 1
      serialized_subgroup_left = mat_add_op_name + "(" + mat_mul_op_name + "(" + serialized_subgroup_left
      serialized_subgroup_right = serialized_subgroup_right + "))"
    elif node.op_type == "Relu":
      relu_op_name = "relu"
      serialized_subgroup_left = relu_op_name + "(" + serialized_subgroup_left
      serialized_subgroup_right = serialized_subgroup_right + ")"
    elif node.op_type == "Concat":
      relu_op_name = "concat"
      serialized_subgroup_left = relu_op_name + "(" + serialized_subgroup_left
      serialized_subgroup_right = serialized_subgroup_right + ")"
    
    if i == len(group) - 1:
      # add the last node output to the intermediate_group_output
      intermediate_group_output.add(node.output[0])

  serialized_expression = "{left_expr}|{right_expr}| as {output}".format(
      left_expr=serialized_subgroup_left,
      right_expr=serialized_subgroup_right,
      output=node.output[0]
  )  
  serialized_model_graph.append(serialized_expression)

  serialized_model_graph = [s.replace("/", "") for s in serialized_model_graph]

with h5py.File('../../../temp/test_onnx_ffnn.h5', 'w') as f:
  for dataset_name, tensor in model_tensors.items():
    print("mapping: ", dataset_name, "->", op_name2tensor_map[dataset_name])
    f.create_dataset(op_name2tensor_map[dataset_name], data=tensor)


# output the serialized model graph
with open("../../../temp/test_onnx_serialized.txt", "w") as f:
  for serialized_expression in serialized_model_graph:
    f.write(serialized_expression + "\n")

with open("../../../temp/test_onnx_dependencies.txt", "w") as f:
  for group_id, dependent_groups in dependency_graph.items():
    temp_r = [group_id] + list(dependent_groups)
    line = " ".join(str(i) for i in temp_r)
    f.write(line + "\n")

print("Model serialization time: ", timer.toc())