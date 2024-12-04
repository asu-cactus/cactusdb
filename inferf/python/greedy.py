import heapq
import json
from collections import defaultdict
import torch


class Edge:
    def __init__(self, _id, parent_node, child_node, direction, num_input_features, num_input_rows, num_output_features, num_output_rows):
        self._id = _id
        self.parent_node = parent_node
        self.child_node = child_node
        self.direction = direction
        self.num_input_features = num_input_features
        self.num_input_rows = num_input_rows
        self.num_output_features = num_output_features
        self.num_output_rows = num_output_rows



def performGreedy(edges, factorized_output_features, utility_threshold=0.5, k1=9.8167, k2=2.1713):

    def getUtility(num_input_features, num_input_rows, num_output_features, num_output_rows):
        #return k1 * (num_output_features - factorized_output_features) + k2 * (num_output_rows*num_output_features - num_input_rows*num_input_features)
        return k1 * (num_output_features - factorized_output_features) + k2 * (num_output_rows*num_output_features - num_input_rows*num_output_features)


    labels = {}
    utility = {}
    node_to_edge = defaultdict(lambda: [])
    gpq = []
    processed = set()

    for e in edges:
        labels[e._id] = 0
        temp_list = node_to_edge[e.parent_node]
        temp_list.append((e._id, e.direction))
        node_to_edge[e.parent_node] = temp_list

        utility[e._id] = getUtility(e.num_input_features, e.num_input_rows, e.num_output_features, e.num_output_rows)
        if utility[e._id] >= utility_threshold:
            heapq.heappush(gpq, (-1 * utility[e._id], e._id))


    child_edge_list = defaultdict(lambda: ["", ""])
    for e in edges:
        child_node = e.child_node
        if child_node in node_to_edge:
            e1, direction1 = node_to_edge[child_node][0]
            e2, direction2 = node_to_edge[child_node][1]
            child_edge_list[e._id][direction1] = e1
            child_edge_list[e._id][direction2] = e2

    def traverseBranch(eLocal):
        if eLocal in processed:
            return labels[eLocal]


        if eLocal not in child_edge_list:
            if utility[eLocal] >= utility_threshold:
                labels[eLocal] = 1
            processed.add(eLocal)
            return labels[eLocal]
        else:
            left, right = child_edge_list[eLocal]

            if left not in processed:
                labels[left] = traverseBranch(left)
            if right not in processed:
                labels[right] = traverseBranch(right)
                    
            if (labels[left] == 1 and labels[right] == 1) or (labels[left] == 1 and labels[right] == 2) or (labels[left] == 2 and labels[right] == 1):
                labels[eLocal] = 2
            else:
                if utility[eLocal] >= utility_threshold:
                    labels[eLocal] = 1
            processed.add(eLocal)
            return labels[eLocal]


    while gpq:
        max_pair = heapq.heappop(gpq)
        uCurrent = max_pair[0]
        eCurrent = max_pair[1]
        labels[eCurrent] = traverseBranch(eCurrent)

    return labels


def getEdgesFromJson(json_data):
    idx = 0
    edge_list = []
    for json_obj in json_data:
        join_id = json_obj['ID']
        left = json_obj['Left']
        right = json_obj['Right']
        tuple_left = json_obj['NumTuplesLeft']
        dim_left = json_obj['NumDimLeft']
        tuple_right = json_obj['NumTuplesRight']
        dim_right = json_obj['NumDimRight']
        tuple_output = json_obj['NumTuplesOutput']
        dim_output = json_obj['NumDimOutput']

        edge = Edge(idx, join_id, left, 0, dim_left, tuple_left, dim_output, tuple_output)
        edge_list.append(edge)
        idx += 1

        edge = Edge(idx, join_id, right, 1, dim_right, tuple_right, dim_output, tuple_output)
        edge_list.append(edge)
        idx += 1
    return edge_list



def getNeuronInputSize(model_path, input_layer):
    state_dict = torch.load(model_path)
    input_weight = state_dict[f"{input_layer}.weight"]
    return input_weight.shape[0]




if __name__ == "__main__":
    file_path = "plans/4_3.txt"
    num_neurons = getNeuronInputSize("plans/dummy.pth", "fc1")

    with open(file_path, "r") as file:
        json_string = file.read()

    if json_string.startswith('R"(') and json_string.endswith(')"'):
        json_string = json_string[3:-2]
    json_data = json.loads(json_string)
    
    edge_list = getEdgesFromJson(json_data)
    labels = performGreedy(edge_list, factorized_output_features=num_neurons, utility_threshold=0.5)

    print("Number of neurons in the split layer:", str(num_neurons))
    for edge in edge_list:
        print(f"Plan: {edge.child_node} ---> {edge.parent_node} = {labels[edge._id]}")


