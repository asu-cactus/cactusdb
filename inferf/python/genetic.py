import json
import random
from collections import defaultdict

# class that define the properties of an input edge
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


# for each input edge, returns the height/distance from corresponding leaf edge. returns 1 for leaf edges
def getHeight(edge_list, child_edge_list):
    height = {}
    def helper(e_id):
        if e_id in height:
            return height[e_id]

        if e_id not in child_edge_list:
            height[e_id] = 1
            return height[e_id]
        else:
            left, right = child_edge_list[e_id]
            height[left] = helper(left)
            height[right] = helper(right)
            height[e_id] = max(height[left], height[right]) + 1
            return height[e_id]

    edge_heights = []
    for edge in edge_list:
        height[edge._id] = helper(edge._id)
        edge_heights.append((edge._id, height[edge._id]))
    return edge_heights


# checks if the labels of various join edges are valid, specially looks for edges where label should be 2
def validateChromosome(current, map_edge, child_edge_list, edge_heights):
    keys = child_edge_list.keys()
    for e_id, _ in edge_heights:
        if e_id in keys:
            left, right = child_edge_list[e_id]
            lLeft, lRight = current[map_edge[left]], current[map_edge[right]]
            if (lLeft == 1 and lRight == 1) or (lLeft == 1 and lRight == 2) or (lLeft == 2 and lRight == 1) or (lLeft == 2 and lRight == 2):
                current[map_edge[e_id]] = 2
            else:
                if current[map_edge[e_id]] == 2:
                    current[map_edge[e_id]] = random.randint(0, 1)
    return current


# method to initialize n valid chromosomes representing various factorization plans
def initChromosome(n, k, map_edge, child_edge_list, edge_heights):
    map_chrm = {}
    count = 0
    while count < n:
        current = [random.choice([0, 1]) for _ in range(k)]
        current_updated = validateChromosome(current, map_edge, child_edge_list, edge_heights)
        if current_updated not in map_chrm.values():
            map_chrm[count] = current_updated
            count += 1
    return map_chrm


# method to exchange subtrees between two parents
def crossoverParents(parent1, parent2, map_edge, child_edge_list, edge_heights):
    keys = list(child_edge_list.keys())
    targetKey = keys.pop(random.randrange(len(keys)))
    offspring1, offspring2 = parent1.copy(), parent2.copy()
    stack, subtree = [targetKey], []

    while stack:
        current = stack.pop()
        subtree.append(current)
        if current in keys:
            stack.extend(childList[current])

    for e_id in subtree:
        offspring1[e_id], offspring2[e_id] = offspring2[e_id], offspring1[e_id]

    return validateChromosome(offspring1, map_edge, child_edge_list, edge_heights), validateChromosome(offspring2, map_edge, child_edge_list, edge_heights)


# method to make random change on the chromosomes generated after crossover between two parent chromosomes
def mutateChromosome(chromosome, map_edge, child_edge_list, edge_heights):
    mutabel_edges = [i for i, label in enumerate(chromosome) if label in [0, 1]]
    if len(mutabel_edges) > 0:
        targetEdge = mutabel_edges.pop(random.randrange(len(mutabel_edges)))
        chromosome[targetEdge] = 1 - chromosome[targetEdge]
    return validateChromosome(chromosome, map_edge, child_edge_list, edge_heights)


def getFitness(chromosome):
    """Fitness function that returns a random value."""
    return random.random()


# method to perform the genetic algorithm
def performGenetic(edges, num_join, factorized_output_features, p=2, max_iter=5, utility_threshold=0.5, k1=9.8167, k2=2.1713):

    labels = {} # stores labels of input join edges
    map_edge = {} # maps edge id to a position index in the chromosome representation
    node_to_edge = defaultdict(lambda: []) # maps a join node to its two child edges

    # initialize labels, map_edges, and node_to_edge
    j = 0
    for e in edges:
        map_edge[e._id] = j
        j += 1
        labels[e._id] = 0
        temp_list = node_to_edge[e.parent_node]
        temp_list.append((e._id, e.direction))
        node_to_edge[e.parent_node] = temp_list

    # child_edge_list holds the left and right child ids for each non-leaf edges 
    child_edge_list = defaultdict(lambda: ["", ""])
    for e in edges:
        child_node = e.child_node
        if child_node in node_to_edge:
            e1, direction1 = node_to_edge[child_node][0]
            e2, direction2 = node_to_edge[child_node][1]
            child_edge_list[e._id][direction1] = e1
            child_edge_list[e._id][direction2] = e2

    edge_heights = getHeight(edge_list, child_edge_list)
    sorted_heights = sorted(edge_heights, key=lambda x: x[1], reverse=False)

    map_chrm = initChromosome(num_join*p, len(edges), map_edge, child_edge_list, edge_heights)
    num_chrm = len(map_chrm.keys())
    bestChrm, bestFitness = map_chrm[0], float("-inf")

    # perform parent selection, crossover of parents, and mutation max_iter times to generate random chromosomes
    for i in range(max_iter):
        idx1, idx2 = random.sample(range(0, num_chrm), 2)
        parent1, parent2 = map_chrm[idx1], map_chrm[idx2]
        parent1, parent2 = crossoverParents(parent1, parent2, map_edge, child_edge_list, edge_heights)
        parent1 = mutateChromosome(parent1, map_edge, child_edge_list, edge_heights)
        parent2 = mutateChromosome(parent2, map_edge, child_edge_list, edge_heights)
        fitness1, fitness2 = getFitness(parent1), getFitness(parent2)

        if fitness1 > bestFitness:
            bestFitness = fitness1
            bestChrm = parent1
        if fitness2 > bestFitness:
            bestFitness = fitness2
            bestChrm = parent2

    return bestChrm, map_edge



def getEdgesFromJson(json_data):
    num_join = 0
    idx = 0
    edge_list = []
    for json_obj in json_data:
        num_join += 1
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
    return edge_list, num_join


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
    
    edge_list, num_join = getEdgesFromJson(json_data)
    selected_plan, map_edge = performGenetic(edge_list, num_join, factorized_output_features=64, utility_threshold=0.5)

    for edge in edge_list:
        print(f"Plan: {edge.child_node} ---> {edge.parent_node} = {selected_plan[map_edge[edge._id]]}")
