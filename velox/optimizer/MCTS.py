import time
import math
import random
from tqdm.auto import tqdm
from typing import Optional
import socket
import json


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


class MCTSTreeNode:
    def __init__(self, state: dict, parent: Optional["MCTSTreeNode"] = None):
        """The state is dictionary where key is the expression and the value is
        the action that will be taken.
        For example: {'mat_mul0(ROW["v"])': 'Mul2JoinAgg'}
        """
        self.state = state
        self.children = []
        self.num_visit = 0
        self.reward = 0
        self.parent = parent
        self.action_space = self.get_action_space()
        self.is_terminal = self.check_terminal()
        self.is_fully_expanded = self.check_and_update_is_fully_expanded()

    def check_terminal(self):
        if len(self.get_action_space()) == 0:
            return True
        return False

    def get_action_space(self) -> dict():
        """Need to compute communicate with the Velox via socket and get the
        possible action for each expression
        Result is stored as a dictionary: tuple(expression, action): bool
        {('mat_mul0(ROW["v"])', 'Mul2JoinAgg'): false, ...}
        the boolean flag is used to indicate whether this action has been taken
        """
        pass
        # TODO

    def check_and_update_is_fully_expanded(self) -> bool:
        """Check if the current node is fully expanded and update the flag"""
        if len(self.children) == len(self.action_space):
            self.is_fully_expanded = True
        else:
            self.is_fully_expanded = False
        return self.is_fully_expanded


class MCTS:
    def __init__(
        self,
        max_iteration_num: int = 2000,
        max_iteration_time: int = None,
        exploration_weight: float = math.sqrt(2),
        rollout_policy: str = "random",
    ):
        """
        Args:
            max_iteration_num (int) : maximum number of search iteration
            max_iteration_time (int) : maximum time (ms) of search iteration
            exploration_weight (float) : factor for the UCB formula
            rollout_policy (str) : rollout policy
        """
        if max_iteration_num == None and max_iteration_time == None:
            raise ValueError("Must have either a time limit or an iteration limit")
        if max_iteration_num < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.max_iteration_num = max_iteration_num
        self.max_iteration_time = max_iteration_time
        self.exploration_weight = exploration_weight
        self.rollout_policy = rollout_policy
        self.iteration_count = 0
        self.timer = Timer()

    def search(self, root_node: MCTSTreeNode):
        """MCTS search algorithm"""
        self.root_node = root_node
        node = self.root_node
        self.timer.tic()
        for iter_idx in tqdm(self.max_iteration_num):
            t_elapsed_time = self.timer.toc()
            if t_elapsed_time >= self.max_iteration_time:
                # exist search if exceeds the maximum search time
                print(
                    "[INFO] maximum search time reached out, current search iteration idx: {}".format(
                        iter_idx
                    )
                )
                break

            # check if the current node is terminal node
            while not node.is_terminal:
                node.check_and_update_is_fully_expanded()
                if node.is_fully_expanded:
                    # select the best node based on UCT
                    node = self.select(node)
                else:
                    # expand the node if the node is not fully expanded
                    new_node = self.expand(node)
                    # run simulation after expand
                    reward = self.simulate(new_node)
                    self.back_propagate(new_node, reward)
                    continue
            # get reward via simulation after reaching the terminal state
            reward = self.simulate(node)
            self.back_propagate(node, reward)

    def select(self, node: MCTSTreeNode) -> MCTSTreeNode:
        """Select best node based on UCT"""
        selected_node = max(
            node.children,
            key=lambda child: child.reward / child.num_visit
            + self.exploration_weight
            * math.sqrt(math.log(node.num_visit) / child.num_visit),
        )
        # child num_visit will be updated during back-propagation
        return selected_node

    def expand(self, node: MCTSTreeNode) -> MCTSTreeNode:
        """Expand not fully explorered node, and return a child node"""
        unexplorered_action = [
            key for key, val in node.action_space.items() if val == False
        ]
        selected_expression, selected_action = random.choice(unexplorered_action)
        new_state = node.state.copy()
        new_state[selected_expression] = selected_action
        new_node = MCTSTreeNode(new_state, parent=node)
        node.children.append(new_node)
        return new_node

    def simulate(self, node: MCTSTreeNode) -> float:
        """Run simulation to get reward"""
        while not node.is_terminal():
            # randomly select an action
            possible_actions = list(node.action_space.keys())
            selected_action = random.choice(possible_actions)
            new_state = node.state.copy()
            new_state[selected_action] = selected_action
            node = MCTSTreeNode(new_state)
        # get reward of terminal node
        reward = self.get_reward(node)
        return reward

    def get_reward(self, node: MCTSTreeNode) -> float:
        """Communicate with Velox to get reward with given state"""
        # TODO
        return 0

    def back_propagate(self, node: MCTSTreeNode, reward: float):
        """Back propagate to update reward and num_visit through the path"""
        while node is not None:
            node.num_visit += 1
            node.reward += reward
            node = node.parent


def send_message_by_socket(message, client_socket):
    client_socket.sendall(json.dumps(message).encode("utf-8"))


def receive_message_by_socket(client_socket):
    received_message_str = client_socket.recv(1024).decode("utf-8")
    json_message = json.loads(received_message_str)
    return json_message


if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("localhost", 12345)

    server_socket.bind(server_address)
    server_socket.listen(1)

    print("Python server listening on port 12345...")
    client_socket, client_address = server_socket.accept()

    print(f"Connected to C++ client: {client_address}")

    optimization_is_finished = False
    while not optimization_is_finished:
        # Receive message
        received_json_message = receive_message_by_socket(client_socket)
        print("[INFO] Recevied message", received_json_message)
        # Do something
        send_message = dict()
        mctsAction = received_json_message["mctsAction"]
        if mctsAction == "start":
            initQueryPlan = received_json_message["queryPlan"]
            rootNode = MCTSTreeNode({"queryPlan": initQueryPlan})
            optimization_is_finished = True
            send_message = {"optimizationIsFinished": True}
        # elif mctsAction == 'recQueryPlan':
        #     send_message = {"mctsAction": "", "optimizationIsFinished": True}
        # Send message
        send_message_by_socket(send_message, client_socket)
