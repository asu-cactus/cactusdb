import time
import math
import random
from tqdm.auto import tqdm
from typing import Optional
import socket
import json
import os


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
    # Static variable
    node_id = 0

    def __init__(
        self,
        state: dict,
        client_socket: socket.socket,
        is_terminal: bool = None,
        from_action: tuple = None,
        is_temp_node: bool = False,  # temp node is created during rollout
        parent: Optional["MCTSTreeNode"] = None,
        verbose: int = 0,
    ):
        """The state is dictionary where key is the expression and the value is
        the action that will be taken.
        For example: {'mat_mul0(ROW["v"])': 'Mul2JoinAgg'}
        """
        self.state = state
        self.children = []
        self.num_visit = 0
        self.reward = 0
        self.parent = parent
        self.client_socket = client_socket
        self.printout = False
        self.verbose = verbose
        self.action_space = self.get_action_space()
        self.is_terminal = (
            is_terminal if is_terminal is not None else self.check_terminal()
        )
        self.is_fully_expanded = self.check_and_update_is_fully_expanded()
        self.from_action = from_action
        if is_temp_node:
            self.node_id = -1
        else:
            self.node_id = MCTSTreeNode.node_id
            MCTSTreeNode.node_id += 1

        if self.verbose >= 2:
            print(
                "[INFO] node states: id: {} \n \t\t terminal: {} \n  \t\t parsed action space: {}".format(
                    self.node_id, self.is_terminal, self.action_space
                )
            )

    def check_terminal(self):
        if len(self.action_space) == 0:
            return True
        return False

    def get_action_space(self) -> dict():
        """Need to compute communicate with the Velox via socket and get the
        possible action for each expression
        Result is stored as a dictionary: tuple(expression, action): bool
        {('mat_mul0(ROW["v"])', 'Mul2JoinAgg'): false, ...}
        the boolean flag is used to indicate whether this action has been taken
        """

        send_message = dict()
        send_message["mctsAction"] = "getActionSpace"
        send_message["optimizationIsFinished"] = False
        send_message_by_socket(send_message, self.client_socket, self.verbose)
        received_message = receive_message_by_socket(self.client_socket, self.verbose)
        received_action_space = received_message["actionSpace"]
        action_space = dict()
        for action_pair in received_action_space:
            target_expression = action_pair["expression"]
            list_actions = action_pair["action"]
            for action in list_actions:
                action_space[(target_expression, action)] = False
        # add no-action state for the root node
        if self.parent is None or len(action_space) != 0:
            action_space[("None", "None")] = False
        return action_space

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
        client_socket: socket.socket,
        max_iteration_num: int = 2000,
        max_iteration_time: int = 1 * 60 * 1000,  # 1 mins
        max_sim_iteration_num: int = 100,
        max_sim_iteration_time: int = 100 * 30 * 1000,  # 30 seconds
        exploration_weight: float = math.sqrt(2),
        rollout_policy: str = "random",
        reward_mode: str = "offline",
        verbose: int = 0,
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
        self.client_socket = client_socket
        self.max_iteration_num = max_iteration_num
        self.max_iteration_time = max_iteration_time
        self.exploration_weight = exploration_weight
        self.max_sim_iteration_num = max_sim_iteration_num
        self.max_sim_iteration_time = max_sim_iteration_time
        self.rollout_policy = rollout_policy
        self.iteration_count = 0
        self.reward_mode = reward_mode
        self.timer = Timer()
        self.verbose = verbose

    def train(self, root_node: MCTSTreeNode):
        """MCTS training algorithm"""
        self.root_node = root_node
        self.timer.tic()
        if self.verbose >= 1:
            print("[INFO] ==========Start MCTS Training==========")
        for iter_idx in tqdm(range(self.max_iteration_num)):
            if self.verbose >= 1:
                print("[INFO] search iteration idx: ", iter_idx)
            node = self.root_node
            # each new iteration needs to reset the query plan from root node
            send_message = dict()
            send_message["mctsAction"] = "resetPlan"
            send_message["optimizationIsFinished"] = False
            send_message_by_socket(send_message, self.client_socket, self.verbose)
            t_elapsed_time = self.timer.toc()
            if t_elapsed_time >= self.max_iteration_time:
                # exist search if exceeds the maximum search time
                print(
                    "[INFO] maximum search time reached out, current search iteration idx: {}, current time: {}, max allowed time: {}".format(
                        iter_idx, t_elapsed_time, self.max_iteration_time
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
                    node = new_node
                    break
            # get reward via simulation after reaching the terminal state
            reward = self.simulate(node)
            self.back_propagate(node, reward)
        t_elapsed_time = self.timer.toc()
        print(
            "[INFO] MCTS training finished, elapsed time: {} ms".format(t_elapsed_time)
        )

    def search(self, root_node: MCTSTreeNode):
        """MCTS search algorithm"""
        self.root_node = root_node
        self.timer.tic()
        if self.verbose >= 1:
            print("[INFO] ==========Start MCTS Search==========")
        node = self.root_node
        # each new iteration needs to reset the query plan from root node
        send_message = dict()
        send_message["mctsAction"] = "resetPlan"
        send_message["optimizationIsFinished"] = False
        send_message_by_socket(send_message, self.client_socket, self.verbose)

        # check if the current node is terminal node
        while not node.is_terminal:
            t_elapsed_time = self.timer.toc()
            if t_elapsed_time >= self.max_iteration_time:
                # exist search if exceeds the maximum search time
                print("[INFO] maximum search time reached out")
                break
            # return if no children
            if len(node.children) == 0:
                break
            node.check_and_update_is_fully_expanded()
            node = self.select(node, use_factor=False)

        # get reward via simulation after reaching the terminal state
        t_elapsed_time = self.timer.toc()
        latency = self.run_plan()

        print("[INFO] MCTS search time: {} ms".format(t_elapsed_time))
        print("[INFO] MCTS search finished, query execution time: {} s".format(latency))
        current_query_plan = self.get_current_query_plan()
        print("[INFO] Searched query plan: {}".format(current_query_plan))

    def select(self, node: MCTSTreeNode, use_factor=True) -> MCTSTreeNode:
        """Select best node based on UCT

        Args:
            node (MCTSTreeNode): node to perform the selection
            use_factor (bool): whether use the exploration_weight, it should be True
                               when training
        """
        exploration_weight = self.exploration_weight if use_factor else 0
        selected_node = max(
            node.children,
            key=lambda child: child.reward / child.num_visit
            + exploration_weight
            * math.sqrt(math.log(node.num_visit) / child.num_visit),
        )
        # child num_visit will be updated during back-propagation
        selected_expression, selected_action = selected_node.from_action
        send_message = dict()
        send_message["mctsAction"] = "takeAction"
        send_message["targetString"] = selected_expression
        send_message["targetAction"] = selected_action
        send_message["optimizationIsFinished"] = False
        send_message_by_socket(send_message, self.client_socket, self.verbose)

        if self.verbose >= 2:
            print(
                "[INFO] performed SELECTION, selected action: {}".format(
                    (selected_expression, selected_action)
                )
            )
        return selected_node

    def expand(self, node: MCTSTreeNode) -> MCTSTreeNode:
        """Expand not fully explorered node, and return a child node"""
        unexplorered_action = [
            key for key, val in node.action_space.items() if val == False
        ]
        selected_expression, selected_action = random.choice(unexplorered_action)
        new_state = node.state.copy()
        new_state[selected_expression] = selected_action
        matched_keys = [
            key for key in node.action_space.keys() if key[0] == selected_expression
        ]
        for target_expression, action in matched_keys:
            node.action_space[(target_expression, action)] = True
        # set as terminal node if the selected expression is None
        is_terminal = True if selected_expression == "None" else None
        if self.verbose >= 2:
            print(
                "[INFO] performed EXPAND, selected action: {}".format(
                    (selected_expression, selected_action)
                )
            )
        send_message = dict()
        send_message["mctsAction"] = "takeAction"
        send_message["targetString"] = selected_expression
        send_message["targetAction"] = selected_action
        send_message["optimizationIsFinished"] = False
        send_message_by_socket(send_message, self.client_socket, self.verbose)
        new_node = MCTSTreeNode(
            new_state,
            parent=node,
            client_socket=self.client_socket,
            is_terminal=is_terminal,
            from_action=(selected_expression, selected_action),
            verbose=self.verbose,
        )
        node.children.append(new_node)

        return new_node

    def run_plan(self):
        """Communicate with Velox to get reward with given state"""
        send_message = dict()
        send_message["mctsAction"] = "runPlan"
        send_message["optimizationIsFinished"] = False
        send_message_by_socket(send_message, self.client_socket, self.verbose)
        received_message = receive_message_by_socket(self.client_socket, self.verbose)
        latency = received_message["latency"]
        return latency

    def simulate(self, node: MCTSTreeNode) -> float:
        """Run simulation to get reward"""
        simulation_count = 0
        timer_simulation = Timer()
        timer_simulation.tic()
        while not node.is_terminal:
            if (
                simulation_count > self.max_sim_iteration_num
                or timer_simulation.toc() > self.max_sim_iteration_time
            ):
                break
            # randomly select an action
            possible_actions = list(node.action_space.keys())
            selected_expression, selected_action = random.choice(possible_actions)
            is_terminal = True if selected_expression == "None" else None
            if self.verbose >= 2:
                print(
                    "[INFO] performed SIMULATION, selected action: {}".format(
                        (selected_expression, selected_action)
                    )
                )
            new_state = node.state.copy()
            new_state[selected_expression] = selected_action
            send_message = dict()
            send_message["mctsAction"] = "takeAction"
            send_message["targetString"] = selected_expression
            send_message["targetAction"] = selected_action
            send_message["optimizationIsFinished"] = False
            send_message_by_socket(send_message, self.client_socket, self.verbose)
            node = MCTSTreeNode(
                new_state,
                parent=node,
                client_socket=self.client_socket,
                from_action=(selected_expression, selected_action),
                is_temp_node=True,
                verbose=self.verbose,
                is_terminal=is_terminal,
            )

            simulation_count += 1
        # get reward of terminal node
        reward = self.get_reward(node)
        return reward

    def get_reward(self, node: MCTSTreeNode) -> float:
        """Communicate with Velox to get reward with given state"""
        send_message = dict()
        send_message["mctsAction"] = "getCost"
        send_message["costMode"] = self.reward_mode
        send_message_by_socket(send_message, self.client_socket, self.verbose)
        received_message = receive_message_by_socket(self.client_socket, self.verbose)
        # TODO: Current reward is the latency, should be changed once integrated with cost model
        reward = -received_message["reward"]
        if self.verbose >= 2:
            print("[INFO] reward value: ", reward)
        return reward

    def back_propagate(self, node: MCTSTreeNode, reward: float):
        """Back propagate to update reward and num_visit through the path"""
        while node is not None:
            node.num_visit += 1
            node.reward += reward
            if self.verbose >= 3:
                print(
                    "[INFO] current iterated node: num_visit {}, reward {}".format(
                        node.num_visit, node.reward
                    )
                )
            node = node.parent

    def get_current_query_plan(self):
        send_message = dict()
        send_message["mctsAction"] = "getQueryPlan"
        send_message["optimizationIsFinished"] = False
        send_message_by_socket(send_message, self.client_socket, self.verbose)
        received_message = receive_message_by_socket(
            self.client_socket, verbose=self.verbose, buffsize=1024 * 1024
        )
        current_query_plan = received_message["queryPlan"]

        return current_query_plan


def send_message_by_socket(message, client_socket, verbose=0):
    client_socket.sendall(json.dumps(message).encode("utf-8"))
    # wait for ackonowledgment
    ack = client_socket.recv(1024)
    if verbose >= 3:
        print("[DEBUG] Sent message: ", message)


def receive_message_by_socket(client_socket, verbose=0, buffsize=10240):
    received_message_str = client_socket.recv(buffsize).decode("utf-8")
    json_message = json.loads(received_message_str)
    if verbose >= 3:
        print("[INFO] Received Message: ", json_message)
    return json_message


if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ("localhost", 12345)

    server_socket.bind(server_address)
    server_socket.listen(1)

    print("Python server listening on port 12345...")
    client_socket, client_address = server_socket.accept()

    print(f"Connected to C++ client: {client_address}")
    VERBOSE = (
        2  # 0: no output, 1: simplified output, 2: detailed output, 3: every output
    )
    # os.environ["mcts_debug"] = "True"
    optimization_is_finished = False
    while not optimization_is_finished:
        # Receive message
        received_json_message = receive_message_by_socket(client_socket, VERBOSE)
        # Do something
        send_message = dict()
        mctsAction = received_json_message["mctsAction"]
        if mctsAction == "start":
            initQueryPlan = received_json_message["queryPlan"]
            rootNode = MCTSTreeNode(
                state={"queryPlan": initQueryPlan},
                client_socket=client_socket,
                verbose=VERBOSE,
            )
            mcts = MCTS(
                client_socket=client_socket,
                max_iteration_num=6,
                max_sim_iteration_num=3,
                # reward_mode="online",
                reward_mode="offline",
                max_iteration_time=1000 * 3600,
                verbose=VERBOSE,
            )
            mcts.train(rootNode)
            print("[DEBUG] num_visit: ", rootNode.num_visit)
            mcts.search(rootNode)
            optimization_is_finished = True
            send_message = {"optimizationIsFinished": True, "mctsAction": "finished"}
        # Send message
        send_message_by_socket(send_message, client_socket, VERBOSE)
