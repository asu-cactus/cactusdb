from __future__ import division
import time
import math
import random
from copy import deepcopy,copy
from tracemalloc import start
import numpy as np
from math import log
import torch
import copy
from abc import ABC, abstractmethod


class Rule(ABC):
    def __init__(self, rule_type, rule_name, pattern):
        self._pattern = pattern
        self._rule_type = rule_type
        self._name = rule_name

    @property
    def rule_type(self):
        return self._rule_type

    @property
    def pattern(self):
        return self._pattern
    
    @property
    def name(self):
        return self._name
    
    def updateName(self, index):
        self._name += str(index)

    @abstractmethod
    def noRules(self):
        raise NotImplementedError
    

    @abstractmethod
    def check(self, before) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(self, before):
        raise NotImplementedError

class Pattern:
    def __init__(self, opr_type):
        self._opr_type = opr_type
        self._children = []

    def append_child(self, child):
        self._children.append(child)

    @property
    def children(self):
        return self._children

    @property
    def opr_type(self):
        return self._opr_type
    

class Mul2JoinAgg(Rule):
    def __init__(self):
        pattern = Pattern("Project")
        pattern.append_child(Pattern("source"))
        super().__init__("Mul2JoinAgg", "Mul2JoinAgg", pattern)

    def check(self, condition):
        if "mul()" in condition:
            return True
        return False

    def apply(self, before):
        after = before.replace("mul()", "result")
        return after
    
    def noRules(self):
        noRules = [JoinAgg2Mul()]
        return noRules

class JoinAgg2Mul(Rule):
    def __init__(self):
        pattern = Pattern("Project")
        pattern.append_child(Pattern("source"))
        super().__init__("JoinAgg2Mul", "JoinAgg2Mul", pattern)

    def check(self, condition):
        return False

    def apply(self, before):
        after = "mul()"
        return after
    
    def noRules(self):
        noRules = [Mul2JoinAgg()]
        return noRules

class Merge2Single(Rule):
    def __init__(self):
        pattern = Pattern("Project")
        pattern.append_child(Pattern("source"))
        super().__init__("Merge2Single", "Merge2Single", pattern)

    def check(self, condition):
        if "relu(add(mul()))" in condition:
            return True
        return False

    def apply(self, before):
        after = "torch()"
        return after
    
    def noRules(self):
        noRules = [Split2Multi()]
        return noRules

class Split2Multi(Rule):
    def __init__(self):
        pattern = Pattern("Project")
        pattern.append_child(Pattern("source"))
        super().__init__("Split2Multi", "Split2Multi", pattern)

    def check(self, condition):
        if "torch()" in condition:
            return True
        return False

    def apply(self, before):
        after = "relu(add(mul()))"
        return after

    def noRules(self):
        noRules = [Merge2Single()]
        return noRules

class rulesManager():
    def __init__(self):
        self._rules = [Mul2JoinAgg(), JoinAgg2Mul(), Merge2Single(), Split2Multi()]

    def add_rule(self, rule):
        self._rules.append(rule)
    
    def remove_rule(self, rule):
        self._rules = [r for r in self._rules if r.name != rule.name]

    @property
    def rules(self):
        return self._rules

def getReward(state):
    startTime = time.time()
    prediction = random.uniform(0, 1)
    return prediction,time.time()-startTime

def randomPolicy(node):
    t1 = 0
    while not node.isTerminal:
        startTime = time.time()
        temp = node.state.getPossibleActions()
        non_empty_indices = [index for index, s in enumerate(temp) if s]
        if non_empty_indices:
            selected_index = random.choice(non_empty_indices)
            selected_set = temp[selected_index]

        if selected_set:
            action = random.choice(list(selected_set))
            newNode = treeNode(node.state.takeAction(action, selected_index), node)
            node.children[action.name+str(selected_index)] = newNode

        if sum(len(s) for s in node.state.getPossibleActions()) == len(node.children):
            node.isFullyExpanded = True
        node = newNode
    # reward = state.getReward()
    startTime = time.time()
    reward,nntime = getReward(node.state)
    t1+= time.time()-startTime
    # print(reward)
    return node,reward,t1

getPossibleActionsTime = 0
takeActionTime = 0
class planState:
    def __init__(self, queryContext, rulesManager):

        self.order_list = [[] for _ in range(len(queryContext))]

        self.inputState1 = queryContext
        self.validRules = [rulesManager for _ in range(len(queryContext))]
        self.currentStep = 0
        self.possibleActions = []
        
    def getPossibleActions(self):
        global getPossibleActionsTime
        startTime = time.time()
        # print(self.nodes)
        if len(self.possibleActions)>0 and self.currentStep>1:
            return self.possibleActions
        
        possibleActions = []
        for j in range(len(self.inputState1)):
            state = self.inputState1[j]
            actions = set()
            for rule in self.validRules[j].rules:
                if(rule.check(state)):
                    actions.add(rule)
            possibleActions.append(actions)


            # for i in range(1, 7):
            #     if i == 1 and "mul()" in state and (not (j+0.2) in self.order_list[j]):
            #         actions.add(j+i*0.1)
            #     elif i == 2 and "result" in state and (not (j+0.1) in self.order_list[j]):
            #         actions.add(j+i*0.1)
            #     elif i == 3 and "torch()" in state and (not (j+0.4) in self.order_list[j]):
            #         actions.add(j+i*0.1)
            #     elif i == 4 and "relu(add(mul()))" in state and (not (j+0.3) in self.order_list[j]):
            #         actions.add(j+i*0.1)
            #     elif i == 5 and "cnn" in state:
            #         actions.add(j+i*0.1)
            #     elif i == 6 and "df" in state:
            #         actions.add(j+i*0.1)
            # possibleActions.append(actions)




        self.possibleActions = possibleActions
        getPossibleActionsTime += time.time()-startTime
        return possibleActions

    def getPossibleActionsInState(self, targetState):
        state = self.inputState1[targetState]
        actions = set()
        for rule in self.validRules[targetState].rules:
                if(rule.check(state)):
                    actions.add(rule)



        # for i in range(1, 7):
        #     if i == 1 and "mul()" in state and (not (targetState+0.2) in self.order_list[targetState]):
        #         actions.add(targetState+i*0.1)
        #     elif i == 2 and "result" in state and (not (targetState+0.1) in self.order_list[targetState]):
        #         actions.add(targetState+i*0.1)
        #     elif i == 3 and "torch()" in state and (not (targetState+0.4) in self.order_list[targetState]):
        #         actions.add(targetState+i*0.1)
        #     elif i == 4 and "relu(add(mul()))" in state and (not (targetState+0.3) in self.order_list[targetState]):
        #         actions.add(targetState+i*0.1)
        #     elif i == 5 and "cnn" in state:
        #         actions.add(targetState+i*0.1)
        #     elif i == 6 and "df" in state:
        #         actions.add(targetState+i*0.1)

        return actions

    def takeAction(self, action, targetState):
        global takeActionTime
        startTime = time.time()
        newState = copy.deepcopy(self)

        newState.order_list[targetState].append(action)
        newState.currentStep = self.currentStep + 1

        newState.inputState1[targetState] = action.apply(newState.inputState1[targetState])
        for rule in action.noRules():
            newState.validRules[targetState].remove_rule(rule)

        # if action == (targetState+0.1):
        #     newState.inputState1[targetState] = newState.inputState1[targetState].replace("mul()", "result")
        # elif action == (targetState+0.2):
        #     newState.inputState1[targetState] = "mul()"
        # elif action == (targetState+0.3):
        #     newState.inputState1[targetState] = "relu(add(mul()))"
        # elif action == (targetState+0.4):
        #     newState.inputState1[targetState] = "torch()"
        # elif action == (targetState+0.5):
        #     newState.inputState1[targetState] = ""
        # elif action == (targetState+0.6):
        #     newState.inputState1[targetState] = ""

        newState.possibleActions[targetState] = newState.getPossibleActionsInState(targetState)
        takeActionTime += time.time()-startTime
        return newState

    def isTerminal(self):
        all_empty = all(len(actions) == 0 for actions in self.possibleActions)
        if all_empty and self.currentStep > 0:
            return True
        return False


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}

class mcts():
    def __init__(self, iterationLimit=None, explorationConstant=1 / math.sqrt(16),
                 rolloutPolicy=randomPolicy):
        if iterationLimit == None:
            raise ValueError("Must have either a time limit or an iteration limit")
        # number of iterations of the search
        if iterationLimit < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.searchLimit = iterationLimit
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.nntime = 0
        self.nntime_no_feature =0
        global getPossibleActionsTime
        getPossibleActionsTime = 0
        global takeActionTime
        takeActionTime = 0

    def search(self, initialState):
        self.root = treeNode(initialState, None)
        for i in range(self.searchLimit):
            self.executeRound()

        # bestChild = self.getBestChild(self.root, 0)
        # return self.getAction(self.root, bestChild)
    def continueSearch(self):
        for i in range(self.searchLimit):
            self.executeRound()

    def executeRound(self):
        node = self.selectNode(self.root)
        # newState = deepcopy(node.state)
        startTime = time.time()
        node,reward,nntime_no_feature = self.rollout(node)
        self.nntime += time.time()-startTime
        self.nntime_no_feature += nntime_no_feature
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        totalNum = sum(len(s) for s in actions)
        for i in range(len(actions)):
            for action in actions[i]:
                if (action.name+str(i)) not in node.children:
                    newNode = treeNode(node.state.takeAction(action, i), node)
                    node.children[action.name+str(i)] = newNode
                    if totalNum == len(node.children):
                        node.isFullyExpanded = True
                # if newNode.isTerminal:
                #     print(newNode)
                    return newNode
        print(len(actions),len(node.children))
        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        # print(reward)
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action


if __name__ == "__main__":


    # Define your PlanState initialization data
    # totalNumberOfTables = 10  # Replace with your actual data
    # numberOfTables = 3  # Replace with your actual data
    # queryEncode = [1.0, 2.0, 3.0]  # Replace with your actual data
    # all_joins = [(0, 1), (1, 2), (2, 3)]  # Replace with your actual data
    # joins_with_predicate = [(0, 2), (1, 3)]  # Replace with your actual data
    # nodes = {0, 1, 2, 3}  # Replace with your actual data

    # initialState = planState(totalNumberOfTables, numberOfTables, queryEncode, all_joins, joins_with_predicate, nodes)

    # # Create an MCTS instance with parameters
    # mcts_instance = mcts(iterationLimit=1000, explorationConstant=1 / math.sqrt(16), rolloutPolicy=randomPolicy)

    # # Perform the MCTS search
    # mcts_instance.search(initialState)

    # # Retrieve the best action from the MCTS
    # best_action = mcts_instance.getAction(mcts_instance.root, mcts_instance.getBestChild(mcts_instance.root, 0))

    # # Print or use the best action as needed
    # print("Best Action:", best_action)
    queryContext = ["relu(add(mul()))", "mul()"]
    rulesManager = rulesManager()
    initialState = planState(queryContext, rulesManager)
    mcts_instance = mcts(iterationLimit=10, explorationConstant=1 / math.sqrt(16), rolloutPolicy=randomPolicy)
    mcts_instance.search(initialState)
    node = mcts_instance.root
    actions = []
    while not node.isTerminal:
        bestChild = mcts_instance.getBestChild(node, 1 / math.sqrt(16))
        action =  mcts_instance.getAction(node, bestChild)
        actions.append(action)
        node = bestChild
    print(actions)
    # best_action = mcts_instance.getAction(mcts_instance.root, mcts_instance.getBestChild(mcts_instance.root, 0))




    

    
    
