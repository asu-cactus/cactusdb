#include <iostream>
#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <ctime>
#include <cstring>
#include <functional>


class planState {
public:
    std::vector<std::vector<double>> order_list;
    std::vector<std::string> inputState1;
    int currentStep;
    std::vector<std::unordered_set<double>> possibleActions;

    planState(std::vector<std::string> queryContext)
        : inputState1(queryContext), currentStep(0), possibleActions(queryContext.size()) {
        order_list.resize(queryContext.size());
    }

    planState(const planState& other) {
        // Copy primitive types directly
        currentStep = other.currentStep;

        // Deep copy vector of vectors
        order_list = other.order_list;

        // Deep copy vector of strings
        inputState1 = other.inputState1;

        // Deep copy vector of unordered sets
        possibleActions.clear();
    }

    std::vector<std::unordered_set<double>> getPossibleActions() {
        // Assuming time and other variables are defined globally
        if (possibleActions.size() > 0 && currentStep > 1) {
            return possibleActions;
        }

        for (size_t j = 0; j < inputState1.size(); ++j) {
            const std::string& state = inputState1[j];
            std::unordered_set<double> actions;
            for (int i = 1; i <= 6; ++i) {
                double actionValue = j + i * 0.1;
                if (i == 1 && state.find("mul()") != std::string::npos && std::find(order_list[j].begin(), order_list[j].end(), j + 0.2) == order_list[j].end()) {
                    actions.insert(actionValue);
                } else if (i == 2 && state.find("result") != std::string::npos && std::find(order_list[j].begin(), order_list[j].end(), j + 0.1) == order_list[j].end()) {
                    actions.insert(actionValue);
                } else if (i == 3 && state.find("torch()") != std::string::npos && std::find(order_list[j].begin(), order_list[j].end(), j + 0.4) == order_list[j].end()) {
                    actions.insert(actionValue);
                } else if (i == 4 && state.find("relu(add(mul()))") != std::string::npos && std::find(order_list[j].begin(), order_list[j].end(), j + 0.3) == order_list[j].end()) {
                    actions.insert(actionValue);
                } else if (i == 5 && state.find("cnn") != std::string::npos) {
                    actions.insert(actionValue);
                } else if (i == 6 && state.find("df") != std::string::npos) {
                    actions.insert(actionValue);
                }
            }
            possibleActions[j] = actions;
        }

        return possibleActions;
    }

    std::unordered_set<double> getPossibleActionsInState(size_t targetState) {
        const std::string& state = inputState1[targetState];
        std::unordered_set<double> actions;
        for (int i = 1; i <= 6; ++i) {
            double actionValue = targetState + i * 0.1;
            if (i == 1 && state.find("mul()") != std::string::npos && std::find(order_list[targetState].begin(), order_list[targetState].end(), targetState + 0.2) == order_list[targetState].end()) {
                actions.insert(actionValue);
            } else if (i == 2 && state.find("result") != std::string::npos && std::find(order_list[targetState].begin(), order_list[targetState].end(), targetState + 0.1) == order_list[targetState].end()) {
                actions.insert(actionValue);
            } else if (i == 3 && state.find("torch()") != std::string::npos && std::find(order_list[targetState].begin(), order_list[targetState].end(), targetState + 0.4) == order_list[targetState].end()) {
                actions.insert(actionValue);
            } else if (i == 4 && state.find("relu(add(mul()))") != std::string::npos && std::find(order_list[targetState].begin(), order_list[targetState].end(), targetState + 0.3) == order_list[targetState].end()) {
                actions.insert(actionValue);
            } else if (i == 5 && state.find("cnn") != std::string::npos) {
                actions.insert(actionValue);
            } else if (i == 6 && state.find("df") != std::string::npos) {
                actions.insert(actionValue);
            }
        }
        return actions;
    }

    planState takeAction(double action, size_t targetState) {
        // Assuming time and other variables are defined globally
        planState newState(*this);
        newState.order_list[targetState].push_back(action);
        newState.currentStep = currentStep + 1;

        if (action == (targetState + 0.1)) {
            size_t pos = newState.inputState1[targetState].find("mul()");
            if (pos != std::string::npos) {
                newState.inputState1[targetState].replace(pos, 4, "result");
            }
        } else if (action == (targetState + 0.2)) {
            newState.inputState1[targetState] = "mul()";
        } else if (action == (targetState + 0.3)) {
            newState.inputState1[targetState] = "relu(add(mul()))";
        } else if (action == (targetState + 0.4)) {
            newState.inputState1[targetState] = "torch()";
        } else if (action == (targetState + 0.5)) {
            newState.inputState1[targetState] = "";
        } else if (action == (targetState + 0.6)) {
            newState.inputState1[targetState] = "";
        }

        newState.possibleActions[targetState] = newState.getPossibleActionsInState(targetState);
        return newState;
    }

    bool isTerminal() {
        bool all_empty = std::all_of(possibleActions.begin(), possibleActions.end(),
                                     [](const std::unordered_set<double>& actions) {
                                         return actions.empty();
                                     });

        return all_empty && currentStep > 0;
    }
};


class treeNode {
public:
    planState state;
    bool isTerminal;
    bool isFullyExpanded;
    treeNode* parent;
    int numVisits;
    double totalReward;
    double currentReward;
    std::unordered_map<double, treeNode*> children;

    treeNode(planState state, treeNode* parent) : state(state), parent(parent), numVisits(0), totalReward(0) {
        isTerminal = state.isTerminal();
        isFullyExpanded = isTerminal;
    }
};

double getReward(planState state) {
    // Get the current time in milliseconds
    auto startTime = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(startTime).count();

    // Seed the random number generator with the current time
    std::mt19937 generator(static_cast<unsigned>(millis));

    // Generate a random prediction between 0 and 1
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double prediction = distribution(generator);

    // Get the current time again and calculate the elapsed time
    auto endTime = std::chrono::high_resolution_clock::now().time_since_epoch();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0;

    // Return the random prediction and elapsed time
    return prediction;
}

treeNode randomPolicy(treeNode node) {
    double t1 = 0.0;
    while (!node.isTerminal) {
        auto startTime = std::chrono::high_resolution_clock::now();
        auto temp = node.state.getPossibleActions();
        std::vector<size_t> nonEmptyIndices;

        for (size_t index = 0; index < temp.size(); ++index) {
            if (!temp[index].empty()) {
                nonEmptyIndices.push_back(index);
            }
        }

        if (!nonEmptyIndices.empty()) {
            size_t selected_index = nonEmptyIndices[rand() % nonEmptyIndices.size()];
            const auto& selected_set = temp[selected_index];

            if (!selected_set.empty()) {
                double action = *std::next(selected_set.begin(), rand() % selected_set.size());
                treeNode* newNode = new treeNode(node.state.takeAction(action, selected_index), &node);
                node.children[action] = newNode;

                if (std::accumulate(temp.begin(), temp.end(), 0, [](size_t sum, const std::unordered_set<double>& s) { return sum + s.size(); }) == node.children.size()) {
                    node.isFullyExpanded = true;
                }

                node = *newNode;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        t1 += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    double reward = getReward(node.state);
    auto endTime = std::chrono::high_resolution_clock::now();
    t1 += std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0;
    node.currentReward = reward;
    // Perform any additional actions with the results if needed

    return node;
}

// Function to get a random reward


class mcts {
public:
    int searchLimit;
    double explorationConstant;
    // randomPolicy rollout;
    double nntime;
    double nntime_no_feature;
    treeNode* root;

    treeNode (*rollout)(treeNode) = nullptr;

    mcts(int iterationLimit, double explorationConstant, treeNode (*randomPolicy)(treeNode))
        : searchLimit(iterationLimit), explorationConstant(explorationConstant), rollout(randomPolicy), nntime(0), nntime_no_feature(0), root(nullptr) {
        if (iterationLimit < 1) {
            throw std::invalid_argument("Iteration limit must be greater than one");
        }
    }

    void search(planState initialState) {
        root = new treeNode(initialState, nullptr);

        for (int i = 0; i < searchLimit; ++i) {
            executeRound();
        }
    }

    void continueSearch() {
        for (int i = 0; i < searchLimit; ++i) {
            executeRound();
        }
    }

    void executeRound() {
        treeNode* node = selectNode(root);
        auto startTime = std::chrono::high_resolution_clock::now();
        auto result = rollout(*node);
        // nntime += std::get<2>(result);
        // nntime_no_feature += std::get<2>(result);
        backpropagate(&result, result.currentReward);
    }

    treeNode* selectNode(treeNode* node) {
        while (!node->isTerminal) {
            if (node->isFullyExpanded) {
                node = getBestChild(node, explorationConstant);
            } else {
                return expand(node);
            }
        }
        return node;
    }

    treeNode* expand(treeNode* node) {
        auto actions = node->state.getPossibleActions();
        size_t totalNum = std::accumulate(actions.begin(), actions.end(), 0, [](size_t sum, const std::unordered_set<double>& s) { return sum + s.size(); });

        for (size_t i = 0; i < actions.size(); ++i) {
            for (const auto& action : actions[i]) {
                if (node->children.find(action) == node->children.end()) {
                    treeNode* newNode = new treeNode(node->state.takeAction(action, i), node);
                    node->children[action] = newNode;

                    if (totalNum == node->children.size()) {
                        node->isFullyExpanded = true;
                    }
                    return newNode;
                }
            }
        }

        throw std::runtime_error("Should never reach here");
    }

    void backpropagate(treeNode* node, double reward) {
        while (node != nullptr) {
            node->numVisits += 1;
            node->totalReward += reward;
            node = node->parent;
        }
    }

    treeNode* getBestChild(treeNode* node, double explorationValue) {
        double bestValue = std::numeric_limits<double>::lowest();
        std::vector<treeNode*> bestNodes;

        for (const auto& child : node->children) {
            treeNode* childNode = child.second;
            double nodeValue = childNode->totalReward / childNode->numVisits + explorationValue * std::sqrt(2 * std::log(node->numVisits) / childNode->numVisits);

            if (nodeValue > bestValue) {
                bestValue = nodeValue;
                bestNodes = {childNode};
            } else if (nodeValue == bestValue) {
                bestNodes.push_back(childNode);
            }
        }

        return bestNodes[rand() % bestNodes.size()];
    }

    double getAction(treeNode* root, treeNode* bestChild) {
        for (const auto& child : root->children) {
            if (child.second == bestChild) {
                return child.first;
            }
        }

        throw std::runtime_error("Should never reach here");
    }
};

int main() {
    // Example usage of the mcts class
    std::vector<std::string> queryContext = {"relu(add(mul()))", "mul()"};
    planState initialState(queryContext);
    mcts mctsInstance(10, 1 / std::sqrt(16), randomPolicy);
    mctsInstance.search(initialState);
    treeNode* node = mctsInstance.root;
    std::vector<double> actions;

    while (!node->isTerminal) {
        treeNode* bestChild = mctsInstance.getBestChild(node, 1 / std::sqrt(16));
        double action = mctsInstance.getAction(node, bestChild);
        actions.push_back(action);
        node = bestChild;
    }

    // Print or use the actions as needed
    std::cout << "Best Actions: ";
    for (const auto& action : actions) {
        std::cout << action << " ";
    }
    std::cout << std::endl;

    return 0;
}