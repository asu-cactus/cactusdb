/*
 * Copyright (c) 2025 ASU Cactus Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file
 * @brief Implementation of a decision tree for machine learning predictions.
 */

#pragma once

#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/BaseFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml {

#define MAX_NUM_NODES_PER_TREE 512

class Tree;
typedef std::shared_ptr<Tree> TreePtr;

/**
 * @struct Node
 * @brief Represents a node in a decision tree.
 */
typedef struct {
  union {
    float threshold;  ///< Threshold value for non-leaf nodes.
    float leafValue;  ///< Value for leaf nodes.
  };
  int indexID;          ///< Index of the feature to compare.
  int leftChild;        ///< Index of the left child node.
  int rightChild;       ///< Index of the right child node.
  bool isLeaf;          ///< Whether the node is a leaf.
  bool isMissTrackLeft; ///< Whether to track left if feature value is missing.
} Node;

/**
 * @class Tree
 * @brief Represents a decision tree used for predictions.
 */
class Tree {
 public:
  Node tree[MAX_NUM_NODES_PER_TREE]; ///< Array of tree nodes.
  int treeId; ///< ID of the tree in the forest.

  /**
   * @brief Default constructor.
   */
  Tree() {}

  /**
   * @brief Constructor that initializes the tree from an xgboost model dump.
   * @param id The ID of the tree.
   * @param treePath Path to the file containing the tree structure.
   */
  Tree(int id, std::string treePath) : treeId{id} {
    this->constructTreeFromPath(treePath, this->tree);
  }

  /**
   * @brief Constructs a tree from a file dumped from an xgboost model.
   * @param treePathIn Path to the file containing the tree structure.
   * @param tree Pointer to the array of nodes to be populated.
   */
  static void constructTreeFromPath(std::string treePathIn, Node* tree) {
    std::vector<std::string> relationships;
    std::vector<std::string> innerNodes;
    std::vector<std::string> leafNodes;
    constructTreeFromPathHelper(
        treePathIn, relationships, innerNodes, leafNodes);
    processInnerNodes(innerNodes, tree);
    processLeafNodes(leafNodes, tree);
    processRelationships(relationships, tree);
  }

  /**
   * @brief Parses the file and categorizes the lines into relationships, inner nodes, and leaf nodes.
   * @param treePathIn Path to the file containing the tree structure.
   * @param relationships Vector to store relationship lines.
   * @param innerNodes Vector to store inner node lines.
   * @param leafNodes Vector to store leaf node lines.
   */
  static void constructTreeFromPathHelper(
      std::string treePathIn,
      std::vector<std::string>& relationships,
      std::vector<std::string>& innerNodes,
      std::vector<std::string>& leafNodes) {
    std::ifstream inputFile;
    inputFile.open(treePathIn.data());
    assert(inputFile.is_open());

    std::string line;
    while (getline(inputFile, line)) {
      if ((line.size() == 0) || (line.find("graph") != std::string::npos) ||
          (line.find("}") != std::string::npos)) {
      } else {
        if (line.find("->") != std::string::npos) {
          relationships.push_back(line);
        } else if (line.find("leaf") != std::string::npos) {
          leafNodes.push_back(line);
        } else if (line.find("label") != std::string::npos) {
          innerNodes.push_back(line);
        } else {
          // skip the case of empty line, somehow it won't be captured by the
          // first condition
        }
      }
    }

    inputFile.close();
  }

  /**
   * @brief Parses the lines corresponding to tree inner nodes.
   * @param innerNodes Vector of strings representing inner nodes.
   * @param tree Pointer to the array of nodes to be populated.
   */
  static void processInnerNodes(
      std::vector<std::string>& innerNodes,
      Node* tree) {
    int findStartPosition;
    int findMidPosition;
    int findEndPosition;

    for (int i = 0; i < innerNodes.size(); ++i) {
      const std::string& currentLine = innerNodes[i];
      int nodeID;
      int indexID;
      float threshold;

      if ((findEndPosition = currentLine.find("[ label")) !=
          std::string::npos) {
        nodeID = std::stoi(currentLine.substr(4, findEndPosition - 1 - 4));
      } else {
        LOG(ERROR) << "[ERROR] Error in extracting inner node nodeID\n";
        exit(1);
      }

      if ((findStartPosition = currentLine.find("f")) != std::string::npos &&
          (findEndPosition = currentLine.find("<")) != std::string::npos) {
        indexID = std::stoi(currentLine.substr(
            findStartPosition + 1, findEndPosition - findStartPosition - 1));
      } else {
        LOG(ERROR) << "[Error] Error in extracting inner node indexID\n";
        exit(1);
      }

      if ((findStartPosition = currentLine.find("<")) != std::string::npos &&
          (findEndPosition = currentLine.find("\" ]")) != std::string::npos) {
        threshold = std::stod(currentLine.substr(
            findStartPosition + 1, findEndPosition - findStartPosition - 1));
      } else {
        LOG(ERROR) << "[ERROR] Error in extracting inner node threshold\n";
        exit(1);
      }
      tree[nodeID].isMissTrackLeft = false; // XGBoost default is noMissing/right

      tree[nodeID].indexID = indexID;
      tree[nodeID].isLeaf = false;
      tree[nodeID].leftChild = -1;
      tree[nodeID].rightChild = -1;
      tree[nodeID].threshold = threshold;
    }
  }

  /**
   * @brief Parses the lines corresponding to tree leaf nodes.
   * @param leafNodes Vector of strings representing leaf nodes.
   * @param tree Pointer to the array of nodes to be populated.
   */
  static void processLeafNodes(
      std::vector<std::string>& leafNodes,
      Node* tree) {
    int findStartPosition;
    int findMidPosition;
    int findEndPosition;

    for (int i = 0; i < leafNodes.size(); ++i) {
      const std::string& currentLine = leafNodes[i];
      int nodeID;
      float leafValue = -1.0f;

      if ((findEndPosition = currentLine.find("[")) != std::string::npos) {
        nodeID = std::stoi(currentLine.substr(4, findEndPosition - 1 - 4));
      } else {
        LOG(ERROR) << "[ERROR] Error in extracting leaf node nodeID\n";
        exit(1);
      }

      if ((findStartPosition = currentLine.find("leaf=")) !=
              std::string::npos &&
          (findEndPosition = currentLine.find("\" ]")) != std::string::npos) {
        leafValue = std::stod(currentLine.substr(
            findStartPosition + 5,
            findEndPosition - 3 - findStartPosition - 5));
      } else {
        std::cout << "Error in extracting leaf node leafValue\n";
        exit(1);
      }

      tree[nodeID].indexID = -1;
      tree[nodeID].isLeaf = true;
      tree[nodeID].leftChild = -1;
      tree[nodeID].rightChild = -1;
      tree[nodeID].leafValue = leafValue;
      tree[nodeID].isMissTrackLeft = true; // Doesn't matter to leave nodes
    }
  }

  /**
   * @brief Parses the lines corresponding to tree relationships.
   * @param relationships Vector of strings representing relationships.
   * @param tree Pointer to the array of nodes to be populated.
   */
  static void processRelationships(
      std::vector<std::string>& relationships,
      Node* tree) {
    int findStartPosition;
    int findMidPosition;
    int findEndPosition;

    for (int i = 0; i < relationships.size(); ++i) {
      const std::string& currentLine = relationships[i];
      int parentNodeID;
      int childNodeID;

      if ((findMidPosition = currentLine.find("->")) != std::string::npos) {
        parentNodeID =
            std::stoi(currentLine.substr(4, findMidPosition - 1 - 4));
      } else {
        std::cout << "Error in extracting parentNodeID\n";
        exit(1);
      }

      if ((findEndPosition = currentLine.find("[")) != std::string::npos) {
        childNodeID = std::stoi(currentLine.substr(
            findMidPosition + 3, findEndPosition - 1 - findMidPosition - 3));
      } else {
        std::cout << "Error in extracting childNodeID\n";
        exit(1);
      }

      if (currentLine.find("yes, missing") != std::string::npos) {
        tree[parentNodeID].isMissTrackLeft =
            true; // in processInnerNodes(), default value is set to no/right
      }

      if (tree[parentNodeID].leftChild == -1) {
        tree[parentNodeID].leftChild = childNodeID;
      } else if (tree[parentNodeID].rightChild == -1) {
        tree[parentNodeID].rightChild = childNodeID;
      } else {
        std::cout
            << "Error in parsing trees: children nodes were updated again: "
            << parentNodeID << "->" << childNodeID << std::endl;
      }
    }
  }

  /**
   * @brief Predicts the output for a single input.
   * @param input Pointer to the input feature values.
   * @param curBase Base index for the current input.
   * @return Predicted value.
   */
  inline float predictSingle(float* input, int curBase) {
    int curIndex = 0;
    while (!tree[curIndex].isLeaf) {
      const float featureValue = input[curBase + tree[curIndex].indexID];
      curIndex = featureValue < tree[curIndex].threshold
          ? tree[curIndex].leftChild
          : tree[curIndex].rightChild;
    }
    float result = (float)(tree[curIndex].leafValue);
    return result;
  }

  /**
   * @brief Predicts the output for multiple inputs.
   * @param input Vector of input feature values.
   * @param resultVector Vector to store the predicted values.
   * @param numInputs Number of inputs.
   * @param numFeatures Number of features per input.
   */
  inline void predict(
      VectorPtr& input,
      std::vector<float>& resultVector,
      int numInputs,
      int numFeatures) {
    auto inputFeatures = input->as<ArrayVector>()->elements();
    float* inputValues = inputFeatures->values()->asMutable<float>();
    float* outData = resultVector.data();

    for (int rowIndex = 0; rowIndex < numInputs; rowIndex++) {
      int curIndex = 0;
      int curBase = rowIndex * numFeatures;
      while (!tree[curIndex].isLeaf) {
        const float featureValue =
            inputValues[curBase + tree[curIndex].indexID];
        curIndex = featureValue < tree[curIndex].threshold
            ? tree[curIndex].leftChild
            : tree[curIndex].rightChild;
      }
      outData[rowIndex] = (float)(tree[curIndex].leafValue);
    }
  }

  /**
   * @brief Predicts the output for multiple inputs, handling missing values.
   * @param input Vector of input feature values.
   * @param resultVector Vector to store the predicted values.
   * @param numInputs Number of inputs.
   * @param numFeatures Number of features per input.
   */
  inline void predictMissing(
      VectorPtr& input,
      std::vector<float>& resultVector,
      int numInputs,
      int numFeatures) {
    auto inputFeatures = input->as<ArrayVector>()->elements();
    float* inputValues = inputFeatures->values()->asMutable<float>();
    float* outData = resultVector.data();

    for (int rowIndex = 0; rowIndex < numInputs; rowIndex++) {
      int curIndex = 0;
      int curBase = rowIndex * numFeatures;
      while (!tree[curIndex].isLeaf) {
        const float featureValue =
            inputValues[curBase + tree[curIndex].indexID];
        if (std::isnan(featureValue)) {
          curIndex = tree[curIndex].isMissTrackLeft ? tree[curIndex].leftChild
                                                    : tree[curIndex].rightChild;

        } else {
          curIndex = featureValue < tree[curIndex].threshold
              ? tree[curIndex].leftChild
              : tree[curIndex].rightChild;
        }
      }
      outData[rowIndex] = (float)(tree[curIndex].leafValue);
    }
  }
};

/**
 * @class TreePrediction
 * @brief Implements a machine learning function for tree-based predictions.
 */
class TreePrediction : public MLFunction {
 public:
  /**
   * @brief Constructor for TreePrediction.
   * @param treeId ID of the tree.
   * @param treePath Path to the file containing the tree structure.
   * @param numFeatures Number of features per input.
   * @param hasMissing Whether the input data contains missing values.
   */
  TreePrediction(
      int treeId,
      std::string treePath,
      int numFeatures,
      bool hasMissing) {
    this->tree = std::make_shared<Tree>(treeId, treePath);
    this->numFeatures = numFeatures;
    this->hasMissing = hasMissing;
  }

  /**
   * @brief Applies the tree prediction function to the input data.
   * @param rows Selectivity vector indicating which rows to process.
   * @param args Vector of input arguments.
   * @param type Type of the output vector.
   * @param context Evaluation context.
   * @param output Output vector to store the results.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int numInputs = rows.size();
    std::vector<float> resultVector(numInputs);

    if (hasMissing) {
      this->tree->predictMissing(
          args[0], resultVector, numInputs, this->numFeatures);
    } else {
      this->tree->predict(args[0], resultVector, numInputs, this->numFeatures);
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<float>(resultVector, REAL());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("REAL")
                .build()};
  }

  /**
   * @brief Returns the tensor associated with the function.
   * @return Pointer to the tensor.
   */
  float* getTensor() const override {
    return new float[0]; // will this lead to memory leak?
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "tree_predict";
  }

  /**
   * @brief Returns the function name.
   * @return Function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Estimates the cost of the function.
   * @param inputDims Dimensions of the input.
   * @return Cost estimate.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO
    return CostEstimate(1, inputDims[0], dims[1]);
  }

 private:
  TreePtr tree; ///< Pointer to the decision tree.
  int numFeatures; ///< Number of features per input.
  bool hasMissing; ///< Whether the input data contains missing values.
};

} // namespace ml
