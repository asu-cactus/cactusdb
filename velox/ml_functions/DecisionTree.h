/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include "velox/ml_functions/functions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace std;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml {

#define MAX_NUM_NODES_PER_TREE 512
class Tree;
typedef std::shared_ptr<Tree> TreePtr;

// Definition of Node (tree node) in our decision forest implementation

typedef struct {
  // returnClass will be the vaule to compare while this is not a leaf node
  union {
    float threshold;
    float leafValue;
  };
  int indexID;
  int leftChild;
  int rightChild;
  bool isLeaf;
  // When feature value is missing, whether track/traverseTo the left node
  bool isMissTrackLeft;
} Node;

// Implementation of Tree
class Tree {
 public:
  // An array of tree nodes
  Node tree[MAX_NUM_NODES_PER_TREE];

  // The ID of the tree in the forest
  int treeId;

  // Default constructor
  Tree() {}

  // Constructor by parsing xgboost model dump
  Tree(int id, std::string treePath) : treeId{id} {
    this->constructTreeFromPath(treePath, this->tree);
  }

  // Construct a tree from a file dumped from an xgboost model
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

  // Parsing the file and categorize the lines from the file
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
        if (line.find("->") != string::npos) {
          relationships.push_back(line);
        } else if (line.find("leaf") != string::npos) {
          leafNodes.push_back(line);
        } else if (line.find("label") != string::npos) {
          innerNodes.push_back(line);
        } else {
          // skip the case of empty line, somehow it won't be captured by the
          // first condition
        }
      }
    }

    inputFile.close();
  }

  // Parsing the lines corresponding to tree inner nodes
  static void processInnerNodes(
      std::vector<std::string>& innerNodes,
      Node* tree) {
    int findStartPosition;
    int findMidPosition;
    int findEndPosition;

    // Constructing inner nodes
    for (int i = 0; i < innerNodes.size(); ++i) {
      const string& currentLine = innerNodes[i];
      int nodeID;
      int indexID;
      float threshold;

      // To get nodeID
      if ((findEndPosition = currentLine.find("[ label")) != string::npos) {
        nodeID = std::stoi(currentLine.substr(4, findEndPosition - 1 - 4));
      } else {
        LOG(ERROR) << "[ERROR] Error in extracting inner node nodeID\n";
        exit(1);
      }

      // To get nodeIndex
      if ((findStartPosition = currentLine.find("f")) != string::npos &&
          (findEndPosition = currentLine.find("<")) != string::npos) {
        indexID = std::stoi(currentLine.substr(
            findStartPosition + 1, findEndPosition - findStartPosition - 1));
      } else {
        LOG(ERROR) << "[Error] Error in extracting inner node indexID\n";
        exit(1);
      }

      // To get threshold
      if ((findStartPosition = currentLine.find("<")) != string::npos &&
          (findEndPosition = currentLine.find("\" ]")) != string::npos) {
        threshold = std::stod(currentLine.substr(
            findStartPosition + 1, findEndPosition - findStartPosition - 1));
      } else {
        LOG(ERROR) << "[ERROR] Error in extracting inner node threshold\n";
        exit(1);
      }
      tree[nodeID].isMissTrackLeft =
          false; // XGBoost default is noMissing/right

      tree[nodeID].indexID = indexID;
      tree[nodeID].isLeaf = false;
      tree[nodeID].leftChild = -1;
      tree[nodeID].rightChild = -1;
      tree[nodeID].threshold = threshold;
    }
  }

  static void processLeafNodes(
      std::vector<std::string>& leafNodes,
      Node* tree) {
    int findStartPosition;
    int findMidPosition;
    int findEndPosition;

    // Constructing leaf nodes
    for (int i = 0; i < leafNodes.size(); ++i) {
      // Construct leaf nodes
      const string& currentLine = leafNodes[i];
      int nodeID;
      float leafValue = -1.0f;

      if ((findEndPosition = currentLine.find("[")) != string::npos) {
        nodeID = std::stoi(currentLine.substr(4, findEndPosition - 1 - 4));
      } else {
        LOG(ERROR) << "[ERROR] Error in extracting leaf node nodeID\n";
        exit(1);
      }

      // Output Class of XGBoost always a Double/Float. ProbabilityValue for
      // Classification, ResultValue for Regression
      if ((findStartPosition = currentLine.find("leaf=")) != string::npos &&
          (findEndPosition = currentLine.find("\" ]")) != string::npos) {
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

  static void processRelationships(
      std::vector<std::string>& relationships,
      Node* tree) {
    int findStartPosition;
    int findMidPosition;
    int findEndPosition;

    // Constructing edges
    for (int i = 0; i < relationships.size(); ++i) {
      // Construct Directed Edges between Nodes
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

  inline float predictSingle(float* input, int curBase) {
    int curIndex = 0;
    while (!tree[curIndex].isLeaf) {
      const float featureValue = input[curBase + tree[curIndex].indexID];
      curIndex = featureValue < tree[curIndex].threshold
          ? tree[curIndex].leftChild
          : tree[curIndex].rightChild;
    }
    float result = (float)(tree[curIndex].leafValue);
    //std::cout << curBase << ":" << this->treeId << "=" << result << std::endl;
    return result;
  }

  inline void predict(
      VectorPtr& input,
      std::vector<float>& resultVector,
      int numInputs,
      int numFeatures) {
    // get the input features
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

  inline void predictMissing(
      VectorPtr& input,
      std::vector<float>& resultVector,
      int numInputs,
      int numFeatures) {
    // get the input features
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

class TreePrediction : public MLFunction {
 public:
  TreePrediction(
      int treeId,
      std::string treePath,
      int numFeatures,
      bool hasMissing) {
    this->tree = std::make_shared<Tree>(treeId, treePath);
    this->numFeatures = numFeatures;
    this->hasMissing = hasMissing;
  }

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

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("REAL")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  float* getTensor() const override {
    return new float[0]; // will this lead to memory leak?
  }

  static std::string getName() {
    return "tree_predict";
  }

  std::string getFuncName() {
    return getName();
  };

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO
    return CostEstimate(1, inputDims[0], dims[1]);
  }

 private:
  TreePtr tree;
  int numFeatures;
  bool hasMissing;
};

} // namespace 
