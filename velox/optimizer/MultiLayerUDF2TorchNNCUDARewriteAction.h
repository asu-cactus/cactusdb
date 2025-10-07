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

#include <iostream>
#include <memory>
#include <regex>

#include "RewriteAction.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/core/PlanNode.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/optimizer/Helper.h"

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;
using namespace optimization;

namespace optimization {

class MultiLayerUDF2TorchNNCUDARewriteAction : public RewriteAction {
 public:
  MultiLayerUDF2TorchNNCUDARewriteAction() {}

  void clearVectors() {
    dims.clear();
    weights.clear();
    bias.clear();
  }

  /**
   * @brief A function to apply a rule for rewriting the logical plan.
   *
   * @param curNode A pointer to the current plan node, usually point to the
   * last node of logical plan.
   * @param prevNode A pointer to the previous plan node, usually point to the
   * previous node before current node.
   * @param maker A pointer to the VectorMaker, which is a helper class used to
   * build the data source vector.
   * @param planBuilder A pointer to the planBuilder, which is a helper class
   * used to build the logical plan.
   * @param pool_ A pointer to the memory pool, which is used to build the
   * logical plan.
   * @param planNodeIdGenerator A pointer to the planNodeIdGenerator, which is
   * used to track the ID of the plan Node.
   * @param targets A vector for multiple strings, representing the target UDF
   * name that can apply this rewritten rule.
   * @param cataLog Reference to a CataLog object to store metadata and
   * information.
   *
   * @return A boolean value indicating whether the rewrite was successful.
   */
  bool apply(
      std::shared_ptr<const core::PlanNode> curNode,
      std::shared_ptr<const core::PlanNode> prevNode,
      VectorMaker& maker,
      PlanBuilder& planBuilder,
      std::shared_ptr<memory::MemoryPool> pool_,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      std::vector<std::string> targets,
      CataLog& cataLog) override {
    clearVectors();
    bool transformationApplied = false;
    std::set<std::string> finalProjectExprSets;
    // Iterate over each target in the targets container
    for (auto target : targets) {
      // Start from the current node
      if (curNode) {
        // Get the name of node
        std::string_view nodeName = curNode->name();
        // We first search project node
        if (nodeName == "Project") {
          // Cast node as project node
          if (auto myProjectNode =
                  std::dynamic_pointer_cast<const ProjectNode>(curNode)) {
            // Get projections in project node
            const std::vector<TypedExprPtr>& projections =
                myProjectNode->projections();
            // Get the names of projections
            const std::vector<std::string>& projectionsNames =
                myProjectNode->names();
            int numProjections = projections.size();
            // Check if the number of projections is equal to the number of
            // projection names
            assert(numProjections == projectionsNames.size());
            // Flag indicates whether the target UDF is found
            bool findRewriteTarget = false;
            // Iterate each projection
            for (int exprIdx = 0; exprIdx < numProjections; exprIdx++) {
              auto expression = projections[exprIdx];
              // Get the string of expression
              std::string exprStr = expression->toString();
              // Check if target exist in the expression
              if (exprStr.find(target) != std::string::npos) {
                // There is one limitation here: the current rewrite can only
                // rewrite whole single expression or partial expression
                // starting from the innermost UDF. It does not support rewrite
                // partial of the expression starting from the middle.
                // TODO: to support this should update check function
                // correspondingly.

                // Example:
                // Support:        |<------------Fuse:------------->|
                // softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))
                // Not-Support:    |<------Fuse:---->|
                // softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))

                targetExprStr = exprStr;
                core::QueryConfig config({});
                std::vector<std::string> parsedSingleExprs;
                std::vector<std::string> matchedExprs;
                // parse the target string into a std::vecotor<DLKernel(string)>
                parseDLExpressions(target, parsedSingleExprs, matchedExprs);
                std::reverse(
                    parsedSingleExprs.begin(), parsedSingleExprs.end());

                std::vector<velox::dl::KernelType> kernelTypes;
                std::vector<float*> weights;
                std::vector<int> dims;
                bool hasArgmax = false;
                // process each expression from the innermost DL kernel
                for (int i = 0; i < parsedSingleExprs.size(); i++) {
                  // double check it is supported DL kernel
                  assert(isSupportedDLKernel(parsedSingleExprs[i]));
                  auto dlKernelName = parsedSingleExprs[i];
                  std::vector<int> udfDims;
                  if (dlKernelName.find("mat_mul") != std::string::npos) {
                    auto myDL = getVectorFunction(
                        dlKernelName, {ARRAY(REAL())}, {}, config);
                    assert(myDL);
                    auto myDLFunc =
                        std::dynamic_pointer_cast<MatrixMultiply>(myDL);
                    assert(myDLFunc);
                    weights.push_back(myDLFunc->getTensor());
                    udfDims = myDLFunc->getDims();
                    kernelTypes.push_back(velox::dl::KernelType::MatMul);
                  } else if (
                      dlKernelName.find("mat_add") != std::string::npos ||
                      dlKernelName.find("mat_vector_add") !=
                          std::string::npos) {
                    auto myDL = getVectorFunction(
                        dlKernelName, {ARRAY(REAL())}, {}, config);
                    assert(myDL);
                    auto myDLFunc =
                        std::dynamic_pointer_cast<MatrixVectorAddition>(myDL);
                    assert(myDLFunc);
                    weights.push_back(myDLFunc->getTensor());
                    udfDims = myDLFunc->getDims();
                    kernelTypes.push_back(velox::dl::KernelType::MatAdd);
                  } else if (dlKernelName.find("relu") != std::string::npos) {
                    // Relu itself does not have dims stored in the UDF will use
                    // the last element in dims. current limitation: relu cannot
                    // be the innermost UDF.
                    assert(!dims.empty());
                    udfDims = {dims.back()};
                    kernelTypes.push_back(velox::dl::KernelType::ReLU);
                  } else if (
                      dlKernelName.find("batch_norm") != std::string::npos) {
                    // BachNorm
                    auto myDL = getVectorFunction(
                        dlKernelName, {ARRAY(REAL())}, {}, config);
                    assert(myDL);
                    auto myDLFunc =
                        std::dynamic_pointer_cast<BatchNorm1D>(myDL);
                    assert(myDLFunc);
                    weights.push_back(myDLFunc->getWeight());
                    weights.push_back(myDLFunc->getBias());
                    udfDims = myDLFunc->getDims();
                    kernelTypes.push_back(velox::dl::KernelType::BatchNorm);
                  } else if (
                      dlKernelName.find("softmax") != std::string::npos) {
                    // Softmax itself does not have dims stored in the UDF will
                    // use the last element in dims. current limitation: softmax
                    // cannot be the innermost UDF.
                    udfDims = {dims.back()};
                    kernelTypes.push_back(velox::dl::KernelType::Softmax);
                  } else if (dlKernelName.find("argmax") != std::string::npos) {
                    // Argmax itself does not have dims stored in the UDF will
                    // use the last element in dims. current limitation: argmax
                    // cannot be the innermost UDF.
                    udfDims = {dims.back(), 1};
                    kernelTypes.push_back(velox::dl::KernelType::Argmax);
                    hasArgmax = true;
                  } else if (
                      dlKernelName.find("sigmoid") != std::string::npos) {
                    udfDims = {dims.back(), 1};
                    kernelTypes.push_back(velox::dl::KernelType::Sigmoid);
                  } else {
                    std::cout
                        << "ERROR, Unsupported DL kernel: " << dlKernelName
                        << std::endl;
                  }

                  // Size of dimension should equal to 2*(Number of DL Ops)
                  // dims with index 2*i and 2*i+1 are the input and output

                  if (udfDims.size() == 2) {
                    // For DLs have two dimensions, like MatMul
                    dims.push_back(udfDims[0]);
                    dims.push_back(udfDims[1]);
                  } else {
                    // For DLs have one dimension, like Relu
                    dims.push_back(udfDims[0]);
                    dims.push_back(udfDims[0]);
                  }
                }

                std::string torchDNNName =
                    fmt::format("torchDNN_CUDA_{}", rewriteTorchDNNCounter++);

                exec::registerVectorFunction(
                    torchDNNName,
                    TorchDNNV2CUDA::signatures(),
                    std::make_unique<TorchDNNV2CUDA>(
                        kernelTypes, weights, dims));

                // Capture the data src
                std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
                std::smatch matches;
                // Object to capture the matched data source
                std::string matchedDataSrc;
                std::string targetExprName = projectionsNames[exprIdx];
                // Search out the matched data source and store in matches
                if (std::regex_search(
                        targetExprStr, matches, patternToMatchRawSource)) {
                  matchedDataSrc = matches[1].str();
                } else {
                  LOG(ERROR) << "Uncaptured data source" << std::endl;
                }

                std::regex pattern(escapeRegex(target));
                // To distinguish the argmax and non-argmax case, the TorchDNN
                // function requires different signatures to handle the output
                // type TorchDNNV2(input: array(REAL)) -> array(REAL)
                // TorchDNNV2(input: array(REAL), 1 :INTEGER) -> INTEGER
                // Replace the expression
                if (hasArgmax) {
                  targetExprStr = std::regex_replace(
                      targetExprStr,
                      pattern,
                      fmt::format(
                          "{}({},{})", torchDNNName, matchedDataSrc, 1));
                } else {
                  targetExprStr = std::regex_replace(
                      targetExprStr,
                      pattern,
                      fmt::format("{}({})", torchDNNName, matchedDataSrc));
                }

                finalProjectExprSets.insert(
                    targetExprStr + " AS " + targetExprName);
                // std::cout << fmt::format("exprStr: {}", targetExprStr)
                //           << std::endl;

                findRewriteTarget = true;
              } else {
                // Parse the non-target expressions
                std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
                std::smatch matches;
                if (std::regex_search(
                        exprStr, matches, patternToMatchRawSource)) {
                  auto matchedDataSrc = matches[1].str();
                  auto rewriteExpr = std::regex_replace(
                      exprStr, patternToMatchRawSource, matchedDataSrc);

                  finalProjectExprSets.insert(
                      rewriteExpr + " AS " + projectionsNames[exprIdx]);
                } else {
                  LOG(ERROR)
                      << "Error: undefined-edge case detected: " << exprStr
                      << std::endl;
                }
              }
            }

            if ((curNode->sources().size()) > 0 && findRewriteTarget) {
              // get the source node of the current node
              auto srcNode = curNode->sources()[0];
              auto rewritePlan =
                  exec::test::PlanBuilder(planNodeIdGenerator, pool_.get());
              rewritePlan = rewritePlan.setRoot(srcNode);

              std::vector<std::string> finalProjectExprs(
                  finalProjectExprSets.begin(), finalProjectExprSets.end());
              rewritePlan = rewritePlan.project(finalProjectExprs);
              if (prevNode == nullptr) {
                planBuilder.setRoot(rewritePlan.planNode());
              } else {
                // use seralization and deserialization to replace the source
                // node
                auto serializedPlan = planBuilder.planNode()->serialize();
                auto serializedNewSource = rewritePlan.planNode()->serialize();
                auto srcNodeIdToBeReplaced = curNode->id();
                replaceSourceWithIdInSerializedPlan(
                    serializedPlan, serializedNewSource, srcNodeIdToBeReplaced);
                auto deserlizedUpdatedPlanNode =
                    ISerializable::deserialize<core::PlanNode>(
                        serializedPlan, pool_.get());
                planBuilder.setRoot(deserlizedUpdatedPlanNode);
              }
              transformationApplied = true;
            }
          }
        }
        // Search for a filter node, which is similar to the project node
        if (nodeName == "Filter") {
          std::shared_ptr<const FilterNode> myFilterNode =
              std::dynamic_pointer_cast<const FilterNode>(curNode);

          const TypedExprPtr& filterExpr = myFilterNode->filter();

          targetExprStr = filterExpr->toString();

          if (targetExprStr == target) {
            if (auto call =
                    std::dynamic_pointer_cast<const core::CallTypedExpr>(
                        filterExpr)) {
              core::QueryConfig config({});
              // Stored the names for mat_add and mat_mul
              std::vector<std::string> mat_add_occurrences;
              std::vector<std::string> mat_mul_occurrences;

              std::regex pattern_add(R"(mat_add\d+)");
              std::regex pattern_mul(R"(mat_mul\d+)");

              auto words_begin_add = std::sregex_iterator(
                  target.begin(), target.end(), pattern_add);
              auto words_end_add = std::sregex_iterator();
              for (std::sregex_iterator i = words_begin_add; i != words_end_add;
                   ++i) {
                std::smatch match = *i;
                mat_add_occurrences.push_back(match.str());
              }

              auto words_begin_mul = std::sregex_iterator(
                  target.begin(), target.end(), pattern_mul);
              auto words_end_mul = std::sregex_iterator();
              for (std::sregex_iterator i = words_begin_mul; i != words_end_mul;
                   ++i) {
                std::smatch match = *i;
                mat_mul_occurrences.push_back(match.str());
              }
              // Search the pointer for registed add function
              std::vector<std::shared_ptr<VectorFunction>> myAddFunc;
              for (std::string mat_add_name : mat_add_occurrences) {
                std::shared_ptr<VectorFunction> myAdd = getVectorFunction(
                    mat_add_name, {ARRAY(REAL())}, {}, config);
                if (myAdd) {
                  myAddFunc.push_back(myAdd);
                }
              }
              // Iterating over myAddFunc in reverse, the inner function is the
              // first layer add function
              for (auto it = myAddFunc.rbegin(); it != myAddFunc.rend(); ++it) {
                // Dynamically cast to MatrixVectorAddition
                auto myAddUDF =
                    std::dynamic_pointer_cast<MatrixVectorAddition>(*it);
                if (myAddUDF) {
                  bias.push_back(myAddUDF->getTensor());
                }
              }
              // Search the pointer for registed mul function
              std::vector<std::shared_ptr<VectorFunction>> myMulFunc;
              for (std::string mat_mul_name : mat_mul_occurrences) {
                std::shared_ptr<VectorFunction> myMul = getVectorFunction(
                    mat_mul_name, {ARRAY(REAL())}, {}, config);
                if (myMul) {
                  myMulFunc.push_back(myMul);
                }
              }
              // Iterating over myMulFunc in reverse, the inner function is the
              // first layer mul function
              for (auto it = myMulFunc.rbegin(); it != myMulFunc.rend(); ++it) {
                // Dynamically cast to MatrixVectorAddition
                auto myMulUDF = std::dynamic_pointer_cast<MatrixMultiply>(*it);
                if (myMulUDF) {
                  weights.push_back(myMulUDF->getTensor());
                  if (dims.empty()) {
                    // We extract two parameters from the first layer, and we
                    // only need one parameter for later layer
                    dims.push_back(myMulUDF->getDims()[0]);
                    dims.push_back(myMulUDF->getDims()[1]);
                  } else {
                    dims.push_back(myMulUDF->getDims()[1]);
                  }
                }
              }
              // Register torchdnn_multi
              registerVectorFunction(
                  "torchDNN",
                  TorchDNN::signatures(),
                  std::make_unique<TorchDNN>(weights, bias, dims));
              // TODO:catalog process

              if (curNode->sources().size() > 0) {
                // Here, we focus on the inner case. For example:
                // project({softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))})
                // ---> project({torchnnx(v)}) The format for inner case
                // is project(twolayer())
                // TODO: Medium case - project(func1(twolayer(func2())))
                // TODO: Outer case -
                // project({twolayer(func1(func2()))}) Plan Builder
                // start from the previous node.
                planBuilder = planBuilder.setRoot(curNode->sources()[0]);
                // Regular expression match
                std::regex pattern(R"(softmax\d+\(.*\)\))");
                // Replace the expression
                targetExprStr =
                    std::regex_replace(targetExprStr, pattern, "torchDNN(v)");
                // Plan Builder add the new node.
                planBuilder = planBuilder.project({targetExprStr});

                transformationApplied = true;
              }
            }
            if (curNode->sources().size() > 0) {
              // Here, we focus on the inner case. For example:
              // project({softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))})
              // ---> project({torchnnx(v)}) The format for inner case is
              // project(twolayer())
              // TODO: Medium case - project(func1(twolayer(func2())))
              // TODO: Outer case - project({twolayer(func1(func2()))})
              // Plan Builder start from the previous node.
              planBuilder = planBuilder.setRoot(curNode->sources()[0]);

              std::regex pattern(R"(softmax\d+\(.*\)\))");

              targetExprStr =
                  std::regex_replace(targetExprStr, pattern, "torchDNN(v)");

              planBuilder = planBuilder.project({targetExprStr});

              transformationApplied = true;
            }
          }
        }
      }
      // Serach lower level plan node
      std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();
      // Until leaf node
      if (sources.size() == 0)
        return false;
      // recursive search
      for (auto source : sources)

        transformationApplied |= apply(
            source,
            curNode,
            maker,
            planBuilder,
            pool_,
            planNodeIdGenerator,
            targets,
            cataLog);
    }

    return transformationApplied;
  }

  /**
   * @brief A function to get the name of rewritten rule.
   *
   * @return A string value denoting the name of the rule
   */
  std::string name() override {
    return "MultiLayerUDF2TorchNNCUDARewriteAction";
  }

  bool isSupportedDLKernel(std::string dlKernelName) {
    for (auto supportedDLKernel : supportedDLKernels) {
      // Does not support block-based mat_mul
      if (dlKernelName.find(supportedDLKernel) != std::string::npos &&
          dlKernelName.find("_h") == std::string::npos) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief A function to check if this rule can be applied in a logical plan
   * and to store the possible UDF name.
   *
   * @param rootNode A pointer to the logical plan.
   * @param targetActions A pointer to the vector used to store possible UDF
   * names applicable for this rule.
   * @param cataLog Reference to a CataLog object to store metadata and
   * information.
   *
   * @return A boolean value indicating whether the check was successful.
   */
  bool check(
      std::shared_ptr<const core::PlanNode> rootNode,
      std::vector<std::string>& targetActions,
      CataLog& cataLog) override {
    try {
      bool checkSuccess = true;
      if (!rootNode) {
        throw std::invalid_argument("rootNode is null");
      }

      std::string_view nodeName = rootNode->name();

      std::vector<TypedExprPtr> expressions;
      // We first check the project node
      if (nodeName == "Project") {
        auto myProjectNode =
            std::dynamic_pointer_cast<const ProjectNode>(rootNode);

        if (!myProjectNode) {
          throw std::runtime_error("Failed to cast to ProjectNode");
        }

        expressions = myProjectNode->projections();
      } else if (nodeName == "Filter") {
        auto myFilterNode =
            std::dynamic_pointer_cast<const FilterNode>(rootNode);

        if (!myFilterNode) {
          throw std::runtime_error("Failed to cast to FilterNode");
        }

        expressions = {myFilterNode->filter()};
      }
      // Search each expressions
      for (const auto& expression : expressions) {
        std::string expr = expression->toString();
        // std::cout << "expr: " << expr << std::endl;
        std::vector<std::string> parsedSingleExprs;
        std::vector<std::string> matchedExprs;
        parseDLExpressions(expr, parsedSingleExprs, matchedExprs);
        // reverse to get the innermost function first
        std::reverse(parsedSingleExprs.begin(), parsedSingleExprs.end());
        std::reverse(matchedExprs.begin(), matchedExprs.end());
        std::string targetExprStr;
        for (int i = 0; i < parsedSingleExprs.size(); i++) {
          if (isSupportedDLKernel(parsedSingleExprs[i])) {
            targetExprStr = matchedExprs[i];
          }
        }

        if (!targetExprStr.empty()) {
          targetActions.push_back(targetExprStr);
        }
        // std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
        // std::smatch matches;
        // if (std::regex_search(
        //         exprStr, matches, patternToMatchRawSource)) {
        //   auto matchedDataSrc = matches[1].str();
        //   auto rewriteExpr = std::regex_replace(
        //       exprStr, patternToMatchRawSource, matchedDataSrc);

        //   finalProjectExprSets.insert(
        //       rewriteExpr + " AS " + projectionsNames[exprIdx]);
        // } else {
        //   LOG(ERROR)
        //       << "Error: undefined-edge case detected: " << exprStr
        //       << std::endl;
        // }

        // // Regular expression match
        // std::regex pattern(R"(softmax\d+\(.*\)\))");

        // auto wordsBegin =
        //     std::sregex_iterator(expr.begin(), expr.end(), pattern);

        // auto wordsEnd = std::sregex_iterator();
        // // Retrieve the possible UDF name applicable for this rule, stored in
        // // targetAction.
        // for (auto it = wordsBegin; it != wordsEnd; ++it) {
        //   targetActions.push_back(it->str());
        // }
      }

      std::vector<std::shared_ptr<const core::PlanNode>> sources =
          rootNode->sources();

      if (sources.empty()) {
        // Safe exit for leaf node
        return true;
      }

      for (const auto& source : sources) {
        checkSuccess &= check(source, targetActions, cataLog);
      }

      return checkSuccess;

    } catch (const std::exception& e) {
      std::cerr << "Error in check function: " << e.what() << std::endl;

      return false; // Return false for any error
    }
  }

 private:
  std::vector<int> dims;
  std::string targetExprStr;
  std::vector<float*> weights;
  std::vector<float*> bias;
  std::vector<std::string> supportedDLKernels = {
      "mat_mul",
      "mat_vector_add",
      "mat_add",
      "relu",
      "batch_norm",
      "softmax",
      "argmax",
      "sigmoid"};
  static inline int rewriteTorchDNNCounter = 0;
};

} // namespace optimization