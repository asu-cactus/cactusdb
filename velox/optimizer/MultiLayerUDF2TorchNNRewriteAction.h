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

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;
using namespace optimization;

namespace optimization {

class MultiLayerUDF2TorchNNRewriteAction : public RewriteAction {
 public:
  MultiLayerUDF2TorchNNRewriteAction() {}

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
    // Iterate over each target in the targets container
    for (auto target : targets) {
      // Start from the current node
      if (curNode) {
        // Get the name of node
        std::string_view nodeName = curNode->name();
        // We frist search project node
        if (nodeName == "Project") {
          // Cast node as project node
          if (auto myProjectNode =
                  std::dynamic_pointer_cast<const ProjectNode>(curNode)) {
            // Get projections in project node
            const std::vector<TypedExprPtr>& projections =
                myProjectNode->projections();
            // Search each expression in projections
            for (auto expression : projections) {
              // Get the string of expression
              exprStr = expression->toString();
              // Tree serach in one expression until leaf expression (size == 0)
              while (expression->inputs().size() > 0) {
                // String match the target UDF name
                if (exprStr == target) {
                  // Cast this matched expression to CallTypedExpr, which is
                  // used to get the pointer for the UDF functions
                  if (auto call =
                          std::dynamic_pointer_cast<const core::CallTypedExpr>(
                              expression)) {
                    // We only consider one projection expression in the project
                    // node.
                    if (projections.size() == 1) {
                      core::QueryConfig config({});
                      // Stored the names for mat_add and mat_mul
                      std::vector<std::string> mat_add_occurrences;
                      std::vector<std::string> mat_mul_occurrences;

                      std::regex pattern_add(R"(mat_add\d+)");
                      std::regex pattern_mul(R"(mat_mul\d+)");

                      auto words_begin_add = std::sregex_iterator(
                          target.begin(), target.end(), pattern_add);
                      auto words_end_add = std::sregex_iterator();
                      for (std::sregex_iterator i = words_begin_add;
                           i != words_end_add;
                           ++i) {
                        std::smatch match = *i;
                        mat_add_occurrences.push_back(match.str());
                      }

                      auto words_begin_mul = std::sregex_iterator(
                          target.begin(), target.end(), pattern_mul);
                      auto words_end_mul = std::sregex_iterator();
                      for (std::sregex_iterator i = words_begin_mul;
                           i != words_end_mul;
                           ++i) {
                        std::smatch match = *i;
                        mat_mul_occurrences.push_back(match.str());
                      }
                      // Search the pointer for registed add function
                      std::vector<std::shared_ptr<VectorFunction>> myAddFunc;
                      for (std::string mat_add_name : mat_add_occurrences) {
                        std::shared_ptr<VectorFunction> myAdd =
                            getVectorFunction(
                                mat_add_name, {ARRAY(REAL())}, {}, config);
                        if (myAdd) {
                          myAddFunc.push_back(myAdd);
                        }
                      }
                      // Iterating over myAddFunc in reverse, the inner function
                      // is the first layer add function
                      for (auto it = myAddFunc.rbegin(); it != myAddFunc.rend();
                           ++it) {
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
                        std::shared_ptr<VectorFunction> myMul =
                            getVectorFunction(
                                mat_mul_name, {ARRAY(REAL())}, {}, config);
                        if (myMul) {
                          myMulFunc.push_back(myMul);
                        }
                      }
                      // Iterating over myMulFunc in reverse, the inner function
                      // is the first layer mul function
                      for (auto it = myMulFunc.rbegin(); it != myMulFunc.rend();
                           ++it) {
                        // Dynamically cast to MatrixVectorAddition
                        auto myMulUDF =
                            std::dynamic_pointer_cast<MatrixMultiply>(*it);
                        if (myMulUDF) {
                          weights.push_back(myMulUDF->getTensor());
                          if (dims.empty()) {
                            // We extract two parameters from the first layer,
                            // and we only need one parameter for later layer
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
                          std::make_unique<TorchDNN>(
                              weights, bias, dims));
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
                        planBuilder =
                            planBuilder.setRoot(curNode->sources()[0]);
                        // Regular expression match
                        std::regex pattern(R"(softmax\d+\(.*\)\))");
                        // Replace the expression
                        exprStr =
                            std::regex_replace(exprStr, pattern, "torchDNN(v)");
                        // Plan Builder add the new node.
                        planBuilder = planBuilder.project({exprStr});

                        transformationApplied = true;
                      }
                    }
                  }
                }
                // Search the lower level expression in the same plan node
                expression = expression->inputs()[0];
              }
            }
          }
        }
        // Search for a filter node, which is similar to the project node
        if (nodeName == "Filter") {
          std::shared_ptr<const FilterNode> myFilterNode =
              std::dynamic_pointer_cast<const FilterNode>(curNode);

          const TypedExprPtr& filterExpr = myFilterNode->filter();

          exprStr = filterExpr->toString();

          if (exprStr == target) {
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
                auto myAddUDF = std::dynamic_pointer_cast<MatrixVectorAddition>(*it);
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
                exprStr = std::regex_replace(exprStr, pattern, "torchDNN(v)");
                // Plan Builder add the new node.
                planBuilder = planBuilder.project({exprStr});

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

              exprStr = std::regex_replace(exprStr, pattern, "torchDNN(v)");

              planBuilder = planBuilder.project({exprStr});

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
    return "MultiLayerUDF2TorchNNRewriteAction";
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
      // We first check the project node
      if (nodeName == "Project") {
        auto myProjectNode =
            std::dynamic_pointer_cast<const ProjectNode>(rootNode);

        if (!myProjectNode) {
          throw std::runtime_error("Failed to cast to ProjectNode");
        }

        const std::vector<TypedExprPtr>& projections =
            myProjectNode->projections();
        // Search each expressions
        for (const auto& expression : projections) {
          std::string expr = expression->toString();
          // Regular expression match
          std::regex pattern(R"(softmax\d+\(.*\)\))");

          auto wordsBegin =
              std::sregex_iterator(expr.begin(), expr.end(), pattern);

          auto wordsEnd = std::sregex_iterator();
          // Retrieve the possible UDF name applicable for this rule, stored in
          // targetAction.
          for (auto it = wordsBegin; it != wordsEnd; ++it) {
            targetActions.push_back(it->str());
          }
        }
      }
      // We then check the filter node
      if (nodeName == "Filter") {
        auto myFilterNode =
            std::dynamic_pointer_cast<const FilterNode>(rootNode);

        if (!myFilterNode) {
          throw std::runtime_error("Failed to cast to FilterNode");
        }

        const TypedExprPtr& filterExpr = myFilterNode->filter();

        std::string expr = filterExpr->toString();
        // Regular expression match
        std::regex pattern(R"(softmax\d+\(.*\)\))");

        auto wordsBegin =
            std::sregex_iterator(expr.begin(), expr.end(), pattern);

        auto wordsEnd = std::sregex_iterator();
        // Retrieve the possible UDF name applicable for this rule, stored in
        // targetAction.
        for (auto it = wordsBegin; it != wordsEnd; ++it) {
          targetActions.push_back(it->str());
        }
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
  std::string exprStr;
  std::vector<float*> weights;
  std::vector<float*> bias;
};

} 