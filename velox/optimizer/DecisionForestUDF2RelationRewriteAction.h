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
#include "velox/common/file/FileSystems.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/optimizer/Helper.h"
#include "velox/parse/TypeResolver.h"

using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

namespace optimization {

class DecisionForestUDF2RelationRewriteAction : public RewriteAction {
 public:
  DecisionForestUDF2RelationRewriteAction() {}

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
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
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
    bool transformationApplied = false;
    for (auto target : targets) {
      if (curNode) {
        std::string_view nodeName = curNode->name();

        if (nodeName == "Project") {
          std::shared_ptr<const ProjectNode> myProjectNode =
              std::dynamic_pointer_cast<const ProjectNode>(curNode);

          const std::vector<TypedExprPtr>& projections =
              myProjectNode->projections();

          for (auto expression : projections) {
            if (auto call =
                    std::dynamic_pointer_cast<const core::CallTypedExpr>(
                        expression)) {
              std::string callName = call->name();

              if (callName.find(target) != std::string::npos) {
                /* "decision_forest_predict"
                 * Case I: If this function call is the only expression in the
                 * projection (e.g.,project({"decision_forest_predict(x)"})), we
                 * need to extract the model path from the expression to
                 * initialize treeRowVector as a member of this object, and then
                 * replace this node by the following plan:
                 *
                 * nestedLoopJoin(
                 *    exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                 *    .values({treeRowVector})
                 *    .project({"tree_id as tree_id",
                 * "velox_decision_tree_construct(tree_path) as tree"})
                 *    .planNode(), {"idx", "x", "tree_id", "tree"})
                 *    .project({"idx as idx", "tree_id as tree_id",
                 * "velox_decision_tree_predict(x, tree) as prediction"})
                 *    .aggregation({"idx"}, {"sum(prediction) as sum"},{},
                 * core::AggregationNode::Step::kPartial, false) .project({"idx
                 * as idx", "if (sum > 0.0, 1.0, 0.0)"})
                 */

                if (projections.size() == 1) {
                  // We are the only expression in the project operator

                  // We shall extract the path
                  core::QueryConfig config({});

                  std::shared_ptr<VectorFunction> myUDF =
                      getVectorFunction(callName, {ARRAY(REAL())}, {}, config);
                  int numTrees;
                  std::vector<std::shared_ptr<TempFilePath>> treePaths;

                  if (myUDF) {
                    std::cout << "In MyUDF" << std::endl;

                    std::shared_ptr<ForestPrediction> myDecisionForestUDF =
                        dynamic_pointer_cast<ForestPrediction>(myUDF);

                    if (myDecisionForestUDF) {
                      std::string modelPath =
                          myDecisionForestUDF->getForestPath();

                      std::vector<std::string> pathVectors;

                      Forest::vectorizeForestFolder(modelPath, pathVectors);

                      numTrees = pathVectors.size();
                      // TODO: can set as a parameter in the future
                      int numTreeSplits = 8;
                      uint32_t numTreeRowsPerSplit =
                          ceil(numTrees / numTreeSplits);
                      uint32_t treeIndex = 0;
                      optimization::MyFileTest myFile;
                      for (size_t i = 0; i < numTreeSplits; i++) {
                        auto startIdx = treeIndex;
                        auto endIdx = treeIndex + (i + 1) * numTreeRowsPerSplit;
                        endIdx = (endIdx < numTrees) ? endIdx : numTrees;
                        size_t numDataInPartition = endIdx - startIdx;

                        auto model =
                            maker.flatVector<StringView>(numDataInPartition);
                        auto treeIndexVector =
                            maker.flatVector<int16_t>(numDataInPartition);
                        for (int j = 0; j < numDataInPartition; j++) {
                          model->set(
                              j, StringView(pathVectors[treeIndex].c_str()));
                          treeIndexVector->set(j, treeIndex);
                          treeIndex += 1;
                        }

                        treeRowVector = maker.rowVector(
                            {"tree_id", "tree_path"}, {treeIndexVector, model});

                        auto file = TempFilePath::create();
                        auto config =
                            std::make_shared<facebook::velox::dwrf::Config>();
                        myFile.writeToFile(file->path, {treeRowVector}, config);
                        treePaths.push_back(file);
                      }
                      assert(treeIndex == numTrees);
                    }
                  }

                  // We remove the current node from the plan
                  //
                  if (curNode->sources().size() > 0) {
                    planBuilder = planBuilder.setRoot((curNode->sources())[0]);
                    std::cout << "debug, current plan"
                              << planBuilder.planNode()->toString(true, true)
                              << std::endl;

                    // We build the plan from this point
                    core::PlanNodeId p1;

                    planBuilder =
                        exec::test::PlanBuilder(
                            planNodeIdGenerator, pool_.get())
                            .tableScan(asRowType(treeRowVector->type()))
                            .capturePlanNodeId(p1)
                            .project(
                                {"tree_id as tree_id",
                                 "velox_decision_tree_construct(tree_path) as tree"})
                            .nestedLoopJoin(
                                planBuilder.planNode(),
                                {"idx", "v", "tree_id", "tree"})
                            .project(
                                {"idx as idx",
                                 "tree_id as tree_id",
                                 "velox_decision_tree_predict(v, tree) as prediction"})
                            .aggregation(
                                {"idx"},
                                {"sum(prediction) as sum"},
                                {},
                                core::AggregationNode::Step::kPartial,
                                false)
                            .project(
                                {"idx as idx", "if (sum > 0.0, 1.0, 0.0)"});
                    // 										planBuilder
                    // =
                    // planBuilder.nestedLoopJoin(exec::test::PlanBuilder(planNodeIdGenerator,
                    // pool_.get()) 														.tableScan(asRowType(tree){treeRowVector})
                    // 														.capturePlanNodeId(p1)
                    // 														.project({"tree_id
                    // as tree_id", "velox_decision_tree_construct(tree_path) as
                    // tree"}) 														.planNode(), {"idx", "v", "tree_id", "tree"})
                    // 														.project({"idx
                    // as idx", "tree_id as tree_id",
                    // "velox_decision_tree_predict(v, tree) as prediction"})
                    // 														.aggregation({"idx"},
                    // {"sum(prediction) as sum"},{},
                    // core::AggregationNode::Step::kPartial, false)
                    // 														.project({"idx
                    // as idx", "if (sum > 0.0, 1.0, 0.0)"});
                    std::shared_ptr<OutputStat> inputStat =
                        std::make_shared<OutputStat>(OutputStat(numTrees, 1));
                    Source inputSource =
                        Source(p1, Source::Type::FILE, std::move(inputStat));
                    cataLog.addSource(std::make_shared<Source>(inputSource));
                    cataLog.setIdAddressMap(p1, treePaths);

                    transformationApplied = true;
                  }
                }

                /*
                 * Case II: If this function call is the last expression in the
                 * projection (e.g.,
                 * project({"decision_forest_predict(func1(func2(x)))"})), we
                 * need to split this project node into two project nodes (e.g.,
                 * project({"func1(func2(x)) as
                 * y"}).project("decision_forest_predict(y)")) Then, we apply
                 * the rewrite action as described in Case I.
                 */

                // TODO

                /*
                 * Case III: If this function call is a middle expression in the
                 * projection (e.g.,
                 * project({"func4(func3(decision_forest_predict(func1(func2(x)))))"})),
                 * we need to split this project node into three project nodes
                 * (e.g., project({"func1(func2(x)) as
                 * z0"}).project("decision_forest_predict(z0) as
                 * z1").project("func4(func3(z1))")) Then, we apply the rewrite
                 * action as described in Case I.
                 */

                // TODO
              }
            }
          }
        }

        std::vector<std::shared_ptr<const PlanNode>> sources =
            curNode->sources();

        if (sources.size() == 0)
          return false;

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
    }
    return transformationApplied;
  }

  /**
   * @brief A function to get the name of rewritten rule.
   *
   * @return A string value denoting the name of the rule
   */
  std::string name() override {
    return "DecisionForestUDF2RelationRewriteAction";
  }

  /**
   * @brief A function to check if this rule can be applied in a logical plan
   * and to store the possible UDF name.
   *
   * @param rootNode A pointer to the logical plan.
   * @param targetActions A pointer to the vector used to store possible UDF
   * names applicable for this rule.
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
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

      if (nodeName == "Project") {
        auto myProjectNode =
            std::dynamic_pointer_cast<const ProjectNode>(rootNode);

        if (!myProjectNode) {
          throw std::runtime_error("Failed to cast to ProjectNode");
        }

        const std::vector<TypedExprPtr>& projections =
            myProjectNode->projections();

        for (const auto& expression : projections) {
          std::string expr = expression->toString();
          // For this rule, we only check wheather decision_forest_predict UDF
          // function is in expressions
          std::regex pattern(R"(decision_forest_predict)");

          auto wordsBegin =
              std::sregex_iterator(expr.begin(), expr.end(), pattern);

          auto wordsEnd = std::sregex_iterator();
          // Find applicable UDF name and store in targetActions
          for (auto it = wordsBegin; it != wordsEnd; ++it) {
            targetActions.push_back(it->str());
          }
        }
      }

      if (nodeName == "Filter") {
        auto myFilterNode =
            std::dynamic_pointer_cast<const FilterNode>(rootNode);

        if (!myFilterNode) {
          throw std::runtime_error("Failed to cast to FilterNode");
        }

        const TypedExprPtr& filterExpr = myFilterNode->filter();

        std::string expr = filterExpr->toString();
        // For this rule, we only check wheather decision_forest_predict UDF
        // function is in expressions
        std::regex pattern(R"(decision_forest_predict)");

        auto wordsBegin =
            std::sregex_iterator(expr.begin(), expr.end(), pattern);

        auto wordsEnd = std::sregex_iterator();
        // Find applicable UDF name and store in targetActions
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
  RowVectorPtr treeRowVector;
};

} // namespace optimization
