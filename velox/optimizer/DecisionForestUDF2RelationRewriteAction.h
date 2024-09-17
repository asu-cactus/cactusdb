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
    bool transformationApplied = false;
    std::set<std::string> constructTreeExprSets;
    std::set<std::string> nestedLoopProjectExprSets;
    std::set<std::string> singleTreePredictProjectExprSets;
    std::set<std::string> treePredictAggKeySets;
    std::set<std::string> treePredictAggExprSets;
    std::set<std::string> finalProjectExprSets;
    for (auto target : targets) {
      if (curNode) {
        std::string_view nodeName = curNode->name();

        if (nodeName == "Project") {
          std::shared_ptr<const ProjectNode> myProjectNode =
              std::dynamic_pointer_cast<const ProjectNode>(curNode);

          const std::vector<TypedExprPtr>& projections =
              myProjectNode->projections();
          // Get the names of projections
          const std::vector<std::string>& projectionsNames =
              myProjectNode->names();
          // Search each expression in projections
          int numProjections = projections.size();

          bool findRewriteTarget = false;
          int numTrees;
          std::vector<std::shared_ptr<TempFilePath>> treePaths;

          std::string treeIdName;
          std::string treePathName;
          std::string decisionTreePredictFuncName;
          for (int exprIdx = 0; exprIdx < numProjections; exprIdx++) {
            auto expression = projections[exprIdx];
            // Get the string of expression
            std::string exprStr = expression->toString();

            if (exprStr.find(target) != std::string::npos) {
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
               *    .planNode(), {"idx", "x", treeIdName, "tree"})
               *    .project({"idx as idx", "tree_id as tree_id",
               * "velox_decision_tree_predict(x, tree) as prediction"})
               *    .aggregation({"idx"}, {"sum(prediction) as sum"},{},
               * core::AggregationNode::Step::kPartial, false) .project({"idx
               * as idx", "if (sum > 0.0, 1.0, 0.0)"})
               */

              // We are the only expression in the project operator

              // We shall extract the path
              core::QueryConfig config({});

              // Parse the function name from target
              // Extract the function name
              size_t openParen = target.find('(');
              std::string funcName = target.substr(0, openParen);
              funcName = trim(funcName);

              std::shared_ptr<VectorFunction> myUDF =
                  getVectorFunction(funcName, {ARRAY(REAL())}, {}, config);
              assert(myUDF);

              // Get the original decision forest UDF
              std::shared_ptr<ForestPrediction> myDecisionForestUDF =
                  dynamic_pointer_cast<ForestPrediction>(myUDF);
              assert(myDecisionForestUDF);

              int numCols = myDecisionForestUDF->getNumFeatures();
              decisionTreePredictFuncName = fmt::format("velox_decision_tree_predict{}", rewriteDecisionForestCounter);

              exec::registerVectorFunction(
                                     decisionTreePredictFuncName,
                                     VeloxTreePrediction::signatures(),
                                     std::make_unique<VeloxTreePrediction>(numCols));

              // Get the target expression name
              std::string targetExprName = projectionsNames[exprIdx];
              // capture the data src
              std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
              std::smatch matches;
              // object to capture the matched data source
              std::string matchedDataSrc;
              // Search out the matched data source and store in matches
              if (std::regex_search(target, matches, patternToMatchRawSource)) {
                matchedDataSrc = matches[1].str();
              } else {
                LOG(ERROR) << "Uncaptured data source" << std::endl;
              }
              nestedLoopProjectExprSets.insert(matchedDataSrc);

              std::string modelPath = myDecisionForestUDF->getForestPath();
              std::vector<std::string> pathVectors;
              Forest::vectorizeForestFolder(modelPath, pathVectors);

              numTrees = pathVectors.size();
              // TODO: can set as a parameter in the future
              int numTreeSplits = 8;
              uint32_t numTreeRowsPerSplit = ceil(numTrees / numTreeSplits);
              uint32_t treeIndex = 0;
              optimization::MyFileTest myFile;

              treeIdName =
                  fmt::format("tree_id{}", rewriteDecisionForestCounter);
              treePathName =
                  fmt::format("tree_path{}", rewriteDecisionForestCounter);
              std::string intermediateTreeName = fmt::format(
                  "tree_intermediate{}", rewriteDecisionForestCounter);
              std::string intermediateTreePredictName = fmt::format(
                  "tree_intermediate_predict{}", rewriteDecisionForestCounter);
              std::string aggregatedTreePredictName = fmt::format(
                  "aggregated_tree_predict{}", rewriteDecisionForestCounter);
              rewriteDecisionForestCounter++;

              for (size_t i = 0; i < numTreeSplits; i++) {
                auto startIdx = treeIndex;
                auto endIdx = treeIndex + (i + 1) * numTreeRowsPerSplit;
                endIdx = (endIdx < numTrees) ? endIdx : numTrees;
                size_t numDataInPartition = endIdx - startIdx;

                auto model = maker.flatVector<StringView>(numDataInPartition);
                auto treeIndexVector =
                    maker.flatVector<int16_t>(numDataInPartition);
                for (int j = 0; j < numDataInPartition; j++) {
                  model->set(j, StringView(pathVectors[treeIndex].c_str()));
                  treeIndexVector->set(j, treeIndex);
                  treeIndex += 1;
                }

                treeRowVector = maker.rowVector(
                    {treeIdName, treePathName}, {treeIndexVector, model});

                auto file = TempFilePath::create();
                auto config = std::make_shared<facebook::velox::dwrf::Config>();
                myFile.writeToFile(file->path, {treeRowVector}, config);
                treePaths.push_back(file);
              }

              constructTreeExprSets.insert(treeIdName);
              constructTreeExprSets.insert(fmt::format(
                  "velox_decision_tree_construct({}) as {}",
                  treePathName,
                  intermediateTreeName));
              nestedLoopProjectExprSets.insert(treeIdName);
              nestedLoopProjectExprSets.insert(intermediateTreeName);
              nestedLoopProjectExprSets.insert(matchedDataSrc);
              singleTreePredictProjectExprSets.insert(treeIdName);
              singleTreePredictProjectExprSets.insert(fmt::format(
                  "{}({}, {}) as {}",
                  decisionTreePredictFuncName,
                  matchedDataSrc,
                  intermediateTreeName,
                  intermediateTreePredictName));
              treePredictAggExprSets.insert(fmt::format(
                  "sum({}) as {}",
                  intermediateTreePredictName,
                  aggregatedTreePredictName));
              finalProjectExprSets.insert(fmt::format(
                  "if ({} > 0.0, 1.0, 0.0) AS {}",
                  aggregatedTreePredictName,
                  targetExprName));

              assert(treeIndex == numTrees);
              findRewriteTarget = true;

              // TODO
            } else {
              // Parse the non-target expressions
              auto rewriteExpr = exprStr;
              std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
              std::smatch matches;
              // Start position for the search
              std::string::const_iterator searchStart(exprStr.cbegin());

              // Search out the matched data source and store in matches
              while (std::regex_search(
                  searchStart,
                  exprStr.cend(),
                  matches,
                  patternToMatchRawSource)) {
                auto matchedDataSrc = matches[1].str();
                std::regex patternOfReplaceExpr(escapeRegex(matches[0].str()));

                rewriteExpr = std::regex_replace(
                    rewriteExpr, patternOfReplaceExpr, matchedDataSrc);
                // Update the search start position
                searchStart = matches.suffix().first;
                nestedLoopProjectExprSets.insert(matchedDataSrc);
                singleTreePredictProjectExprSets.insert(matchedDataSrc);
                treePredictAggKeySets.insert(matchedDataSrc);
              }
              rewriteExpr = replaceDoubleQuotes(rewriteExpr);

              finalProjectExprSets.insert(
                  rewriteExpr + " AS " + projectionsNames[exprIdx]);
            }
          }

          // Found rewrite
          if (curNode->sources().size() > 0 && findRewriteTarget) {
            // do an inference to obtain the id column to aggregate tree
            // predictions assume the id column contains "id"
            std::string matMulAggKey;
            for (std::string colName : nestedLoopProjectExprSets) {
              if (containsStrButNotEqual(colName, "id")) {
                matMulAggKey = colName;
                break;
              }
            }
            treePredictAggKeySets.insert(matMulAggKey);

            std::vector<std::string> constructTreeExprs(
                constructTreeExprSets.begin(), constructTreeExprSets.end());
            std::vector<std::string> nestedLoopProjectExprs(
                nestedLoopProjectExprSets.begin(),
                nestedLoopProjectExprSets.end());
            std::vector<std::string> singleTreePredictProjectExprs(
                singleTreePredictProjectExprSets.begin(),
                singleTreePredictProjectExprSets.end());
            std::vector<std::string> treePredictAggKeyExprs(
                treePredictAggKeySets.begin(), treePredictAggKeySets.end());
            std::vector<std::string> treePredictAggExprs(
                treePredictAggExprSets.begin(), treePredictAggExprSets.end());
            std::vector<std::string> finalProjectExprs(
                finalProjectExprSets.begin(), finalProjectExprSets.end());

            // Remove the current node
            planBuilder = planBuilder.setRoot((curNode->sources())[0]);

            // We build the plan from this point
            core::PlanNodeId p1;

            auto rewritePlan =
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(asRowType(treeRowVector->type()))
                    .capturePlanNodeId(p1)
                    .project(constructTreeExprs)
                    .nestedLoopJoin(
                        planBuilder.planNode(), nestedLoopProjectExprs)
                    .project(singleTreePredictProjectExprs)
                    .aggregation(
                        treePredictAggKeyExprs,
                        treePredictAggExprs,
                        {},
                        core::AggregationNode::Step::kPartial,
                        false)
                    .project(finalProjectExprs);
            std::shared_ptr<OutputStat> inputStat =
                std::make_shared<OutputStat>(OutputStat(numTrees, 1));
            Source inputSource =
                Source(p1, Source::Type::FILE, std::move(inputStat));
            cataLog.addSource(std::make_shared<Source>(inputSource));
            cataLog.setIdAddressMap(p1, treePaths);


            if (prevNode == nullptr) {
              planBuilder.setRoot(rewritePlan.planNode());
            } else {
              // use seralization and deserialization to replace the source
              // node
              auto serializedPlan = planBuilder.planNode()->serialize();
              auto serializedNewSource = rewritePlan.planNode()->serialize();
              // FIXME: due the missing implementation of serialization and deserialization
              // of Tree Type, the current branch is not reachble which limited by check `
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
          std::string exprStr = expression->toString();
          // For this rule, we only check wheather decision_forest_predict UDF
          // function is in expressions and limite to the case there is no
          // other UDF in the expression

          std::regex patternToMatchDecisionForest(
              "(decision_forest_predict\\d*\\(ROW\\[\"(.*?)\"\\]\\))");
          std::smatch matches;
          if (std::regex_search(
                  exprStr, matches, patternToMatchDecisionForest)) {
            auto matchedExpr = matches[1].str();
            if (matchedExpr == exprStr & prevNode == nullptr) {
              // FIXME: due the serialization and deserialization bug, we can only support
              // rewrite the decision forest where the project node is the last 
              // node
              targetActions.push_back(matchedExpr);
            } else {
              LOG(ERROR) << "Error: undefined-edge case detected: " << exprStr
                         << std::endl;
            }
          }
        }
      }

      if (nodeName == "Filter") {
        // FIXME: needed to be refactorized
        // auto myFilterNode =
        //     std::dynamic_pointer_cast<const FilterNode>(rootNode);

        // if (!myFilterNode) {
        //   throw std::runtime_error("Failed to cast to FilterNode");
        // }

        // const TypedExprPtr& filterExpr = myFilterNode->filter();

        // std::string expr = filterExpr->toString();
        // // For this rule, we only check wheather decision_forest_predict UDF
        // // function is in expressions
        // std::regex pattern(R"(decision_forest_predict)");

        // auto wordsBegin =
        //     std::sregex_iterator(expr.begin(), expr.end(), pattern);

        // auto wordsEnd = std::sregex_iterator();
        // // Find applicable UDF name and store in targetActions
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
  RowVectorPtr treeRowVector;
  static inline int rewriteDecisionForestCounter = 0;
};

} // namespace optimization
