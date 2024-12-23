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
// using namespace optimization;

namespace optimization {

class MLDecompositionPushdownRewriteAction : public RewriteAction {
 public:
  MLDecompositionPushdownRewriteAction() {}

  void clearVectors() {}

  /**
   * @brief A function to recursively search the plan to find the pushdown Node
   * for the target expr
   *
   * @param curNode A pointer to the current plan node
   * @param expr A string representing the target expression to be pushed down
   * @param exprSources A vector of strings representing the data sources of the
   * expr
   * @param rootNodeId A string representing the nodeId of the expr node
   * @param finalPushdownNodeId A string reference to store the final pushdown
   * node id
   *
   */

  std::string findPushdownNodeId(
      std::shared_ptr<const core::PlanNode> curNode,
      std::string expr,
      std::vector<std::string> exprSources,
      std::string rootNodeId,
      std::string& finalPushdownNodeId) {
    std::string pushdownNodeId;
    std::string curNodeId = curNode->id();
    std::string_view curNodeName = curNode->name();
    std::vector<std::shared_ptr<const core::PlanNode>> sources =
        curNode->sources();
    // try to search the target expression to find the pushdown node
    if (rootNodeId != curNodeId) {
      // variable to store the expressions and alias
      std::vector<TypedExprPtr> expressions;
      std::vector<std::string> expressionAlias;
      if (curNodeName == "Project") {
        // cast to project node
        auto myProjectNode =
            std::dynamic_pointer_cast<const ProjectNode>(curNode);
        expressions = myProjectNode->projections();
        expressionAlias = myProjectNode->names();

        // check: all expression sources are presented, otherwise the expression
        // cannot be pushdown
        int numSourcesPresented = 0;
        for (auto exprSrc : exprSources) {
          for (int i = 0; i < expressionAlias.size(); i++) {
            if (expressionAlias[i] == exprSrc) {
              numSourcesPresented += 1;
            }
          }
        }

        if (numSourcesPresented == exprSources.size()) {
          // if all sources are presented, then assign the pushdownNodeId
          pushdownNodeId = curNodeId;
          // std::cout << "[DEBUG] found a match, expr: " << expr
          //           << " pushdownNodeId: " << curNodeId
          //           << " expressionAlias: " << expressionAlias << std::endl;
        }
      } else {
        // Since the pushdown is applied by creating a new project node after
        // the pushdown node, and the expression alias are not available in the
        // filter node, we only need to find the project node
      }
    }

    // traverse the source nodes to find the lowest pushdown node
    for (const auto& source : sources) {
      std::string returnedPushdownNodeId = findPushdownNodeId(
          source, expr, exprSources, rootNodeId, finalPushdownNodeId);
      // A pushdown through a JOIN node is considered as a valid pushdown
      // update the finalPushdownNodeId
      if (curNodeName.find("Join") != std::string::npos &&
          returnedPushdownNodeId != "") {
        finalPushdownNodeId = returnedPushdownNodeId;
      } else {
        // If find a expr can be further pushed down, then update the
        // pushdownNodeId
        if (returnedPushdownNodeId != "") {
          pushdownNodeId = returnedPushdownNodeId;
        }
      }
    }

    return pushdownNodeId;
  }

  /**
   * @brief A function to check if the target expression can be pushed down
   *
   * @param curNode A pointer to the current plan node
   * @param expr A string representing the target expression to be pushed down
   * @param exprSources A vector of strings representing the data sources of the
   * expr
   * @param rootNodeId A string representing the nodeId of the expr node
   */
  bool mayPushdown(
      std::shared_ptr<const core::PlanNode> curNode,
      std::string expr,
      std::vector<std::string> exprSources,
      std::string rootNodeId) {
    std::string finalPushdownNodeId;
    findPushdownNodeId(
        curNode, expr, exprSources, rootNodeId, finalPushdownNodeId);
    return finalPushdownNodeId != "";
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

    // The set to store the target project node expression
    std::set<std::string> pushdownProjectExprSets;
    std::set<std::string> targetProjectExprSets;
    std::map<std::string, std::shared_ptr<const core::PlanNode>>
        pushdownNodesMap;
    // Iterate over each target in the targets container
    for (auto target : targets) {
      // Start from the current node
      if (curNode) {
        // Get the name of node
        std::string_view nodeName = curNode->name();
        // The pushdown rule is applicable to Project node and Filter node
        if (nodeName == "Project") {
          // Cast node as project node
          auto myProjectNode =
              std::dynamic_pointer_cast<const ProjectNode>(curNode);
          assert(myProjectNode);

          // Get projections in project node
          const std::vector<TypedExprPtr>& projections =
              myProjectNode->projections();
          // Get the names of projections
          const std::vector<std::string>& projectionsNames =
              myProjectNode->names();
          int numProjections = projections.size();
          // Check if the number of projections is equal to the number
          // of projection names
          assert(numProjections == projectionsNames.size());
          // Get the names of projections
          std::vector<std::string> pushdownNodeProjectionsNames;
          // Flag: indicates whether the target UDF is found
          bool findRewriteTarget = false;

          std::string finalPushdownNodeId;
          std::shared_ptr<const core::PlanNode> pushdownPlanNode;
          std::string pushdownResultName;
          // Iterate each projection
          for (int exprIdx = 0; exprIdx < numProjections; exprIdx++) {
            auto expression = projections[exprIdx];
            // Get the string of expression
            std::string exprStr = expression->toString();
            std::string targetExprName = projectionsNames[exprIdx];
            // Check if target exist in the expression
            if (exprStr.find(target) != std::string::npos) {
              // Capture the data src
              std::vector<string> matchedDataSources =
                  findDataSrcFromExpr(exprStr);

              // Find the pushdown node for the target expression
              findPushdownNodeId(
                  curNode,
                  exprStr,
                  matchedDataSources,
                  curNode->id(),
                  finalPushdownNodeId);

              if (finalPushdownNodeId != "") {
                // The expr can be pushed down

                // std::cout << "[DEBUG] found pushdownNodeId: "
                //           << finalPushdownNodeId << std::endl;

                // Get the candidate pushdown node via nodeId
                pushdownPlanNode =
                    findPlanNodeById(curNode, finalPushdownNodeId);
                // Check: node if found
                assert(pushdownPlanNode);
                // Store the name of the pushdowned expression
                pushdownResultName = targetExprName;

                // Get the expressions of the pushdown Node
                auto pushdownProjectNode =
                    std::dynamic_pointer_cast<const ProjectNode>(
                        pushdownPlanNode);
                assert(pushdownProjectNode);

                // Get the names of projections
                pushdownNodeProjectionsNames = pushdownProjectNode->names();

                for (auto pushdownName : pushdownNodeProjectionsNames) {
                  pushdownProjectExprSets.insert(pushdownName);
                }

                // Parse the target pushdown expressions and extract the data
                // sources the expression is written as
                // mat_mul0(ROW["features"]), we need to extract the data source
                // from the expression via regex
                std::string pushDownExpression = target;
                
                // rewrite lambda function if it exists
                pushDownExpression = rewriteLambdaInExpStr(pushDownExpression);

                std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
                std::smatch matches;
                // Start position for the search
                std::string::const_iterator searchStart(target.cbegin());
                int rewriteSrcIdx = 0;
                // Search out the matched data source and store in matches
                while (std::regex_search(
                    searchStart,
                    target.cend(),
                    matches,
                    patternToMatchRawSource)) {
                  auto matchedDataSrc = matches[1].str();
                  std::regex patternOfReplaceExpr(
                      escapeRegex(matches[0].str()));
                  pushDownExpression = std::regex_replace(
                      pushDownExpression, patternOfReplaceExpr, matchedDataSrc);
                  // Update the search start position
                  searchStart = matches.suffix().first;
                }

                // Replace the double quotes with single quotes
                pushDownExpression = replaceDoubleQuotes(pushDownExpression);

                // Add the alias to the expression
                pushDownExpression =
                    pushDownExpression + " AS " + pushdownResultName;
                pushdownProjectExprSets.insert(pushDownExpression);

                // There is a case the whole expression is partially pushed down
                // For example: mat_mul2(mat_mul1(in1), in2), only mat_mul1(in1)
                // can be pushed down, we need to process the rest part
                // extract the computation after target rewrite
                // UDF
                std::string escapedRegex = escapeRegex(target);
                std::regex patternOfRewriteFinalExpr(escapedRegex);
                auto targetExprStr = std::regex_replace(
                    exprStr, patternOfRewriteFinalExpr, pushdownResultName);

                // set flag to false if the whole expression is pushed down
                if (targetExprStr == pushdownResultName) {
                  targetProjectExprSets.insert(targetExprName);
                } else {
                  targetProjectExprSets.insert(
                      targetExprStr + " AS " + targetExprName);
                }

                LOG(INFO) << "[INFO] target expression: nodeName: " << nodeName
                          << " rewriteExpr: "
                          << targetExprStr + " AS " + targetExprName
                          << std::endl;
                findRewriteTarget = true;
              } else {
                throw std::invalid_argument(
                    "[Error] pushdown node not found for target expression: " +
                    target);
              }
            } else {
              // Parse the non-target expressions
              auto rewriteExpr = exprStr;
              // rewrite lambda function if it exists
              rewriteExpr = rewriteLambdaInExpStr(rewriteExpr);

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
              }
              rewriteExpr = replaceDoubleQuotes(rewriteExpr);

              targetProjectExprSets.insert(
                  rewriteExpr + " AS " + projectionsNames[exprIdx]);

              LOG(INFO) << "[INFO] non-target expression: nodeName: "
                        << nodeName << " rewriteExpr: "
                        << rewriteExpr + " AS " + projectionsNames[exprIdx]
                        << std::endl;
            }
          }

          if (findRewriteTarget) {
            // if find the rewrite target, start to rewrite the query plan
            std::vector<std::string> pushdownProjectExprs(
                pushdownProjectExprSets.begin(), pushdownProjectExprSets.end());

            std::vector<std::string> pushdownNodeProjectionsNameCopy =
                pushdownNodeProjectionsNames;
            pushdownNodeProjectionsNameCopy.push_back(pushdownResultName);

            // To ensure the pushdowned expression is carried through its
            // original node, it must be added to the nodes in between.
            // An easier approach is to add the expression to its serialized
            // format and then deserialize it. However, we need to capture the
            // serialized format of the newly added expression. Therefore, we
            // create a helper project node to facilitate this capturing.

            // Create a project node after pushdown node
            auto pushdownNodePlanBuilder =
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setRoot(pushdownPlanNode)
                    .project({pushdownProjectExprs})
                    // Add a temporary project node to capture the serialized
                    // format of the newly added expression. This node will be
                    // removed after capturing the necessary information.
                    .project({pushdownNodeProjectionsNameCopy});

            // std::cout << "[DEBUG] pushdownNodePlanBuilder: "
            //           << pushdownNodePlanBuilder.planNode()->toString(
            //                  true, true)
            //           << std::endl;

            auto serializedPushdownPlan =
                pushdownNodePlanBuilder.planNode()->serialize();

            // Got the filed in folly::dynamic
            folly::dynamic pushdownExprFiled = folly::dynamic::object;
            for (auto& field : serializedPushdownPlan["projections"]) {
              if (field["fieldName"] == pushdownResultName) {
                pushdownExprFiled = field;
                break;
              }
            }

            LOG(INFO) << "[INFO] pushdownExprFiled: " << pushdownExprFiled
                      << std::endl;

            assert(!pushdownExprFiled.empty());
            // remove the last helper node
            pushdownNodePlanBuilder = pushdownNodePlanBuilder.setRoot(
                pushdownNodePlanBuilder.planNode()->sources()[0]);
            serializedPushdownPlan =
                pushdownNodePlanBuilder.planNode()->serialize();

            auto serializedPlan = planBuilder.planNode()->serialize();
            // update the plan by replacing the source node with the new
            // pushdown node
            replaceSourceWithIdInSerializedPlan(
                serializedPlan, serializedPushdownPlan, finalPushdownNodeId);
            auto deserlizedUpdatedPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());

            LOG(INFO) << "[INFO] query plan after 1st changes: "
                      << deserlizedUpdatedPlanNode->toString(true, true)
                      << std::endl;

            auto curNodeInUpdatePlan =
                findPlanNodeById(deserlizedUpdatedPlanNode, curNode->id());

            std::vector<std::string> nodeIdsBetweenSourceAndTarget;
            findNodeIdsBetweenIds(
                curNodeInUpdatePlan,
                pushdownNodePlanBuilder.planNode()->id(),
                curNodeInUpdatePlan->id(),
                nodeIdsBetweenSourceAndTarget);

            LOG(INFO) << "[INFO] nodeIdsBetweenSourceAndTarget: "
                      << nodeIdsBetweenSourceAndTarget << std::endl;
            addProjectionFiledInSerializedPlan(
                serializedPlan,
                pushdownExprFiled,
                nodeIdsBetweenSourceAndTarget);

            deserlizedUpdatedPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());

            std::string curNodeId = curNode->id();
            auto curNodeInUpdatedPlan =
                findPlanNodeById(deserlizedUpdatedPlanNode, curNodeId);

            std::vector<std::string> targetProjectExprs(
                targetProjectExprSets.begin(), targetProjectExprSets.end());

            auto rewritePlan =
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setRoot(curNodeInUpdatedPlan->sources()[0])
                    .project(targetProjectExprs);

            auto serializedNewSource = rewritePlan.planNode()->serialize();

            replaceSourceWithIdInSerializedPlan(
                serializedPlan, serializedNewSource, curNodeId);

            auto deserlizedFinalPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());
            planBuilder.setRoot(deserlizedFinalPlanNode);
          }
        } else if (nodeName == "Filter") {
          auto myFilterNode =
              std::dynamic_pointer_cast<const FilterNode>(curNode);

          if (!myFilterNode) {
            throw std::runtime_error("Failed to cast to FilterNode");
          }

          // Each filter node has only one expression
          auto filterExpression = myFilterNode->filter();
          std::string exprStr = filterExpression->toString();
          std::string finalPushdownNodeId;
          std::shared_ptr<const core::PlanNode> pushdownPlanNode;
          if (exprStr.find(target) != std::string::npos) {
            // Capture the data src
            std::vector<string> matchedDataSources =
                findDataSrcFromExpr(exprStr);

            findPushdownNodeId(
                curNode,
                exprStr,
                matchedDataSources,
                curNode->id(),
                finalPushdownNodeId);
            if (finalPushdownNodeId != "") {
              // get the candidateNode
              pushdownPlanNode = findPlanNodeById(curNode, finalPushdownNodeId);
              // it should be found
              assert(pushdownPlanNode);
            }

            // Parse the target pushdown expressions to make it can be
            // recognized by Velox by removing ROW
            std::string pushDownExpression = target;
            std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
            std::smatch matches;
            // Start position for the search
            std::string::const_iterator searchStart(target.cbegin());
            int rewriteSrcIdx = 0;
            // Search out the matched data source and store in matches
            while (std::regex_search(
                searchStart, target.cend(), matches, patternToMatchRawSource)) {
              auto matchedDataSrc = matches[1].str();
              std::regex patternOfReplaceExpr(escapeRegex(matches[0].str()));
              pushDownExpression = std::regex_replace(
                  pushDownExpression, patternOfReplaceExpr, matchedDataSrc);
              // Update the search start position
              searchStart = matches.suffix().first;
              // pushDownExpression = escapeRegex(pushDownExpression);
            }

            // Invoke the cast function to fix the parsing issue
            if (pushDownExpression.find("cast") != std::string::npos) {
              // pushDownExpression =
              //     fix_cast_function_parsing(pushDownExpression);
              // Reformat filter expression
              pushDownExpression = reformatComparisonExpr(pushDownExpression);
              // std::cout << "[DEBUG] pushDownExpression: " <<
              // pushDownExpression
              //           << std::endl;
            }
            pushDownExpression = replaceDoubleQuotes(pushDownExpression);

            // Add a filter node after the pushdown node
            auto pushdownNodePlanBuilder =
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setRoot(pushdownPlanNode)
                    .filter(pushDownExpression);

            auto serializedPushdownPlan =
                pushdownNodePlanBuilder.planNode()->serialize();
            auto serializedPlan = planBuilder.planNode()->serialize();
            replaceSourceWithIdInSerializedPlan(
                serializedPlan, serializedPushdownPlan, finalPushdownNodeId);

            auto deserlizedUpdatedPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());

            auto curNodeInUpdatedPlan =
                findPlanNodeById(deserlizedUpdatedPlanNode, curNode->id());
            auto rewritePlanByRemovePushdownedFilter =
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setRoot(curNodeInUpdatedPlan->sources()[0]);
            auto serializedNewSource =
                rewritePlanByRemovePushdownedFilter.planNode()->serialize();
            replaceSourceWithIdInSerializedPlan(
                serializedPlan, serializedNewSource, curNode->id());

            auto deserlizedFinalPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());
            planBuilder.setRoot(deserlizedFinalPlanNode);
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
    return "MLDecompositionPushdownRewriteAction";
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
      // Currently we support Project and Filter nodes pushdown,
      // need to use different cast for different node types to
      // obtain the expressions
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
      std::string curNodeId = rootNode->id();

      // Search each expressions
      for (const auto& expression : expressions) {
        // Note: current implementation support pushdown starting from the
        // innermost function Example:
        // Support: |<-----------Pushdown:------------>|
        // softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))
        // Not-Support:    |<----Pushdown:---->|
        // softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))
        std::string expr = expression->toString();
        // Note: the ordering of stored expression is starting from including
        // all UDFs then excluding the outermost UDFs. Example of matchedExprs:
        // {softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))),
        // mat_add(mat_mul(relu(mat_add(mat_mul(v)))),
        // mat_mul(relu(mat_add(mat_mul(v)))}
        std::vector<std::string> parsedSingleExprs;
        std::vector<std::string> matchedExprs;
        parseDLExpressions(expr, parsedSingleExprs, matchedExprs);
        std::string targetExprStr;

        std::vector<std::string> matchedDataSources = findDataSrcFromExpr(expr);

        for (int i = 0; i < parsedSingleExprs.size(); i++) {
          targetExprStr = matchedExprs[i];
          if (targetExprStr.find("_partial_agg") != std::string::npos) {
            // Edge case: when converting  the MatMul to relational approach,
            // its intermediate aggregation name is following the pattern
            // "[Table_Source]_partial_agg[Number]", and it will later be used
            // in the following computation, actually it is not a pushdown
            // target, so we need to skip it.
            continue;
          }
          if (mayPushdown(
                  rootNode, targetExprStr, matchedDataSources, curNodeId)) {
            targetActions.push_back(targetExprStr);
            // If found a pushdown target, then break the loop, since the
            // afterward expressions are included in the current iterated
            // expression
            break;
          }
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
  std::string targetExprStr;
  std::vector<float*> weights;
  std::vector<float*> bias;
};

} // namespace optimization