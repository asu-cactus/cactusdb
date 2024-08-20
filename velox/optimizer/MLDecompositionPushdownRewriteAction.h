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

  std::string findPushdownNodeId(
      std::shared_ptr<const core::PlanNode> curNode,
      std::string expr,
      std::vector<std::string> exprSources,
      std::string rootNodeId,
      std::string& finalPushdownNodeId) {
    std::string pushdownNodeId;
    // std::string pushdownNodeId = curNode->id();
    std::string curNodeId = curNode->id();
    std::string_view curNodeName = curNode->name();
    std::vector<std::shared_ptr<const core::PlanNode>> sources =
        curNode->sources();
    // try to search the target expression to find the pushdown node
    if (rootNodeId != curNodeId) {
      std::vector<TypedExprPtr> expressions;
      std::vector<std::string> expressionAlias;
      // obtain expressions
      if (curNodeName == "Project") {
        auto myProjectNode =
            std::dynamic_pointer_cast<const ProjectNode>(curNode);
        expressions = myProjectNode->projections();
        expressionAlias = myProjectNode->names();

        // need to check all expression sources are presented
        int numSourcesPresented = 0;
        for (auto exprSrc : exprSources) {
          for (int i = 0; i < expressionAlias.size(); i++) {
            if (expressionAlias[i] == exprSrc) {
              numSourcesPresented += 1;
            }
          }
        }

        if (numSourcesPresented == exprSources.size()) {
          // std::cout << "[DEBUG] found a match, expr: " << expr
          //           << " pushdownNodeId: " << curNodeId
          //           << " expressionAlias: " << expressionAlias << std::endl;
          pushdownNodeId = curNodeId;
        }
      } else {
        // Only do pushdown for project node
      }
    }
    for (const auto& source : sources) {
      std::string returnedPushdownNodeId = findPushdownNodeId(
          source, expr, exprSources, rootNodeId, finalPushdownNodeId);
      // Only the pushdownable node through a JOIN node will be added as a valid
      // node to the vector
      if (curNodeName.find("Join") != std::string::npos &&
          returnedPushdownNodeId != "") {
        finalPushdownNodeId = returnedPushdownNodeId;
      } else {
        pushdownNodeId = pushdownNodeId;
      }
    }

    return pushdownNodeId;
  }

  bool mayPushdown(
      std::shared_ptr<const core::PlanNode> curNode,
      std::string expr,
      std::vector<std::string> exprSources,
      std::string rootNodeId) {
    std::string finalPushdownNodeId;
    findPushdownNodeId(
        curNode, expr, exprSources, rootNodeId, finalPushdownNodeId);
    // std::cout << "[DEBUG] finalPushdownNodeId: " << finalPushdownNodeId
    //           << std::endl;
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
        // The pushdown rule is only applicable to Project node
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
          // Flag indicates whether the target UDF is found
          bool findRewriteTarget = false;
          bool isPartialPushDown = true;
          // The pushdown udf could exist in multiple expressions,
          // so we need to store the pushdown node id for each expression
          // std::vector<std::string> pushdownNodeIds;

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

              findPushdownNodeId(
                  curNode,
                  exprStr,
                  matchedDataSources,
                  curNode->id(),
                  finalPushdownNodeId);

              // find applicable pushdown Node
              if (finalPushdownNodeId != "") {
                // std::cout << "[DEBUG] found pushdownNodeId: "
                //           << finalPushdownNodeId << std::endl;

                // get the candidateNode
                pushdownPlanNode =
                    findPlanNodeById(curNode, finalPushdownNodeId);
                // it should be found
                assert(pushdownPlanNode);
                
                pushdownResultName = targetExprName;

                // get the expressions of the pushdown Node
                auto pushdownProjectNode =
                    std::dynamic_pointer_cast<const ProjectNode>(
                        pushdownPlanNode);
                assert(pushdownProjectNode);


                // Get the names of projections
               pushdownNodeProjectionsNames =
                    pushdownProjectNode->names();

                for (auto pushdownName : pushdownNodeProjectionsNames) {
                  pushdownProjectExprSets.insert(pushdownName);
                }

                // Parse the target pushdown expressions
                std::string pushDownExpression = target;
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
                  // pushDownExpression = escapeRegex(pushDownExpression);
                }
                pushDownExpression = replaceDoubleQuotes(pushDownExpression);

                pushDownExpression =
                    pushDownExpression + " AS " + pushdownResultName;
                pushdownProjectExprSets.insert(pushDownExpression);

                // extract the computation after target rewrite
                // UDF
                std::string escapedRegex = escapeRegex(target);
                std::regex patternOfRewriteFinalExpr(escapedRegex);
                auto targetExprStr = std::regex_replace(
                    exprStr, patternOfRewriteFinalExpr, pushdownResultName);

                // set flag to false if the whole expression is pushed down
                if (targetExprStr == pushdownResultName) {
                  // isPartialPushDown = false;
                  targetProjectExprSets.insert(targetExprName);
                } else {
                  targetProjectExprSets.insert(
                      targetExprStr + " AS " + targetExprName);
                }

                // std::cout << "[DEBUG] target expression: nodeName: " << nodeName
                //           << " rewriteExpr: "
                //           << targetExprStr + " AS " + targetExprName
                //           << std::endl;
                findRewriteTarget = true;
              } else {
                throw std::invalid_argument(
                    "[Error] pushdown node not found for target expression: " +
                    target);
              }
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
              }
              rewriteExpr = replaceDoubleQuotes(rewriteExpr);

              targetProjectExprSets.insert(
                  rewriteExpr + " AS " + projectionsNames[exprIdx]);

              // std::cout << "[DEBUG] non-target expression: nodeName: "
              //           << nodeName << " rewriteExpr: "
              //           << rewriteExpr + " AS " + projectionsNames[exprIdx]
              //           << std::endl;
            }
          }

          if (findRewriteTarget) {
            std::vector<std::string> pushdownProjectExprs(
                pushdownProjectExprSets.begin(), pushdownProjectExprSets.end());

            std::vector<std::string> pushdownNodeProjectionsNameCopy = pushdownNodeProjectionsNames;
            pushdownNodeProjectionsNameCopy.push_back(pushdownResultName);
            
            // std::cout << "[DEBUG]pushdownProjectExprs: " 
            //           << pushdownProjectExprs << std::endl;

            // std::cout << "[DEBUG] pushdownNodeProjectionsNameCopy: "
            //           << pushdownNodeProjectionsNameCopy << std::endl;

            // create a project node after pushdown node
            auto pushdownNodePlanBuilder =
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setRoot(pushdownPlanNode)
                    .project({pushdownProjectExprs})
                    // Add a helper project node. The reason is using to capture the filed in 
                    // serialized plan in a more convenient way, the following node will be 
                    // removed after capture it.
                    .project({pushdownNodeProjectionsNameCopy}); 

            // std::cout << "[DEBUG] pushdownNodePlanBuilder: "
            //           << pushdownNodePlanBuilder.planNode()->toString(
            //                  true, true)
            //           << std::endl;


            auto serializedPushdownPlan =
                pushdownNodePlanBuilder.planNode()->serialize();

            // Got the filed
            folly::dynamic pushdownExprFiled = folly::dynamic::object;
            for (auto& field : serializedPushdownPlan["projections"]) {
              if (field["fieldName"] == pushdownResultName) {
                pushdownExprFiled = field;
                break;
              }
            }

            // std::cout << "[DEBUG] pushdownExprFiled: "
            //           << pushdownExprFiled
            //           << std::endl;

            assert(!pushdownExprFiled.empty());
            // remove the last helper node
            pushdownNodePlanBuilder = pushdownNodePlanBuilder.setRoot(pushdownNodePlanBuilder.planNode()->sources()[0]);
            serializedPushdownPlan = pushdownNodePlanBuilder.planNode()->serialize();

            // auto newPushdownNodeId = pushdownNodePlanBuilder.planNode()->id();
            // .project({"pushdown_0", "user_id", "user_description"});
            // std::cout << "[DEBUG] pushdownNodePlanBuilder: "
            //           << pushdownNodePlanBuilder.planNode()->toString(
            //                  true, true)
            //           << std::endl;

            
            auto serializedPlan = planBuilder.planNode()->serialize();
            replaceSourceWithIdInSerializedPlan(
                serializedPlan, serializedPushdownPlan, finalPushdownNodeId);
            auto deserlizedUpdatedPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());

            // std::cout << "[DEBUG] query plan after 1st changes: "
            //           << deserlizedUpdatedPlanNode->toString(true, true)
            //           << std::endl;

            // std::cout << "[DEBUG] seralized plan after 1st changes:\n"
            //           << serializedPlan
            //           << std::endl;

            auto curNodeInUpdatePlan =
                    findPlanNodeById(deserlizedUpdatedPlanNode, curNode->id());

            std::vector<std::string> nodeIdsBetweenSourceAndTarget;
            findNodeIdsBetweenIds(
                curNodeInUpdatePlan,
                pushdownNodePlanBuilder.planNode()->id(),
                curNodeInUpdatePlan->id(),
                nodeIdsBetweenSourceAndTarget);
            // std::cout << "[DEBUG] src Node Id: " << pushdownNodePlanBuilder.planNode()->id()
            //           << " target Node Id: " << curNodeInUpdatePlan->id() << std::endl;

            // std::cout << "[DEBUG] nodeIdsBetweenSourceAndTarget: "
            //           << nodeIdsBetweenSourceAndTarget << std::endl;

            

            // std::cout << "[DEBUG] pushdownExprFiled: "
            //           << pushdownExprFiled
            //           << std::endl;

            // std::cout << "[DEBUG] query plan before add filed: "
            //           << serializedPlan
            //           << std::endl;

            addProjectionFiledInSerializedPlan(serializedPlan, pushdownExprFiled, nodeIdsBetweenSourceAndTarget);

            // std::cout << "[DEBUG] query plan after add filed: "
            //           << serializedPlan
            //           << std::endl;

            deserlizedUpdatedPlanNode =
                ISerializable::deserialize<core::PlanNode>(
                    serializedPlan, pool_.get());
            
            // std::cout << "[DEBUG] success of add filed" << std::endl;

            // if (isPartialPushDown) {
              // if it is partially pushdown, we need to create a new project
              // node to replace the curNode in new plan to finish the computation
              // after the pushdown expression
              std::string curNodeId = curNode->id();
              auto curNodeInUpdatedPlan =
                      findPlanNodeById(deserlizedUpdatedPlanNode, curNodeId);
              
              std::vector<std::string> targetProjectExprs(
                  targetProjectExprSets.begin(), targetProjectExprSets.end());

              // std::cout << "[DEBUG] findRewriteTarget: " << findRewriteTarget
              //           << std::endl;
              // std::cout << "[DEBUG] pushdownProjectExprSets: "
              //           << pushdownProjectExprSets << std::endl;
              // std::cout << "[DEBUG] targetProjectExprSets: "
              //           << targetProjectExprSets << std::endl;

              auto rewritePlan =
                    exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .setRoot(curNodeInUpdatedPlan->sources()[0])
                    .project(targetProjectExprs);

              // std::cout << "[DEBUG] rewritePlan: "
              //           << rewritePlan.planNode()->toString(true, true)
              //           << std::endl;
              
              auto serializedNewSource =
                  rewritePlan.planNode()->serialize();
              
              replaceSourceWithIdInSerializedPlan(
                  serializedPlan, serializedNewSource, curNodeId);
              
              auto deserlizedFinalPlanNode =
                      ISerializable::deserialize<core::PlanNode>(
                          serializedPlan, pool_.get());
              planBuilder.setRoot(deserlizedFinalPlanNode);
            // } else {
            //   planBuilder.setRoot(deserlizedUpdatedPlanNode);
            // }
            // std::cout << "[INFO] final serialized query plan: \n" << serializedPlan << std::endl;

            // std::cout << "[DEBUG] query plan after add filed: "
            //           << deserlizedUpdatedPlanNode->toString(true, true)
            //           << std::endl;

            // std::cout << "[debug] captured filed: " << pushdownExprFiled
            //           << std::endl;
            // folly::dynamic&

            // std::cout << "[DEBUG] final plan: \n"
            //           << planBuilder.planNode()->toString(true,true) << std::endl;

            
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
      std::string curNodeId = rootNode->id();

      // Search each expressions
      for (const auto& expression : expressions) {
        // Note: current implementation support pushdown starting from the
        // innermost function Example:
        // Support: |<----Pushdown:--->|
        // softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))
        // Not-Support:    |<----Pushdown:---->|
        // softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))
        std::string expr = expression->toString();
        // std::cout << "expr: " << expr << std::endl;
        std::vector<std::string> parsedSingleExprs;
        std::vector<std::string> matchedExprs;
        parseDLExpressions(expr, parsedSingleExprs, matchedExprs);
        // reverse to get the innermost function first
        std::reverse(parsedSingleExprs.begin(), parsedSingleExprs.end());
        std::reverse(matchedExprs.begin(), matchedExprs.end());
        std::string targetExprStr;

        std::vector<std::string> matchedDataSources = findDataSrcFromExpr(expr);

        for (int i = 0; i < parsedSingleExprs.size(); i++) {
          targetExprStr = matchedExprs[i];
          // std::cout << "reached here: " << targetExprStr
          //           << " matchedDataSources: " << matchedDataSources
          //           << " curNodeId: " << curNodeId << std::endl;
          if (mayPushdown(
                  rootNode, targetExprStr, matchedDataSources, curNodeId)) {
            targetActions.push_back(targetExprStr);
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