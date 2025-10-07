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

class MLFactorizationRewriteAction : public RewriteAction {
 public:
  MLFactorizationRewriteAction() {}

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

  std::string findPushdownNodeIds(
      std::shared_ptr<const core::PlanNode> curNode,
      std::string expr,
      std::vector<std::string> exprSources,
      std::string rootNodeId,
      std::vector<std::string>& pushdownCandidateNodeIds) {
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
      // std::cout << "[DEBUG]: curNodeName: " << curNodeName
      // << " curNodeId: " << curNodeId << std::endl;
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
        }
      } else if (curNodeName.find("Aggregation") != std::string::npos) {
        // do not the pushdown through aggregation node
        return "";
      } else if (curNodeName.find("Values") != std::string::npos) {
        auto myValuesNode =
            std::dynamic_pointer_cast<const ValuesNode>(curNode);
        auto values = myValuesNode->values();
        std::vector<std::string> valuesNames;
        // obtain all value names
        for (auto val : values) {
          auto rowVector = std::dynamic_pointer_cast<const RowVector>(val);
          if (rowVector) {
            auto rowType =
                std::dynamic_pointer_cast<const RowType>(rowVector->type());

            for (int i = 0; i < rowVector->childrenSize(); ++i) {
              auto child = rowVector->childAt(i);
              auto childName = rowType->nameOf(i);
              valuesNames.push_back(childName);
            }
          }
        }

        int numSourcesPresented = 0;
        for (auto exprSrc : exprSources) {
          for (int i = 0; i < valuesNames.size(); i++) {
            if (valuesNames[i] == exprSrc) {
              numSourcesPresented += 1;
            }
          }
        }

        if (numSourcesPresented == exprSources.size()) {
          // if all sources are presented, then assign the pushdownNodeId
          pushdownNodeId = curNodeId;
        }

      } else {
        // Since the pushdown is applied by creating a new project node after
        // the pushdown node, and the expression alias are not available in the
        // filter node, we only need to find the project node
      }
    }

    // traverse the source nodes to find the lowest pushdown node
    for (const auto& source : sources) {
      std::string returnedPushdownNodeId = findPushdownNodeIds(
          source, expr, exprSources, rootNodeId, pushdownCandidateNodeIds);
      // A pushdown through a JOIN node is considered as a valid pushdown
      // update the finalPushdownNodeId
      if (curNodeName.find("Join") != std::string::npos &&
          returnedPushdownNodeId != "") {
        // add all the pushdown nodeId though the join to the
        // pushdownCandidateNodeIds
        pushdownCandidateNodeIds.push_back(returnedPushdownNodeId);
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
    // std::string finalPushdownNodeId;
    std::vector<std::string> pushdownCandidateNodeIds;
    findPushdownNodeIds(
        curNode, expr, exprSources, rootNodeId, pushdownCandidateNodeIds);
    return pushdownCandidateNodeIds.size() != 0;
  }

  bool isFactorizableExpr(std::string udfExpr) {
    if (udfExpr.find("mat_mul") != std::string::npos) {
      return true;
    } else if (udfExpr.find("mat_vector_add") != std::string::npos) {
      return true;
    } else {
      return false;
    }
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

    /* Note: current factorization and pushdown is relied on the other query
    rewriter to generate the query plan

    */

    return transformationApplied;
  }

  /**
   * @brief A function to get the name of rewritten rule.
   *
   * @return A string value denoting the name of the rule
   */
  std::string name() override {
    return "MLFactorizationRewriteAction";
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
      // std::cout << "MLFactorzationRewriteAction: check function called"
      // << std::endl;
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

        // std::cout << "[DEBUG] check: parsedSingleExprs: " <<
        // parsedSingleExprs
        //           << std::endl;
        // std::cout << "[DEBUG] check: matchedExprs: " << matchedExprs
        //           << std::endl;
        // std::cout << "[BUDEG] check: matchedDataSources: " <<
        // matchedDataSources
        //           << std::endl;

        if (matchedDataSources.size() == 1) {
          // Whole group can be pushed down
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
            std::cout << "[DEBUG] check: targetExprStr: " << targetExprStr
                      << std::endl;
            if (mayPushdown(
                    rootNode, targetExprStr, matchedDataSources, curNodeId)) {
              targetActions.push_back(targetExprStr);
              // If found a pushdown target, then break the loop, since the
              // afterward expressions are included in the current iterated
              // expression
              break;
            }
          }

        } else if (matchedDataSources.size() > 1) {
          std::string factorizableExpr;
          bool isFactorizable = false;
          // Multiple data sources, need to check if all of them are
          // Starting from the outer most UDF
          auto lastUDFExpr = parsedSingleExprs.back();
          if (lastUDFExpr.find("concat") != std::string::npos) {
            // If the last UDF is concat, then we can further check
            for (int i = parsedSingleExprs.size() - 2; i >= 0; i--) {
              auto expr = parsedSingleExprs[i];
              if (isFactorizableExpr(expr)) {
                factorizableExpr = expr;
                break;
              }
            }
          }
          // std::cout << "[DEBUG] found one factorizableExpr " <<
          // factorizableExpr
          // << std::endl;
          if (factorizableExpr != "") {
            // find pushdownable node
            for (auto dataSrc : matchedDataSources) {
              std::vector<std::string> pushdownCandidateNodeIds;
              findPushdownNodeIds(
                  rootNode,
                  expr,
                  {dataSrc},
                  curNodeId,
                  pushdownCandidateNodeIds);
              // std::cout << "[DEBUG] given: factorizableExpr: "
              // << factorizableExpr << " pushdownCandidateNodeIds: "
              // << pushdownCandidateNodeIds << std::endl;
              if (pushdownCandidateNodeIds.size() != 0) {
                cataLog.addFactorizableSrcPushdownNodes(
                    dataSrc, pushdownCandidateNodeIds);
              }
            }
            cataLog.addFactorizableOpSrc(factorizableExpr, matchedDataSources);
          }

        } else {
          // No data source found, skip this expression
          // TODO: need further investigate the case
          std::cout << "[DEBUG] check: no data source found, skip this "
                       "expression: "
                    << expr << std::endl;
          continue;
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