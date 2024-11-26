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
#include "velox/optimizer/Helper.h"

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

namespace optimization {

class Mul2JoinAggHorizontalRewriteAction : public RewriteAction {
 public:
  Mul2JoinAggHorizontalRewriteAction() {}

  void clearVectors() {
    dims.clear();
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
    clearVectors();
    bool transformationApplied = false;
    // Initialize the sets for expressions at different stages
    // a matrix multiplication UDF will be rewritten to a join and aggregation
    // where nestedLoopProjectExprSets is the set of expressions for the
    // projection at the nested loop join, preComputeExprSets is the set of
    // expressions before the rewrite target mat_mul UDF, mulProjectExprSets is
    // the set of expressions for the projection at the block-wise matrix
    // multiplication, and finalProjectExprSets is the set of expressions for
    // UDF computation after the rewrite mat_mul UDF. Given a following example:
    // relu(mat_mul2(mat_mul1(v))), with the rewrite of mat_mul2, mat_mul1(v)
    // should be parsed and stored in preComputeExprSets, and relu will be
    // parsed and stored in the finalProjectExprSets. Meanwhile, if there are
    // multiple projections in the same node, these expressions will be stored
    // in the finalProjectExprSets. And their sources are stored in the other
    // exprSets.
    std::set<std::string> nestedLoopProjectExprSets;
    std::set<std::string> preComputeExprSets;
    std::set<std::string> mulProjectExprSets;
    // key and agg exprs for block-based MatMul aggregation
    std::set<std::string> matMulAggKeySets;
    std::set<std::string> matMulAggExprSets;
    std::set<std::string> finalProjectExprSets;
    // matched data source of the targeting MatMul
    std::string rewriteMatMulSrc;

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
            // Search each expression in projections
            int numProjections = projections.size();
            // Check if the number of projections is equal to the number of
            // projection names
            assert(numProjections == projectionsNames.size());
            // Flag indicates whether the target UDF is found
            bool findRewriteTarget = false;
            // Flag indicates whether there are computation before the target
            // UDF
            bool hasPrecomputeProject = false;
            // Iterate each projection
            for (int exprIdx = 0; exprIdx < numProjections; exprIdx++) {
              auto expression = projections[exprIdx];
              // Get the string of expression
              std::string exprStr = expression->toString();
              // std::cout << "[DEBUG] exprStr: " << exprStr << std::endl;

              // Check if target exist in the expression
              // filter out block weight
              if (exprStr.find(target) != std::string::npos &&
                  exprStr.find(target + "_wb") == std::string::npos) {
                // std::cout << "[debug] found matched str: " << exprStr << ","
                // << target << std::endl;

                targetExprStr = exprStr;
                core::QueryConfig config({});
                // Search for UDF functions by names
                std::shared_ptr<VectorFunction> myMul =
                    getVectorFunction(target, {ARRAY(REAL())}, {}, config);

                if (myMul) {
                  // Get the specific MatrixMultiply UDF
                  std::shared_ptr<MatrixMultiply> myMulUDF =
                      std::dynamic_pointer_cast<MatrixMultiply>(myMul);

                  if (myMulUDF) {
                    // Get the dimensions, weights from this MatrixMultiply UDF
                    dims = myMulUDF->getDims();
                    weights = myMulUDF->getTensor();
                    // Get information (defaultBlocksnumber, number of samples)
                    // from cataLog
                    int blocks = cataLog.getDefaultBlocksNum();
                    int blockSize = cataLog.getDefaultBlocksSize();

                    // Register matrix blocks multiply function
                    // the registered block-based mat_mul will be renamed as
                    // target_h
                    std::string blockMatMulName = fmt::format("{}_h", target);
                    registerVectorFunction(
                        blockMatMulName,
                        MatrixMultiply_h::signatures(),
                        std::make_unique<MatrixMultiply_h>(
                            dims[0], dims[1], cataLog.getDefaultBlocksSize()));
                    // Get the target expression name
                    std::string targetExprName = projectionsNames[exprIdx];
                    // retrieve the expression within the target expression
                    std::string exprsWithinTarget =
                        extractExprWithinTarget(exprStr, target);
                    // capture the data src
                    std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
                    std::smatch matches;
                    // object to capture the matched data source
                    std::string matchedDataSrc;
                    // Search out the matched data source and store in matches
                    if (std::regex_search(
                            exprsWithinTarget,
                            matches,
                            patternToMatchRawSource)) {
                      matchedDataSrc = matches[1].str();
                      rewriteMatMulSrc = matchedDataSrc;
                    } else {
                      LOG(ERROR) << "Uncaptured data source" << std::endl;
                    }

                    // the name of blocked weights follow the formatting:
                    // target_wb target_wb_row is the row id of the weight block
                    // target_wb_col is the column id of the weight block
                    std::string weightBlockName = target + "_wb";
                    nestedLoopProjectExprSets.insert(target + "_wb_row");
                    nestedLoopProjectExprSets.insert(target + "_wb_col");
                    preComputeExprSets.insert(target + "_wb_row");
                    preComputeExprSets.insert(target + "_wb_col");
                    mulProjectExprSets.insert(target + "_wb_col");
                    mulProjectExprSets.insert(target + "_wb_row");

                    nestedLoopProjectExprSets.insert(matchedDataSrc);
                    nestedLoopProjectExprSets.insert(weightBlockName);
                    preComputeExprSets.insert(weightBlockName);
                    // name the intermediate aggregation results from the data
                    // source name for example, the data_source name is
                    // user_features, the intermediate name will be
                    // user_features_partial_agg1
                    std::string parsedDataSrc =
                        splitString(matchedDataSrc, '_')[0];
                    std::string intermediateAggregationName = fmt::format(
                        "{}_partial_agg{}",
                        parsedDataSrc,
                        rewriteMatMulCounter++);
                    // set block-based matrix multiply input name
                    std::string blockMatMulInputName = matchedDataSrc;

                    // exprsWithinTarget starts with ROW, if so, there is no
                    // other UDFs before the target UDF
                    if (exprsWithinTarget.rfind("ROW", 0) != 0) {
                      // there are other UDFs before the target udf
                      std::string exprsWithinTargetWithRewriteSrc =
                          std::regex_replace(
                              exprsWithinTarget,
                              patternToMatchRawSource,
                              matchedDataSrc);
                      // if has precompute project, we need to have a new name for
                      // its results and used in the block-based matrix multiply
                      blockMatMulInputName = matchedDataSrc + "_block_input";
                      preComputeExprSets.insert(
                          exprsWithinTargetWithRewriteSrc + " AS " +
                          blockMatMulInputName);
                      // set the flag to true
                      hasPrecomputeProject = true;
                    }

                     mulProjectExprSets.insert(fmt::format(
                        "{}({}, {}) AS {}",
                        blockMatMulName,
                        blockMatMulInputName,
                        weightBlockName,
                        intermediateAggregationName));

                    // Add UDF associate information (UDF with input values) to
                    // cataLog std::string nameSuffix = "_vertical";
                    // cataLog.add(target,
                    // cataLog.getDataSourceBlocksSchema("values"),
                    // cataLog.getDataSourceBlocksFileAddr("values"), 0,
                    // "_vertical"); Add the parsed expression to

                    // finalProjectExprSets
                    std::string matMulAggExpr = fmt::format(
                        "array_cat({}, {}) AS {}",
                        intermediateAggregationName,
                        target + "_wb_col",
                        intermediateAggregationName);
                    matMulAggExprSets.insert(matMulAggExpr);

                    // Regular expression match
                    // extract the following computation after target rewrite
                    // UDF
                    std::string escapedRegex =
                        escapeRegex(target + "(" + exprsWithinTarget + ")");
                    std::regex patternOfRewriteFinalExpr(escapedRegex);
                    // std::cout << fmt::format("[DEBUG] targetExprStr: {},
                    // patternOfRewriteFinalExpr: {}, targetExprName: {}",
                    // targetExprStr, target +  + "("  + exprsWithinTarget +
                    // ")", targetExprName) << std::endl;
                    targetExprStr = std::regex_replace(
                        targetExprStr,
                        patternOfRewriteFinalExpr,
                        intermediateAggregationName);

                    finalProjectExprSets.insert(
                        targetExprStr + " AS " + targetExprName);
                    findRewriteTarget = true;
                  }
                }
              } else {
                // Parse the non-target expressions
                // Match the source
                std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
                std::smatch matches;
                if (std::regex_search(
                        exprStr, matches, patternToMatchRawSource)) {
                  auto matchedDataSrc = matches[1].str();
                  auto rewriteExpr = std::regex_replace(
                      exprStr, patternToMatchRawSource, matchedDataSrc);
                  // Store the data source in the exprSets and the
                  // computation in the finalProjectExprSets
                  preComputeExprSets.insert(matchedDataSrc);
                  nestedLoopProjectExprSets.insert(matchedDataSrc);
                  mulProjectExprSets.insert(matchedDataSrc);
                  matMulAggKeySets.insert(matchedDataSrc);
                  finalProjectExprSets.insert(
                      rewriteExpr + " AS " + projectionsNames[exprIdx]);

                } else {
                  LOG(ERROR)
                      << "Error: undefined-edge case detected: " << exprStr
                      << std::endl;
                }
              }
              // }
            }
            if ((curNode->sources().size()) > 0 && findRewriteTarget) {
              // Get schema of values and weights from cataLog
              weightSchema = cataLog.getUDFSchema(target + "_weights_vertical");

              // Build new plan
              // Note: In NestedLoopJoin, each row from the left side is
              // iterated over and joined with every row from the right side.
              // Given that the weight matrix is already partitioned and has
              // fewer rows to iterate compared to the input, it is recommended
              // to place the partitioned weight matrix on the left side to
              // enhance the performance of the nested loop join due to reduced
              // iteration overhead.

              // do an inference to obtain the id column of the src
              // we assume the id column follows the naming convention that
              // the source name overlaps with the id column name and id
              // is included in the column name as well.
              std::string srcName = splitString(rewriteMatMulSrc, '_')[0];
              std::string matMulAggKey;
              for (std::string colName : nestedLoopProjectExprSets) {
                if (containsStrButNotEqual(colName, srcName) &&
                    containsStrButNotEqual(colName, "id")) {
                  matMulAggKey = colName;
                  break;
                }
              }
              if (matMulAggKey.empty()) {
                matMulAggKey = "idx";
                LOG(INFO)
                    << "[WARN] inference of MatMul agg key failed, use default value: idx"
                    << std::endl;
              }

              matMulAggKeySets.insert(matMulAggKey);
              preComputeExprSets.insert(matMulAggKey);
              nestedLoopProjectExprSets.insert(matMulAggKey);
              mulProjectExprSets.insert(matMulAggKey);
              // std::cout << "[DEBUG] matMulAggKey: " << matMulAggKey <<
              // std::endl; std::cout << "[DEBUG] srcName: " << srcName <<
              // std::endl; std::cout << "[DEBUG] nestedLoopProjectExprsSets: "
              // << nestedLoopProjectExprSets << std::endl; std::cout <<
              // "[DEBUG] preComputeExprsSets: " << preComputeExprSets <<
              // std::endl; std::cout << "[DEBUG] mulProjectExprSets: " <<
              // mulProjectExprSets << std::endl; std::cout << "[DEBUG]
              // matMulAggKeySets: " << matMulAggKeySets << std::endl; std::cout
              // << "[DEBUG] matMulAggExprSets: " << matMulAggExprSets <<
              // std::endl; std::cout << "[DEBUG] finalProjectExprsSets: " <<
              // finalProjectExprSets << std::endl;

              // get the source node of the current node
              auto srcNode = curNode->sources()[0];

              // Initial PlanNode Id for splits
              core::PlanNodeId p1;
              core::PlanNodeId p2;

              // Convert the exprSets into exprVector for PlanBuilder
              std::vector<std::string> preComputeExprs(
                  preComputeExprSets.begin(), preComputeExprSets.end());
              std::vector<std::string> nestedLoopProjectExprs(
                  nestedLoopProjectExprSets.begin(),
                  nestedLoopProjectExprSets.end());
              std::vector<std::string> mulProjectExprs(
                  mulProjectExprSets.begin(), mulProjectExprSets.end());
              std::vector<std::string> matMulAggKeys(
                  matMulAggKeySets.begin(), matMulAggKeySets.end());
              std::vector<std::string> matMulAggExprs(
                  matMulAggExprSets.begin(), matMulAggExprSets.end());
              std::vector<std::string> finalProjectExprs(
                  finalProjectExprSets.begin(), finalProjectExprSets.end());
              LOG(INFO)
                  << fmt::format(
                         "[DEBUG] preComputeExprs: | {}, \n nestedLoopProjectExprs: | {}, \n mulProjectExprs: | {}, \n finalProjectExprs: | {}, \n",
                         preComputeExprs,
                         nestedLoopProjectExprs,
                         mulProjectExprs,
                         finalProjectExprs)
                  << std::endl;
              // TODO accomodate automatic join reording here
              auto rewritePlan =
                  exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(weightSchema)
                      .capturePlanNodeId(p2);
              // Re-create the plan from the source node of target rewrite
              // project node
              rewritePlan.nestedLoopJoin(srcNode, nestedLoopProjectExprs);
              // Add the pre-computation expressions if they exist
              if (hasPrecomputeProject) {
                rewritePlan.project(preComputeExprs);
              }

              // Add the rewrite nodes of mat_mul
              rewritePlan
                  .project(mulProjectExprs)
                  // .localPartition({"idx"})
                  // .singleAggregation({"idx"}, {"array_cat(t, w_col) AS v"})
                  .partialAggregation(matMulAggKeys, matMulAggExprs)
                  .localPartition(matMulAggKeys)
                  .intermediateAggregation()
                  .finalAggregation()
                  .project(finalProjectExprs);
              cataLog.setIdAddressMap(
                  p2, cataLog.getUDFFileAddr(target + "_weights_vertical"));
              int blockSize = cataLog.getDefaultBlocksSize();
              int numBlocks = ceil(dims[1] / blockSize);

              std::shared_ptr<OutputStat> weightStat =
                  std::make_shared<OutputStat>(
                      OutputStat(numBlocks, blockSize));
              Source weightSource =
                  Source(p2, Source::Type::FILE, std::move(weightStat));
              cataLog.addSource(std::make_shared<Source>(weightSource));
              cataLog.addNodeIdRelationName(p2, target + "_weights_vertical");

              LOG(INFO)
                  << "[Rewrite] mul2joinAggHorizontal rewrite updated catalog: "
                  << std::endl;
              LOG(INFO) << fmt::format(
                               "\t\t p2: {}, {}, {}", p2, numBlocks, blockSize)
                        << std::endl;

              transformationApplied = true;

              // Add the edge between the rewrite plan's root node the the node
              // after the rewrite node in the original plan
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
            }
          }
        }

        // Search for a filter node, which is similar to the project node
        if (nodeName == "Filter") {
          std::shared_ptr<const FilterNode> myFilterNode =
              std::dynamic_pointer_cast<const FilterNode>(curNode);

          const TypedExprPtr& filterExpr = myFilterNode->filter();

          targetExprStr = filterExpr->toString();

          if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(
                  filterExpr)) {
            if (targetExprStr.find(target) != std::string::npos) {
              std::cout << "[WARNING] Not supported yet" << std::endl;
              // core::QueryConfig config({});
              // // Search for UDF functions by names
              // std::shared_ptr<VectorFunction> myMul =
              //     getVectorFunction(target, {ARRAY(REAL())}, {}, config);

              // if (myMul) {
              //   // Get the specific MatrixMultiply UDF
              //   std::shared_ptr<MatrixMultiply> myMulUDF =
              //       std::dynamic_pointer_cast<MatrixMultiply>(myMul);

              //   if (myMulUDF) {
              //     // Get the dimensions, weights from this MatrixMultiply UDF
              //     dims = myMulUDF->getDims();
              //     weights = myMulUDF->getTensor();
              //     // Get information (defaultBlocksnumber, number of samples)
              //     // from cataLog
              //     int blocks = cataLog.getDefaultBlocksNum();
              //     int blockSize = cataLog.getDefaultBlocksSize();

              //     // Register matrix blocks multiply function
              //     registerVectorFunction(
              //         "mat_mul_h",
              //         MatrixMultiply_h::signatures(),
              //         // std::make_unique<MatrixMultiply_h>(dims[0]/blocks,
              //         // dims[1], samples, weights, blocks)
              //         std::make_unique<MatrixMultiply_h>(
              //             dims[0], dims[1], cataLog.getDefaultBlocksSize()));
              //     // Add UDF associate information (UDF with input values) to
              //     // cataLog Should blocking source here catalog source will
              //     // invoke a intern function to blocking itself, then return
              //     // schema and address in here cataLog.add(target,
              //     // cataLog.getDataSourceBlocksSchema("values"),
              //     // cataLog.getDataSourceBlocksFileAddr("values"), 0);
              //   }
              // }
              // if (curNode->sources().size() > 0) {
              //   // Initial PlanNode Id for splits
              //   core::PlanNodeId p1;
              //   core::PlanNodeId p2;
              //   // Get schema of values and weights from cataLog
              //   valueSchema = cataLog.getUDFSchema(target + "_values");
              //   weightSchema =
              //       cataLog.getUDFSchema(target + "_weights_vertical");
              //   // Regular expression match
              //   std::regex pattern(target + R"(\([^)]+\))");
              //   targetExprStr =
              //       std::regex_replace(targetExprStr, pattern, "R1");

              //   // Build new plan
              //   planBuilder =
              //       exec::test::PlanBuilder(planNodeIdGenerator)
              //           .tableScan(weightSchema)
              //           .capturePlanNodeId(p2)
              //           // automatically generate row number for input values
              //           .nestedLoopJoin(
              //               PlanBuilder(planNodeIdGenerator)
              //                   .tableScan(valueSchema)
              //                   .capturePlanNodeId(p1)
              //                   .planNode(),
              //               {"idx", "v", "w", "w_row", "w_col"})
              //           .project({"mat_mul_h(v, w) AS t", "idx", "w_col"})
              //           // .localPartition({"idx"})
              //           // .singleAggregation({"idx"}, {"array_cat(t, w_col)
              //           AS
              //           // v"})
              //           .partialAggregation(
              //               {"idx"}, {"array_cat(t, w_col) AS v"})
              //           .localPartition({"idx"})
              //           .intermediateAggregation()
              //           .finalAggregation()
              //           .project({targetExprStr})
              //           ;

              //   // Delete old nodeId-fileAddress map
              //   auto valueFileAddr =
              //       cataLog.getFileAddress(cataLog.getVectorIdMap("v"));
              //   cataLog.deleteIdAddressMap(cataLog.getVectorIdMap("v"));
              //   // Insert new nodeId-fileAddress maps
              //   cataLog.setIdAddressMap(p1, valueFileAddr);
              //   cataLog.setIdAddressMap(
              //       p2, cataLog.getUDFFileAddr(target +
              //       "_weights_vertical"));

              //   transformationApplied = true;
              // }
            }
          }
        }
        // Serach lower level plan node
        std::vector<std::shared_ptr<const PlanNode>> sources =
            curNode->sources();
        // Until leaf node
        if (sources.size() == 0)
          return false;
        // recursive search
        for (auto source : sources) {
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
    }
    return transformationApplied;
  }

  std::string name() override {
    return "Mul2JoinAggHorizontalRewriteAction";
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
      bool checkApplied = true;
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
          // match mat_mul and filter out the _wb (weight block) and _h
          // (block-based mat_mul) case
          std::regex pattern(R"(mat_mul\d+\_\d+\()");
          auto wordsBegin =
              std::sregex_iterator(expr.begin(), expr.end(), pattern);
          auto wordsEnd = std::sregex_iterator();
          // Retrieve the possible UDF name applicable for this rule, and check
          // if there existed block files, stored in targetAction.
          for (auto it = wordsBegin; it != wordsEnd; ++it) {
            // remove the last character ')'
            std::string functionName =
                it->str().substr(0, it->str().size() - 1);
            if (cataLog.checkExistsUDFFileAddr(
                    functionName + "_weights_vertical")) {
              targetActions.push_back(functionName);
            } else {
              LOG(ERROR) << "[ERROR]: " << functionName + "_weights_vertical"
                         << "does not exist in catalog" << std::endl;
            }
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
        std::regex pattern(R"(mat_mul\d+)");

        auto wordsBegin =
            std::sregex_iterator(expr.begin(), expr.end(), pattern);

        auto wordsEnd = std::sregex_iterator();
        // Retrieve the possible UDF name applicable for this rule, and check if
        // there existed block files, stored in targetAction.
        for (auto it = wordsBegin; it != wordsEnd; ++it) {
          if (cataLog.checkExistsUDFFileAddr(it->str() + "_weights_vertical")) {
            targetActions.push_back(it->str());
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
        checkApplied &= check(source, targetActions, cataLog);
      }

      return checkApplied;

    } catch (const std::exception& e) {
      std::cerr << "Error in check function: " << e.what() << std::endl;

      return false; // Return false for any error
    }
  }

 private:
  std::vector<int> dims;
  std::string targetExprStr;
  int blocks;
  float* weights;
  RowTypePtr valueSchema;
  RowTypePtr weightSchema;
  static inline int rewriteMatMulCounter = 0;
};

} // namespace optimization