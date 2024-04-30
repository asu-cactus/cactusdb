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

#include <memory>
#include <iostream>
#include "RewriteAction.h"
#include "velox/core/PlanNode.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include <regex>

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


namespace optimization {

class Mul2JoinAggRewriteAction : public RewriteAction {

public:

    Mul2JoinAggRewriteAction (){}

	void clearVectors() {
		dims.clear();
	}

	/**
	 * @brief A function to apply a rule for rewriting the logical plan.
	 * 
	 * @param curNode A pointer to the current plan node, usually point to the last node of logical plan.
	 * @param prevNode A pointer to the previous plan node, usually point to the previous node before current node.
	 * @param maker A pointer to the VectorMaker, which is a helper class used to build the data source vector.
	 * @param planBuilder A pointer to the planBuilder, which is a helper class used to build the logical plan.
	 * @param pool_ A pointer to the memory pool, which is used to build the logical plan.
	 * @param planNodeIdGenerator A pointer to the planNodeIdGenerator, which is used to track the ID of the plan Node.
	 * @param targets A vector for multiple strings, representing the target UDF name that can apply this rewritten rule.
	 * @param cataLog A class storing metadata and information related to UDFs and data sources.
	 * 
	 * @return A boolean value indicating whether the rewrite was successful.
	*/
    bool apply(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
		   std::vector<std::string> targets,
		   CataLog &cataLog) override {
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
						if (auto myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode)) {
							// Get projections in project node
							const std::vector<TypedExprPtr> & projections = myProjectNode->projections();
							// Search each expression in projections
							for (auto expression : projections) {
								// Get the string of expression
								exprStr = expression->toString();
								if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)){

									std::string callName = call->name();
									// String match the target UDF name
									if (exprStr.find(target) != std::string::npos) {
										// We only consider one projection expression in the project node.
										if (projections.size() == 1) {

											core::QueryConfig config({});
											// Search for UDF functions by names
											std::shared_ptr<VectorFunction> myMul = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

											if (myMul) {
												// Get the specific MatrixMultiply UDF
												std::shared_ptr<MatrixMultiply> myMulUDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul);

												if (myMulUDF) {
													// Get the dimensions, weights from this MatrixMultiply UDF
													dims = myMulUDF->getDims();
													weights = myMulUDF->getTensor();
													// Get information (defaultBlocksnumber, number of samples) from cataLog
													int blocks = cataLog.getDefaultBlocksNum();
													int samples = cataLog.getDataSourceStat("values")[0];

													// Register matrix blocks multiply function
													registerVectorFunction(
														"mat_mul_block",
														MatrixMultiply_Block::signatures(),
														std::make_unique<MatrixMultiply_Block>(dims[0]/blocks, dims[1], samples, blocks)
													);
													// Add UDF associate information (UDF with input values) to cataLog
													std::string nameSuffix = "_vertical";
													cataLog.add(target, cataLog.getDataSourceBlocksSchema("values"), cataLog.getDataSourceBlocksFileAddr("values"), 0, nameSuffix);
												}
											}
											if (curNode->sources().size() > 0) {
												// Initial PlanNode Id for splits
												core::PlanNodeId p1;
												core::PlanNodeId p2;
												// Get schema of values and weights from cataLog
												valueSchema = cataLog.getUDFSchema(target+"_values_vertical");
												weightSchema = cataLog.getUDFSchema(target+"_weights_horizontal");
												// Regular expression match
												std::regex pattern(target + R"(\([^)]+\))");
												exprStr = std::regex_replace(exprStr, pattern, "r1");
												// Build new plan
												planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
																.tableScan(valueSchema)
																.capturePlanNodeId(p1)
																.hashJoin(
																	{"v_col"},
																	{"w_row"},
																	exec::test::PlanBuilder(planNodeIdGenerator)
																	.tableScan(weightSchema)
																	.capturePlanNodeId(p2)
																	.planNode(),
																"", // extra filter
																{"v_row", "w_col", "v", "w"})
																.project({"v_row", "w_col", "mat_mul_block(v, w) AS mp"})
																// .localPartition({})
																// .singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS R1"})
																.partialAggregation({"w_col","v_row"}, {"array_sum(mp) AS R1"})
																.localPartition({})
																.intermediateAggregation()
																.finalAggregation()
																// .project({"r1"})
																.unnest({}, {"r1"}) // after unnest velox will automatically add + "_e" to its original name. PlanBuilder.cpp/unnest
																// .rowNumber({}) // TODO: need further investigation to see if performance can be improved
																// .localPartition({"row_number"})
																.project({"r1_e AS r1"})
																// TODO: performance degradation observed, temporary disable the repartition
																// .localPartitionRoundRobinRow() 
																.project({exprStr})
																;
												// Here are several approaches to correctly aggregat the multiplied values.
												// Note: if the values to be aggregated are already partitioned in batches.
												// A singleAggregation is enough otherwise shuffling is needed to aggregate
												// the values cross different threads/batches.
												// Approach 1: run shuffling and a single aggregation: 
												//   .localPartition({}).singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS R1"})
												// Approach 2: run via partialAggregation: 
												//   .partialAggregation({"w_col","v_row"}, {"array_sum(mp) AS R1"})
												//	 .localPartition({})
												//	 .intermediateAggregation()
												//	 .finalAggregation()
												
												// Delete old nodeId-fileAddress map
												cataLog.deleteIdAddressMap(cataLog.getVectorIdMap("v"));
												// Insert new nodeId-fileAddress maps
												cataLog.setIdAddressMap(p1, cataLog.getUDFFileAddr(target+"_values_vertical"));
												cataLog.setIdAddressMap(p2, cataLog.getUDFFileAddr(target+"_weights_horizontal"));

												transformationApplied = true;
											}
										}
										
									}

								}
							}
						}						
					}
				
					// Search for a filter node, which is similar to the project node
					if (nodeName == "Filter") {

						std::shared_ptr<const FilterNode> myFilterNode = std::dynamic_pointer_cast<const FilterNode> (curNode);

            			const TypedExprPtr & filterExpr = myFilterNode->filter();

						exprStr = filterExpr->toString();
		 			
		        		if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr)) {
							
							if (exprStr.find(target) != std::string::npos) {

								core::QueryConfig config({});
								// Search for UDF functions by names
								std::shared_ptr<VectorFunction> myMul = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

								if (myMul) {
									// Get the specific MatrixMultiply UDF
									std::shared_ptr<MatrixMultiply> myMulUDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul);

									if (myMulUDF) {
										// Get the dimensions, weights from this MatrixMultiply UDF
										dims = myMulUDF->getDims();
										weights = myMulUDF->getTensor();
										// Get information (defaultBlocksnumber, number of samples) from cataLog
										int blocks = cataLog.getDefaultBlocksNum();
										int samples = cataLog.getDataSourceStat("values")[0];

										// Register matrix blocks multiply function

										registerVectorFunction(
														"mat_mul_block",
														MatrixMultiply_Block::signatures(),
														std::make_unique<MatrixMultiply_Block>(dims[0]/blocks, dims[1], samples, blocks)
													);
										// Add UDF associate information (UDF with input values) to cataLog
										// Should blocking source here
										// catalog source will invoke a intern function to blocking itself, then return schema and address in here
										std::string nameSuffix = "_vertical";
										cataLog.add(target, cataLog.getDataSourceBlocksSchema("values"), cataLog.getDataSourceBlocksFileAddr("values"), 0, nameSuffix);

									}
								}
								if (curNode->sources().size() > 0) {
									// Initial PlanNode Id for splits
									core::PlanNodeId p1;
									core::PlanNodeId p2;
									// Get schema of values and weights from cataLog
									valueSchema = cataLog.getUDFSchema(target+"_values");
									weightSchema = cataLog.getUDFSchema(target+"_weights");
									// Regular expression match
									std::regex pattern(target + R"(\([^)]+\))");
									exprStr = std::regex_replace(exprStr, pattern, "R1");
									// Build new plan
									planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
													.tableScan(valueSchema)
													.capturePlanNodeId(p1)
													.hashJoin(
														{"v_col"},
														{"w_row"},
														exec::test::PlanBuilder(planNodeIdGenerator)
													.tableScan(weightSchema)
													.capturePlanNodeId(p2)
													.planNode(),
														"", // extra filter
														{"v_row", "w_col", "v", "w"})
													.project({"v_row", "w_col", "mat_mul_b(v, w) AS mp"})
													.singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS R1"})
													.project({exprStr});
									// Delete old nodeId-fileAddress map
									cataLog.deleteIdAddressMap(cataLog.getVectorIdMap("v"));
									// Insert new nodeId-fileAddress maps
									cataLog.setIdAddressMap(p1, cataLog.getUDFFileAddr(target+"_values_vertical"));
									cataLog.setIdAddressMap(p2, cataLog.getUDFFileAddr(target+"_weights_horizontal"));

									transformationApplied = true;
								}

								
							}

						}
					}
					// Serach lower level plan node
					std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();
					// Until leaf node
					if (sources.size() == 0) return false;
						// recursive search
						for (auto source : sources)       		 
						
							transformationApplied |= apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator, targets, cataLog);
			
				}
			
			}
			return transformationApplied;
    	}


    std::string name() override {
    
        return "Mul2JoinAggRewriteAction";
    
    }

	/**
	 * @brief A function to check if this rule can be applied in a logical plan and to store the possible UDF name.
	 * 
	 * @param rootNode A pointer to the logical plan.
	 * @param targetActions A pointer to the vector used to store possible UDF names applicable for this rule.
	 * @param cataLog A class storing metadata and information related to UDFs and data sources.
	 * 
	 * @return A boolean value indicating whether the check was successful.
	*/
	bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> &targetActions, CataLog &cataLog) override {
		try {
			bool checkSuccess = true;
			if (!rootNode) {

				throw std::invalid_argument("rootNode is null");

			}

			std::string_view nodeName = rootNode->name();
			// We first check the project node
			if (nodeName == "Project") {

				auto myProjectNode = std::dynamic_pointer_cast<const ProjectNode>(rootNode);

				if (!myProjectNode) {

					throw std::runtime_error("Failed to cast to ProjectNode");

				}

				const std::vector<TypedExprPtr> &projections = myProjectNode->projections();
				// Search each expressions
				for (const auto &expression : projections) {

					std::string expr = expression->toString();
					// Regular expression match
					std::regex pattern(R"(mat_mul\d+)");

					auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);

					auto wordsEnd = std::sregex_iterator();
					// Retrieve the possible UDF name applicable for this rule, and check if there existed block files, stored in targetAction.
					for (auto it = wordsBegin; it != wordsEnd; ++it) {

						if (cataLog.checkExistsUDFFileAddr(it->str()+"_weights_horizontal")) {

							targetActions.push_back(it->str());
						}
					}
				}
			}
			// We then check the filter node
			if (nodeName == "Filter") {

				auto myFilterNode = std::dynamic_pointer_cast<const FilterNode>(rootNode);

				if (!myFilterNode) {

					throw std::runtime_error("Failed to cast to FilterNode");
				}

				const TypedExprPtr &filterExpr = myFilterNode->filter();

				std::string expr = filterExpr->toString();
				// Regular expression match
				std::regex pattern(R"(mat_mul\d+)");

				auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);

				auto wordsEnd = std::sregex_iterator();
				// Retrieve the possible UDF name applicable for this rule, and check if there existed block files, stored in targetAction.
				for (auto it = wordsBegin; it != wordsEnd; ++it) {

					if (cataLog.checkExistsUDFFileAddr(it->str()+"_weights_horizontal")) {

						targetActions.push_back(it->str());
					}

				}
			}

			std::vector<std::shared_ptr<const core::PlanNode>> sources = rootNode->sources();

			if (sources.empty()) {
				// Safe exit for leaf node
				return true;

			}

			for (const auto &source : sources) {
				checkSuccess &= check(source, targetActions, cataLog);
			}

			return checkSuccess; 

		} catch (const std::exception &e) {

			std::cerr << "Error in check function: " << e.what() << std::endl;

			return false;  // Return false for any error

		}
	}


private: 

    std::vector<int> dims;
    std::string exprStr;
	int blocks;
	float* weights;
	RowTypePtr valueSchema;
	RowTypePtr weightSchema;

};


}