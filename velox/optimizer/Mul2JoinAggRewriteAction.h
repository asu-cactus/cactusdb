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


    bool apply(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
		   std::vector<std::string> targets,
		   CataLog &cataLog) override {
			for (auto target : targets) {
				if (curNode) {

					std::string_view nodeName = curNode->name();

					if (nodeName == "Project") {
					
						if (auto myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode)) {
							const std::vector<TypedExprPtr> & projections = myProjectNode->projections();
							for (auto expression : projections) {
								exprStr = expression->toString();
								if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)){
									std::string callName = call->name();
									if (exprStr.find(target) != std::string::npos) {
										if (projections.size() == 1) {
											core::QueryConfig config({});
											std::shared_ptr<VectorFunction> myMul = getVectorFunction(target, {ARRAY(REAL())}, {}, config);
											if (myMul) {
												std::shared_ptr<MatrixMultiply> myMulUDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul);
												if (myMulUDF) {
													dims = myMulUDF->getDims();
													weights = myMulUDF->getTensor();
													int blocks = cataLog.getDefaultBlocksNum();
													int samples = cataLog.getDataSourceStat("values")[0];


													registerVectorFunction(
														"mat_mul_b",
														MatrixMultiply_b::signatures(),
														std::make_unique<MatrixMultiply_b>(dims[0]/blocks, dims[1], samples, weights, blocks)
													);

													cataLog.add(target, cataLog.getDataSourceBlocksSchema("values"), cataLog.getDataSourceBlocksFileAddr("values"), 0);

												}
											}
											if (curNode->sources().size() > 0) {
												core::PlanNodeId p1;
												core::PlanNodeId p2;
												valueSchema = cataLog.getUDFSchema(target+"_values");
												weightSchema = cataLog.getUDFSchema(target+"_weights");

												std::regex pattern(target + R"(\([^)]+\))");
												exprStr = std::regex_replace(exprStr, pattern, "R1");
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

												cataLog.deleteIdAddressMap(cataLog.getVectorIdMap("v"));
												cataLog.setIdAddressMap(p1, cataLog.getUDFFileAddr(target+"_values"));
												cataLog.setIdAddressMap(p2, cataLog.getUDFFileAddr(target+"_weights"));

											return true;
											}
										}
										
									}

								}
							}
						}						
					}
				
					std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();

					if (sources.size() == 0) return false;

						for (auto source : sources)       		 
						
							apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator, targets, cataLog);
			
				}
			
			}
    	}


    std::string name() override {
    
        return "Mul2JoinAggRewriteAction";
    
    }

	/**
	 * @brief A function to check if this rule can be applied in a logical plan and to store the possible UDF name.
	 * 
	 * @param rootNode A pointer to the logical plan.
	 * @param targetActions A pointer to the vector used to store possible UDF names applicable for this rule.
	 * 
	 * @return A boolean value indicating whether the check was successful.
	*/
	bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> &targetActions, CataLog &cataLog) override {
		try {
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
					// Retrieve the possible UDF name applicable for this rule, stored in targetAction.
					for (auto it = wordsBegin; it != wordsEnd; ++it) {
						if (cataLog.checkExistsUDFFileAddr(it->str()+"_weights")) {
							targetActions.push_back(it->str());
						}
					}
				}

				return true;  // Return true for successful execution
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
				// Retrieve the possible UDF name applicable for this rule, stored in targetAction.
				for (auto it = wordsBegin; it != wordsEnd; ++it) {

					if (cataLog.checkExistsUDFFileAddr(it->str()+"_weights")) {
						targetActions.push_back(it->str());
					}

				}

				return true;  // Return true for successful execution
			}

			std::vector<std::shared_ptr<const core::PlanNode>> sources = rootNode->sources();

			if (sources.empty()) {
				// Safe exit for leaf node
				return true;

			}

			for (const auto &source : sources) {

				if (!check(source, targetActions, cataLog)) {
					// Propagate false if any child node returns false
					return false;

				}
			}

			return true;  // Return true for successful execution

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