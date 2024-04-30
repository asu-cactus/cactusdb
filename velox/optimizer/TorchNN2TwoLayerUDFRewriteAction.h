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
#include "velox/ml_functions/NNBuilder.h"
#include <regex>

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


namespace optimization {

class TorchNN2TwoLayerUDFRewriteAction : public RewriteAction {

public:

    TorchNN2TwoLayerUDFRewriteAction () {}

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
     * @param cataLog Reference to a CataLog object to store metadata and information.
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
					if (auto myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode)){
						// Get projections in project node
						const std::vector<TypedExprPtr> & projections = myProjectNode->projections();
						// Search each expression in projections
						for (auto expression : projections) {
							// Get the string of expression
							exprStr = expression->toString();
							// Tree serach in one expression until leaf expression (size == 0)
							while (expression->inputs().size() > 0) {
								
								if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {
									// Torchdnn is a single UDF name, we can compare it by call->name()
									std::string callName = call->name();
									// String match the target UDF name
									if (callName == target) {
										// We only consider one projection expression in the project node.
										if (projections.size() == 1) {
								
											core::QueryConfig config({});
											// Get the specific torchdnn UDF
											std::shared_ptr<VectorFunction> UDF = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

											if (UDF) {
								
												std::shared_ptr<TorchDNN2Level> TorchUDF = std::dynamic_pointer_cast<TorchDNN2Level>(UDF);

												if (TorchUDF) {
													// Get the dimensions, weights and bias from this torchdnn UDF
													dims = TorchUDF->getDims();

													float** weights = TorchUDF->getWeights();

													float** bias = TorchUDF->getBias();
													// Create two layers UDF
													compute =  NNBuilder()
													.denseLayer(dims[1], dims[0], weights[0], bias[0], NNBuilder::RELU)
													.denseLayer(dims[2], dims[1], weights[1], bias[1], NNBuilder::SOFTMAX)
													.build();
												}
								
											}

											if (curNode->sources().size() > 0) {
												// Here, we focus on the inner case. For example:
												// project({torchnnx(v)}) ---> project({softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))})
												// The format for inner case is project(torchnnx())
												// TODO: Medium case - project(func1(torchnnx(func2())))
												// TODO: Outer case - project({torchnnx(func1(func2()))})						

												// Plan Builder start from the previous node.
												// The rewritten plan should be project({softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))} AS R1).Project(R1)
												planBuilder = planBuilder.setRoot((curNode->sources())[0]);

												std::string twoLayers = fmt::format(compute, "v");

												twoLayers += " AS R1";
												// Regular expression match
												std::regex pattern(target + R"(\([^)]+\))");
												// Replace the expression
												exprStr = std::regex_replace(exprStr, pattern, "R1");
												// Plan Builder add the new node.
												planBuilder = planBuilder.project({twoLayers}).project({exprStr});
									
												transformationApplied = true;
											}
								
										}
																															
									}

								}

								expression = expression->inputs()[0];
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

			        	std::string callName = call->name();

			        	if (callName == target) {
			     
                            core::QueryConfig config({});

				            std::shared_ptr<VectorFunction> UDF = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

				            if (UDF) {
				   
				                std::shared_ptr<TorchDNN2Level> TorchUDF = std::dynamic_pointer_cast<TorchDNN2Level>(UDF);

					            if (TorchUDF) {
					 
					                dims = TorchUDF->getDims();

                                    float** weights = TorchUDF->getWeights();

                                    float** bias = TorchUDF->getBias();
                                    
                                    compute =  NNBuilder()
                                    .denseLayer(dims[1], dims[0], weights[0], bias[0], NNBuilder::RELU)
                                    .denseLayer(dims[2], dims[1], weights[1], bias[1], NNBuilder::SOFTMAX)
                                    .build();
					            }
				   
				            }

                            if (curNode->sources().size() > 0) {

                                planBuilder = planBuilder.setRoot((curNode->sources())[0]);								

								std::string twoLayers = fmt::format(compute, "v");

								twoLayers += " AS R1";

								std::regex pattern(target + R"(\([^)]+\))");

								exprStr = std::regex_replace(exprStr, pattern, "R1");

								planBuilder = planBuilder.project({twoLayers}).filter({exprStr});
							
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

	/**
	 * @brief A function to get the name of rewritten rule.
	 * 
	 * @return A string value denoting the name of the rule
	*/
    std::string name() override {
    
        return "TorchNN2TwoLayerUDFRewriteAction";
    
    }

	/**
	 * @brief A function to check if this rule can be applied in a logical plan and to store the possible UDF name.
	 * 
	 * @param rootNode A pointer to the logical plan.
	 * @param targetActions A pointer to the vector used to store possible UDF names applicable for this rule.
     * @param cataLog Reference to a CataLog object to store metadata and information.
	 * 
	 * @return A boolean value indicating whether the check was successful.
	*/
	bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> &targetActions, CataLog& cataLog) override {
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
					// For this rule, we only check wheather torchnn UDF function is in expressions
					std::regex pattern(R"(torchdnn\d+)");

					auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);

					auto wordsEnd = std::sregex_iterator();
					// Retrieve the possible UDF name applicable for this rule, stored in targetAction.
					for (auto it = wordsBegin; it != wordsEnd; ++it) {

						targetActions.push_back(it->str());
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
				std::regex pattern(R"(torchdnn\d+)");

				auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);

				auto wordsEnd = std::sregex_iterator();
				// Retrieve the possible UDF name applicable for this rule, stored in targetAction.
				for (auto it = wordsBegin; it != wordsEnd; ++it) {

					targetActions.push_back(it->str());
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
    std::string compute;
	std::string exprStr;

};


}