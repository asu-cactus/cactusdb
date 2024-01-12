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

class TwoLayerUDF2TorchNNRewriteAction : public RewriteAction {

public:

    TwoLayerUDF2TorchNNRewriteAction () {}


    bool apply(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
		   std::vector<std::string> targets) override {

			for (auto target : targets) {

				if (curNode) {

					std::string_view nodeName = curNode->name();

				if (nodeName == "Project") {
				
					if (auto myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode)){

						const std::vector<TypedExprPtr> & projections = myProjectNode->projections();

						for (auto expression : projections) {

							exprStr = expression->toString();

							while (expression->inputs().size() > 0) {

								if (exprStr == target) {

									if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {
								
										if (projections.size() == 1) {
								
                            				core::QueryConfig config({});

											if (auto call1 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0])) {
												std::string firstMatAddName = call1->name();
												if (auto call2 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0])) {
													std::string firstMatMulName = call2->name();
													if (auto call3 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])) {
														std::string secondMatAddName = call3->name();
														if (auto call4 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])) {
															std::string secondMatMulName = call4->name();
															std::shared_ptr<VectorFunction> myMul1 = getVectorFunction(secondMatMulName, {ARRAY(REAL())}, {}, config);
															std::shared_ptr<VectorFunction> myMul2 = getVectorFunction(firstMatMulName, {ARRAY(REAL())}, {}, config);

															std::shared_ptr<VectorFunction> myAdd1 = getVectorFunction(secondMatAddName, {ARRAY(REAL())}, {}, config);
															std::shared_ptr<VectorFunction> myAdd2 = getVectorFunction(firstMatAddName, {ARRAY(REAL())}, {}, config);

															if (myMul1 && myMul2 && myAdd1 && myAdd2) {
																if (auto myMul1UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul1)) {
																	if (auto myMul2UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul2)) {
																		if (auto myAdd1UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd1)) {
																			if (auto myAdd2UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd2)) {
																				dims.push_back(myMul1UDF->getDims()[0]);
																				dims.push_back(myMul1UDF->getDims()[1]);
																				dims.push_back(myMul2UDF->getDims()[1]);

																				weights[0] = myMul1UDF->getTensor();
																				weights[1] = myMul2UDF->getTensor();

																				bias[0] = myAdd1UDF->getTensor();
																				bias[1] = myAdd2UDF->getTensor();

																				registerVectorFunction(
																				"torchDNN",
																				TorchDNN::signatures(),
																				std::make_unique<TorchDNN>(weights, bias, dims)
																				);
																			}
																		}
																	}
																}
																	
															}
														}
													}
												}
											}
																	
											if (curNode->sources().size() > 0) {

												planBuilder = planBuilder.setRoot(curNode->sources()[0]);

												std::regex pattern(R"(softmax\d+\(mat_add\d+\(mat_mul\d+\(relu\d+\(mat_add\d+\(mat_mul\d+.*\)\)\)\)\))");

												exprStr = std::regex_replace(exprStr, pattern, "torchDNN(v)");

												planBuilder = planBuilder.project({exprStr});

												return true;
											}
										}
																															
									}

								}

								expression = expression->inputs()[0];
							}
				
						}		 
					
					}
				}

				if (nodeName == "Filter") {
		 
            		std::shared_ptr<const FilterNode> myFilterNode = std::dynamic_pointer_cast<const FilterNode> (curNode);

            		const TypedExprPtr & filterExpr = myFilterNode->filter();

					exprStr = filterExpr->toString();

					if (exprStr == target) {

						if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr)) {				
					
							core::QueryConfig config({});

							if (auto call1 = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0])) {
								std::string firstMatAddName = call1->name();
								if (auto call2 = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0]->inputs()[0])) {
									std::string firstMatMulName = call2->name();
									if (auto call3 = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])) {
										std::string secondMatAddName = call3->name();
										if (auto call4 = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])) {
											std::string secondMatMulName = call4->name();
											std::shared_ptr<VectorFunction> myMul1 = getVectorFunction(secondMatMulName, {ARRAY(REAL())}, {}, config);
											std::shared_ptr<VectorFunction> myMul2 = getVectorFunction(firstMatMulName, {ARRAY(REAL())}, {}, config);

											std::shared_ptr<VectorFunction> myAdd1 = getVectorFunction(secondMatAddName, {ARRAY(REAL())}, {}, config);
											std::shared_ptr<VectorFunction> myAdd2 = getVectorFunction(firstMatAddName, {ARRAY(REAL())}, {}, config);

											if (myMul1 && myMul2 && myAdd1 && myAdd2) {
												if (auto myMul1UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul1)) {
													if (auto myMul2UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul2)) {
														if (auto myAdd1UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd1)) {
															if (auto myAdd2UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd2)) {
																dims.push_back(myMul1UDF->getDims()[0]);
																dims.push_back(myMul1UDF->getDims()[1]);
																dims.push_back(myMul2UDF->getDims()[1]);

																weights[0] = myMul1UDF->getTensor();
																weights[1] = myMul2UDF->getTensor();

																bias[0] = myAdd1UDF->getTensor();
																bias[1] = myAdd2UDF->getTensor();

																registerVectorFunction(
																"torchDNN",
																TorchDNN::signatures(),
																std::make_unique<TorchDNN>(weights, bias, dims)
																);
															}
														}
													}
												}
													
											}
										}
									}
								}
							}
													
							if (curNode->sources().size() > 0) {

								planBuilder = planBuilder.setRoot(curNode->sources()[0]);

								std::regex pattern(R"(softmax\d+\(mat_add\d+\(mat_mul\d+\(relu\d+\(mat_add\d+\(mat_mul\d+.*\)\)\)\)\))");

								exprStr = std::regex_replace(exprStr, pattern, "torchDNN(v)");

								planBuilder = planBuilder.project({exprStr});

								return true;
							}
						
																												
						}

					}		 		
		     
            	}

	    		std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();

            	if (sources.size() == 0) return false;

            	for (auto source : sources)       		 
            
	         		apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator, targets);
	
				}
			}
    
    }


    std::string name() override {
    
        return "TwoLayerUDF2TorchNNRewriteAction";
    
    }

bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> &targetActions) override {
    try {
        if (!rootNode) {
            throw std::invalid_argument("rootNode is null");
        }

        std::string_view nodeName = rootNode->name();

        if (nodeName == "Project") {
            auto myProjectNode = std::dynamic_pointer_cast<const ProjectNode>(rootNode);

            if (!myProjectNode) {
                throw std::runtime_error("Failed to cast to ProjectNode");
            }

            const std::vector<TypedExprPtr> &projections = myProjectNode->projections();

            for (const auto &expression : projections) {
                std::string expr = expression->toString();
				std::regex pattern(R"(softmax\d+\(mat_add\d+\(mat_mul\d+\(relu\d+\(mat_add\d+\(mat_mul\d+.*\)\)\)\)\))");
                auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);
                auto wordsEnd = std::sregex_iterator();

                for (auto it = wordsBegin; it != wordsEnd; ++it) {
                    targetActions.push_back(it->str());
                }
            }

            return true;  // Return true for successful execution
        }

        if (nodeName == "Filter") {
            auto myFilterNode = std::dynamic_pointer_cast<const FilterNode>(rootNode);

            if (!myFilterNode) {
                throw std::runtime_error("Failed to cast to FilterNode");
            }

            const TypedExprPtr &filterExpr = myFilterNode->filter();
            std::string expr = filterExpr->toString();
            std::regex pattern(R"(softmax\d+\(mat_add\d+\(mat_mul\d+\(relu\d+\(mat_add\d+\(mat_mul\d+.*\)\)\)\)\))");
            auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);
            auto wordsEnd = std::sregex_iterator();

            for (auto it = wordsBegin; it != wordsEnd; ++it) {
                targetActions.push_back(it->str());
            }

            return true;  // Return true for successful execution
        }

        std::vector<std::shared_ptr<const core::PlanNode>> sources = rootNode->sources();

        if (sources.empty()) {
            // Safe exit for leaf node
            return true;
        }

        for (const auto &source : sources) {
            if (!check(source, targetActions)) {
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
	float* weights[2];
	float* bias[2];

};


}