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

							if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {

								std::string callName = call->name();

								if (callName == target) {
							
									if (projections.size() == 1) {
							
										core::QueryConfig config({});

										std::shared_ptr<VectorFunction> UDF = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

										if (UDF) {
							
											std::shared_ptr<TorchDNN> TorchUDF = std::dynamic_pointer_cast<TorchDNN>(UDF);

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
                                            // only focus on inner case :project({torchnnx(v)}) ---> project({softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))})
                                            //TODO: medi case: project(func1(torchnnx(func2())))
                                            //      outer case: project({torchnnx(func1(func2()))})
											planBuilder = planBuilder.setRoot((curNode->sources())[0]);

											std::string twoLayers = fmt::format(compute, "v");

											twoLayers += " AS R1";

											std::regex pattern(target + R"(\([^)]+\))");

											exprStr = std::regex_replace(exprStr, pattern, "R1");

											planBuilder = planBuilder.project({twoLayers}).project({exprStr});
								
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
		 			
		        	if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr)) {

			        	std::string callName = call->name();

			        	if (callName == target) {
			     
                            core::QueryConfig config({});

				            std::shared_ptr<VectorFunction> UDF = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

				            if (UDF) {
				   
				                std::shared_ptr<TorchDNN> TorchUDF = std::dynamic_pointer_cast<TorchDNN>(UDF);

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
    
        return "TorchNN2TwoLayerUDFRewriteAction";
    
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
                std::regex pattern(R"(torchdnn\d+)");
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
            std::regex pattern(R"(torchdnn\d+)");
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
    std::string compute;
	std::string exprStr;

};


}