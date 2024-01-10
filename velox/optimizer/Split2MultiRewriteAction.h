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

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


namespace optimization {

class Split2MultiRewriteAction : public RewriteAction {

public:

    Split2MultiRewriteAction () {}


    bool apply(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
		   std::string target ) override {

        if (curNode) {

            std::string_view nodeName = curNode->name();

	    if (nodeName == "Project") {
		 
            std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode);

            const std::vector<TypedExprPtr> & projections = myProjectNode->projections();

            for (auto expression : projections) {
				while (expression->inputs().size() > 0) {
		        if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {

			        std::string callName = call->name();

			        if (callName.find(target) != std::string::npos) {
			     
			      /* 
			       * Case I: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
			       * we need to split it to the fully extended expression, (e.g.,project({"softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))"})).
			       */
                              
			            if (projections.size() == 1) {
			       
			            // We are the only expression in the project operator

				        // We shall extract the path
                            core::QueryConfig config({});

				            std::shared_ptr<VectorFunction> myUDF = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

				            if (myUDF) {
				   
				                std::shared_ptr<TorchDNN> myTorchUDF = std::dynamic_pointer_cast<TorchDNN>(myUDF);

					            if (myTorchUDF) {
					 
					                dims = myTorchUDF->getDims();

                                    float** weights = myTorchUDF->getWeights();

                                    float** bias = myTorchUDF->getBias();
                                    
                                    compute =  NNBuilder()
                                    .denseLayer(dims[1], dims[0], weights[0], bias[0], NNBuilder::RELU)
                                    .denseLayer(dims[2], dims[1], weights[1], bias[1], NNBuilder::SOFTMAX)
                                    .build();
					            }
				   
				            }

                                   // We remove the current node from the plan
				   //
                            if (curNode->sources().size() > 0) {

                                planBuilder = planBuilder.setRoot((curNode->sources())[0]);

                                    // We build the plan from this point

                                planBuilder = planBuilder
                                             .project({fmt::format(compute, "v")});


                                return true;
                            }
			       
			       }
                                                                                             			       

		               /*
				* Case II: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
				* we need to split it to one dense layer expression and multi core functions expression, (e.g.,project({"dense1(relu(mat_add(mat_mul(v))))))"})), 
                * or project({"softmax(mat_add(mat_mul(dense0(v))))"})
				* 
				*/

			       //TODO


			    /*
                * Case III: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
				* we need to split it to two dense layer expressions, (e.g.,project({"dense1(dense0(v))"})),
                */

			       //TODO
			        


			  }

		     	}
				expression = expression->inputs()[0];
				}
		 
		 	}		 
		     
        }

		// if (nodeName == "Filter") {
		 
        //     std::shared_ptr<const FilterNode> myFilterNode = std::dynamic_pointer_cast<const FilterNode> (curNode);

        //     const TypedExprPtr & filterExpr = myFilterNode->filter();
		 
		//         if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr)) {

		// 	        std::string callName = call->name();

		// 	        if (callName.find(target) != std::string::npos) {
			     
		// 	      /* 
		// 	       * Case I: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
		// 	       * we need to split it to the fully extended expression, (e.g.,project({"softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))"})).
		// 	       */
			       
		// 	            // We are the only expression in the filter operator
        //                     core::QueryConfig config({});

		// 		            std::shared_ptr<VectorFunction> myUDF = getVectorFunction(target, {ARRAY(REAL())}, {}, config);

		// 		            if (myUDF) {
				   
		// 		                std::shared_ptr<TorchDNN> myTorchUDF = std::dynamic_pointer_cast<TorchDNN>(myUDF);

		// 			            if (myTorchUDF) {
					 
		// 			                dims = myTorchUDF->getDims();

        //                             float** weights = myTorchUDF->getWeights();

        //                             float** bias = myTorchUDF->getBias();
                                    
        //                             compute =  NNBuilder()
        //                             .denseLayer(dims[1], dims[0], weights[0], bias[0], NNBuilder::RELU)
        //                             .denseLayer(dims[2], dims[1], weights[1], bias[1], NNBuilder::SOFTMAX)
        //                             .build();
		// 			            }
				   
		// 		            }

        //                            // We remove the current node from the plan
		// 		   //
        //                     if (curNode->sources().size() > 0) {

        //                         planBuilder = planBuilder.setRoot((curNode->sources())[0]);

        //                             // We build the plan from this point

        //                         planBuilder = planBuilder
        //                                      .project({fmt::format(compute, "v")});


        //                         return true;
        //                     }
			       
			       
                                                                                             			       

		//                /*
		// 		* Case II: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
		// 		* we need to split it to one dense layer expression and multi core functions expression, (e.g.,project({"dense1(relu(mat_add(mat_mul(v))))))"})), 
        //         * or project({"softmax(mat_add(mat_mul(dense0(v))))"})
		// 		* 
		// 		*/

		// 	       //TODO


		// 	    /*
        //         * Case III: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
		// 		* we need to split it to two dense layer expressions, (e.g.,project({"dense1(dense0(v))"})),
        //         */

		// 	       //TODO
			        


		// 	  }
		 
		//      }
		 
		 		 
		     
        //     }

	    std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();

            if (sources.size() == 0) return false;

            for (auto source : sources)       		 
            
	         apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator, target);
	
	}
    
    
    }


    std::string name() override {
    
        return "Split2MultiRewriteAction";
    
    }

	bool check(TypedExprPtr expression) override {
		if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {
			std::string callName = call->name();
			if (callName.find("torchdnn") != std::string::npos) {
				return true;
			}
			else {
				return false;
			}
		}
		else {
			return false;
		}


	}

	// bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> & actionPairs) override {
	// 	if (rootNode) {
	// 		std::string_view nodeName = curNode->name();
	// 		if (nodeName == "Project") {
	// 			std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (rootNode);
	// 			const std::vector<TypedExprPtr> & projections = myProjectNode->projections();
	// 			for (auto expression : projections) {
	// 				while (expression->inputs().size() > 0) {
	// 					if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {
	// 						std::string callName = call->name();
	// 						if (callName.find("torchdnn") != std::string::npos) {
	// 							actionPairs.push_back(callName)
	// 						}
	// 					}
	// 					expression = expression->inputs()[0];
	// 				}
	// 			}
	// 		}
	// 		if (nodeName == "Filter") {
	// 			std::shared_ptr<const FilterNode> myFilterNode = std::dynamic_pointer_cast<const FilterNode> (rootNode);
	// 		}
	// 	}
	// 	if (actionPairs.size() > 0)
	// 		return true;
	// 	else
	// 		return false;
	// }


private: 

    std::vector<int> dims;
    std::string compute;

};


}