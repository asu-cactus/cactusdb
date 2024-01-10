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

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


namespace optimization {

class Merge2SingleRewriteAction : public RewriteAction {

public:

    Merge2SingleRewriteAction () {}


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

					if (callName.find(target) != std::string::npos){
			      /*
			       * Case I: If this function call is two layer expression in the projection (e.g.,project({"softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))"})), 
			       * we need to merge it to the compact expression, (e.g.,project({"torch(v)"})).
			       */
                              
			            if (projections.size() == 1) {
			       
			            // We are the only expression in the project operator


                            core::QueryConfig config({});
							// TODO: replace by call and call2, std::dynamic_pointer_cast<const core::CallTypedExpr>(call->inputs()[0])->name() ---mat_add4
							std::string firstMatMulName = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0])->name();
							std::string firstMatAddName = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0])->name();

							std::string secondMatMulName = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])->name();
							std::string secondMatAddName = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])->name();

							std::shared_ptr<VectorFunction> myMul1 = getVectorFunction(secondMatMulName, {ARRAY(REAL())}, {}, config);
							std::shared_ptr<VectorFunction> myMul2 = getVectorFunction(firstMatMulName, {ARRAY(REAL())}, {}, config);

							std::shared_ptr<VectorFunction> myAdd1 = getVectorFunction(secondMatAddName, {ARRAY(REAL())}, {}, config);
							std::shared_ptr<VectorFunction> myAdd2 = getVectorFunction(firstMatAddName, {ARRAY(REAL())}, {}, config);


				            if (myMul1 && myMul2 && myAdd1 && myAdd2) {
								std::shared_ptr<MatrixMultiply> myMul1UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul1);
								std::shared_ptr<MatrixMultiply> myMul2UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul2);

								std::shared_ptr<MatrixAddition> myAdd1UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd1);
								std::shared_ptr<MatrixAddition> myAdd2UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd2);

					            if (myMul1UDF && myMul2UDF && myAdd1UDF && myAdd2UDF) {
					 
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

                                   // We remove the current node from the plan
				   //
                            if (curNode->sources().size() > 0) {

                                planBuilder = planBuilder.setRoot(curNode->sources()[0]);

                                    // We build the plan from this point

                                planBuilder = planBuilder.project({"torchDNN(v)"});


                                return true;
                            }
			       
			       }
                                                                                             			       

		               /*
				* Case II: If this function call is the only expression in the projection (e.g.,project({"dense1(relu(mat_add(mat_mul(v))))))"})) 
				* or ({"softmax(mat_add(mat_mul(dense0(v))))"}), 
				* we need to merge it to one compact expression (e.g.,project({"torch(v)"}))
				* 
				*/

			       //TODO


			    /*
                * Case III: If this function call is the only expression in the projection (e.g.,project({"dense1(dense0(v))"})), 
				* we need to merge it to one compact expressions, (e.g.,project({"torch(v)"})),
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

		// 	while (filterExpr->inputs().size() > 0) {
		 
		//         if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr)) {

		// 	        std::string callName = call->name();

		// 			if (callName.find(target) != std::string::npos){
			     
		// 	      /* 
		// 	       * Case I: If this function call is two layer expression in the projection (e.g.,project({"softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))"})), 
		// 	       * we need to merge it to the compact expression, (e.g.,project({"torch(v)"})).
		// 	       */
			       
		// 	            // We are the only expression in the project operator


        //                     core::QueryConfig config({});
		// 					// TODO: replace by call and call2, std::dynamic_pointer_cast<const core::CallTypedExpr>(call->inputs()[0])->name() ---mat_add4
		// 					std::string firstMatMulName = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0]->inputs()[0])->name();
		// 					std::string firstMatAddName = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0])->name();

		// 					std::string secondMatMulName = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])->name();
		// 					std::string secondMatAddName = std::dynamic_pointer_cast<const core::CallTypedExpr>(filterExpr->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])->name();

		// 					std::shared_ptr<VectorFunction> myMul1 = getVectorFunction(secondMatMulName, {ARRAY(REAL())}, {}, config);
		// 					std::shared_ptr<VectorFunction> myMul2 = getVectorFunction(firstMatMulName, {ARRAY(REAL())}, {}, config);

		// 					std::shared_ptr<VectorFunction> myAdd1 = getVectorFunction(secondMatAddName, {ARRAY(REAL())}, {}, config);
		// 					std::shared_ptr<VectorFunction> myAdd2 = getVectorFunction(firstMatAddName, {ARRAY(REAL())}, {}, config);


		// 		            if (myMul1 && myMul2 && myAdd1 && myAdd2) {
		// 						std::shared_ptr<MatrixMultiply> myMul1UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul1);
		// 						std::shared_ptr<MatrixMultiply> myMul2UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul2);

		// 						std::shared_ptr<MatrixAddition> myAdd1UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd1);
		// 						std::shared_ptr<MatrixAddition> myAdd2UDF = std::dynamic_pointer_cast<MatrixAddition>(myAdd2);

		// 			            if (myMul1UDF && myMul2UDF && myAdd1UDF && myAdd2UDF) {
					 
		// 							dims.push_back(myMul1UDF->getDims()[0]);
		// 							dims.push_back(myMul1UDF->getDims()[1]);
		// 							dims.push_back(myMul2UDF->getDims()[1]);

		// 							weights[0] = myMul1UDF->getTensor();
		// 							weights[1] = myMul2UDF->getTensor();

		// 							bias[0] = myAdd1UDF->getTensor();
		// 							bias[1] = myAdd2UDF->getTensor();

		// 							registerVectorFunction(
		// 							  "torchDNN",
		// 							  TorchDNN::signatures(),
		// 							  std::make_unique<TorchDNN>(weights, bias, dims)
		// 							);
		// 			            }
				   
		// 		            }

        //                            // We remove the current node from the plan
		// 		   //
        //                     if (curNode->sources().size() > 0) {

        //                         planBuilder = planBuilder.setRoot(curNode->sources()[0]);

        //                             // We build the plan from this point

        //                         planBuilder = planBuilder.project({"torchDNN(v)"});


        //                         return true;
        //                     }
			       
			       
                                                                                             			       

		//                /*
		// 		* Case II: If this function call is the only expression in the projection (e.g.,project({"dense1(relu(mat_add(mat_mul(v))))))"})) 
		// 		* or ({"softmax(mat_add(mat_mul(dense0(v))))"}), 
		// 		* we need to merge it to one compact expression (e.g.,project({"torch(v)"}))
		// 		* 
		// 		*/

		// 	       //TODO


		// 	    /*
        //         * Case III: If this function call is the only expression in the projection (e.g.,project({"dense1(dense0(v))"})), 
		// 		* we need to merge it to one compact expressions, (e.g.,project({"torch(v)"})),
        //         */

		// 	       //TODO
			        


			  
		// 	}
		 
		//      }
		// 	 filterExpr = filterExpr->inputs()[0];
		 
		// 	} 
		     
        //     }

	    std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();

            if (sources.size() == 0) return false;

            for (auto source : sources)       		 
            
	         apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator, target);
	
	}
    
    
    }


    std::string name() override {
    
        return "Merge2SingleRewriteAction";
    
    }

	bool check(TypedExprPtr expression) override {
		if (auto call0 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {
			std::string callName0 = call0->name();
			if (auto call1 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0])) {
				std::string callName1 = call1->name();
				if ((callName0.find("softmax") != std::string::npos) && (callName1.find("mat_mul") != std::string::npos)) {
					if (auto call2 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0]->inputs()[0])) {
						std::string callName2 = call2->name();
						if (auto call3 = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0]->inputs()[0])) {
							std::string callName3 = call3->name();
							if ((callName2.find("relu") != std::string::npos) && (callName3.find("mat_mul")!= std::string::npos)) {
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
					else {
						return false;
					}
				}
				else {
					return false;
				}
			}
			else {
				return false;
			}
		}
		else {
			return false;
		}
	}


private: 

    std::vector<int> dims;
	float* weights[2];
	float* bias[2];
};


}