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

class Mul2JoinAggRewriteAction : public RewriteAction {

public:

    Mul2JoinAggRewriteAction (int blocks, core::PlanNodeId* p1, core::PlanNodeId* p2, RowTypePtr v, RowTypePtr w): blocks(blocks), p1(p1), p2(p2), valueSchema(v), weightSchema(w){}


    bool apply(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator ) override {

        if (curNode) {

            std::string_view nodeName = curNode->name();

	    if (nodeName == "Project") {
		 
            std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode);

            const std::vector<TypedExprPtr> & projections = myProjectNode->projections();

            for (auto expression : projections) {
		 
		        if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {

			        std::string callName = call->name();

			        if (callName.find("softmax") != std::string::npos) {
			     
			      /* 
			       * Case I: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
			       * we need to split it to the fully extended expression, (e.g.,project({"softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))"})).
			       */
                              
			            if (projections.size() == 1) {
			       
			            // We are the only expression in the project operator

				        // We shall extract the path
                            core::QueryConfig config({});

				            std::shared_ptr<VectorFunction> myMul1 = getVectorFunction("mat_mul0", {ARRAY(REAL())}, {}, config);
							std::shared_ptr<VectorFunction> myMul2 = getVectorFunction("mat_mul3", {ARRAY(REAL())}, {}, config);

				            if (myMul1 && myMul2) {
				   
				                std::shared_ptr<MatrixMultiply> myMul1UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul1);
								std::shared_ptr<MatrixMultiply> myMul2UDF = std::dynamic_pointer_cast<MatrixMultiply>(myMul2);

					            if (myMul1UDF && myMul2UDF) {
					 
					                dims = myMul1UDF->getDims();
									weights = myMul1UDF->getTensor();
									registerVectorFunction(
										"mat_mul_b",
										MatrixMultiply_b::signatures(),
										std::make_unique<MatrixMultiply_b>(dims[0]/blocks, dims[1], 1000, weights, blocks)
									);

									str.push_back(expression->toString());

									std::string searchString = "mat_mul0(ROW[\"v\"])";
      								std::string replaceString = "result";

									std::size_t found = str[0].find(searchString);
									while (found != std::string::npos) {
										str[0].replace(found, searchString.length(), replaceString);
										found = str[0].find(searchString, found + replaceString.length());
									}

					            }
				   
				            }

                                   // We remove the current node from the plan
				   //
                            if (curNode->sources().size() > 0) {

									planBuilder = exec::test::PlanBuilder(planNodeIdGenerator)
													.tableScan(valueSchema)
													.capturePlanNodeId(*p1)
													.hashJoin(
														{"v_col"},
														{"w_row"},
														exec::test::PlanBuilder(planNodeIdGenerator)
													.tableScan(weightSchema)
													.capturePlanNodeId(*p2)
													.planNode(),
														"", // extra filter
														{"v_row", "w_col", "v", "w"})
													.project({"v_row", "w_col", "mat_mul_b(v, w) AS mp"})
													.singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS result"})
													.project({str[0]});

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
		 
		 }		 
		     
            }
	
	    std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();

            if (sources.size() == 0) return false;

            for (auto source : sources)       		 
            
	         apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator);
	
	}
    
    
    }


    std::string name() override {
    
        return "Mul2JoinAggRewriteAction";
    
    }


private: 

    std::vector<int> dims;
    std::vector<std::string> str;
	int blocks;
	float* weights;
	core::PlanNodeId* p1;
	core::PlanNodeId* p2;
	RowTypePtr valueSchema;
	RowTypePtr weightSchema;

};


}