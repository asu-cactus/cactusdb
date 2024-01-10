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

class JoinAgg2MulRewriteAction : public RewriteAction {

public:

    JoinAgg2MulRewriteAction (core::PlanNodeId* p0, RowTypePtr v):p0(p0), valueSchema(v){}


    bool apply(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
		   std::string target) override {

        if (curNode) {

            std::string_view nodeName = curNode->name();

	    if (nodeName == "Project") {
		 
            std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode);

            const std::vector<TypedExprPtr> & projections = myProjectNode->projections();

            for (auto expression : projections) {
				str.push_back(expression->toString());
				while (expression->inputs().size() > 0){
					expression = expression->inputs()[0];
				}
		        if (auto call = std::dynamic_pointer_cast<const core::InputTypedExpr>(expression)) {

			        if (str[0].find("result") != std::string::npos) {
			     
			      /* 
			       * Case I: If this function call is the only expression in the projection (e.g.,project({"torch(v)"})), 
			       * we need to split it to the fully extended expression, (e.g.,project({"softmax(mat_add(mat_mul(relu(mat_add(mat_mul(v))))))"})).
			       */
                              
			            if (projections.size() == 1) {
			       
			            // We are the only expression in the project operator

				        // We shall extract the path
                            core::QueryConfig config({});

				            std::shared_ptr<VectorFunction> myMul1 = getVectorFunction("mat_mul_b", {ARRAY(REAL()), ARRAY(REAL())}, {}, config);


				            if (myMul1) {
				   
				                std::shared_ptr<MatrixMultiply_b> myMul1UDF = std::dynamic_pointer_cast<MatrixMultiply_b>(myMul1);


					            if (myMul1UDF) {
					 
					                dims = myMul1UDF->getDims();
									weights = myMul1UDF->getTensor();
									registerVectorFunction(
										"mat_mul0",
										MatrixMultiply::signatures(),
										std::make_unique<MatrixMultiply>(weights, dims[0]*dims[3], dims[1])
									);

									std::string searchString = "ROW[\"result\"]";
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
													.capturePlanNodeId(*p0)
													.project({"mat_mul0(v) AS result"})
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
            
	         apply(source, curNode, maker, planBuilder, pool_, planNodeIdGenerator, target);
	
	}
    
    
    }


    std::string name() override {
    
        return "JoinAgg2MulRewriteAction";
    
    }

	bool check(TypedExprPtr expression) override {
		return true;
	}


private: 

    std::vector<int> dims;
    std::vector<std::string> str;
	int blocks;
	float* weights;
	core::PlanNodeId* p0;
	RowTypePtr valueSchema;


};


}