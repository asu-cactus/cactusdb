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
#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/Expressions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/core/ITypedExpr.h"
#include <regex>

using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


namespace optimization {

class DecisionForestUDF2RelationRewriteAction : public RewriteAction {

public:

    DecisionForestUDF2RelationRewriteAction () {}


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
		 
            			std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode);

            			const std::vector<TypedExprPtr> & projections = myProjectNode->projections();

            			for (auto expression : projections) {
		 
		     				if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {

			  					std::string callName = call->name();

			  					if (callName.find(target) != std::string::npos) {
			     
									/* "decision_forest_predict"
									* Case I: If this function call is the only expression in the projection (e.g.,project({"decision_forest_predict(x)"})), 
									* we need to extract the model path from the expression to initialize treeRowVector as a member of this object, 
									* and then replace this node by the following plan:
									*
									* nestedLoopJoin(
									*    exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
									*    .values({treeRowVector})
									*    .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
									*    .planNode(), {"row_id", "x", "tree_id", "tree"})
									*    .project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
									*    .aggregation({"row_id"}, {"sum(prediction) as sum"},{}, core::AggregationNode::Step::kPartial, false)
									*    .project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"})
									*/
												
									if (projections.size() == 1) {
									
										// We are the only expression in the project operator

										// We shall extract the path
										core::QueryConfig config({});

										std::shared_ptr<VectorFunction> myUDF = getVectorFunction(callName, {ARRAY(REAL())}, {}, config);

										if (myUDF) {
									
											std::shared_ptr<ForestPrediction> myDecisionForestUDF = dynamic_pointer_cast<ForestPrediction>(myUDF);

											if (myDecisionForestUDF) {
										
												std::string modelPath = myDecisionForestUDF->getForestPath();

												std::vector<std::string> pathVectors;
																
												Forest::vectorizeForestFolder(modelPath, pathVectors);

												int numTrees = pathVectors.size();

												auto model = maker.flatVector<StringView> (pathVectors.size());

												for (int i = 0; i < numTrees; i++) {

													model->set(i, StringView(pathVectors[i].c_str()));

												}

												auto treeIndexVector = maker.flatVector<int16_t>(numTrees);

												for (int i = 0; i < numTrees; i++) {

													treeIndexVector->set(i, i);

												}

												treeRowVector = maker.rowVector({"tree_id", "tree_path"}, {treeIndexVector, model});
										
											}
									
										}

													// We remove the current node from the plan
									//
										if (curNode->sources().size() > 0) {

											planBuilder = planBuilder.setRoot((curNode->sources())[0]);

											// We build the plan from this point
									
											planBuilder = planBuilder.nestedLoopJoin(exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
															.values({treeRowVector})
															.project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
															.planNode(), {"row_id", "x", "tree_id", "tree"})
															.project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
															.aggregation({"row_id"}, {"sum(prediction) as sum"},{}, core::AggregationNode::Step::kPartial, false)
															.project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"});

											return true;
										}
									
									}
																																

										/*
									* Case II: If this function call is the last expression in the projection (e.g., project({"decision_forest_predict(func1(func2(x)))"})), 
									* we need to split this project node into two project nodes (e.g., project({"func1(func2(x)) as y"}).project("decision_forest_predict(y)"))
									* Then, we apply the rewrite action as described in Case I.
									*/

									//TODO


									/*
									* Case III: If this function call is a middle expression in the projection 
									* (e.g., project({"func4(func3(decision_forest_predict(func1(func2(x)))))"})), 
									* we need to split this project node into three project nodes 
									* (e.g., project({"func1(func2(x)) as z0"}).project("decision_forest_predict(z0) as z1").project("func4(func3(z1))"))
									* Then, we apply the rewrite action as described in Case I.
									*/

									//TODO
			        
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
    
        return "DecisionForestUDF2RelationRewriteAction";
    
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
					// For this rule, we only check wheather decision_forest_predict UDF function is in expressions
					std::regex pattern(R"(decision_forest_predict)");

					auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);

					auto wordsEnd = std::sregex_iterator();
					// Find applicable UDF name and store in targetActions
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
				// For this rule, we only check wheather decision_forest_predict UDF function is in expressions
				std::regex pattern(R"(decision_forest_predict)");

				auto wordsBegin = std::sregex_iterator(expr.begin(), expr.end(), pattern);

				auto wordsEnd = std::sregex_iterator();
				// Find applicable UDF name and store in targetActions
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

    RowVectorPtr treeRowVector;


};


}
