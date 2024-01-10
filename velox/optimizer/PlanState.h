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
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "RuleManager.h"

namespace optimization {

class PlanState {

public:

    PlanState(RuleManager ruleManager):ruleManager(ruleManager){}

    void getPossibleActions(std::shared_ptr<const core::PlanNode> curNode){
        if (curNode) {
            std::string_view nodeName = curNode->name();
            if (nodeName == "Project") {
                std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode);
                const std::vector<TypedExprPtr> & projections = myProjectNode->projections();
                for (auto expression : projections) {
                    while (expression->inputs().size() > 0) {
                        for (auto& rulePtr : ruleManager.rules) {
                            auto& rule = *rulePtr;
                            if (rule.check(expression)) {
                                std::string key = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)->name();
                                std::string value = rule.name();
                                auto it = actions.find(key);
                                if (it != actions.end()) {
                                    // Key already exists, append the value to the set
                                    it->second.insert(value);
                                } else {
                                    // Key doesn't exist, insert a new key-value pair
                                    actions[key] = {value};
                                }
                            }
                        }
                        expression = expression->inputs()[0];
                    }
                }
            }
        }
    }

    void takeAction(std::shared_ptr<const core::PlanNode> curNode, 
	       std::shared_ptr<const core::PlanNode> prevNode, 
	       VectorMaker & maker,
	       PlanBuilder & planBuilder,
	       std::shared_ptr<memory::MemoryPool> pool_,
	       std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
		   std::string targetString, 
           std::string tagetRule){

        auto rule = ruleManager.pickRule(tagetRule);
        (*rule).apply(curNode, nullptr, maker, planBuilder, pool_, planNodeIdGenerator, targetString);
        preAction = tagetRule;
        actions.erase(targetString);
    }

    void update(PlanBuilder & planBuilder){
        auto curNode = planBuilder.planNode();
        getPossibleActions(curNode);
    }



    std::map<std::string, std::set<std::string>> actions;
    RuleManager ruleManager;
    std::string preAction;

};


}