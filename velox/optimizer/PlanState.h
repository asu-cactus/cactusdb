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
    PlanState(RuleManager ruleManager) : ruleManager(ruleManager) {}

    void getPossibleActions(std::shared_ptr<const core::PlanNode> rootNode) {
        for (auto& rulePair : ruleManager.rules) {
            auto& rule = *rulePair.second;
            if (rule.check(rootNode, actions)) {
                for (const auto& action : actions) {
                    actionsPair[action] = rulePair.first;
                }
                actions.clear();
            }
        }
    }

    void takeAction(std::shared_ptr<const core::PlanNode> curNode,
                    std::shared_ptr<const core::PlanNode> prevNode,
                    VectorMaker& maker,
                    PlanBuilder& planBuilder,
                    std::shared_ptr<memory::MemoryPool> pool_,
                    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
                    const std::vector<std::string>& targetString) {

        std::string targetRule = actionsPair[targetString[0]];
        auto rule = ruleManager.pickRule(targetRule);
        if (rule) {
            rule->apply(curNode, nullptr, maker, planBuilder, pool_, planNodeIdGenerator, targetString);
            preAction = targetRule;
            //TODO: forbidden preAction in next step.
        } else {
            // Handle the case when the rule is not found
            std::cerr << "Error: Rule not found for targetString: " << targetString[0] << std::endl;

        }

        actionsPair.clear();
    }

    void update(PlanBuilder& planBuilder) {
        auto curNode = planBuilder.planNode();
        getPossibleActions(curNode);
    }

    std::map<std::string, std::string> actionsPair;
    std::vector<std::string> actions;
    RuleManager ruleManager;
    std::string preAction;
};


}