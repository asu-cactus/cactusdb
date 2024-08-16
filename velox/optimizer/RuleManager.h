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
#include "RewriteAction.h"

#include "TorchNN2TwoLayerUDFRewriteAction.h"
#include "TwoLayerUDF2TorchNNRewriteAction.h"
#include "DecisionForestUDF2RelationRewriteAction.h"
#include "Mul2JoinAggRewriteAction.h"
#include "Mul2JoinAggHorizontalRewriteAction.h"
#include "MultiLayerUDF2TorchNNRewriteAction.h"
#include "MLDecompositionPushdownRewriteAction.h"


namespace optimization {

class RuleManager {
public:
    RuleManager() {
        // Initialize the rules
        rules.emplace("TorchNN2TwoLayerUDFRewriteAction", std::make_shared<TorchNN2TwoLayerUDFRewriteAction>());

        rules.emplace("Mul2JoinAggHorizontalRewriteAction", std::make_shared<Mul2JoinAggHorizontalRewriteAction>());

        // rules.emplace("TwoLayerUDF2TorchNNRewriteAction", std::make_shared<TwoLayerUDF2TorchNNRewriteAction>());

        rules.emplace("DecisionForestUDF2RelationRewriteAction", std::make_shared<DecisionForestUDF2RelationRewriteAction>());

        rules.emplace("Mul2JoinAggRewriteAction", std::make_shared<Mul2JoinAggRewriteAction>());

        rules.emplace("MultiLayerUDF2TorchNNRewriteAction", std::make_shared<MultiLayerUDF2TorchNNRewriteAction>());

        rules.emplace("MLDecompositionPushdownRewriteAction", std::make_shared<MLDecompositionPushdownRewriteAction>());

        // Add more rules if needed
    }
    /**
     * @brief A function to obtain the pointer for a rule given its name.
     * 
     * @param ruleName A string for the rule name.
     * 
     * @return A pointer to the rule.
    */
    std::shared_ptr<RewriteAction> pickRule(const std::string& ruleName) {
        // Search for the rule by name
        auto it = rules.find(ruleName);

        if (it != rules.end()) {

            return it->second;
            
        }

        // Rule not found, handle the case (throw an exception, return a default rule, etc.)
        // For now, just return nullptr
        return nullptr;
    }

    std::map<std::string, std::shared_ptr<RewriteAction>> rules;
};


}