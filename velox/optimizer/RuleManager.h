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
// #include "Split2MultiRewriteAction.h"
// #include "Merge2SingleRewriteAction.h"
// #include "Mul2JoinAggRewriteAction.h"
// #include "JoinAgg2MulRewriteAction.h"
#include "TorchNN2TwoLayerUDFRewriteAction.h"
#include "TwoLayerUDF2TorchNNRewriteAction.h"

using namespace optimization;

namespace optimization {

class RuleManager {
public:
    RuleManager() {
        // Initialize the rules
        rules.emplace("TorchNN2TwoLayerUDFRewriteAction", std::make_shared<optimization::TorchNN2TwoLayerUDFRewriteAction>());
        rules.emplace("TwoLayerUDF2TorchNNRewriteAction", std::make_shared<optimization::TwoLayerUDF2TorchNNRewriteAction>());
        // rules.emplace("Merge2Single", std::make_shared<optimization::Merge2SingleRewriteAction>());
        // Add more rules if needed
    }

    std::shared_ptr<optimization::RewriteAction> pickRule(const std::string& ruleName) {
        // Search for the rule by name
        auto it = rules.find(ruleName);
        if (it != rules.end()) {
            return it->second;
        }

        // Rule not found, handle the case (you can throw an exception, return a default rule, etc.)
        // For now, just return nullptr
        return nullptr;
    }

    std::map<std::string, std::shared_ptr<optimization::RewriteAction>> rules;
};


}