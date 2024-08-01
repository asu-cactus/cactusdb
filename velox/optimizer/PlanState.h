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

#include <fmt/core.h>
#include <iostream>
#include <memory>
#include "CataLog.h"
#include "RuleManager.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
namespace optimization {

class PlanState {
 public:
  /**
   * @brief Constructor for the PlanState class.
   *
   * Initializes a PlanState object with a specified RuleManager.
   *
   * @param ruleManager The RuleManager to be associated with the PlanState.
   */
  PlanState(RuleManager ruleManager) : ruleManager(ruleManager) {}

  /**
   * @brief Get possible actions based on a given root node.
   *
   * This function iterates through each rule in the rule manager, checks if the
   * rule can be applied in the given plan represented by the root node, and
   * stores the target UDF names in the 'actions' container, then it maps each
   * action to its corresponding rule name in 'actionsPair'.
   *
   * @param rootNode A shared pointer to the root node of the plan.
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
   */
  void getPossibleActions(
      std::shared_ptr<const core::PlanNode> rootNode,
      CataLog& cataLog) {
    actionsPair.clear();
    // Search for each rule
    for (auto& rulePair : ruleManager.rules) {
      // Get the pointers for rules
      auto& rule = *rulePair.second;
      // Check if rule can be applied in this plan, store target UDF name in
      // actions
      if (rule.check(rootNode, actions, cataLog)) {
        // Create a map to store actions and target UDF names, key is UDF name,
        // value is rule name
        for (const auto& action : actions) {
          LOG(INFO) << "[INFO] PlanState: pushed Action: " << action
                    << " Rule: " << rulePair.first << std::endl;
          actionsPair[action].push_back(rulePair.first);
        }
        // clear target UDF name, prepare for next rule
        actions.clear();
      }
    }
  }

  /**
   * @brief Check whether a rule is valid in actionPairs when apply the rule
   *
   * @param targetRule A rule will be applied
   * @param targetString The target string expression where the rule applies
   */

  bool checkIsValidRule(std::string targetRule, std::string targetString) {
    // action pair is structures as: targetStr: [applicable rule1, applicable
    // rule2, ...]
    auto it = actionsPair.find(targetString);
    if (it != actionsPair.end()) {
      const std::vector<std::string>& applicableRules = it->second;
      if (std::find(
              applicableRules.begin(), applicableRules.end(), targetRule) !=
          applicableRules.end()) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Take action based on the specified target strings to rewrite the
   * logical plan.
   *
   * This function applies rules to the current plan node based on the target
   * strings and rewrites the plan accordingly.
   *
   * Target actions come with [(expr1, action1), (expr2, action2), (expr3,
   * action1)]. It laters will be grouped by the action, hence each rule.apply()
   * will be called only once.
   *
   * @param curNode A pointer to the current plan node, usually point to the
   * last node of logical plan.
   * @param prevNode A pointer to the previous plan node, usually point to the
   * previous node before current node.
   * @param maker A pointer to the VectorMaker, which is a helper class used to
   * build the data source vector.
   * @param planBuilder A pointer to the planBuilder, which is a helper class
   * used to build the logical plan.
   * @param pool_ A pointer to the memory pool, which is used to build the
   * logical plan.
   * @param planNodeIdGenerator A pointer to the planNodeIdGenerator, which is
   * used to track the ID of the plan Node.
   * @param targets A vector for multiple strings, representing the target UDF
   * name that can apply this rewritten rule.
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
   */
  void takeAction(
      std::shared_ptr<const core::PlanNode> curNode,
      std::shared_ptr<const core::PlanNode> prevNode,
      VectorMaker& maker,
      PlanBuilder& planBuilder,
      std::shared_ptr<memory::MemoryPool> pool_,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      const std::vector<std::pair<std::string, std::string>>& targetActions,
      CataLog& cataLog) {
    // Group by actions and expressions by action
    std::unordered_map<std::string, std::vector<std::string>> groupedActions;
    for (auto targetAction : targetActions) {
      auto [targetString, targetRule] = targetAction;
      groupedActions[targetRule].push_back(targetString);
    }

    // Apply selected rules
    for (auto groupedAction : groupedActions) {
      std::string targetRule = groupedAction.first;
      std::vector<std::string> targetStrings = groupedAction.second;

      for (std::string targetString : targetStrings) {
        if (!checkIsValidRule(targetRule, targetString)) {
          throw std::runtime_error(fmt::format(
              "[ERROR]: {} is not a valid rule on: {}\n",
              targetRule,
              targetString));
        }
      }
      // Get the pointer for this rule
      auto rule = ruleManager.pickRule(targetRule);

      if (rule) {
        // Apply rule on this plan
        rule->apply(
            curNode,
            nullptr,
            maker,
            planBuilder,
            pool_,
            planNodeIdGenerator,
            targetStrings,
            cataLog);
        // Store this rule name as the previous action, prepare for next
        // rewritten
        preAction = targetRule;
        // TODO: forbidden preAction in next step. (Avoid cycle)
      } else {
        // Handle the case when the rule is not found
        std::cerr << fmt::format(
                         "[Error]: Rule {} not found for targetString: {}",
                         targetRule,
                         targetStrings)
                  << std::endl;
      }
    }

    actionsPair.clear();
  }

  /**
   * @brief Update the plan state and renew possible actions.
   *
   * This function updates the plan state using the provided PlanBuilder and
   * then renews possible actions based on the updated plan state.
   *
   * @param planBuilder The PlanBuilder used to construct and modify the plan.
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
   */
  void update(PlanBuilder& planBuilder, CataLog& cataLog) {
    // Get the current plan
    auto curNode = planBuilder.planNode();
    // renew possible actions in new state
    getPossibleActions(curNode, cataLog);
  }

  std::map<std::string, std::vector<std::string>> actionsPair;
  std::vector<std::string> actions;
  RuleManager ruleManager;
  std::string preAction;
};

} // namespace optimization