#include <vector>

class Pattern {
public:
    Pattern(OperatorType opr_type) : _opr_type(opr_type) {}

    void append_child(Pattern child) {
        _children.push_back(child);
    }

    const std::vector<Pattern>& children() const {
        return _children;
    }

    OperatorType opr_type() const {
        return _opr_type;
    }

private:
    OperatorType _opr_type;
    std::vector<Pattern> _children;
    
};

#include <stdexcept>

class Rule {
public:
    Rule(RuleType rule_type, Pattern pattern = Pattern(OperatorType::UNKNOWN))
        : _rule_type(rule_type), _pattern(pattern) {}

    RuleType rule_type() const {
        return _rule_type;
    }

    const Pattern& pattern() const {
        return _pattern;
    }

    bool top_match(const Operator* opr) const {
        return opr->opr_type == _pattern.opr_type;
    }

    bool is_implementation_rule() const {
        return _rule_type > RuleType::TRANSFORMATION_DELIMITER;
    }

    bool is_logical_rule() const {
        return _rule_type > RuleType::REWRITE_DELIMITER && _rule_type < RuleType::TRANSFORMATION_DELIMITER;
    }

    bool is_stage_two_rewrite_rules() const {
        return _rule_type > RuleType::TOP_DOWN_DELIMITER && _rule_type < RuleType::REWRITE_DELIMITER;
    }

    bool is_stage_one_rewrite_rules() const {
        return _rule_type < RuleType::TOP_DOWN_DELIMITER;
    }

    virtual int promise() const = 0;

    virtual bool check(const Operator* before, const OptimizerContext* context) const = 0;

    virtual std::vector<Operator*> apply(const Operator* before, const OptimizerContext* context) const = 0; //should be care about the out type
};


class CacheFunctionExpressionInApply : public Rule {
public:
    CacheFunctionExpressionInApply() : Rule(RuleType::CACHE_FUNCTION_EXPRESSION_IN_APPLY, createPattern()) {}

    Promise promise() const override {
        return Promise::CACHE_FUNCTION_EXPRESSION_IN_APPLY;
    }

    bool check(const LogicalApplyAndMerge* before, const OptimizerContext* context) const override {
        const ExpressionTree* expr = before->func_expr;
        // Already cache enabled or not a cacheable function
        if (expr->has_cache() || std::find(CACHEABLE_FUNCTIONS.begin(), CACHEABLE_FUNCTIONS.end(), expr->name) == CACHEABLE_FUNCTIONS.end()) {
            return false;
        }
        // We do not support caching function expression instances with multiple arguments or nested function expressions
        if (expr->children.size() > 1 || !dynamic_cast<const TupleValueExpression*>(expr->children[0])) {
            return false;
        }
        return true;
    }

    std::vector<Operator*> apply(const LogicalApplyAndMerge* before, const OptimizerContext* context) const override {
        // TODO: This will create a catalog entry even in the case of an explain command.
        // We should run this code conditionally.
        ExpressionTree* new_func_expr = enable_cache(context, before->func_expr);
        LogicalApplyAndMerge* after = new LogicalApplyAndMerge(new_func_expr, before->alias, before->do_unnest);
        after->append_child(before->children[0]);
        std::vector<Operator*> result;
        result.push_back(after);
        return result;
    }

private:
    Pattern createPattern() {
        Pattern pattern(OperatorType::LOGICAL_APPLY_AND_MERGE);
        pattern.append_child(Pattern(OperatorType::DUMMY));
        return pattern;
    }
};

#include <vector>
#include <stdexcept>
#include <memory>

class RulesManager {
public:
    RulesManager(const std::unordered_map<std::string, bool>& configs = {}) {
        _logical_rules = {
            std::make_unique<LogicalInnerJoinCommutativity>(),
            std::make_unique<CacheFunctionExpressionInApply>(),
            std::make_unique<CacheFunctionExpressionInFilter>(),
            std::make_unique<CacheFunctionExpressionInProject>(),
        };

        _stage_one_rewrite_rules = {
            std::make_unique<XformLateralJoinToLinearFlow>(),
            std::make_unique<XformExtractObjectToLinearFlow>(),
        };

        _stage_two_rewrite_rules = {
            std::make_unique<EmbedFilterIntoGet>(),
            // std::make_unique<EmbedFilterIntoDerivedGet>(),
            std::make_unique<EmbedSampleIntoGet>(),
            std::make_unique<PushDownFilterThroughJoin>(),
            std::make_unique<PushDownFilterThroughApplyAndMerge>(),
            std::make_unique<CombineSimilarityOrderByAndLimitToVectorIndexScan>(),
            std::make_unique<ReorderPredicates>(),
        };

        _implementation_rules = {
            std::make_unique<LogicalCreateToPhysical>(),
            std::make_unique<LogicalCreateFromSelectToPhysical>(),
            std::make_unique<LogicalRenameToPhysical>(),
            std::make_unique<LogicalCreateFunctionToPhysical>(),
            std::make_unique<LogicalCreateFunctionFromSelectToPhysical>(),
            std::make_unique<LogicalDropObjectToPhysical>(),
            std::make_unique<LogicalInsertToPhysical>(),
            std::make_unique<LogicalDeleteToPhysical>(),
            std::make_unique<LogicalLoadToPhysical>(),
            std::make_unique<LogicalGetToSeqScan>(),
            std::make_unique<LogicalDerivedGetToPhysical>(),
            std::make_unique<LogicalUnionToPhysical>(),
            std::make_unique<LogicalGroupByToPhysical>(),
            std::make_unique<LogicalOrderByToPhysical>(),
            std::make_unique<LogicalLimitToPhysical>(),
            std::make_unique<LogicalJoinToPhysicalNestedLoopJoin>(),
            std::make_unique<LogicalLateralJoinToPhysical>(),
            std::make_unique<LogicalJoinToPhysicalHashJoin>(),
            std::make_unique<LogicalFunctionScanToPhysical>(),
            std::make_unique<LogicalFilterToPhysical>(),
            std::make_unique<LogicalShowToPhysical>(),
            std::make_unique<LogicalExplainToPhysical>(),
            std::make_unique<LogicalCreateIndexToVectorIndex>(),
            std::make_unique<LogicalVectorIndexScanToPhysical>(),
            std::make_unique<LogicalProjectNoTableToPhysical>(),
        };

        bool is_ray_enabled = configs.find("ray") != configs.end() ? configs.at("ray") : false;
        if (is_ray_enabled) {
            _implementation_rules.insert(_implementation_rules.end(), {
                std::make_unique<LogicalExchangeToPhysical>(),
                std::make_unique<LogicalApplyAndMergeToRayPhysical>(),
                std::make_unique<LogicalProjectToRayPhysical>(),
            });
        } else {
            _implementation_rules.insert(_implementation_rules.end(), {
                std::make_unique<LogicalApplyAndMergeToPhysical>(),
                std::make_unique<LogicalProjectToPhysical>(),
            });
        }

        _all_rules.insert(_all_rules.end(), _stage_one_rewrite_rules.begin(), _stage_one_rewrite_rules.end());
        _all_rules.insert(_all_rules.end(), _stage_two_rewrite_rules.begin(), _stage_two_rewrite_rules.end());
        _all_rules.insert(_all_rules.end(), _logical_rules.begin(), _logical_rules.end());
        _all_rules.insert(_all_rules.end(), _implementation_rules.begin(), _implementation_rules.end());
    }

    const std::vector<std::unique_ptr<Rule>>& stage_one_rewrite_rules() const {
        return _stage_one_rewrite_rules;
    }

    const std::vector<std::unique_ptr<Rule>>& stage_two_rewrite_rules() const {
        return _stage_two_rewrite_rules;
    }

    const std::vector<std::unique_ptr<Rule>>& implementation_rules() const {
        return _implementation_rules;
    }

    const std::vector<std::unique_ptr<Rule>>& logical_rules() const {
        return _logical_rules;
    }

    void disable_rules(const std::vector<Rule*>& rules) {
        auto remove_from_list = [](std::vector<std::unique_ptr<Rule>>& rule_list, const Rule* rule_to_remove) {
            auto it = std::remove_if(rule_list.begin(), rule_list.end(), [rule_to_remove](const std::unique_ptr<Rule>& rule) {
                return rule->rule_type() == rule_to_remove->rule_type();
            });
            rule_list.erase(it, rule_list.end());
        };

        for (const Rule* rule : rules) {
            if (rule->is_implementation_rule() || rule->is_stage_one_rewrite_rules() || rule->is_stage_two_rewrite_rules() || rule->is_logical_rule()) {
                throw std::runtime_error("Provided Invalid rule");
            }

            if (rule->is_implementation_rule()) {
                remove_from_list(_implementation_rules, rule);
            } else if (rule->is_stage_one_rewrite_rules()) {
                remove_from_list(_stage_one_rewrite_rules, rule);
            } else if (rule->is_stage_two_rewrite_rules()) {
                remove_from_list(_stage_two_rewrite_rules, rule);
            } else if (rule->is_logical_rule()) {
                remove_from_list(_logical_rules, rule);
            }
        }
    }

    void add_rules(const std::vector<Rule*>& rules) {
        auto add_to_list = [](std::vector<std::unique_ptr<Rule>>& rule_list, const Rule* rule_to_add) {
            if (std::none_of(rule_list.begin(), rule_list.end(), [rule_to_add](const std::unique_ptr<Rule>& rule) {
                return rule->rule_type() == rule_to_add->rule_type();
            })) {
                rule_list.emplace_back(std::make_unique<Rule>(*rule_to_add));
            }
        };

        for (const Rule* rule : rules) {
              if (rule->is_implementation_rule()) {
                add_to_list(_implementation_rules, rule);
            } else if (rule->is_stage_one_rewrite_rules()) {
                add_to_list(_stage_one_rewrite_rules, rule);
            } else if (rule->is_stage_two_rewrite_rules()) {
                add_to_list(_stage_two_rewrite_rules, rule);
            } else if (rule->is_logical_rule()) {
                add_to_list(_logical_rules, rule);
            }
        }
    }

private:
    std::vector<std::unique_ptr<Rule>> _logical_rules;
    std::vector<std::unique_ptr<Rule>> _stage_one_rewrite_rules;
    std::vector<std::unique_ptr<Rule>> _stage_two_rewrite_rules;
    std::vector<std::unique_ptr<Rule>> _implementation_rules;
    std::vector<std::unique_ptr<Rule>> _all_rules;
};


