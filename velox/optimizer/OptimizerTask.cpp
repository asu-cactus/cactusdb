#include <vector>
#include <algorithm>

class OptimizerTask {
public:
    OptimizerTask(OptimizerContext* optimizer_context, OptimizerTaskType task_type)
        : _task_type(task_type), _optimizer_context(optimizer_context) {}

    OptimizerContext* optimizer_context() {
        return _optimizer_context;
    }

    virtual void execute() = 0;

private:
    OptimizerTaskType _task_type;
    OptimizerContext* _optimizer_context;
};

class OptimizerTaskStack {
public:
    OptimizerTaskStack() {}

    void push(std::shared_ptr<OptimizerTask> task) {
        _task_stack.push(task);
    }

    std::shared_ptr<OptimizerTask> pop() {
        if (!_task_stack.empty()) {
            std::shared_ptr<OptimizerTask> task = _task_stack.top();
            _task_stack.pop();
            return task;
        } else {
            // Handle the case when the stack is empty, you can throw an exception or return a nullptr.
            // For example, you can throw an exception:
            throw std::runtime_error("The task stack is empty.");
        }
    }

    bool empty() const {
        return _task_stack.empty();
    }

private:
    std::stack<std::shared_ptr<OptimizerTask>> _task_stack;
};

class TopDownRewrite : public OptimizerTask {
public:
    TopDownRewrite(
        GroupExpression* root_expr,
        std::vector<Rule>& rule_set,
        OptimizerContext* optimizer_context
    ) : root_expr(root_expr), rule_set(rule_set), OptimizerTask(optimizer_context, OptimizerTaskType::TOP_DOWN_REWRITE) {}

    void execute() override {
        std::vector<Rule> valid_rules;
        for (Rule& rule : rule_set) {
            if (!root_expr->is_rule_explored(rule.rule_type) && rule.top_match(root_expr->opr)) {
                valid_rules.push_back(rule);
            }
        }

        std::sort(valid_rules.begin(), valid_rules.end(), [](const Rule& a, const Rule& b) {
            return a.promise() < b.promise();
        });

        for (Rule& rule : valid_rules) {
            Binder binder(root_expr, rule.pattern, optimizer_context->memo);//need to add iteration method to invoke
            for (auto match : binder) {
                if (!rule.check(match, optimizer_context)) {
                    continue;
                }
                auto after = rule.apply(match, optimizer_context);
                auto plans = after.toList();
                assert(plans.size() <= 1 && "Rewrite rule cannot generate more than one alternate plan.");
                for (const auto& plan : plans) {
                    auto new_expr = optimizer_context->replace_expression(plan, root_expr->group_id);
                    optimizer_context->task_stack.push(new TopDownRewrite(new_expr, rule_set, optimizer_context));
                    // The root has changed, so we cannot apply more rules to the same root, hence return
                    return;
                }

                root_expr->mark_rule_explored(rule.rule_type);
            }
        }

        for (auto child : root_expr->children) {
            auto child_expr = optimizer_context->memo->groups[child]->logical_exprs[0];
            optimizer_context->task_stack.push(new TopDownRewrite(child_expr, rule_set, optimizer_context));
        }
    }

private:
    GroupExpression* root_expr;
    std::vector<Rule>& rule_set;
};


