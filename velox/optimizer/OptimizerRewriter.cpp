class OptimizerRewriter {
public:
    OptimizerRewriter(EvaDBDatabase* db, RulesManager* rules_manager = nullptr, CostModel* cost_model = nullptr)
        : db(db), rules_manager(rules_manager), cost_model(cost_model) {
        this->rules_manager = rules_manager ? rules_manager : new RulesManager({{"ray", is_ray_enabled}});
        this->cost_model = cost_model ? cost_model : new CostModel();
    }

    void execute_task_stack(OptimizerTaskStack& task_stack) {
        while (!task_stack.empty()) {
            OptimizerTask* task = task_stack.pop();
            task->execute();
        }
    }

    Operator* optimize(Operator* logical_plan) {
        // OptimizerContext optimizer_context(db, cost_model, *rules_manager);
        // Memo& memo = optimizer_context.memo;
        // GroupExpression& grp_expr = optimizer_context.add_opr_to_group(logical_plan);
        // int root_grp_id = grp_expr.group_id;
        // const LogicalExpression& root_expr = memo.groups[root_grp_id].logical_exprs[0];
        auto root_expr = getRootExpr(logical_plan);
        // TopDown Rewrite
        optimizer_context.task_stack.push(new TopDownRewrite(root_expr, rules_manager->type_one_rewrite_rules, optimizer_context));
        execute_task_stack(optimizer_context.task_stack);

        // BottomUp Rewrite
        // root_expr = memo.groups[root_grp_id].logical_exprs[0];
        // optimizer_context.task_stack.push(new BottomUpRewrite(root_expr, rules_manager->stage_two_rewrite_rules, optimizer_context));
        // execute_task_stack(optimizer_context.task_stack);

        // // Optimize Expression (logical -> physical transformation)
        // Group& root_group = memo.get_group_by_id(root_grp_id);
        // optimizer_context.task_stack.push(new OptimizeGroup(root_group, optimizer_context));
        // execute_task_stack(optimizer_context.task_stack);

        // // Build Optimal Tree
        // Operator* optimal_plan = build_optimal_physical_plan(root_grp_id, optimizer_context);
        return optimal_plan;
    }

    Operator* build(Operator* logical_plan) {
        try {
            Operator* plan = optimize(logical_plan);
            return plan;
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("Optimizer timed out!");
        }
    }

private:
    RulesManager* rules_manager;
    CostModel* cost_model;
};
