#include <vector>
#include <algorithm>
#include <iterator>

class Binder {
public:
    Binder(GroupExpression* grp_expr, Pattern pattern, Memo* memo)
        : _grp_expr(grp_expr), _pattern(pattern), _memo(memo) {}

    static std::vector<Operator> grp_binder(int idx, Pattern pattern, Memo* memo) {
        GroupExpression* grp = memo->groups[idx];

        std::vector<Operator> result;
        for (auto expr : grp->logical_exprs) {
            auto binders = binder(expr, pattern, memo);
            result.insert(result.end(), binders.begin(), binders.end());
        }

        return result;
    }

    static std::vector<Operator> binder(GroupExpression* expr, Pattern pattern, Memo* memo) {
        assert(expr != nullptr);
        std::vector<Operator> result;
        auto curr_iterator = std::vector<Operator>();
        std::vector<std::vector<Operator>> child_binders;

        if (pattern.opr_type != OperatorType::DUMMY) {
            curr_iterator = {expr->opr};
            if (expr->opr.opr_type != pattern.opr_type) {
                return result;
            }

            if (pattern.children.size() != expr->children.size()) {
                return result;
            }

            for (size_t i = 0; i < expr->children.size(); i++) {
                child_binders.push_back(grp_binder(expr->children[i], pattern.children[i], memo));
            }
        } else {
            // Record the group id in a Dummy Operator
            curr_iterator = {Dummy(expr->group_id, expr->opr)};
        }

        std::vector<Operator> iterators{curr_iterator};
        for (auto& binder : child_binders) {
            iterators.push_back(binder);
        }

        for (const auto& match : product(iterators.begin(), iterators.end())) {
            auto x = build_opr_tree_from_pre_order_repr(match);
            result.push_back(x);
        }

        return result;
    }

    static Operator build_opr_tree_from_pre_order_repr(const std::vector<Operator>& pre_order_repr) {
        Operator opr_tree = pre_order_repr[0];
        if (!opr_tree.isOperator()) {
            throw std::runtime_error("Unknown operator encountered, expected Operator");
        }

        opr_tree.children.clear();

        for (size_t i = 1; i < pre_order_repr.size(); i++) {
            opr_tree.append_child(build_opr_tree_from_pre_order_repr(pre_order_repr[i]));
        }

        return opr_tree;
    }

    std::vector<Operator> operator()() {
        std::vector<Operator> result;
        auto matches = binder(_grp_expr, _pattern, _memo);
        for (const auto& match : matches) {
            auto x = build_opr_tree_from_pre_order_repr(match);
            result.push_back(x);
        }
        return result;
    }

private:
    GroupExpression* _grp_expr;
    Pattern _pattern;
    Memo* _memo;
};
