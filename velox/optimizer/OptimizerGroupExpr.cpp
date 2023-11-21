class GroupExpression {
public:
    GroupExpression(Operator* opr, int group_id = UNDEFINED_GROUP_ID, const std::vector<int>& children = std::vector<int>())
        : _opr(opr), _group_id(group_id), _children(children), _rules_explored(INVALID_RULE) {
        // Remove this assert after fixing optimizer_context:_xform_opr_to_group_expr
        if (!_opr->children().empty()) {
            throw std::runtime_error("Cannot create a group expression from operator with children");
        }
    }

    Operator* opr() {
        return _opr;
    }

    int group_id() {
        return _group_id;
    }

    void set_group_id(int new_id) {
        _group_id = new_id;
    }

    std::vector<int> children() {
        return _children;
    }

    void set_children(const std::vector<int>& new_children) {
        _children = new_children;
    }

    void append_child(int child_id) {
        _children.push_back(child_id);
    }

    RuleType rules_explored() {
        return _rules_explored;
    }

    bool is_logical() {
        return _opr->is_logical();
    }

    void mark_rule_explored(RuleType rule_id) {
        _rules_explored |= rule_id;
    }

    bool is_rule_explored(RuleType rule_id) {
        return (_rules_explored & rule_id) == rule_id;
    }

    bool operator==(const GroupExpression& other) const {
        return _group_id == other.group_id()
            && _opr == other.opr()
            && _children == other.children();
    }

    std::string to_string() const {
        // Implement the logic to convert the GroupExpression to a string.
    }

    size_t hash() const {
        // Implement the logic to compute the hash of the GroupExpression.
    }

private:
    Operator* _opr;
    int _group_id;
    std::vector<int> _children;
    RuleType _rules_explored;

    // Define other private members and functions as needed
};