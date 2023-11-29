
#pragma once

#include <memory>
#include <iostream>
#include "velox/core/PlanFragment.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::optimizer {



class Optimizer {
public:
  Optimizer(std::shared_ptr<core::QueryCtx> queryCtx);
  // std::vector<exec::Operator*> traverse(const core::PlanFragment& planFragment);
  // RowVectorPtr op(const core::PlanFragment& planFragment);
  void getCandidates(std::vector<exec::Operator*> operators);
  // core::PlanNodePtr& applyRules(const core::PlanFragment& planFragment);
  // core::PlanNodePtr& op(const core::PlanFragment& planFragment);
  ~Optimizer();

private:
  std::shared_ptr<core::QueryCtx> queryCtx_;
  // std::vector<OptimizerCandidate> candidates;
};

} // namespace facebook::velox::optimizer