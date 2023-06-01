
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
  void traverse(const core::PlanFragment& planFragment);
  ~Optimizer();

private:
  std::shared_ptr<core::QueryCtx> queryCtx_;
};

} // namespace facebook::velox::optimizer