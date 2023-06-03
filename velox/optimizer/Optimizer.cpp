#include "velox/optimizer/Optimizer.h"
#include "velox/exec/Operator.h"
#include "velox/exec/FilterProject.h"



using namespace facebook::velox::optimizer;

Optimizer::Optimizer(std::shared_ptr<core::QueryCtx> queryCtx) : queryCtx_(queryCtx) {
}

void Optimizer::traverse(const core::PlanFragment& planFragment) {
  auto task_op = std::make_shared<exec::Task>("task_op", planFragment, 0, queryCtx_);
  auto drivers = task_op->op();
  auto operators = drivers[0]->operators();
  for (const auto& op : operators) {
    // TODO
    auto fp = dynamic_cast<exec::FilterProject*>(op);
    if (fp) {
      std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
      std::cout << "The operatorType is: " << op->operatorType() << std::endl;
      std::cout << "\n" << std::endl;
      std::cout << "The expression tree: " << std::endl;
      const std::unique_ptr<exec::ExprSet>& exprs = fp->getExprs();

      std::cout << exprs->toString(false /*compact*/) << std::endl;
      // Use exprs as needed.
    } else {
      std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
      std::cout << "The operatorType is: " << op->operatorType() << std::endl;
      std::cout << "\n" << std::endl;
    }
  }
}

Optimizer::~Optimizer() {
}