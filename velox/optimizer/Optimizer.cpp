#include "velox/optimizer/Optimizer.h"
#include "velox/exec/Operator.h"
#include "velox/exec/FilterProject.h"


using namespace facebook::velox::exec;
using namespace facebook::velox::optimizer;
using namespace facebook::velox;


Optimizer::Optimizer(std::shared_ptr<core::QueryCtx> queryCtx) : queryCtx_(queryCtx) {

}

std::vector<Operator*> Optimizer::traverse(const core::PlanFragment& planFragment) {
  auto task_op = exec::Task::create("task_op", planFragment, 0, queryCtx_);
  // auto task_op = std::make_shared<exec::Task>("task_op", planFragment, 0, queryCtx_);
  auto drivers = task_op->op();
  auto operators = drivers[0]->operators();

  return operators;
}

// void Optimizer::getCandidates(std::vector<Operator*> operators) {
//   for (const auto& op : operators) {
//     // TODO: Add more logic to determine if the operator should be insert to candidates.
//     // Here only check the type of an operator.
//     auto fp = dynamic_cast<exec::FilterProject*>(op);
//     if (fp) {
//         // Store the operator index and its related ExprSet in the candidates vector
//         candidates.emplace_back(op->planNodeId(), fp->getExprs());
//       }
//     if (fp) {
//       std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
//       std::cout << "The operatorType is: " << op->operatorType() << std::endl;
//       std::cout << "\n" << std::endl;
//       std::cout << "The expression tree: " << std::endl;
//       const std::unique_ptr<exec::ExprSet>& exprs = fp->getExprs();

//       std::cout << exprs->toString(false /*compact*/) << std::endl;
//       candidates.emplace_back(op->planNodeId(), fp->getExprs());
//       // Use exprs as needed.
//     } else {
//       std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
//       std::cout << "The operatorType is: " << op->operatorType() << std::endl;
//       std::cout << "\n" << std::endl;
//     }
//   }
// }

// std::vector<Operator*> makeOperators(const core::PlanFragment& planFragment) {

// }

// core::PlanNodePtr& Optimizer::applyRules(const core::PlanFragment& planFragment, OptimizerCandidate candidate) {

//   return optimizedPlan;
// }



// core::PlanNodePtr& Optimizer::op(const core::PlanFragment& planFragment) {
//   auto operators = traverse(planFragment);
//   getCandidates(operators);
//   for (const auto& candidate : candidates){
//     applyRules()
//   }


//   return optimizedPlan;
// }



Optimizer::~Optimizer() {
}