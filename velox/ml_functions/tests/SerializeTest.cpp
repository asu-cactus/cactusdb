#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/Expressions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

class SerializeTest : public VectorTestBase {
 public:
  const std::string kTpchConnectorId = "test-tpch";

  SerializeTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    Type::registerSerDe();
    common::Filter::registerSerDe();
    // connector::hive::HiveTableHandle::registerSerDe();
    connector::hive::LocationHandle::registerSerDe();
    // connector::hive::HiveColumnHandle::registerSerDe();
    connector::hive::HiveInsertTableHandle::registerSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    // registerPartitionFunctionSerDe();

    // Register TPC-H connector.
    auto tpchConnector =
        connector::getConnectorFactory(
            connector::tpch::TpchConnectorFactory::kTpchConnectorName)
            ->newConnector(kTpchConnectorId, nullptr);
    connector::registerConnector(tpchConnector);
  }

  ~SerializeTest() {
    connector::unregisterConnector(kTpchConnectorId);
  }

  /// Parse SQL expression into a typed expression tree using DuckDB SQL parser.
  core::TypedExprPtr parseExpression(
      const std::string& text,
      const RowTypePtr& rowType) {
    parse::ParseOptions options;
    auto untyped = parse::parseExpr(text, options);
    return core::Expressions::inferTypes(untyped, rowType, execCtx_->pool());
  }

  /// Compile typed expression tree into an executable ExprSet.
  std::unique_ptr<exec::ExprSet> compileExpression(
      const std::string& expr,
      const RowTypePtr& rowType) {
    std::vector<core::TypedExprPtr> expressions = {
        parseExpression(expr, rowType)};
    return std::make_unique<exec::ExprSet>(
        std::move(expressions), execCtx_.get());
  }

  /// Evaluate an expression on one batch of data.
  VectorPtr evaluate(exec::ExprSet& exprSet, const RowVectorPtr& input) {
    exec::EvalCtx context(execCtx_.get(), &exprSet, input.get());

    SelectivityVector rows(input->size());
    std::vector<VectorPtr> result(1);
    exprSet.eval(rows, context, result);
    return result[0];
  }

  /// Make TPC-H split to add to TableScan node.
  exec::Split makeTpchSplit() const {
    return exec::Split(std::make_shared<connector::tpch::TpchConnectorSplit>(
        kTpchConnectorId));
  }

  /// Run the demo.
  void run();

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
};

void replaceSourceWithIdInSerializedPlan(
    folly::dynamic& serializedPlan,
    folly::dynamic& serializedNewSource,
    std::string nodeId) {
  for (auto& source : serializedPlan["sources"]) {
    if (source["id"].asString() == nodeId) {
      source = serializedNewSource;
      std::cout << "[info] found and replaced" << std::endl;
      break;
    } else {
      replaceSourceWithIdInSerializedPlan(source, serializedNewSource, nodeId);
    }
  }
}

void SerializeTest::run() {
  auto a = makeFlatVector<int>({0, 1, 2});
  auto b = makeFlatVector<int>({2, 2, 2});

  auto data = makeRowVector({"a", "b"}, {a, b});
  //   auto data = makeRowVector({"b"}, {arrayOfArrays});
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  core::PlanNodeId p1;
  core::PlanNodeId p2;
  core::PlanNodeId p3;
  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .values({data})
                  .project({"a*3 as a", "b*3 as b"})
                  .capturePlanNodeId(p0)
                  .planNode();

  auto plan1 = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({data})
                   .project({"a*4 as a", "b*4 as b"})
                   .capturePlanNodeId(p2)
                   .project({"a*5 as a", "b*5 as b"})
                   .capturePlanNodeId(p3)
                   .planNode();

  auto serializedNewSource = plan->serialize();
  auto serializedPlan = plan1->serialize();

  std::cout << "serializedNewSource : \n"
            << plan->toString(true, true) << std::endl;
  std::cout << "serializedPlan: \n" << plan1->toString(true, true) << std::endl;

  replaceSourceWithIdInSerializedPlan(serializedPlan, serializedNewSource, p2);

  std::cout << "serializedPlan: " << plan1->toString(true, true) << std::endl;

  auto deserlizedUpdatedPlan =
      ISerializable::deserialize<core::PlanNode>(serializedPlan, pool_.get());

  std::cout << "updated query plan: "
            << deserlizedUpdatedPlan->toString(true, true) << std::endl;

  auto result =
      AssertQueryBuilder(deserlizedUpdatedPlan).copyResults(pool_.get());
  std::cout << std::endl << "Results: 1 \n" << result->toString() << std::endl;
  std::cout << result->toString(0, result->size()) << std::endl;
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  SerializeTest demo;
  demo.run();
}
