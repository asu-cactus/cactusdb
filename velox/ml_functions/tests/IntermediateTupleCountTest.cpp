/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <folly/init/Init.h>
#include <torch/torch.h>
#include "MLTestUtility.h"
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/Expressions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/optimizer/CataLog.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

class IntermediateTupleCountTest : public VectorTestBase {
 public:
  const std::string kTpchConnectorId = "test-tpch";

  IntermediateTupleCountTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    // Register TPC-H connector.
    auto tpchConnector =
        connector::getConnectorFactory(
            connector::tpch::TpchConnectorFactory::kTpchConnectorName)
            ->newConnector(kTpchConnectorId, nullptr);
    connector::registerConnector(tpchConnector);
  }

  ~IntermediateTupleCountTest() {
    connector::unregisterConnector(kTpchConnectorId);
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

void IntermediateTupleCountTest::run() {
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  CataLog cataLog;
  // Let’s create two vectors of 64-bit integers and one vector of strings.
  auto a = makeFlatVector<float>({0, 1, 2});
  //   auto b = makeFlatVector<int64_t>({0, 1, 2});
  // auto baseVector = makeArrayVector<float>(
  //     {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}});
  auto baseVector = makeArrayVector<float>(
      {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}});

  // Create an array of array vector using above base vector
  auto arrayOfArrays = makeArrayVector({0}, baseVector);

  // auto data = makeRowVector({"a", "b"}, {a, arrayOfArrays});
  auto data = makeRowVector({"b"}, {arrayOfArrays});

  auto plan = PlanBuilder().values({data});

  std::vector<RowVectorPtr> finalResult;
  float executionTime =
      runPlanWithCataLog(pool_, 8, plan, cataLog, finalResult, 1, 4, true);
  
  for (auto& result : finalResult) {
    std::cout << "Result: " << result->toString(0, result->size()) << std::endl;
  }
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  IntermediateTupleCountTest demo;
  demo.run();
}
