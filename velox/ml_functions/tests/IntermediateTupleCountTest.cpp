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
#include "velox/ml_functions/TupleCounter.h"
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
  void run(int verbose);

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};
  std::unique_ptr<core::ExecCtx> execCtx_{
      std::make_unique<core::ExecCtx>(pool_.get(), queryCtx_.get())};
};

void IntermediateTupleCountTest::run(int verbose) {
  std::random_device rd;
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, rd());
  CataLog cataLog;

  int numRows = 20000;
  std::vector<int> inputIds = randomGenerator.genIntRange(0, numRows);
  std::vector<float> attr1Values = randomGenerator.genFloat1dVector(numRows);
  std::vector<std::vector<float>> featureValues =
      randomGenerator.genFloat2dVector(numRows, 10);

  auto inputIdVector = makeFlatVector<int>(inputIds, INTEGER());
  auto attr1Vector = makeFlatVector<float>(attr1Values, REAL());
  auto featureVector = makeArrayVector<float>(featureValues, REAL());

  auto inputRowVector = makeRowVector(
      {"id", "attr1", "features"}, {inputIdVector, attr1Vector, featureVector});

  auto inputRowVectorBatches = splitRowVectorIntoBatches(inputRowVector, 5);

  exec::registerVectorFunction(
      "tuple_counter",
      TupleCounter::signatures(),
      std::make_unique<TupleCounter>(&cataLog));

  auto plan =
      PlanBuilder()
          .values(inputRowVectorBatches)
          .filter("attr1 > 0.3")
          .project(
              {"id", "attr1", "tuple_counter(features, 'test1') AS features"});

  std::vector<RowVectorPtr> finalResult;
  float executionTime = runPlanWithCataLog(
      pool_, 8, plan, cataLog, finalResult, 1, verbose, true);
  std::cout << "Execution time: " << executionTime << " seconds" << std::endl;
  
  std::cout << "IntermediateStateTupleCounterMap" << std::endl;
  for (const auto& [key, value] :
       cataLog.getIntermediateStateTupleCounterMap()) {
    std::cout << key << ": " << value << std::endl;
  }
  
  auto sketchPlan = PlanBuilder()
                        .values(inputRowVectorBatches)
                        .filter("attr1 > 0.3")
                        .singleAggregation({}, {"approx_set(id, 0.01)"})
                        .project({"cardinality(a0)"});

  auto sketchTime = runPlanWithCataLog(
      pool_, 8, sketchPlan, cataLog, finalResult, 1, verbose, true);
      
  std::cout << "Sketch Execution time: " << sketchTime << " seconds"
            << std::endl;
}

DEFINE_int32(verbose, 2, "Verbose");

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  int verbose = FLAGS_verbose;

  IntermediateTupleCountTest demo;
  demo.run(verbose);
}
