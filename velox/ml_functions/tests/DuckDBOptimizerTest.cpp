#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/QueryPlanner.h"
#include "velox/parse/TypeResolver.h"

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

/*
 * Test the DuckDB Optimizer and how SQL query is converted into a Velox plan.
 */

class DuckDBOptimizerTest : public HiveConnectorTestBase {
 public:
  DuckDBOptimizerTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);
  }

  ~DuckDBOptimizerTest() {}

  // void registerFunctions();
  void testDuckDBOptimizer();
  void run();

  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};

  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};

  // The DuckDbQueryPlanner instance, it will convert SQL query into a Velox
  // plan.
  core::DuckDbQueryPlanner planner_{pool_.get()};
};

// Testing the query of "SELECT decision_tree_predict(x) FROM data1"
void DuckDBOptimizerTest::testDuckDBOptimizer() {
  // Init two tables, users and items, and two tables will be joined
  // on the itemID.
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  int numUser = 1000;
  int numItem = 10000;
  auto userId = randomGenerator.genIntRange(0, numUser);
  auto userAtt1 = randomGenerator.genFloat1dVector(numUser);
  auto userAtt2 = randomGenerator.genFloat1dVector(numUser);
  auto userAtt3 = randomGenerator.gen1DInt(numUser, 0, 10);
  auto userItemId = randomGenerator.gen1DInt(numUser, 0, numItem);

  auto itemId = randomGenerator.genIntRange(0, numItem);
  auto itemAtt1 = randomGenerator.genFloat1dVector(numItem);
  auto itemAtt2 = randomGenerator.genFloat1dVector(numItem);
  auto itemAtt3 = randomGenerator.gen1DInt(numItem, 0, 100);

  auto userRowVector = maker.rowVector(
      {"userId", "userAtt1", "userAtt2", "userAtt3", "userItemId"},
      {maker.flatVector<int>(userId, INTEGER()),
       maker.flatVector<float>(userAtt1, REAL()),
       maker.flatVector<float>(userAtt2, REAL()),
       maker.flatVector<int>(userAtt3, INTEGER()),
       maker.flatVector<int>(userItemId, INTEGER())});

  auto itemRowVector = maker.rowVector(
      {"itemId", "itemAtt1", "itemAtt2", "itemAtt3"},
      {maker.flatVector<int>(itemId, INTEGER()),
       maker.flatVector<float>(itemAtt1, REAL()),
       maker.flatVector<float>(itemAtt2, REAL()),
       maker.flatVector<int>(itemAtt3, INTEGER())});

  this->planner_.registerTable("user_table", {userRowVector});
  this->planner_.registerTable("item_table", {itemRowVector});

  std::string duckSQL =
      "SELECT * FROM user_table, item_table where user_table.userItemId = item_table.itemId";

  auto myPlan = planner_.plan(duckSQL);

  // extract DuckDB query plan with optimizer enabled. It should failed.
  planner_.getConn().Query("PRAGMA enable_optimizer");
  auto duckDBPlan = planner_.getConn().ExtractPlan(duckSQL);
  std::cout << "DuckDB query plan w/ optimizer: \n" << duckDBPlan->ToString();

  // extract DuckDB query plan without optimizer enabled. It should success.
  planner_.getConn().Query("PRAGMA disable_optimizer");
  auto duckDBPlan1 = planner_.getConn().ExtractPlan(duckSQL);
  std::cout << "DuckDB query plan w/o optimizer: \n" << duckDBPlan1->ToString();

  // The following code will fail because the DuckDB optimizer cannot be enabled.
  // auto queryContext{planner_.tables_};
  // auto myPlan1 = planner_.planWithOptimization(duckSQL);

  std::cout << "parsed query plan w/o optimization: \n"
            << myPlan->toString(true, true) << std::endl;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Query Execution (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  std::cout << "Results: \n" << results->toString() << std::endl;
  // std::cout << results->toString(0, results->size()) << std::endl;
}

void DuckDBOptimizerTest::run() {
  // Run two test cases
  testDuckDBOptimizer();
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  DuckDBOptimizerTest demo;
  demo.run();
}
