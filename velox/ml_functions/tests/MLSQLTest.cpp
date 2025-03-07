/*
 * Copyright (c) 2025 ASU Cactus Lab.
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
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <stdlib.h>
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

using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

/*
 * We test SQL queries that involve our ML functions.
 * In the first test case, called testingTreePredictSmall(), we test the query
 * of "SELECT decision_tree_predict(x) FROM data1"; In the second test case,
 * called testingForestPredictSmall(), we test the query of "SELECT
 * decision_forest_predict(x) FROM data2";
 */

class MLSQLTest : public HiveConnectorTestBase {
 public:
  MLSQLTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);
  }

  ~MLSQLTest() {}

  void registerFunctions();

  void run();
  void testingTreePredictSmall();
  void testingForestPredictSmall();

  ArrayVectorPtr parseCSVFile(
      VectorMaker& maker,
      std::string filePath,
      int numRows,
      int numCols);

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

void MLSQLTest::registerFunctions() {
  std::cout << "To register function for TreePrediction" << std::endl;

  exec::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(
          0, "resources/model/fraud_xgboost_10_8/0.txt", 28, false));

  std::cout << "To register function for ForestPrediction" << std::endl;

  exec::registerVectorFunction(
      "decision_forest_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>(
          "resources/model/fraud_xgboost_10_8", 28, true));
}

// Testing the query of "SELECT decision_tree_predict(x) FROM data1"
void MLSQLTest::testingTreePredictSmall() {
  int num_rows = 10;
  int num_cols = 28;
  int size = num_rows * num_cols;

  std::vector<std::vector<float>> inputVectors;
  for (int i = 0; i < num_rows; i++) {
    std::vector<float> inputVector;
    for (int j = 0; j < num_cols; j++) {
      inputVector.push_back(-5.0);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});

  std::vector<RowVectorPtr> vectors;

  vectors.push_back(inputRowVector);

  this->planner_.registerTable("data1", vectors);

  this->planner_.registerScalarFunction(
      "decision_tree_predict", {ARRAY(REAL())}, REAL());

  registerFunctions();

  std::string duckSQL = "SELECT decision_tree_predict(x) FROM data1";

  auto myPlan = planner_.plan(duckSQL);

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Decision Tree Prediction with Small Data (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

// Testing the query of "SELECT decision_forest_predict(x) FROM data1"
void MLSQLTest::testingForestPredictSmall() {
  int num_rows = 10;
  int num_cols = 28;
  int size = num_rows * num_cols;

  std::vector<std::vector<float>> inputVectors;
  for (int i = 0; i < num_rows; i++) {
    std::vector<float> inputVector;
    for (int j = 0; j < num_cols; j++) {
      inputVector.push_back(-2.0);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});

  std::vector<RowVectorPtr> vectors;

  vectors.push_back(inputRowVector);

  this->planner_.registerTable("data2", vectors);

  this->planner_.registerScalarFunction(
      "decision_forest_predict", {ARRAY(REAL())}, REAL());

  registerFunctions();

  std::string duckSQL = "SELECT decision_forest_predict(x) FROM data2";

  auto myPlan = planner_.plan(duckSQL);

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Decision Forest Prediction with Small Data (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void MLSQLTest::run() {
  // We check whether the model exists
  std::string forestFolderPath = "resources/model/fraud_xgboost_10_8";

  DIR* dir = opendir(forestFolderPath.c_str());

  if (!dir) {
    std::cout
        << "Please check whether folder exists in resources/model/fraud_xgboost_10_8 under the velox root directory.\n"
        << "Also, you need to execute the test from the velox root directory like the following:\n"
        << "cd velox\n"
        << "./_build/release/velox/ml_functions/decision_forest_prediction_test\n"
        << std::endl;

    exit(1);
  }
  closedir(dir);

  // Run two test cases
  testingTreePredictSmall();
  testingForestPredictSmall();
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  MLSQLTest demo;
  demo.run();
}
