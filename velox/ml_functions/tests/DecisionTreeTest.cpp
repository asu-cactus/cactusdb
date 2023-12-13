#include <folly/init/Init.h>
#include <torch/torch.h>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
// #include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

// Utility function to generate random float/int values

class DecisionForestTest : public HiveConnectorTestBase {
 public:
  DecisionForestTest() {
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
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~DecisionForestTest() {}

  void registerFunction();

  void run();
  void test_tree_predict_small();
  void test_forest_predict_small();
  void TestBody() override {}

  void SetUp() {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() {
    HiveConnectorTestBase::TearDown();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::addDefaultLeafMemoryPool();
  VectorMaker maker{pool_.get()};
};

void DecisionForestTest::registerFunction() {

  exec::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(0, "/home/ubuntu/model/fraud_xgboost_10_8_netsdb/0.txt", 28, false));


  exec::registerVectorFunction(
      "decision_forest_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>("/home/ubuntu/model/fraud_xgboost_10_8_netsdb", 28, true));
}

void DecisionForestTest::test_tree_predict_small() {

  int num_rows = 10;
  int num_cols = 28;
  int size = num_rows*num_cols;

  std::vector<std::vector<float>> inputVectors;
  for(int i=0; i < num_rows; i++){
    std::vector<float> inputVector;
    for(int j=0; j < num_cols; j++){
      inputVector.push_back(-5.0);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});

  registerFunction();

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"decision_tree_predict(x)"})
                              .planNode();

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Decision Tree Prediction with Small Data (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void DecisionForestTest::test_forest_predict_small() {

  int num_rows = 10;
  int num_cols = 28;
  int size = num_rows*num_cols;

  std::vector<std::vector<float>> inputVectors;
  for(int i=0; i < num_rows; i++){
    std::vector<float> inputVector;
    for(int j=0; j < num_cols; j++){
      inputVector.push_back(-2.0);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});

  registerFunction();

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"decision_forest_predict(x)"})
                              .planNode();

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Decision Forest Prediction with Small Data (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void DecisionForestTest::run() {
   //test_tree_predict_small();
   test_forest_predict_small();
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  DecisionForestTest demo;
  demo.run();
}
