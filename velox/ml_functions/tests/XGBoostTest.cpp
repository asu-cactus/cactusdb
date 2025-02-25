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
#include <folly/init/Init.h>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/functions.h"
#include "velox/parse/TypeResolver.h"

using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

class XGBoostTest : public HiveConnectorTestBase {
 public:
  XGBoostTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~XGBoostTest() {}

  void registerFunctions(std::string forestPath);

  void run();

  void testingForestPredictSmall(std::string forestPath);

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
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};
  VectorMaker maker{pool_.get()};
};

void XGBoostTest::registerFunctions(std::string forestPath) {
  std::cout << "To register function for XGBoostPrediction" << std::endl;

  exec::registerVectorFunction(
      "xgboost_predict",
      XGBoostPrediction::signatures(),
      std::make_unique<XGBoostPrediction>(forestPath.c_str(), 28));
}

void XGBoostTest::testingForestPredictSmall(std::string forestPath) {
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

  registerFunctions(forestPath);

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"xgboost_predict(x)"})
                    .planNode();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for XGBoost Prediction with Small Data (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

ArrayVectorPtr XGBoostTest::parseCSVFile(
    VectorMaker& maker,
    std::string filePath,
    int numRows,
    int numCols) {
  int size = numRows * numCols;

  std::cout << "Loading tensor of size " << size << " from " << filePath
            << std::endl;

  std::ifstream file(filePath.c_str());

  std::vector<std::vector<float>> inputArrayVector;

  int index = 0;

  std::string line;

  while (numRows--) { // Read a line from the file

    std::vector<float> curRow(numCols);

    std::getline(file, line);

    std::istringstream iss(line); // Create an input string stream from the line

    std::string numberStr;

    int colIndex = 0;

    while (std::getline(
        iss, numberStr, ',')) { // Read each number separated by comma
                                //
      float number = std::stof(numberStr); // Convert the string to float

      if (colIndex < numCols)

        curRow[colIndex] = number;

      colIndex++;
    }

    inputArrayVector.push_back(curRow);
  }

  file.close();

  ArrayVectorPtr tensor = maker.arrayVector<float>(inputArrayVector);

  return tensor;
}

void XGBoostTest::run() {
  std::string forestPath = "resources/model/fraud_xgboost_10_8.json";

  std::ifstream file(forestPath.c_str());

  if (file.good()) {
    testingForestPredictSmall(forestPath);
  } else {
    std::cout
        << "Please make sure you run the code from the velox root directory. (e.g., ./_build/release/velox/ml_functions/xgboost_test)"
        << std::endl;
    exit(1);
  }
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  XGBoostTest demo;
  demo.run();
}
