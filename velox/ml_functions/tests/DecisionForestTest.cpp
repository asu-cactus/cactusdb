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
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


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

  void registerFunctions();

  void run();
  void test_tree_predict_small();
  void test_forest_predict_small();
  void test_forest_predict_crossproduct_small();
  void testBody() override {}

  void setUp() {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void tearDown() {
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

void DecisionForestTest::registerFunctions() {


  exec::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(0, "resources/model/fraud_xgboost_10_8/0.txt", 28, false));

  registerCustomType(
      "tree_type", std::make_unique<TreeTypeFactories>());

  exec::registerVectorFunction(
      "velox_decision_tree_predict",
      VeloxTreePrediction::signatures(),
      std::make_unique<VeloxTreePrediction>(28));

  exec::registerVectorFunction(
       "velox_decision_tree_construct",
       VeloxTreeConstruction::signatures(),
       std::make_unique<VeloxTreeConstruction>());

  registerFunction<VeloxTreePredictionSimpleFunction, float, Array<float>, TheTree>(
      {"velox_decision_tree_predict_simple"});

  exec::registerVectorFunction(
      "decision_forest_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>("resources/model/fraud_xgboost_10_8", 28, true));

}

void DecisionForestTest::testingTreePredictSmall() {

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

  registerFunctions();

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

void DecisionForestTest::testingForestPredictSmall() {

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

  registerFunctions();

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


void DecisionForestTest::testingForestPredictCrossproductSmall() {

  int num_rows = 10;
  int num_cols = 28;
  int size = num_rows*num_cols;


  registerFunctions();


  std::vector<std::vector<float>> inputVectors;
  for(int i = 0; i < num_rows; i++){

     std::vector<float> inputVector;

     for(int j=0; j < num_cols; j++){

         inputVector.push_back((float)(i*j)/100.0);

     }

     inputVectors.push_back(inputVector);
  }
  
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

  std::vector<int32_t> indexVector;
  
  for (int i = 0; i < num_rows; i++) {
  
     indexVector.push_back(i);  
  
  }

  auto inputIndexVector = maker.flatVector<int32_t>(indexVector);


  auto inputRowVector = maker.rowVector({"row_id", "x"}, {inputIndexVector, inputArrayVector});

  std::vector<std::string> pathVectors;

  string forestFolderPath = "resources/model/fraud_xgboost_10_8";
  
  Forest::vectorizeForestFolder(forestFolderPath, pathVectors);

  int numTrees = pathVectors.size();
  
  auto model = makeFlatVector<StringView> (pathVectors.size());

  for (int i = 0; i < numTrees; i++) {
  
      model->set(i, StringView(pathVectors[i].c_str()));
  
  }


  auto treeIndexVector = maker.flatVector<int16_t>(numTrees);

   for (int i = 0; i < numTrees; i++) {

     treeIndexVector->set(i, i);

  }

  auto treeRowVector = maker.rowVector({"tree_id", "tree_path"}, {treeIndexVector, model});
   
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
	          .values({inputRowVector})
		  .nestedLoopJoin(
		      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                          .values({treeRowVector})
			  .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
                          .planNode(), {"row_id", "x", "tree_id", "tree"})
		   .project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
	           .aggregation({"row_id"}, 
				{"sum(prediction) as sum"},
				{},
				core::AggregationNode::Step::kPartial,
				false)
		   .project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"})
                   .planNode();
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Decision Forest Prediction with Small Data (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;

  unregisterCustomType("tree_type");
  
}


void DecisionForestTest::run() {

  string forestFolderPath = "resources/model/fraud_xgboost_10_8";

  DIR * dir = opendir(forestFolderPath.c_str());

  if (!dir) {

      std::cout << "Please check whether folder exists in resources/model/fraud_xgboost_10_8 under the velox root directory.\n"
              << "Also, you need to execute the test from the velox root directory like the following:\n"
              << "cd velox\n"
              << "./_build/release/velox/ml_functions/decision_forest_prediction_test\n"
              << std::endl;
      
      exit(1);

  }
  closedir(dir);
   
  testingTreePredictSmall();
  testingForestPredictSmall();
  testingForestPredictCrossproductSmall();
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  DecisionForestTest demo;
  demo.run();
}
