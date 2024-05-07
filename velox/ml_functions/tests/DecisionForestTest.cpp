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
            ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~DecisionForestTest() {}

  void registerFunctions();

  void run( bool whetherToReorderJoin);
  void testingTreePredictSmall();
  void testingForestPredictSmall();
  void testingForestPredictCrossproductSmall();
  void testingForestPredictCrossproductLarge( bool whetherToReorderJoin );

  ArrayVectorPtr parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols);

  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {
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

  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};
  VectorMaker maker{pool_.get()};
};

void DecisionForestTest::registerFunctions() {

  std::cout <<"To register function for TreePrediction" << std::endl;

  exec::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(0, "resources/model/fraud_xgboost_10_8/0.txt", 28, false));

  std::cout << "To register type for Tree" << std::endl;

  registerCustomType(
      "tree_type", std::make_unique<TreeTypeFactories>());


  std::cout << "To register function for VeloxTreePrediction" << std::endl;

  exec::registerVectorFunction(
      "velox_decision_tree_predict",
      VeloxTreePrediction::signatures(),
      std::make_unique<VeloxTreePrediction>(28));

  std::cout << "To register function for VeloxTreeConstruction" << std::endl;

  exec::registerVectorFunction(
       "velox_decision_tree_construct",
       VeloxTreeConstruction::signatures(),
       std::make_unique<VeloxTreeConstruction>());

  std::cout << "To register function for ForestPrediction" << std::endl;

  exec::registerVectorFunction(
      "decision_forest_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>("resources/model/fraud_xgboost_1600_8", 28, true));

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

  std::cout << "To register user defined functions and types" << std::endl;

  registerFunctions();

  std::cout << "To create small scale sample data" << std::endl;

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


  std::cout << "To load model" << std::endl;

  std::vector<std::string> pathVectors;

  string forestFolderPath = "resources/model/fraud_xgboost_1600_8";
  
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

  std::cout << "To create the plan" << std::endl;

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

  std::cout << "To run the plan" << std::endl;
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Time for Decision Forest Prediction with Small Data (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  std::cout << "Results:" << results->toString() << std::endl;

  std::cout << results->toString(0, results->size()) << std::endl;

  unregisterCustomType("tree_type");
  
}

ArrayVectorPtr DecisionForestTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

    int size = numRows * numCols;

    std::cout << "Loading tensor of size " << size << " from " << filePath << std::endl;

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

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
						    //
            float number = std::stof(numberStr);    // Convert the string to float

	    if (colIndex < numCols)					    

                curRow[colIndex] = number;

	    colIndex ++;

        }

	inputArrayVector.push_back(curRow);
    }

    file.close();

    ArrayVectorPtr tensor = maker.arrayVector<float>(inputArrayVector);
    
    return tensor;

} 

void DecisionForestTest::testingForestPredictCrossproductLarge(bool whetherToReorderJoin) {
  
  registerFunctions();

  int numRows = 56962;
  int numCols = 28;

  std::string dataFilePath = "resources/data/creditcard_test.csv";

  ArrayVectorPtr inputArrayVector = parseCSVFile(maker, dataFilePath, numRows, numCols);

  std::vector<int32_t> indexVector;

  for (int i = 0; i < numRows; i++) {

     indexVector.push_back(i);

  }

  auto inputIndexVector = maker.flatVector<int32_t>(indexVector);

  auto inputRowVector = maker.rowVector({"row_id", "x"}, {inputIndexVector, inputArrayVector});

  std::vector<std::string> pathVectors;

  string forestFolderPath = "resources/model/fraud_xgboost_1600_8";

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

  auto dataConfig = std::make_shared<facebook::velox::dwrf::Config>();

  // affects the number of splits
  // number of bites in each stripe (collection of rows)
  // strip size should be <= split size (total_size / total splits)
  // to have the desired number of splits
  uint64_t kDataSizeKB = 512UL;

  int numDataSplits = 16;

  // used for indexing.
  // 2k rows will be processed in every call
  // but doesn't effect number of splits
  // if stripe size is a large value
  uint32_t numDataRows = numRows/numDataSplits+1;

  dataConfig->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 799 * kDataSizeKB);

  dataConfig->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, numDataRows);

  auto dataFile = TempFilePath::create();

  writeToFile(dataFile->path, {inputRowVector}, dataConfig);

  auto dataHiveSplits =  makeHiveConnectorSplits(dataFile->path, numDataSplits, dwio::common::FileFormat::DWRF);

  auto treeConfig = std::make_shared<facebook::velox::dwrf::Config>();

  // affects the number of splits
  // number of bites in each stripe (collection of rows)
  // strip size should be <= split size (total_size / total splits)
  // to have the desired number of splits
  uint64_t kTreeSizeKB = 1UL;
  
  int numTreeSplits = 16;

  // used for indexing. 
  // 2k rows will be processed in every call
  // but doesn't effect number of splits
  // if stripe size is a large value
  uint32_t numTreeRows = numTrees/numTreeSplits+1;

  treeConfig->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 1 * kTreeSizeKB);

  treeConfig->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, numTreeRows);

  auto treeFile = TempFilePath::create();

  writeToFile(treeFile->path, {treeRowVector}, treeConfig);

  auto treeHiveSplits =  makeHiveConnectorSplits(treeFile->path, numTreeSplits, dwio::common::FileFormat::DWRF);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  core::PlanNodeId p0;

  core::PlanNodeId p1;

  core::PlanFragment myPlan;

  if (!whetherToReorderJoin) {
       myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
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
                   .planFragment();
  } else {
      myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(treeRowVector->type()))
		  .capturePlanNodeId(p1)
                  .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
                  .nestedLoopJoin(
                      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                          .tableScan(asRowType(inputRowVector->type()))
                          .capturePlanNodeId(p0)
                          .planNode(), {"row_id", "x", "tree_id", "tree"})
                   .project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
                   .aggregation({"row_id"},
                                {"sum(prediction) as sum"},
                                {},
                                core::AggregationNode::Step::kPartial,
                                false)
                   .project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"})
                   .planFragment();
  }

  // print statistics of a plan
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"}, 
      {core::QueryConfig::kMaxOutputBatchRows, "100000"}});

  auto task = exec::Task::create("0", myPlan , 0, queryCtx_,
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result)
               std::cout << result->toString() << std::endl;
          return exec::BlockingReason::kNotBlocked;
  });

 std::cout << "Data Hive splits:" << std::endl;
 for(auto& split : dataHiveSplits) {
      // std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
 }

 if (whetherToReorderJoin) {
      std::cout << "Tree Hive splits:" << std::endl;
      for(auto& split : treeHiveSplits) {
          task->addSplit(p1, exec::Split(std::move(split)));
      }
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  int veloxThreads = 8;
  
  task->start(veloxThreads);
  

  task->noMoreSplits(p0);

  if (whetherToReorderJoin) {
  
      task->noMoreSplits(p1);
  }

  // Start task with 2 as maximum drivers and wait for execution to finish
  waitForFinishedDrivers(task);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::stringstream ss;
  
  ss << numDataRows << "," << numDataSplits << "," << numTreeRows << "," << numTreeSplits << "," << veloxThreads << ",";
  
  std::cout << "Time for Decision Forest Prediction with Input Data (sec): " << std::endl;
  
  std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  std::cout << ss.str() << std::endl;

  unregisterCustomType("tree_type");

}

void DecisionForestTest::run(bool whetherToReorderJoin) {

  string forestFolderPath = "resources/model/fraud_xgboost_1600_8";

  DIR * dir = opendir(forestFolderPath.c_str());

  if (!dir) {

      std::cout << "Please check whether folder exists in resources/model/fraud_xgboost_1600_8 under the velox root directory.\n"
              << "Also, you need to execute the test from the velox root directory like the following:\n"
              << "cd velox\n"
              << "./_build/release/velox/ml_functions/decision_forest_prediction_test\n"
              << std::endl;
      
      exit(1);

  }
  closedir(dir);
   
  //testingTreePredictSmall();
  //testingForestPredictSmall();
  //testingForestPredictCrossproductSmall();
  testingForestPredictCrossproductLarge(whetherToReorderJoin);
}


DEFINE_bool(reorder, true, "Whether to reorder join");

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  bool reorder = FLAGS_reorder;

  DecisionForestTest demo;

  std::cout << fmt::format("Reorder: {}", reorder) << std::endl;

  demo.run(reorder);

}
