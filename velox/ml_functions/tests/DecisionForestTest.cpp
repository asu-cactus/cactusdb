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
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <folly/init/Init.h>
#include <pstl/pstl_config.h>
#include <torch/torch.h>
#include <cwchar>
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
#include <type_traits>
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

  void registerFunctions(std::string modelFilePath="resources/model/fraud_xgboost_1600_8", int numCols = 28);

  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath );
  void testingTreePredictSmall();
  void testingForestPredictSmall();
  void testingForestPredictLarge(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingForestPredictCrossproductSmall();
  void testingForestPredictCrossproductLarge( bool whetherToReorderJoin, int numDataSplits, int numTreeSplits, 
                                              uint32_t numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath );

  ArrayVectorPtr parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols);

  RowVectorPtr writeDataToFile(std::string csvFilePath, int numRows, int numCols, int numDataSplits, 
                               std::string outPath, int dataBatchSize);

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

void DecisionForestTest::registerFunctions(std::string modelFilePath, int numCols) {

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
      std::make_unique<VeloxTreePrediction>(numCols));

  std::cout << "To register function for VeloxTreeConstruction" << std::endl;

  exec::registerVectorFunction(
       "velox_decision_tree_construct",
       VeloxTreeConstruction::signatures(),
       std::make_unique<VeloxTreeConstruction>());

  std::cout << "To register function for ForestPrediction" << std::endl;

  exec::registerVectorFunction(
      "decision_forest_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>(modelFilePath, numCols, true));

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
  //std::cout << "Results:" << results->toString() << std::endl;
  //std::cout << results->toString(0, results->size()) << std::endl;
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

void DecisionForestTest::testingForestPredictLarge(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     registerFunctions(modelFilePath, numCols);
   
     //int numRows = 10;
     //int numRows = 56962;
     //int numCols = 28;
     
     //std::string dataFilePath = "resources/data/creditcard_test.csv";
     //std::string dataFilePath = "/data/decision-forest-benchmark-paper/datasets/test10.csv";

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     RowVectorPtr inputRowVector = writeDataToFile(dataFilePath, numRows, numCols, numDataSplits, path, dataBatchSize);

     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
 
     core::PlanNodeId p0;

     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                     .tableScan(asRowType(inputRowVector->type()))
                     .capturePlanNodeId(p0)
                     .project({"decision_forest_predict(x)"})
                     .planFragment();

     // print statistics of a plan
     queryCtx_->testingOverrideConfigUnsafe(
         {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"}, 
         {core::QueryConfig::kMaxOutputBatchRows, "100000"}});
   
     auto task = exec::Task::create("0", myPlan , 0, queryCtx_,
           [](RowVectorPtr result, ContinueFuture* /*unused*/) {
           if(result) {
                 //std::cout << result->toString() << std::endl;
                 //std::cout << result->toString(0, result->size()) << std::endl;
           }      
           return exec::BlockingReason::kNotBlocked;
    });
   
    std::cout << "Data Hive splits:" << std::endl;
    for(auto& split : dataHiveSplits) {
         std::cout << split->toString() << std::endl;
         task->addSplit(p0, exec::Split(std::move(split)));
    }
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
 
    int veloxThreads = 8;

    task->start(veloxThreads);
     
 
    task->noMoreSplits(p0);
   
 
    // Start task with 2 as maximum drivers and wait for execution to finish
    waitForFinishedDrivers(task);
   
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
   
    std::stringstream ss;

    ss << numRows << "," << numDataSplits << "," << veloxThreads << ",";
   
    std::cout << "Time for Decision Forest Prediction with Input Data (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
    std::cout << ss.str() << std::endl;
 
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

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

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

RowVectorPtr DecisionForestTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
                                                 int numDataSplits, std::string outPath, int dataBatchSize) {

    ArrayVectorPtr inputArrayVector = parseCSVFile(maker, csvFilePath, numRows, numCols);
  
    std::vector<int32_t> indexVector;
  
    for (int i = 0; i < numRows; i++) {
  
       indexVector.push_back(i);
  
    }
  
    auto inputIndexVector = maker.flatVector<int32_t>(indexVector);
  
    auto inputRowVector = maker.rowVector({"row_id", "x"}, {inputIndexVector, inputArrayVector});

    auto dataConfig = std::make_shared<facebook::velox::dwrf::Config>();
  
    // affects the number of splits
    // number of bites in each stripe (collection of rows)
    // strip size should be <= split size (total_size / total splits)
    // to have the desired number of splits
    uint64_t kDataSizeKB = 512UL;

    uint32_t numDataRows = dataBatchSize;
  
    dataConfig->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 799 * kDataSizeKB);
  
    dataConfig->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, numDataRows);
  
    // auto dataFile = TempFilePath::create();

    // outPath = dataFile->path;

    writeToFile(outPath, {inputRowVector}, dataConfig);

    return inputRowVector;

}


void DecisionForestTest::testingForestPredictCrossproductLarge(bool whetherToReorderJoin, int numDataSplits, int numTreeSplits, 
                                                               uint32_t numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, 
                                                               std::string forestFolderPath) {
  
  registerFunctions(forestFolderPath, numCols);

  //int numRows = 10;
  //int numRows = 56962;
  //int numCols = 28;

  //std::string dataFilePath = "resources/data/creditcard_test.csv";
  //std::string dataFilePath = "/data/decision-forest-benchmark-paper/datasets/test10.csv";

  std::vector<std::string> pathVectors;

  //string forestFolderPath = "resources/model/fraud_xgboost_1600_8";

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


  auto dataFile = TempFilePath::create();
       
  std::string path = dataFile->path;

  auto inputRowVector = writeDataToFile(dataFilePath, numRows, numCols, numDataSplits, path, dataBatchSize);

  auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

  auto treeConfig = std::make_shared<facebook::velox::dwrf::Config>();

  // affects the number of splits
  // number of bites in each stripe (collection of rows)
  // strip size should be <= split size (total_size / total splits)
  // to have the desired number of splits
  uint64_t kTreeSizeKB = 1UL;
  
  // used for indexing. 
  // 2k rows will be processed in every call
  // but doesn't effect number of splits
  // if stripe size is a large value
  //uint32_t numTreeRows = 100;

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
                   .singleAggregation({"row_id"},
                                {"sum(prediction) as sum"})
                   .project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"})
                   .planFragment();
  } else {
      //myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
        //          .tableScan(asRowType(treeRowVector->type()))
		  //        .capturePlanNodeId(p1)
            //      .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
              //    .nestedLoopJoin(
                //      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  //        .tableScan(asRowType(inputRowVector->type()))
                    //      .capturePlanNodeId(p0)
                      //    .planNode(), {"row_id", "x", "tree_id", "tree"})
               //    .project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
                 //  .partialAggregation({"row_id"},
                   //           {"sum(prediction) as weight"})
                 //  .localPartition({"row_id"})
                 //  .singleAggregation({"row_id"},
                   //              {"sum(weight) as sum"})
                 //  .project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"})
                 //  .planFragment();
                 //
     myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(asRowType(treeRowVector->type()))
                .capturePlanNodeId (p1)
                .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
                .nestedLoopJoin(
                      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get()) 
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .planNode(), {"row_id", "x", "tree_id", "tree"}
                               )
                .project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
                .partialAggregation({"row_id"}, {"sum(prediction) as weight"})
                .localPartition({"row_id"})
                .finalAggregation()
                .project({"row_id as row_id", "if (weight > 0.0, 1.0, 0.0)"})
                .planFragment();
  }

  // print statistics of a plan
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kPreferredOutputBatchBytes, "10000000"}, 
      {core::QueryConfig::kMaxOutputBatchRows, "1000000"}});

  auto task = exec::Task::create("0", myPlan , 0, queryCtx_,
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result) {
               //std::cout << result->toString() << std::endl;
               //std::cout << result->toString(0, result->size()) << std::endl;
          }
          return exec::BlockingReason::kNotBlocked;
  });

 std::cout << "Data Hive splits:" << std::endl;
 for(auto& split : dataHiveSplits) {
      std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
 }

 if (whetherToReorderJoin) {
      std::cout << "Tree Hive splits:" << std::endl;
      for(auto& split : treeHiveSplits) {
          std::cout << split->toString() << std::endl;
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
  
  ss << numRows << "," << numDataSplits << "," << numTreeRows << "," << numTreeSplits << "," << veloxThreads << ",";
  
  std::cout << "Time for Decision Forest Prediction with Input Data (sec): " << std::endl;
  
  std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  std::cout << ss.str() << std::endl;

  unregisterCustomType("tree_type");

}

void DecisionForestTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

  std::cout << "Option is " << option << std::endl;

  if (option == 1)

      testingForestPredictCrossproductLarge(true, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);

  else if (option == 2)

      testingForestPredictLarge(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);

  else

      std::cout << "2 for UDF-centric (without rewriting) and 1 for Relation-centric (with rewriting)" << std::endl;
}


DEFINE_int32(rewriteOrNot, 2, "1 for UDF-centric without rewriting and 2 for Relation-centric with rewriting");
DEFINE_int32(numDataSplits, 16, "number of data splits");
DEFINE_int32(numTreeSplits, 16, "number of tree splits");
DEFINE_int32(numTreeRows, 100, "batch size for processing trees");
DEFINE_int32(dataBatchSize, 100, "batch size for processing input samples");
DEFINE_int32(numRows, 10, "number of tuples in the dataset to be predicted");
DEFINE_int32(numCols, 10, "number of columns in the dataset to be predicted");
DEFINE_string(dataFilePath, "resources/data/creditcard_test.csv", "path to input dataset to be predicted");
DEFINE_string(modelFilePath, "resources/model/fraud_xgboost_1600_8", "path to the model used for prediction");

int main(int argc, char** argv) {

  folly::init(&argc, &argv, false);

  memory::MemoryManager::initialize({});

  int option = FLAGS_rewriteOrNot;
  int numDataSplits = FLAGS_numDataSplits;
  int numTreeSplits = FLAGS_numTreeSplits;
  int numTreeRows = FLAGS_numTreeRows;
  int dataBatchSize = FLAGS_dataBatchSize;
  int numRows = FLAGS_numRows;
  int numCols = FLAGS_numCols;
  std::string dataFilePath = FLAGS_dataFilePath;
  std::string modelFilePath = FLAGS_modelFilePath;

  DecisionForestTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);

}
