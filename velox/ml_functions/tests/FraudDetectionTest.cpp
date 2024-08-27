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
#include <vector>
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
#include "velox/ml_functions/XGBoost.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/Concat.h"
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;



class FraudDetectionTest : public HiveConnectorTestBase {
 public:
  FraudDetectionTest() {
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

  ~FraudDetectionTest() {}

  void registerFunctions(std::string modelFilePath="resources/model/fraud_xgboost_1600_8", int numCols = 28);
  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  
  RowVectorPtr getCustomerData(int numCustomers, int numCustomerFeatures);
  RowVectorPtr getTransactionData(int numTransactions, int numTransactionFeatures, int numCustomers);
  
  void testingNestedLoopJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingNestedLoopJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingHashJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);
  void testingHashJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);

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

void FraudDetectionTest::registerFunctions(std::string modelFilePath, int numCols) {

  std::cout << "To register function for XGBoost Prediction" << std::endl;

  exec::registerVectorFunction(
      "xgboost_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>(modelFilePath, numCols, true));

  std::cout << "To register function for Concatenation" << std::endl;

  exec::registerVectorFunction(
      "concat_vectors",
      Concat::signatures(),
      std::make_unique<Concat>(10, 18));

}


ArrayVectorPtr FraudDetectionTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

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

RowVectorPtr FraudDetectionTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
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


RowVectorPtr FraudDetectionTest::getCustomerData(int numCustomers, int numCustomerFeatures) {
    
    // Customer table
     std::vector<int64_t> customerIDs;
     std::vector<std::vector<float>> customerFeatures;
     
     // Populate Customer table
     for (int i = 0; i < numCustomers; ++i) {
         customerIDs.push_back(i);  // Example: Customer IDs from 0 to 499

         std::vector<float> features;
         for(int j=0; j < numCustomerFeatures; j++){
             features.push_back(2.0);
         }
         
         customerFeatures.push_back(features);
     }

     // Prepare Customer table
     auto customerIDVector = maker.flatVector<int64_t>(customerIDs);
     auto customerFeaturesVector = maker.arrayVector<float>(customerFeatures, REAL());
     auto customerRowVector = maker.rowVector(
         {"customer_id", "customer_features"},
         {customerIDVector, customerFeaturesVector}
     );

     return customerRowVector;
}


RowVectorPtr FraudDetectionTest::getTransactionData(int numTransactions, int numTransactionFeatures, int numCustomers) {
    
    // Transaction table
     std::vector<int64_t> transactionIDs;
     std::vector<int64_t> transactionCustomerIDs;
     std::vector<std::vector<float>> transactionFeatures;
     
     // Populate Transaction table
     for (int i = 0; i < numTransactions; ++i) {
         transactionIDs.push_back(i);  // Example: Transaction IDs from 0 to 4999
         
         // Randomly assign each transaction to a customer
         transactionCustomerIDs.push_back(rand() % numCustomers);

         std::vector<float> features;
         for(int j=0; j < numTransactionFeatures; j++){
             features.push_back(5.0);
         }
         
         transactionFeatures.push_back(features);
     }

     // Prepare Transaction table
     auto transactionIDVector = maker.flatVector<int64_t>(transactionIDs);
     auto transactionCustomerIDVector = maker.flatVector<int64_t>(transactionCustomerIDs);
     auto transactionFeaturesVector = maker.arrayVector<float>(transactionFeatures, REAL());
     auto transactionRowVector = maker.rowVector(
         {"transaction_id", "trans_customer_id", "transaction_features"},
         {transactionIDVector, transactionCustomerIDVector, transactionFeaturesVector}
     );

     return transactionRowVector;
}


void FraudDetectionTest::testingNestedLoopJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;

     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

     
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .filter("customer_id > 50")
                         .nestedLoopJoin(exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(), {"transaction_id", "trans_customer_id", "transaction_features", "customer_id", "customer_features"}
                         )
                         .filter("customer_id = trans_customer_id")
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Nested Noop Join and Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingNestedLoopJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

     
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .nestedLoopJoin(exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(), {"transaction_id", "trans_customer_id", "transaction_features", "customer_id", "customer_features"}
                         )
                         .filter("customer_id = trans_customer_id")
                         .filter("customer_id > 50")
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Nested Noop Join and without Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingHashJoinWithPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .filter("customer_id > 50")
                         .hashJoin({"customer_id"},
                         {"trans_customer_id"},
                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(),
                         "",
                         {"customer_id", "customer_features", "transaction_id", "transaction_features"})
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Has Join and Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::testingHashJoinWithoutPredicatePush(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();                                                                      
                      
     std::string path = dataFile->path;

     int numCustomers = 100;
     int numTransactions = 1000;
     int numCustomerFeatures = 10;
     int numTransactionFeatures = 18;
     
     // Retrieve the customer and transaction data
     RowVectorPtr customerRowVector = getCustomerData(numCustomers, numCustomerFeatures);
     RowVectorPtr transactionRowVector = getTransactionData(numTransactions, numTransactionFeatures, numCustomers);
     
     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    
                         
     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({customerRowVector})
                         .hashJoin({"customer_id"},
                         {"trans_customer_id"},
                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({transactionRowVector})
                         .planNode(),
                         "",
                         {"customer_id", "customer_features", "transaction_id", "transaction_features"})
                         .filter("customer_id > 50")
                         .project({"transaction_id AS tid", "concat_vectors(customer_features, transaction_features) AS features"})
                         .project({"tid", "xgboost_predict(features) AS label"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    //std::cout << results->toString(0, results->size()) << std::endl;
   
    std::cout << "Time for Fraudulent Transaction Detection with Has Join and without Predicate Push (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath) {

  std::cout << "Option is " << option << std::endl;
  registerFunctions(modelFilePath, numCols);

  if (option == 0) {
      testingNestedLoopJoinWithPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      testingNestedLoopJoinWithoutPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      testingHashJoinWithPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
      testingHashJoinWithoutPredicatePush(numDataSplits, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);
  }

  else
      std::cout << "Seleceted test is not supported" << std::endl;
}



DEFINE_int32(rewriteOrNot, 0, "0 for all tests");
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

  FraudDetectionTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath);

}
