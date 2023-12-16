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
#include "velox/ml_functions/VeloxDecisionTree.h"

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

  void registerFunctions();

  void run();
  void test_tree_predict_small();
  void test_forest_predict_small();
  void test_forest_predict_crossproduct_small();
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

void DecisionForestTest::registerFunctions() {

  exec::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(0, "/home/ubuntu/model/fraud_xgboost_10_8_netsdb/0.txt", 28, false));

  registerCustomType(
      "tree_type", std::make_unique<TreeTypeFactories>());

  //registerCustomType(
    //  "tree_type", std::make_unique<AlwaysFailingTypeFactories>());


  registerCustomType(
      "fancy_int", std::make_unique<FancyIntTypeFactories>());


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
      std::make_unique<ForestPrediction>("/home/ubuntu/model/fraud_xgboost_10_8_netsdb", 28, true));

/*
  exec::registerVectorFunction(
      "to_fancy_int",
      ToFancyIntFunction::signatures(),
      std::make_unique<ToFancyIntFunction>());
  exec::registerVectorFunction(
      "from_fancy_int",
      FromFancyIntFunction::signatures(),
      std::make_unique<FromFancyIntFunction>());
*/
      
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


void DecisionForestTest::test_forest_predict_crossproduct_small() {

  int num_rows = 10;
  int num_cols = 28;
  int size = num_rows*num_cols;


  registerFunctions();


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

  std::vector<std::string> pathVectors;

  string forestFolderPath = "resources/model/fraud_xgboost_10_8";
  
  DIR * dir = opendir(forestFolderPath.c_str());
  
  if (!dir) {
  
      std::cout << "Please check whether folder exists in resources/model/fraud_xgboost_10_8 under the velox root directory.\n"
	      << "Also, you need to execute the test from the velox root directory like the following:\n"
	      << "cd velox\n"
	      << "./_build/release/velox/ml_functions/decision_forest_prediction_test\n"
	      << std::endl;
  
  }
  
  Forest::vectorizeForestFolder(forestFolderPath, pathVectors);
  
  auto data = makeFlatVector<StringView> (pathVectors.size());

  for (int i = 0; i < pathVectors.size(); i++) {
  
      data->set(i, StringView(pathVectors[i].c_str()));
  
  }

  auto treeRowVector = maker.rowVector({"c0"}, {data});
   
  /*
  auto data = makeFlatVector<int64_t>({1, 2, 3, 4, 5});

  auto data1 = makeFlatVector<std::shared_ptr<FancyInt>>(5);

  for (int i = 0; i < 5; i++) {
  
     data1->set(i, std::make_shared<FancyInt>(i));

  }

  auto inputRowVector1 = maker.rowVector({"c0"}, {data1});

  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .values({inputRowVector1})
		  .project({"from_fancy_int(c0)"})
		  .planNode();
  */
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
	          .values({inputRowVector})
		  .nestedLoopJoin(
		      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                          .values({treeRowVector})
			  .project({"velox_decision_tree_construct(c0)"})
                          .planNode(), {"x", "p0"})
		    .project({"velox_decision_tree_predict(x, p0)"})
		    //.project({"velox_decision_tree_predict_simple(x, tree)"})
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
   
   //test_tree_predict_small();
   //test_forest_predict_small();
   test_forest_predict_crossproduct_small();

}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  DecisionForestTest demo;
  demo.run();
}
