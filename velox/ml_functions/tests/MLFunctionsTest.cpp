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
//#define EIGEN_USE_BLAS
#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/Expressions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/expression/VectorFunction.h"
#include <Eigen/Dense>
#include <cblas.h>
#include <chrono>
#include <torch/torch.h>
#include "velox/exec/Task.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>



using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

// TODO: Refactor
class MLFunctionsTest : public HiveConnectorTestBase {
 public:

  MLFunctionsTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    SetUp();

  }

  ~MLFunctionsTest() {
  }

  /// Run the demo.
  void run();
  void test_mat_mul();
  void test_mat_add();
  void test_relu();
  void test_dense_layer();
  void test_torch_dense_layer();
  void test_mnist();
  void test_multithreading();
  void test_multithreading_oom();
  void test_batching();
  void test_conv2d();
  FlatVectorPtr<float> get_tensor(std::ifstream& file, int size, int lines);


  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() {
     HiveConnectorTestBase::TearDown();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {     
      usleep(100'000); // 0.1 second.
    }
  }

  void TestBody() override {}

  
  std::shared_ptr<folly::Executor> executor_{std::make_shared<folly::CPUThreadPoolExecutor>(std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{std::make_shared<core::QueryCtx>(executor_.get())};
  
  std::shared_ptr<memory::MemoryPool> pool_ =  memory::addDefaultLeafMemoryPool();
  //std::shared_ptr<memory::MemoryPool> childPool = rootPool_->addAggregateChild("HiveConnectorTestBase.Writer");
  VectorMaker maker{pool_.get()};

};

void MLFunctionsTest::test_mat_mul() { 
//Eigen::setNbThreads(48); 
  int output_size = 5000;
  int input_size = 1000;
  int num_samples = 500000;
  int size = output_size*input_size;
  
  auto weights = maker.flatVector<float>(size);
  auto col = maker.flatVector<int>(num_samples);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 
  
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_samples; i++){
    col->set(i, i* 7 - i*(i%3));
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  
  auto inputRowVector = maker.rowVector({"x","col"}, {featureArrayVector,col});

  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .filter("col % 2 + col % 101 > 97")
                  .project({"mat_mul(x)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan)
                  .maxDrivers(4)
                  .config("max_output_batch_rows", std::to_string(10))
                  .config("preferred_output_batch_rows", std::to_string(10))
                  .config("preferred_output_batch_bytes", std::to_string(1000))
                  .copyResults(pool_.get());

  std::cout << "Results:" << results->toString() << std::endl;
  //std::cout << results->toString(0, results->size()) << std::endl;

}

void MLFunctionsTest::test_mat_add() {
//Eigen::setNbThreads(48); 
  int num_rows = 5;
  int num_cols = 10;
  int size = num_rows*num_cols;
  
  auto weights = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 


  std::vector<std::vector<float>> inputVectors;
  for(int i=0; i < num_rows; i++){
    std::vector<float> inputVector;
    for(int j=0; j < num_cols; j++){
      inputVector.push_back(i*j);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  
  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});


  // step1: Register
  exec::registerVectorFunction(
  "mat_add",
  MatrixAddition::signatures(),
  std::make_unique<MatrixAddition>(weights->values()->asMutable<float>(), num_cols));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"mat_add(x)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void MLFunctionsTest::test_relu(){
 
  int num_rows = 10;
  int num_cols = 3; 

  std::vector<std::vector<float>> inputVectors;
  for(int i=0; i < num_rows; i++){
    std::vector<float> inputVector;
    for(int j=0; j < num_cols; j++){
      inputVector.push_back(i*j);
    }
    inputVectors.push_back(inputVector);
  }

  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});
  
  // step1: Register
  exec::registerVectorFunction(
  "relu",
  Relu::signatures(),
  std::make_unique<Softmax>());

  
  // exec::registerVectorFunction(
  // "relu",
  // Relu::signatures(),
  // std::make_unique<Relu>());

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"relu(x)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
  
}

void MLFunctionsTest::test_dense_layer() {

  int output_size = 5;
  int input_size = 10;
  int num_features = 3;
  int size = output_size*input_size;
  
  auto weights = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 

  auto bias = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  bias->set(i, i % output_size);
  } 
  
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_features; i++){
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  exec::registerVectorFunction(
    "mat_add",
    MatrixAddition::signatures(),
    std::make_unique<MatrixAddition>(bias->values()->asMutable<float>(), output_size)
  );  

  exec::registerVectorFunction(
    "relu",
    Relu::signatures(),
    std::make_unique<Relu>()
  );

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"relu(mat_add(mat_mul(x)))"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
  /*
  std::string compute =  NNBuilder()
                          .denseLayer(5,10,weights->values()->asMutable<float>(), bias->values()->asMutable<float>(), NNBuilder::RELU)
                          .build();

  auto Plan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({fmt::format(compute, "x")})
		              .planNode();
  results = exec::test::AssertQueryBuilder(Plan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
  */

}

void MLFunctionsTest::test_torch_dense_layer(){
 
  int input_size = 784; // num_features
  int layer1_size = 20; // num units in hidden layer 1
  int layer2_size = 10;
  int num_samples = 20;
  
  std::vector<int> dimensions;
  dimensions.push_back(input_size);
  dimensions.push_back(layer1_size);
  dimensions.push_back(layer2_size);
  
  
  std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
  std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
  std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 
 
  

  FlatVectorPtr<float> weights_1 = get_tensor(weights_file, layer1_size * input_size, input_size);
  FlatVectorPtr<float> bias_1 = get_tensor(bias_file, layer1_size, 1);
  FlatVectorPtr<float> weights_2 = get_tensor(weights_file, layer2_size * layer1_size, layer1_size);
  FlatVectorPtr<float> bias_2 = get_tensor(bias_file, layer2_size, 1);
  weights_file.close();
  bias_file.close();

  FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples);
  float* data = input->values()->asMutable<float>();

  std::vector<std::vector<float>> featureVectors;
  for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
    std::vector<float> featureVector(data + cursor, data + cursor + input_size);
    featureVectors.push_back(featureVector);
  }


  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
 
  float* weights[2] = {weights_1->values()->asMutable<float>(), weights_2->values()->asMutable<float>()};
  float* bias[2] = {bias_1->values()->asMutable<float>(), bias_2->values()->asMutable<float>()};

  // step1: Register
  exec::registerVectorFunction(
    "torchDNN",
    TorchDNN::signatures(),
    std::make_unique<TorchDNN>(weights, bias, dimensions)
  );

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"torchDNN(x)"})
		              .planNode();

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();     
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Test (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
 // std::cout << results->toString(0, results->size()) << std::endl;
  
}

void MLFunctionsTest::test_mnist() {
    //Eigen::setNbThreads(12);
    std::cout << Eigen::nbThreads() << std::endl;
    int input_size = 784; // num_features
    int layer1_size = 20; // num units in hidden layer 1
    int layer2_size = 10;
    int num_samples = 10;

    std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
    std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
   std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 

    FlatVectorPtr<float> weights_1 = get_tensor(weights_file, layer1_size * input_size, input_size);
    FlatVectorPtr<float> bias_1 = get_tensor(bias_file, layer1_size, 1);
    FlatVectorPtr<float> weights_2 = get_tensor(weights_file, layer2_size * layer1_size, layer1_size);
    FlatVectorPtr<float> bias_2 = get_tensor(bias_file, layer2_size, 1);
    weights_file.close();
    bias_file.close();

    float* bias_1_values = bias_1->values()->asMutable<float>();
    float* bias_2_values = bias_2->values()->asMutable<float>();

    FlatVectorPtr<float> bias_1_mat = maker.flatVector<float>(num_samples * layer1_size);
    for(int i=0; i < bias_1_mat->size(); i++)
      bias_1_mat->set(i, bias_1_values[i%layer1_size]);
    
    FlatVectorPtr<float> bias_2_mat = maker.flatVector<float>(num_samples * layer2_size);
    for(int i=0; i < bias_2_mat->size(); i++)
      bias_2_mat->set(i, bias_2_values[i%layer2_size]);

    FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples);
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

    std::string compute =  NNBuilder()
                          .denseLayer(layer1_size ,input_size, weights_1->values()->asMutable<float>(), 
                            bias_1_mat->values()->asMutable<float>(), NNBuilder::RELU)
                          .denseLayer(layer2_size ,layer1_size, weights_2->values()->asMutable<float>(), 
                            bias_2_mat->values()->asMutable<float>(), NNBuilder::SOFTMAX)
                          .build();

    std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
    auto plan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({fmt::format(compute, "x")}) 
		              .planNode();
    

   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();     
   auto results = exec::test::AssertQueryBuilder(plan).copyResults(pool_.get());
   std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
   std::cout << "Time for Test (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
   std::cout << "Results:" << results->toString() << std::endl;  
   //std::cout << results->toString(0, results->size()) << std::endl;
   
}

FlatVectorPtr<float> MLFunctionsTest::get_tensor(std::ifstream& file, int size, int lines){
    FlatVectorPtr<float> tensor = maker.flatVector<float>(size);
    int index = 0;
    std::string line;
    while (lines--) { // Read a line from the file
        std::getline(file, line);
        std::istringstream iss(line); // Create an input string stream from the line
      
        std::string numberStr;
        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            float number = std::stof(numberStr);    // Convert the string to float
            tensor->set(index++, number);
        }
    }
    return tensor;
}


void MLFunctionsTest::test_multithreading() { 

  int input_size = 1000;
  int output_size = 500;
  int num_samples = 6000;
  // ( 600 * 1000 x 1000 * 500 )
  int size = output_size * input_size;
  
  auto weights = maker.flatVector<float>(size);

  for(int i=0; i < size; i++){
    weights->set(i, i*2);
  } 
  // register Vector Function
  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  // Create input
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_samples; i++){ 
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
  
  // Create Plan
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  auto plan0 = exec::test::PlanBuilder(planNodeIdGenerator)
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
		              .project({"mat_mul(x)"})
                  .planFragment();

  // queryCtx_->testingOverrideConfigUnsafe(
  //     {{core::QueryConfig::kPreferredOutputBatchRows, "400"}, {core::QueryConfig::kPreferredOutputBatchBytes, "2000000"},  {core::QueryConfig::kMaxOutputBatchRows, "300"}});
  // Create task
  std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 500 * MB)};
  queryCtx_->testingOverrideMemoryPool(rootPool);
  
  queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});

  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector});
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  int num_splits = 20;
  for(int i=0; i < num_splits; i++)
    paths.push_back(file);
  auto hiveSplits = makeHiveConnectorSplits(paths);
 
  int concurrency = 4;
  boost::interprocess::interprocess_semaphore semaphore(concurrency);

  auto task = exec::Task::create("0", plan0 , 0, queryCtx_, 
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            semaphore.post();
          }
          // threads finally exit when there is no data
          // so we don't relase the semaphore in this case
          return exec::BlockingReason::kNotBlocked;
  });

  // Create 2 hive splits and add them to task
  task->start(task, concurrency);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }
  task->noMoreSplits(p0);
  std::cout << std::endl;
  // Start task with 2 as maximum drivers and wait for execution to finish
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

// out of memory and gets stuck //
// exception is not handles yet
// stop the program after it is stuck
void MLFunctionsTest::test_multithreading_oom() { 

  int input_size = 1000;
  int output_size = 500;
  int num_samples = 6000;
  // ( 600 * 1000 x 1000 * 500 )
  int size = output_size * input_size;
  
  auto weights = maker.flatVector<float>(size);

  for(int i=0; i < size; i++){
    weights->set(i, i*2);
  } 
  // register Vector Function
  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  // Create input
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_samples; i++){ 
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
  
  // Create Plan
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  auto plan0 = exec::test::PlanBuilder(planNodeIdGenerator)
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
		              .project({"mat_mul(x)"})
                  .planFragment();

  // queryCtx_->testingOverrideConfigUnsafe(
  //     {{core::QueryConfig::kPreferredOutputBatchRows, "400"}, {core::QueryConfig::kPreferredOutputBatchBytes, "2000000"},  {core::QueryConfig::kMaxOutputBatchRows, "300"}});
  // Create task
  std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 500 * MB)};
  queryCtx_->testingOverrideMemoryPool(rootPool);
  
  queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});

  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector});
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  int num_splits = 20;
  for(int i=0; i < num_splits; i++)
    paths.push_back(file);
  auto hiveSplits = makeHiveConnectorSplits(paths);
 
  int concurrency = 48;
  boost::interprocess::interprocess_semaphore semaphore(concurrency);

  auto task = exec::Task::create("0", plan0 , 0, queryCtx_, 
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            semaphore.post();
          }
          return exec::BlockingReason::kNotBlocked;
  });

  // Create 2 hive splits and add them to task
  task->start(task, concurrency);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }
  task->noMoreSplits(p0);
  std::cout << std::endl;
  // Start task with 2 as maximum drivers and wait for execution to finish
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

void MLFunctionsTest::test_batching() { 

  int input_size = 1000;
  int output_size = 500;
  int num_samples = 600;
  // ( 600 * 1000 x 1000 * 500 )
  int size = output_size * input_size;
  
  auto weights = maker.flatVector<float>(size);

  for(int i=0; i < size; i++){
    weights->set(i, i*2);
  } 
  // register Vector Function
  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  // Create input
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_samples; i++){ 
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
  
  // Create Plan
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  auto plan0 = exec::test::PlanBuilder(planNodeIdGenerator)
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
		              .project({"mat_mul(x)"})
                  .planFragment();
  //{core::QueryConfig::kPreferredOutputBatchBytes, "2000000"}
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kPreferredOutputBatchBytes, "4000000"}, {core::QueryConfig::kMaxOutputBatchRows, "300"}});
  // Create task
  std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 500 * MB)};
  queryCtx_->testingOverrideMemoryPool(rootPool);
  
  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector});
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  int num_splits = 1;
  for(int i=0; i < num_splits; i++)
    paths.push_back(file);
  auto hiveSplits = makeHiveConnectorSplits(paths);
 
  int concurrency = 1;
  boost::interprocess::interprocess_semaphore semaphore(concurrency);

  auto task = exec::Task::create("0", plan0 , 0, queryCtx_, 
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            semaphore.post();
          }
          return exec::BlockingReason::kNotBlocked;
  });

  // Create 2 hive splits and add them to task
  task->start(task, concurrency);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }
  task->noMoreSplits(p0);
  std::cout << std::endl;
  // Start task with 2 as maximum drivers and wait for execution to finish
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}


void MLFunctionsTest::test_conv2d() {
    int cnn_layer1_filters = 64;
    int cnn_layer1_filter_dims[] = {3,3,1}; // height * width * channels
    int weights1_size = cnn_layer1_filter_dims[0] * cnn_layer1_filter_dims[1] * cnn_layer1_filter_dims[2] * cnn_layer1_filters;

    int cnn_layer2_filters = 64;
    int cnn_layer2_filter_dims[] = {3,3,64};
    int weights2_size = cnn_layer2_filter_dims[0] * cnn_layer2_filter_dims[1] * cnn_layer2_filter_dims[2] * cnn_layer2_filters;

    int input_dims[] = {28,28,1}; 
    int input_size = input_dims[0] * input_dims[1] * input_dims[2];
    int num_samples = 1;

    int dims1[] = {cnn_layer1_filters, cnn_layer1_filter_dims[0], cnn_layer1_filter_dims[1], cnn_layer1_filter_dims[2], input_dims[0], input_dims[1]};
    int dims2[] = {cnn_layer2_filters, cnn_layer2_filter_dims[0], cnn_layer2_filter_dims[1], cnn_layer2_filter_dims[2], input_dims[0] - cnn_layer1_filter_dims[0] + 1, input_dims[1] - cnn_layer1_filter_dims[1] + 1};
    std::ifstream weights_file("/home/local/ASUAD/snola119/cnn_weights.txt"); 
    std::ifstream bias_file("/home/local/ASUAD/snola119/cnn_bias.txt"); 
    std::ifstream test_file("/home/local/ASUAD/snola119/test_file.txt"); 

    FlatVectorPtr<float> weights_1 = get_tensor(weights_file, weights1_size, cnn_layer1_filters * cnn_layer1_filter_dims[2]);
    FlatVectorPtr<float> bias_1 = get_tensor(bias_file, cnn_layer1_filters, 1);

    FlatVectorPtr<float> weights_2 = get_tensor(weights_file, weights2_size, cnn_layer2_filters * cnn_layer2_filter_dims[2]); 
    FlatVectorPtr<float> bias_2 = get_tensor(bias_file, cnn_layer2_filters, 1);

    int input3_size = 9216; // num_features
    int layer3_size = 512; // num units in hidden layer 1
    int layer4_size = 10;


    FlatVectorPtr<float> weights_3 = get_tensor(weights_file, layer3_size * input3_size, input3_size);
    FlatVectorPtr<float> bias_3 = get_tensor(bias_file, layer3_size, 1);
    FlatVectorPtr<float> weights_4 = get_tensor(weights_file, layer4_size * layer3_size, layer3_size);
    FlatVectorPtr<float> bias_4 = get_tensor(bias_file, layer4_size, 1);

    float* w3 =  weights_3->values()->asMutable<float>();
    float* w4 =  weights_4->values()->asMutable<float>();


    weights_file.close();
    bias_file.close();

    float* bias_1_values = bias_1->values()->asMutable<float>();
    float* bias_2_values = bias_2->values()->asMutable<float>();
    float* bias_3_values = bias_3->values()->asMutable<float>();
    float* bias_4_values = bias_4->values()->asMutable<float>();
    

    FlatVectorPtr<float> bias_3_mat = maker.flatVector<float>(num_samples * layer3_size);
    for(int i=0; i < bias_3_mat->size(); i++)
      bias_3_mat->set(i, bias_3_values[i%layer3_size]);
    
    FlatVectorPtr<float> bias_4_mat = maker.flatVector<float>(num_samples * layer4_size);
    for(int i=0; i < bias_4_mat->size(); i++)
      bias_4_mat->set(i, bias_4_values[i%layer4_size]);


    FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples * input_dims[2]);
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
    
    std::string compute =  NNBuilder()
                          .convLayer(cnn_layer1_filters, dims1, weights_1->values()->asMutable<float>(), 
                            bias_1->values()->asMutable<float>(), NNBuilder::RELU)
                          .convLayer(cnn_layer2_filters, dims2, weights_2->values()->asMutable<float>(), 
                            bias_2->values()->asMutable<float>(), NNBuilder::RELU)
                          .maxPoolLayer(2, dims2[4] - cnn_layer2_filter_dims[0] + 1, dims2[5] - cnn_layer2_filter_dims[1] + 1)
                          .denseLayer(layer3_size ,input3_size, weights_3->values()->asMutable<float>(), 
                            bias_3_mat->values()->asMutable<float>(), NNBuilder::RELU)
                          .denseLayer(layer4_size ,layer3_size, weights_4->values()->asMutable<float>(), 
                            bias_4_mat->values()->asMutable<float>(), NNBuilder::SOFTMAX)
                          .build();

    std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
    auto plan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({fmt::format(compute, "x")}) 
		              .planNode();
    auto results = exec::test::AssertQueryBuilder(plan).copyResults(pool_.get());
    
   std::cout << "Results:" << results->toString() << std::endl;
   std::cout << results->toString(0, results->size()) << std::endl;

}

void MLFunctionsTest::run() {
  // test_mat_mul();
  // test_mat_add();
  // test_relu();
  // test_dense_layer();
  //   test_torch_dense_layer();
  //   test_mnist();
  //   test_multithreading();
  //  test_multithreading_oom();
      test_batching();
  //  test_conv2d();
   

}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  MLFunctionsTest demo;
  demo.run();
}
