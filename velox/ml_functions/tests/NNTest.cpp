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
#define EIGEN_USE_BLAS
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
#include <string>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"



using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

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

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() {
     HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  /// Run the demo.
  void run();
  void test_dense_layer();
  void test_torch_dense_layer();
  void test_mnist_multithreading();
  void test_torch_dense_layer_multithreading();

  FlatVectorPtr<float> get_tensor(std::ifstream& file, int size, int lines);
  FlatVectorPtr<float> get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines);

  std::vector<float>  get_config(int size){
    std::ifstream conf_file("../../../../velox/ml_functions/tests/config.txt");
    FlatVectorPtr<float> conf = get_tensor(conf_file, size, size);
    conf_file.close();
    float* confs =  conf->values()->asMutable<float>();
    std::vector<float> confs_vector(confs, confs + size);
    return confs_vector;
  }

  void execute_plan(core::PlanFragment plan, PlanNodeId p0, RowVectorPtr inputRowVector, std::vector<float> confs) {

    int num_samples = (int) confs[0];
    int num_splits = (int) confs[1];
    int concurrency = (int) confs[2];

    auto config = std::make_shared<facebook::velox::dwrf::Config>();

    // affects the number of splits
    // number of bites in each stripe (collection of rows)
    // strip size should be <= split size (total_size / total splits)
    // to have the desired number of splits
    uint64_t kSizeKB = 1024UL;
    // used for indexing. 
    // 2k rows will be processed in every call
    // but doesn't effect number of splits
    // if stripe size is a large value
    uint32_t rows = num_samples/num_splits;

    config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 100 * kSizeKB);
    config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, rows);
    auto file = TempFilePath::create();
    writeToFile(file->path, {inputRowVector}, config);
    
    auto hiveSplits =  makeHiveConnectorSplits(file->path, num_splits, dwio::common::FileFormat::DWRF);
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, std::to_string((int)confs[3])}, {core::QueryConfig::kMaxOutputBatchRows, std::to_string((int)confs[4])}});
    
    auto task = exec::Task::create("0", plan , 0, queryCtx_, 
          [](RowVectorPtr result, ContinueFuture* /*unused*/) {
            if(result)
                std::cout << result->toString() << std::endl;
            return exec::BlockingReason::kNotBlocked;
    });

    std::cout << "Hive splits:" << std::endl;
    for(auto& split : hiveSplits) {
      std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    task->start(concurrency);
    task->noMoreSplits(p0);
    waitForFinishedDrivers(task);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {

    while (!task->isFinished()) {     
      usleep(1000); // 0.01 second.
    }
  }


  std::unique_ptr<MemoryManager> memoryManager_;
  
  uint64_t kMemoryCapacity = 512 * MB;
  uint64_t kInitMemoryPoolCapacity = 16 * MB;
  uint64_t kMinMemoryPoolCapacityTransferSize = 8 * MB;

  std::shared_ptr<core::QueryCtx> newQueryCtx(int64_t memoryCapacity) {
    
    std::unordered_map<std::string, std::shared_ptr<Config>> configs;
    std::shared_ptr<MemoryPool> pool = memory::defaultMemoryManager().addRootPool(
        "", memoryCapacity, memory::MemoryReclaimer::create());
    std::unordered_map<std::string, std::string> queryConfig = {{core::QueryConfig::kSpillEnabled, "true"}, 
                                      {core::QueryConfig::kJoinSpillEnabled, "true"},  
                                      {core::QueryConfig::kJoinSpillMemoryThreshold, "1"},
                                       {core::QueryConfig::kSpillableReservationGrowthPct, "1"},
                                      /* 
                                      kSpillPartitionBits is removed after PR 5890, 
                                      kJoinSpillPartitionBits and kAggregationSpillPartitionBits are introduced 
                                      Please consider how to replace it by check the following link: 
                                      https://github.com/facebookincubator/velox/pull/5890 
                                      */
                                      //  {core::QueryConfig::kSpillPartitionBits, "1"}
                                      };
    auto queryCtx = std::make_shared<core::QueryCtx>(
        executor_.get(),
        queryConfig,
        configs,
        cache::AsyncDataCache::getInstance(),
        /*
        Note from origin PR: Removing the relationship of AsyncDataCache inheritance from MemoryAllocator.
        Please check the following commit:
        https://github.com/facebookincubator/velox/commit/ad9ffa1fca3fbb3a550ab426a00ebb745b339b34
        */
        // memory::MemoryAllocator::getInstance(),
        std::move(pool));
    return queryCtx;
  }

  std::shared_ptr<folly::Executor> executor_{std::make_shared<folly::CPUThreadPoolExecutor>(std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{std::make_shared<core::QueryCtx>(executor_.get())};
  
  std::shared_ptr<memory::MemoryPool> pool_ =  memory::addDefaultLeafMemoryPool();
  //std::shared_ptr<memory::MemoryPool> childPool = rootPool_->addAggregateChild("HiveConnectorTestBase.Writer");
  VectorMaker maker{pool_.get()};

};

FlatVectorPtr<float> MLFunctionsTest::get_tensor(std::ifstream& file, int size, int lines){
    return get_tensor(maker,file,size,lines);
}

FlatVectorPtr<float> MLFunctionsTest::get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines){
    //std::cout << "Loading tensor of size " << size << std::endl;
    FlatVectorPtr<float> tensor = m.flatVector<float>(size);
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

void MLFunctionsTest::test_torch_dense_layer(){
 
  int input_size = 768; // num_features
  int layer1_size = 3072; // num units in hidden layer 1
  int layer2_size = 768;
  
  std::vector<int> dimensions;
  dimensions.push_back(input_size);
  dimensions.push_back(layer1_size);
  dimensions.push_back(layer2_size);
  

  int num_samples = std::atoi(std::getenv("samples"));
  int num_splits = std::atoi(std::getenv("splits"));
  int velox_threads = std::atoi(std::getenv("vthreads"));
  int torch_threads = std::atoi(std::getenv("tthreads"));
  torch::set_num_threads(torch_threads);
  
  std::ifstream weights_file("/home/ubuntu/bert_weights.txt"); 
  std::ifstream bias_file("/home/ubuntu/bert_bias.txt"); 
  std::ifstream test_file("/home/ubuntu/bert_input.txt"); 

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
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  

  auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                //.project({fmt::format(compute, "x")}) 
                .project({"torchDNN(x)"}) 
                .planFragment();

  auto config = std::make_shared<facebook::velox::dwrf::Config>();

  // affects the number of splits
  // number of bites in each stripe (collection of rows)
  // strip size should be <= split size (total_size / total splits)
  // to have the desired number of splits
  uint64_t kSizeKB = 1024UL;
  // used for indexing. 
  // 2k rows will be processed in every call
  // but doesn't effect number of splits
  // if stripe size is a large value
  uint32_t rows = num_samples/num_splits;
  config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 100 * kSizeKB);
  config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, rows);
  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector}, config);
  
  auto hiveSplits =  makeHiveConnectorSplits(file->path, num_splits, dwio::common::FileFormat::DWRF);

  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kPreferredOutputBatchBytes, "100000000"}, {core::QueryConfig::kMaxOutputBatchRows, "100000"}});
  auto task = exec::Task::create("0", plan , 0, queryCtx_, 
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          //  if(result)
          //     std::cout << result->toString() << std::endl;
          return exec::BlockingReason::kNotBlocked;
  });

  
  //std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
   // std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  task->start(velox_threads);
  task->noMoreSplits(p0);
  // Start task with 2 as maximum drivers and wait for execution to finish
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::stringstream ss;
  ss << num_samples << "," << num_splits << "," << velox_threads << "," << torch_threads << ",";
  std::cout << ss.str() << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

void MLFunctionsTest::test_mnist_multithreading() {
    
    int input_size = 784; // num_features
    int layer1_size = 1024; // num units in hidden layer 1
    int layer2_size = 10;
    
    std::vector<float> confs = get_config(5);
    
    int num_samples = (int) confs[0];
   
    std::ifstream weights_file("/home/ubuntu/w1024.txt"); 
    std::ifstream bias_file("/home/ubuntu/b1024.txt"); 
    std::ifstream test_file("/home/ubuntu/x_test_large.txt"); 

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

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId p0;
  
    std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
    auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  .project({fmt::format(compute, "x")}) 
		              .planFragment();
  
    execute_plan(plan, p0, inputRowVector, confs);
}

void MLFunctionsTest::test_torch_dense_layer_multithreading(){
  
  int input_size = 784; // num_features
  int layer1_size = 1024; // num units in hidden layer 1
  int layer2_size = 10;

  std::vector<float> confs = get_config(6);
  
  int num_samples = (int) confs[0];
  int num_splits = (int) confs[1];
  int concurrency = (int) confs[2];
  int torch_threads = (int) confs[5];

  std::vector<int> dimensions;
  dimensions.push_back(input_size);
  dimensions.push_back(layer1_size);
  dimensions.push_back(layer2_size);

  std::ifstream weights_file("/home/ubuntu/w1024.txt"); 
  std::ifstream bias_file("/home/ubuntu/b1024.txt"); 
  std::ifstream test_file("/home/ubuntu/x_test_large.txt"); 

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

                   
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  
  auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  .project({"torchDNN(x)"})
		              .planFragment();
  execute_plan(plan, p0, inputRowVector, confs);
}

void MLFunctionsTest::run() {
  // test_mnist_multithreading();
  test_torch_dense_layer_multithreading();
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  MLFunctionsTest demo;
  demo.run();
}
