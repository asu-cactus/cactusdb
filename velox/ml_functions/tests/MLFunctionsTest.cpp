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
#include "velox/common/base/Fs.h"
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
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
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
#include "velox/ml_functions/tests/MLTestUtility.h"



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

    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
    filesystems::registerLocalFileSystem();
    dwio::common::LocalFileSink::registerFactory();

    SetUp();

  }

  ~MLFunctionsTest() {
  }

  // Function from ParquetTestBase.h
  std::unique_ptr<dwio::common::FileSink> createSink(
      const std::string& filePath) {
    auto sink = dwio::common::FileSink::create(
        fmt::format("file:{}", filePath), {.pool = rootPool_.get()});
    return sink;
  }

  // Function from ParquetTestBase.h
  std::unique_ptr<facebook::velox::parquet::Writer> createWriter(
      std::unique_ptr<dwio::common::FileSink> sink,
      std::function<
          std::unique_ptr<facebook::velox::parquet::DefaultFlushPolicy>()>
          flushPolicy,
      const RowTypePtr& rowType,
      facebook::velox::common::CompressionKind compressionKind =
          facebook::velox::common::CompressionKind_NONE) {
    facebook::velox::parquet::WriterOptions options;
    options.memoryPool = rootPool_.get();
    options.flushPolicyFactory = flushPolicy;
    options.compression = compressionKind;
    return std::make_unique<facebook::velox::parquet::Writer>(
        std::move(sink), options, rowType);
  }


  /// Run the demo.
  void run(int numDriver, int memoryPoolSizeMB, int spillMemThresholdMB, bool enableSpill, int repeatRun);
  void test_mat_mul();
  void test_mat_add();
  void test_relu();
  void test_softmax();
  void test_dense_layer();
  void test_torch_dense_layer();
  void test_mnist();
  void test_multithreading();
  void test_multithreading_oom();
  void test_batching();
  void test_conv2d();
  void test_deep_bench_conv1();
  void test_spill(int numDriver, int memoryPoolSizeMB, int spillMemThresholdMB, bool enableSpill, int repeatRun);
  void test_mnist_multithreading();
  void test_torch_dense_layer_multithreading();
  void mytest();
  void test_land_cover_conv3();

  void test_mnist_oom_weights();
  void test_complex_torchnn();

  std::unique_ptr<MemoryManager> memoryManager_;
  
  uint64_t kMemoryCapacity = 512 * MB;
  uint64_t kInitMemoryPoolCapacity = 16 * MB;
  uint64_t kMinMemoryPoolCapacityTransferSize = 8 * MB;

  std::shared_ptr<core::QueryCtx> newQueryCtx(
      int64_t memoryCapacity) {
    
    std::unordered_map<std::string, std::shared_ptr<Config>> configs;
    std::shared_ptr<MemoryPool> pool = memory::MemoryManager::getInstance()->addRootPool(
        "", memoryCapacity, memory::MemoryReclaimer::create());
    std::unordered_map<std::string, std::string> queryConfigValues = {};
   std::unordered_map<std::string, std::string> myMapWithValues = {{core::QueryConfig::kSpillEnabled, "true"}, 
                                      {core::QueryConfig::kJoinSpillEnabled, "true"},  
                                      {core::QueryConfig::kJoinSpillMemoryThreshold, "10485760"},
                                      //  {core::QueryConfig::kSpillableReservationGrowthPct, "1"},
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
        queryConfigValues,
        configs,
        cache::AsyncDataCache::getInstance(),
        std::move(pool));
    return queryCtx;
  }
  FlatVectorPtr<float> get_tensor(std::ifstream& file, int size, int lines);
  FlatVectorPtr<float> get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines);


  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() {
     HiveConnectorTestBase::TearDown();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {

    while (!task->isFinished()) {     
      usleep(1000); // 0.01 second.
    }
  }

  void TestBody() override {}

  
  std::shared_ptr<folly::Executor> executor_{std::make_shared<folly::CPUThreadPoolExecutor>(std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{std::make_shared<core::QueryCtx>(executor_.get())};
  
  std::shared_ptr<memory::MemoryPool> pool_ =  memory::MemoryManager::getInstance()->addLeafPool();
  //std::shared_ptr<memory::MemoryPool> childPool = rootPool_->addAggregateChild("HiveConnectorTestBase.Writer");
  VectorMaker maker{pool_.get()};

};

void MLFunctionsTest::test_mat_mul() { 
//Eigen::setNbThreads(48); 
  int output_size = 500;
  int input_size = 100;
  int num_samples = 500;
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
                  // using CPU for mat_mul
                  .project({"mat_mul(x)"})
                  // using GPU for mat_mul
                  // .project({"mat_mul(x, true)"})
		              .planNode();

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan)
                  .maxDrivers(4)
                  .config("max_output_batch_rows", std::to_string(10))
                  .config("preferred_output_batch_rows", std::to_string(10))
                  .config("preferred_output_batch_bytes", std::to_string(1000))
                  .copyResults(pool_.get());

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Matrix multiply (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  // std::cout << "Results:" << results->toString() << std::endl;
  // std::cout << results->toString(0, results->size()) << std::endl;

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

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Matrix Addition (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  // std::cout << "Results:" << results->toString() << std::endl;
  // std::cout << results->toString(0, results->size()) << std::endl;
}

void MLFunctionsTest::test_relu(){
 
  int num_rows = 1000;
  int num_cols = 5000; 

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
  std::make_unique<Relu>());

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"relu(x)"})
		              .planNode();
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Relu (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  // std::cout << "Results:" << results->toString() << std::endl;
  // std::cout << results->toString(0, results->size()) << std::endl;
}

void MLFunctionsTest::test_softmax(){
 
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
  "softmax",
  Relu::signatures(),
  std::make_unique<Softmax>());

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"softmax(x)"})
		              .planNode();
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Softmax (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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
  
  // std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
  // std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
  // std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 
 
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
    TorchDNN2Level::signatures(),
    std::make_unique<TorchDNN2Level>(weights, bias, dimensions)
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

void MLFunctionsTest::test_mnist() {
    //Eigen::setNbThreads(1);
    std::cout << Eigen::nbThreads() << std::endl;
    int input_size = 784; // num_features
    int layer1_size = 1024; // num units in hidden layer 1
    int layer2_size = 10;
    int num_samples = 60000;

    // std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
    // std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
    // std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 
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

void MLFunctionsTest::test_multithreading() { 

  int input_size = 100;
  int output_size = 500;
  int num_samples = 30000;
  // ( 6000 * 1000 x 1000 * 500 )
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

  
  std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root", 500 * MB)};
  queryCtx_->testingOverrideMemoryPool(rootPool);
  
  auto file = TempFilePath::create();
  auto config = std::make_shared<facebook::velox::dwrf::Config>();

  // affects the number of splits
  // number of bites in each stripe (collection of rows)
  // strip size should be <= split size (total_size / total splits)
  // to have the desired number of splits
  uint64_t kSizeKB = 1024UL;

  // used for indexing. 
  // 2k rows will be processed in every call
  // but doesn't effect number of splits ...
  // if stripe size is a large value
  uint32_t rows = 2000;

  config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 100 * kSizeKB);
  config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, rows);
  writeToFile(file->path, {inputRowVector}, config);
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  auto hiveSplits =  makeHiveConnectorSplits(file->path, 3, dwio::common::FileFormat::DWRF);
  int concurrency = 3;
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

  task->start(concurrency);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }
  task->noMoreSplits(p0);
  std::cout << std::endl;
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

void MLFunctionsTest::mytest() {
  constexpr int32_t kNumRows = 2000; 
  constexpr int64_t kMaxBytes = 1LL << 30; // 1GB
  auto rowType = ROW({"c0", "c1", "c2"}, {INTEGER(), INTEGER(), INTEGER()}); 
  VectorFuzzer fuzzer({}, pool_.get());
  const int32_t numBatches = 5;
  std::vector<RowVectorPtr> batches;
  for (int32_t i = 0; i < numBatches; ++i) {
    batches.push_back(fuzzer.fuzzRow(rowType));
  }
  struct {
    uint64_t orderByMemLimit;
    bool expectSpill;

    std::string debugString() const {
      return fmt::format(
          "orderByMemLimit:{}, expectSpill:{}", orderByMemLimit, expectSpill);
    }
  } testSettings[] = {// Memory limit is disabled so spilling is not triggered.
                      {0, false},
                      // Memory limit is too small so always trigger spilling.
                      {1, true},
                      // Memory limit is too large so spilling is not triggered.
                      {1'000'000'000, false}};

  for (const auto& testData : testSettings) {
    SCOPED_TRACE(testData.debugString());
    auto tempDirectory = exec::test::TempDirectoryPath::create();
    auto queryCtx = std::make_shared<core::QueryCtx>(executor_.get());
    queryCtx->testingOverrideMemoryPool(
        memory::MemoryManager::getInstance()->addRootPool(
            queryCtx->queryId(), kMaxBytes));
    auto results =
        AssertQueryBuilder(
            PlanBuilder()
                .values(batches)
                .orderBy({fmt::format("{} ASC NULLS LAST", "c0")}, false)
                .planNode())
            .queryCtx(queryCtx)
            .copyResults(pool_.get());
    auto task =
        AssertQueryBuilder(
            PlanBuilder()
                .values(batches)
                .orderBy({fmt::format("{} ASC NULLS LAST", "c0")}, false)
                .planNode())
            .queryCtx(queryCtx)
            .spillDirectory(tempDirectory->path)
            .config(core::QueryConfig::kSpillEnabled, "true")
            .config(core::QueryConfig::kOrderBySpillEnabled, "true")
            .config(
                QueryConfig::kOrderBySpillMemoryThreshold,
                std::to_string(testData.orderByMemLimit))
            .assertResults(results);

    auto stats = task->taskStats().pipelineStats;
    for(auto stat : stats){
    for(auto ops : stat.operatorStats){
      std::cout << ops.spilledBytes << " ";
    }
    std::cout << std::endl;
    }
    ASSERT_EQ(testData.expectSpill, stats[0].operatorStats[1].spilledBytes > 0);
  }
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
  std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root", 500 * MB)};
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
  task->start(concurrency);
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
  std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root", 500 * MB)};
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
  task->start(concurrency);
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

void MLFunctionsTest::test_spill(int numDriver = 4, int memoryPoolSizeMB = 100, int spillMemThresholdMB = 20, bool enableSpill = false, int repeatRun = 5){
  std::cout << fmt::format("[INFO] Test Spill, memoryPoolSizeMB: {}, spillMemThresholdMB: {}, enable_spill: {}", memoryPoolSizeMB, spillMemThresholdMB, enableSpill) << std::endl;
  int lsize = 10000;
  int rsize = 1000;

  auto a = maker.flatVector<int>(lsize);
  auto b = maker.flatVector<int>(lsize);

  auto c = maker.flatVector<int>(rsize*rsize);
  auto d = maker.flatVector<int>(rsize*rsize);


  for(int i=0; i < lsize; i++){
    a->set(i, i);
    b->set(i, i*10);
  } 

  for(int i=0; i < rsize; i++) {
    for(int j=0; j < rsize; j++){
      c->set(i*rsize + j, j);
      d->set(i*rsize + j, i*10);
    }
  } 

  auto leftVectors = maker.rowVector({"l1","l2"}, {a, b});
  auto rightVectors = maker.rowVector({"r1", "r2"}, {c, d});

  auto leftRowType =
      ROW({"l1", "l2"}, {INTEGER(), INTEGER()});

  auto rightRowType =
      ROW({"r1", "r2"}, {INTEGER(), INTEGER()});

  auto tempPath = exec::test::TempDirectoryPath::create();
  auto leftFilePath =
      fs::path(fmt::format("{}/left.parquet", tempPath->path));
  auto sink = createSink(leftFilePath);
  auto sinkPtr = sink.get();
  uint64_t kRowsInRowGroup = 100;
  uint64_t kBytesInRowGroup = 128 * 1024 * 1024;
  auto writer = createWriter(std::move(sink), [&]() {
    return std::make_unique<facebook::velox::parquet::LambdaFlushPolicy>(
        kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
  }, leftRowType);
  writer->write(leftVectors);
  writer->flush();
  writer->close();

  auto rightFilePath =
      fs::path(fmt::format("{}/right.parquet", tempPath->path));
  sink = createSink(rightFilePath);
  sinkPtr = sink.get();
  writer = createWriter(std::move(sink), [&]() {
    return std::make_unique<facebook::velox::parquet::LambdaFlushPolicy>(
        kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
  }, rightRowType);
  writer->write(rightVectors);
  writer->flush();
  writer->close();


  const auto spillDirectory = exec::test::TempDirectoryPath::create();

  auto qctx = newQueryCtx(memoryPoolSizeMB * MB);
  if (enableSpill) {
    qctx->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "true"}, 
                                      {core::QueryConfig::kJoinSpillEnabled, "true"},  
                                      {core::QueryConfig::kJoinSpillMemoryThreshold, std::to_string(spillMemThresholdMB * MB)},
    });
  }

  core::PlanNodeId rightTableScanNodeId;
  core::PlanNodeId leftTableScanNodeId;

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto buildSide = PlanBuilder(planNodeIdGenerator)
                       .tableScan(rightRowType, {}, "")
                       .capturePlanNodeId(rightTableScanNodeId)
                       .project({"r1 AS u_r1", "r2 AS u_r2"})
                       .planNode();

  auto joinPlan = PlanBuilder(planNodeIdGenerator)
                        .tableScan(leftRowType, {}, "")
                        .capturePlanNodeId(leftTableScanNodeId)
                        .project({"l1 AS u_l1", "l2 AS u_l2"})
                        .hashJoin(
                            {"u_l1"},
                            {"u_r1"},
                            buildSide,
                            "",
                            {"u_l1", "u_r1", "u_l2"},
                            core::JoinType::kFull);


  auto rightDataHiveSplits = makeHiveConnectorSplits(
        {rightFilePath},
        5,
        dwio::common::FileFormat::PARQUET);

  auto leftDataHiveSplits = makeHiveConnectorSplits(
      {leftFilePath},
      5,
      dwio::common::FileFormat::PARQUET);
  
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (int i = 0; i < repeatRun; i++) {
    
    CursorParameters params;
    params.maxDrivers = numDriver;
    params.planNode = joinPlan.planNode();
    params.queryCtx = qctx;
    params.spillDirectory = spillDirectory->path;
    bool noMoreSplits = false;
    auto addSplits = [&noMoreSplits, &rightDataHiveSplits, &rightTableScanNodeId, 
                      &leftDataHiveSplits, &leftTableScanNodeId](exec::Task* task) {
      if (!noMoreSplits) {
        for (auto& split : rightDataHiveSplits) {
          task->addSplit(rightTableScanNodeId, exec::Split(std::move(split)));
        }
        task->noMoreSplits(rightTableScanNodeId);

        for (auto& split : leftDataHiveSplits) {
          task->addSplit(leftTableScanNodeId, exec::Split(std::move(split)));
        }
        task->noMoreSplits(leftTableScanNodeId);
      }
      noMoreSplits = true;
    };

    auto [cursor, actualResults] = readCursor(params, addSplits);
    waitForTaskCompletion(cursor->task().get());
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  auto elapsedTime =
          (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()) /
          1000000.0;
  auto avgElapsedTime = elapsedTime / repeatRun;
  std::cout << "Time for Test (sec) = " <<  avgElapsedTime << std::endl;
}

void MLFunctionsTest::test_mnist_multithreading() {
    
    int input_size = 784; // num_features
    int layer1_size = 1024; // num units in hidden layer 1
    int layer2_size = 10;
    
    std::ifstream conf_file("/home/ubuntu/samples.txt");
    FlatVectorPtr<float> conf = get_tensor(conf_file, 2, 2);
    float* confs = conf->values()->asMutable<float>();
    conf_file.close();
    std::cout << (int)confs[0] << (int)confs[1];
    int num_samples = (int) confs[0];
    int num_splits = (int) confs[1];
   
    // std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
    // std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
    // std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 
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
  task->start(num_splits);
  task->noMoreSplits(p0);
  // Start task with 2 as maximum drivers and wait for execution to finish
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

void MLFunctionsTest::test_torch_dense_layer_multithreading(){
  //torch::set_num_threads(1);
  int input_size = 784; // num_features
  int layer1_size = 1024; // num units in hidden layer 1
  int layer2_size = 10;
  
  // std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
  // std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
  // std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 
  std::ifstream conf_file("/home/ubuntu/samples.txt");
  FlatVectorPtr<float> conf = get_tensor(conf_file, 5, 5);
  float* confs = conf->values()->asMutable<float>();
  conf_file.close();
  

  int num_samples = (int) confs[0];
  int num_splits = (int) confs[1];
  int concurrency = (int) confs[2];
  std::cout << num_samples << num_splits << concurrency << std::endl;

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
    TorchDNN2Level::signatures(),
    std::make_unique<TorchDNN2Level>(weights, bias, dimensions)
  );

                   
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  
  auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
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
  

  config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 100 * kSizeKB);
  config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, (uint32_t) (num_samples/num_splits));
  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector}, config);
  
  auto hiveSplits =  makeHiveConnectorSplits(file->path, num_splits, dwio::common::FileFormat::DWRF);
  
  
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kPreferredOutputBatchBytes, std::to_string((int)confs[3])}, {core::QueryConfig::kMaxOutputBatchRows, std::to_string((int)confs[4])}});
  auto task = exec::Task::create("0", plan , 0, queryCtx_, 
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          // if(result){
          //   std::cout << result->toString(0, result->size()) << std::endl;
          // }
          return exec::BlockingReason::kNotBlocked;
  });

  // Create 2 hive splits and add them to task
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


void MLFunctionsTest::test_mnist_oom_weights() {
    //Eigen::setNbThreads(12);
    std::cout << Eigen::nbThreads() << std::endl;
    int input_size = 784; // num_features
    int layer1_size = 1024; // num units in hidden layer 1
    int layer2_size = 10;
    int num_samples = 2;

    // std::ifstream weights_file("../../../../velox/ml_functions/tests/weights.txt"); 
    // std::ifstream bias_file("../../../../velox/ml_functions/tests/bias.txt"); 
    // std::ifstream test_file("../../../../velox/ml_functions/tests/test_samples.txt"); 
    std::string weights_file_name = "/home/ubuntu/w1024.txt";
    std::string bias_file_name = "/home/ubuntu/b1024.txt";
    std::string test_file_name = "/home/ubuntu/test_samples.txt";

    
    std::ifstream test_file(test_file_name); 

    FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples);
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

    std::string compute =  NNBuilder(weights_file_name, bias_file_name)
                          .denseLayer(layer1_size ,input_size, NNBuilder::RELU)
                          .denseLayer(layer2_size ,layer1_size, NNBuilder::SOFTMAX)
                          .build();
    std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))
    
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId p0;

    // total memory is 5MB
    std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root", 4 * MB)};
    auto childPool = rootPool->addLeafChild("leaf");
    queryCtx_->testingOverrideMemoryPool(rootPool);


    auto plan = exec::test::PlanBuilder(planNodeIdGenerator, childPool.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  .project({fmt::format(compute, "x")}) 
		              .planFragment();

  // optimizer will init the weights
  // config is introduced by the following commit
  // https://github.com/facebookincubator/velox/commit/9f88bfd54f8389d4ab523343d27d6d40177e1781
  core::QueryConfig config({});

  auto mat_mul0 = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul0", {ARRAY(REAL())}, {}, config));
  auto mat_add1 = std::dynamic_pointer_cast<MatrixAddition>(exec::getVectorFunction("mat_add1", {ARRAY(REAL())}, {}, config));
  auto mat_mul3 = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul3", {ARRAY(REAL())}, {}, config));
  auto mat_add4 = std::dynamic_pointer_cast<MatrixAddition>(exec::getVectorFunction("mat_add4", {ARRAY(REAL())}, {}, config));

  std::ifstream weights_file(mat_mul0->getWeightsFile()); 
  std::ifstream bias_file(mat_add1->getWeightsFile()); 

  VectorMaker m{childPool.get()};

  FlatVectorPtr<float> weights_1 = get_tensor(m, weights_file, layer1_size * input_size, input_size);
  FlatVectorPtr<float> bias_1 = get_tensor(m, bias_file, layer1_size, 1);
  FlatVectorPtr<float> weights_2 = get_tensor(m, weights_file, layer2_size * layer1_size, layer1_size);
  FlatVectorPtr<float> bias_2 = get_tensor(m, bias_file, layer2_size, 1);
  weights_file.close();
  bias_file.close();

  float* bias_1_values = bias_1->values()->asMutable<float>();
  float* bias_2_values = bias_2->values()->asMutable<float>();

  FlatVectorPtr<float> bias_1_mat = m.flatVector<float>(num_samples * layer1_size);
  for(int i=0; i < bias_1_mat->size(); i++)
    bias_1_mat->set(i, bias_1_values[i%layer1_size]);
  
  FlatVectorPtr<float> bias_2_mat = m.flatVector<float>(num_samples * layer2_size);
  for(int i=0; i < bias_2_mat->size(); i++)
    bias_2_mat->set(i, bias_2_values[i%layer2_size]);
  
  mat_mul0->setWeights(weights_1->values()->asMutable<float>());
  mat_mul3->setWeights(weights_2->values()->asMutable<float>());
  mat_add1->setWeights(bias_1_mat->values()->asMutable<float>());
  mat_add4->setWeights(bias_2_mat->values()->asMutable<float>());

  

  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector});
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  int num_splits = 1;
  for(int i=0; i < num_splits; i++)
    paths.push_back(file);
  auto hiveSplits = makeHiveConnectorSplits(paths);

  
 
  auto task = exec::Task::create("0", plan , 0, queryCtx_, 
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            std::cout << result->toString(0, result->size()) << std::endl;
          }
          return exec::BlockingReason::kNotBlocked;
  });
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  task->start(num_splits);
  task->noMoreSplits(p0);
  // Start task with 2 as maximum drivers and wait for execution to finish
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
    std::ifstream weights_file("/home/ubuntu/cnn_weights.txt"); 
    std::ifstream bias_file("/home/ubuntu/cnn_bias.txt"); 
    std::ifstream test_file("/home/ubuntu/test_file.txt"); 

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

void MLFunctionsTest::test_deep_bench_conv1() {
    int cnn_filters = 64;
    int cnn_filter_dims[] = {1,1,64}; // height * width * channels
    int weights_size = cnn_filter_dims[0] * cnn_filter_dims[1] * cnn_filter_dims[2] * cnn_filters;

    int input_dims[] = {112,112,64}; 
    int input_size = input_dims[0] * input_dims[1] * input_dims[2];

    // std::ifstream conf_file("/home/ubuntu/db_conv1_samples.txt");
    // FlatVectorPtr<float> conf = get_tensor(conf_file, 6, 6);
    // float* confs = conf->values()->asMutable<float>();
    // conf_file.close();
    

    int num_samples = std::atoi(std::getenv("samples"));
    int num_splits = std::atoi(std::getenv("splits"));
    int velox_threads = std::atoi(std::getenv("vthreads"));
    int torch_threads = std::atoi(std::getenv("tthreads"));
    torch::set_num_threads(torch_threads);

    int dims[] = {cnn_filters, cnn_filter_dims[0], cnn_filter_dims[1], cnn_filter_dims[2], input_dims[0], input_dims[1]};

    std::ifstream weights_file("/home/ubuntu/db_conv1_weights.txt"); 
    std::ifstream bias_file("/home/ubuntu/db_conv1_bias.txt"); 
    std::ifstream test_file("/home/ubuntu/db_conv1_input.txt"); 

    FlatVectorPtr<float> weights = get_tensor(weights_file, weights_size, cnn_filters * cnn_filter_dims[2]);
    FlatVectorPtr<float> bias = get_tensor(bias_file, cnn_filters, 1);

    weights_file.close();
    bias_file.close();

    float* bias_values = bias->values()->asMutable<float>();
   
    FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples * input_dims[2]);
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
    
    // std::string compute =  NNBuilder()
    //                       .convLayer(cnn_filters, dims, weights->values()->asMutable<float>(), 
    //                         bias->values()->asMutable<float>(), NNBuilder::NONE)
    //                       .build();

    // std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId p0;
  
    //std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))

    exec::registerVectorFunction(
    "torchConvolute",
    TorchDNN2Level::signatures(),
    std::make_unique<TorchConvolute>(weights->values()->asMutable<float>(), dims)
    );

    auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  //.project({fmt::format(compute, "x")}) 
                  .project({"torchConvolute(x)"}) 
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


void MLFunctionsTest::test_land_cover_conv3() {
    int cnn_filters = 2048;
    int cnn_filter_dims[] = {1,1,3}; // height * width * channels
    int weights_size = cnn_filter_dims[0] * cnn_filter_dims[1] * cnn_filter_dims[2] * cnn_filters;

    int input_dims[] = {2500,2500,3}; 
    int input_size = input_dims[0] * input_dims[1] * input_dims[2];

    std::ifstream conf_file("/home/ubuntu/db_conv1_samples.txt");
    FlatVectorPtr<float> conf = get_tensor(conf_file, 6, 6);
    float* confs = conf->values()->asMutable<float>();
    conf_file.close();
    torch::set_num_threads(confs[5]);

    int num_samples = (int) confs[0];
    int num_splits = (int) confs[1];
    std::cout << num_samples << " " << num_splits;

    int dims[] = {cnn_filters, cnn_filter_dims[0], cnn_filter_dims[1], cnn_filter_dims[2], input_dims[0], input_dims[1]};

    std::ifstream weights_file("/home/ubuntu/lc_conv3_weights.txt"); 
    std::ifstream bias_file("/home/ubuntu/lc_conv3_bias.txt"); 
    std::ifstream test_file("/home/ubuntu/lc_conv3_input.txt"); 

    FlatVectorPtr<float> weights = get_tensor(weights_file, weights_size, cnn_filters * cnn_filter_dims[2]);
    FlatVectorPtr<float> bias = get_tensor(bias_file, cnn_filters, 1);

    weights_file.close();
    bias_file.close();

    float* bias_values = bias->values()->asMutable<float>();
   
    FlatVectorPtr<float> input = get_tensor(test_file, input_size * num_samples, num_samples * input_dims[2]);
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
    
    // std::string compute =  NNBuilder()
    //                       .convLayer(cnn_filters, dims, weights->values()->asMutable<float>(), 
    //                         bias->values()->asMutable<float>(), NNBuilder::NONE)
    //                       .build();

    // std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId p0;
  
    //std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))

    exec::registerVectorFunction(
    "torchConvolute",
    TorchDNN2Level::signatures(),
    std::make_unique<TorchConvolute>(weights->values()->asMutable<float>(), dims)
    );

    auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  //.project({fmt::format(compute, "x")}) 
                  .project({"torchConvolute(x)"}) 
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
  task->start(confs[2]);
  task->noMoreSplits(p0);
  // Start task with 2 as maximum drivers and wait for execution to finish
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

void MLFunctionsTest::test_complex_torchnn() {
  std::cout << "Test of Complex TorchNN" << std::endl;
  int batchSize = 500;
  int featureSize = 1024;
  int layer1Size = 1024;
  int layer2Size = 3;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  VectorMaker maker{pool_.get()};
  std::vector<std::vector<float>> inputValue =
      randomGenerator.genFloat2dVector(batchSize, featureSize);
  auto inputValueVector = maker.arrayVector<float>(inputValue, REAL());

  float* w1Weight = randomGenerator.genFloat1dArray(featureSize*layer1Size);
  float* w1Bias = randomGenerator.genFloat1dArray(layer1Size);
  float* w2Weight = randomGenerator.genFloat1dArray(layer1Size*layer2Size);
  float* w2Bias = randomGenerator.genFloat1dArray(layer2Size);

  std::vector<velox::dl::KernelType> kernelTypes = 
      {velox::dl::KernelType::MatMul, 
      velox::dl::KernelType::MatAdd, 
      velox::dl::KernelType::ReLU, 
      velox::dl::KernelType::MatMul, 
      velox::dl::KernelType::MatAdd, 
      velox::dl::KernelType::Softmax};

  std::vector<float*> weights = {w1Weight, w1Bias, w2Weight, w2Bias};
  std::vector<int> dims = {featureSize, layer1Size, layer1Size, layer1Size,
  layer2Size, layer2Size, layer2Size};

  exec::registerVectorFunction(
    "complex_torchNN",
    TorchDNNV2::signatures(),
    std::make_unique<TorchDNNV2>(kernelTypes, weights, dims)
    );

  auto inputRowVector = maker.rowVector({"x"}, {inputValueVector});

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"complex_torchNN(x)"});
  auto results = exec::test::AssertQueryBuilder(myPlan.planNode()).copyResults(pool_.get());

  std::cout << "Results: \n" << results->toString(0, results->size()) << std::endl;

}

void MLFunctionsTest::run(int numDriver, int memoryPoolSizeMB, int spillMemThresholdMB, bool enableSpill, int repeatRun) {
  //  test_mat_mul();
  //  test_mat_add();
  //  test_relu();
  //  test_softmax()
  //  test_dense_layer();
  //  test_torch_dense_layer_multithreading();
  //  test_mnist();
  //  test_multithreading();
  //  test_multithreading_oom();
  //  test_batching();
  //  test_conv2d();
  //  test_deep_bench_conv1();
  //  test_land_cover_conv3();
  //  test_spill(numDriver, memoryPoolSizeMB, spillMemThresholdMB, enableSpill, repeatRun);
  //  mytest();
  //  test_mnist_multithreading();
  //  test_mnist_oom_weights();
  // test_torch_dense_layer();
  test_complex_torchnn();

}

DEFINE_bool(spill, false, "Whether enable spilling");
DEFINE_int32(spill_threshold, 30, "Set spill memory threshold");
DEFINE_int32(memory_pool, 100, "Set memory pool size");
DEFINE_int32(repeat, 5, "Number of repeat run");
DEFINE_int32(num_driver, 1, "Number of driver");
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  bool enableSpill = FLAGS_spill;
  int memoryPoolSizeMB = FLAGS_memory_pool;
  int spillMemoryThresholdMB = FLAGS_spill_threshold;
  int repeatRun = FLAGS_repeat;
  int numDriver = FLAGS_num_driver;
  MLFunctionsTest demo;
  demo.run(numDriver, memoryPoolSizeMB, spillMemoryThresholdMB, enableSpill, repeatRun);
}
