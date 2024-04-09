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
#include "velox/cost_model/CostEstimator.h"



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
  void ffnn(int input_size, int layer1_size, int layer2_size);
  void torch_ffnn(int input_size, int layer1_size, int layer2_size);
  void traverse(std::shared_ptr<const core::PlanNode>& node);

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

    config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, kSizeKB);
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

  struct DataFrame {
    std::vector<std::vector<float>> features;
    std::vector<float*> weights;
    std::vector<float*> bias;
    float* featuresFloat;
  };
  
  DataFrame data_generate(int features, int samples, int first_layer, int second_layer){
    int input_features_size = features;
    int num_samples = samples;

    int first_layer_output_size = first_layer;
    int second_layer_output_size = second_layer;
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer
    // ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer
    int input_total_size = input_features_size * num_samples;

    int weight_layer1_size = input_features_size * first_layer_output_size;
    int weight_layer2_size = first_layer_output_size * second_layer_output_size;

    int bias_layer1_size = num_samples * first_layer_output_size;
    int bias_layer2_size = num_samples * second_layer_output_size;

    std::random_device rd;  // Seed the random number generator
    std::mt19937 gen(rd()); // Initialize the Mersenne Twister engine
    // std::uniform_real_distribution<float> distribution(0.00000009, 0.00000011); // Define the range
    std::uniform_real_distribution<float> distribution(0.0009, 0.0011);

    //generate input
    std::vector<std::vector<float>> featureVectors;
    for (int i = 0; i < num_samples; i++) {
          std::vector<float> featureVector;
          for (int j = 0; j < input_features_size; j++) {
                  featureVector.push_back(i*input_features_size+j);
                  // float random_value = distribution(gen);
                  // featureVector.push_back(0.0000001);
                  // featureVector.push_back(random_value);
          }
          featureVectors.push_back(featureVector);
      }

    float* floatArray = new float[num_samples * input_features_size];
    int index = 0;

    for (const auto& row : featureVectors) {
        for (const float& value : row) {
            floatArray[index++] = value;
        }
    }

    //generate weight
    float* weight_layer1 = new float[weight_layer1_size];
    for (int i = 0; i < weight_layer1_size; ++i) {
        weight_layer1[i] = 0.000001; 
    }
    float* weight_layer2 = new float[weight_layer2_size];
    for (int i = 0; i < weight_layer2_size; ++i) {
        weight_layer2[i] = 0.000001; 
    }
    std::vector<float*> weights;
    weights.push_back(weight_layer1);
    weights.push_back(weight_layer2);

    //generate bias
    float* bias_layer1 = new float[bias_layer1_size];
    for (int i = 0; i < bias_layer1_size; ++i) {
        bias_layer1[i] = 0.00001; 
    }
    float* bias_layer2 = new float[bias_layer2_size];
    for (int i = 0; i < bias_layer2_size; ++i) {
        bias_layer2[i] = 0.00001; 
    }
    std::vector<float*> bias;
    bias.push_back(bias_layer1);
    bias.push_back(bias_layer2);
    // create dataframe
    DataFrame data;
    data.features = featureVectors;
    data.weights = weights;
    data.bias = bias;
    data.featuresFloat = floatArray;

    return data;
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
    std::shared_ptr<MemoryPool> pool = memory::MemoryManager::getInstance()->addRootPool(
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
  
  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};
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

void MLFunctionsTest::torch_ffnn(int input_size, int layer1_size, int layer2_size){
  
  std::vector<float> confs = get_config(6);
  
  int num_samples = (int) confs[0];
  int num_splits = (int) confs[1];
  int concurrency = (int) confs[2];
  int torch_threads = (int) confs[5];

  std::vector<int> dimensions;
  dimensions.push_back(input_size);
  dimensions.push_back(layer1_size);
  dimensions.push_back(layer2_size);

  auto data = data_generate(input_size, num_samples, layer1_size, layer2_size);

  auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
 
  float* weights[2] = {data.weights[0], data.weights[1]};
  float* bias[2] = {data.bias[0], data.bias[1]};

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
  execute_plan(plan, p0, inputRowVector, confs);
}

// for 2 layers only. Can be generalized for multiple layers
void MLFunctionsTest::ffnn(int input_size, int layer1_size, int layer2_size) {
    
    std::vector<float> confs = get_config(5);
    
    int num_samples = (int) confs[0];

    auto data = data_generate(input_size, num_samples, layer1_size, layer2_size);

    float* bias_1_values = data.bias[0];
    float* bias_2_values = data.bias[1];

    FlatVectorPtr<float> bias_1_mat = maker.flatVector<float>(num_samples * layer1_size);
    for(int i=0; i < bias_1_mat->size(); i++)
      bias_1_mat->set(i, bias_1_values[i%layer1_size]);
    
    FlatVectorPtr<float> bias_2_mat = maker.flatVector<float>(num_samples * layer2_size);
    for(int i=0; i < bias_2_mat->size(); i++)
      bias_2_mat->set(i, bias_2_values[i%layer2_size]);

  
    auto featureArrayVector = maker.arrayVector<float>(data.features , REAL());
    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

    std::string compute =  NNBuilder()
                          .denseLayer(layer1_size ,input_size, data.weights[0], 
                            bias_1_mat->values()->asMutable<float>(), NNBuilder::RELU)
                          .denseLayer(layer2_size ,layer1_size, data.weights[1], 
                            bias_2_mat->values()->asMutable<float>(), NNBuilder::SOFTMAX)
                          .build();

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId p0;
  
    //std::cout << compute << std::endl; 
    auto plan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  .project({fmt::format(compute, "x")}) 
		              .planNode();

    CataLog catalog = CataLog("test-catalog");
    // std::shared_ptr<CataLog> catalog = std::make_shared<CataLog>(CataLog("test-catalog"));
    std::shared_ptr<OutputStat> stat = std::make_shared<OutputStat>(OutputStat(1,2));
    Source src1 = Source(p0, Source::Type::FILE, std::move(stat));
    catalog.addSource(std::make_shared<Source>(src1));
    CostModel* cm = new SimpleCostModel(catalog);
    CostEstimator* ce = new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));
    CostEstimate cost = ce->estimateCost(plan);
    std::cout << cost.cost << std::endl;

    //execute_plan(plan, p0, inputRowVector, confs);
}

void MLFunctionsTest::traverse(std::shared_ptr<const core::PlanNode>& node) {
  if(!node)
    return;
  
  std::vector<std::string> vec;
  for (auto source : node->sources()) {
        // store node stat in vector
        // returned by this call
        std::string temp(source->name());
        vec.push_back(temp + ":" + source->id() + " ");
        traverse(source);
  }
  std::cout << "Sources for " << node->name() << " - " << node->id() << std::endl;
  for(std::string v : vec){
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

void MLFunctionsTest::run() {
    ffnn(784,1024,10);
    return;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId nationScanId;
  core::PlanNodeId regionScanId;
  auto plan = PlanBuilder(planNodeIdGenerator)
             .tpchTableScan(
                 tpch::Table::TBL_NATION, {"n_regionkey"}, 1 /*scaleFactor*/)
             .capturePlanNodeId(nationScanId)
             .hashJoin(
                 {"n_regionkey"},
                 {"r_regionkey"},
                 PlanBuilder(planNodeIdGenerator)
                     .tpchTableScan(
                         tpch::Table::TBL_REGION,
                         {"r_regionkey", "r_name"},
                         1 /*scaleFactor*/)
                     .capturePlanNodeId(regionScanId)
                     .planNode(),
                 "", // extra filter
                 {"r_name"})
             .singleAggregation({"r_name"}, {"count(1) as nation_cnt"})
             .orderBy({"r_name"}, false)
             .planNode();

    CataLog catalog = CataLog("test-catalog");
    std::shared_ptr<OutputStat> stat = std::make_shared<OutputStat>(OutputStat(1,2));
    std::shared_ptr<OutputStat> stat2 = std::make_shared<OutputStat>(OutputStat(1,2));
    
    Source src1 = Source(nationScanId, Source::Type::FILE, std::move(stat));
    Source src2 = Source(regionScanId, Source::Type::FILE, std::move(stat2));
   

    catalog.addSource(std::make_shared<Source>(src1));
    catalog.addSource(std::make_shared<Source>(src2));


    CostModel* cm = new SimpleCostModel(catalog);
    CostEstimator* ce = new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

    CostEstimate cost = ce->estimateCost(plan);
    std::cout << cost.cost << std::endl;
    //traverse(plan);
    // std::cout << "Here is the source" << std::endl;
    // std::cout << plan->name();
    // for (auto source : plan->sources()) {
    //     std::cout << source->name() << std::endl;
      
    // }     



  // Large
  //ffnn(597540,1024,14588);
  // small
  // ffnn(784,1024,10);

  //large
  //torch_ffnn(597540,1024,14588);
  // small
  // torch_ffnn(784,1024,10);
}

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  MLFunctionsTest demo;
  demo.run();
}