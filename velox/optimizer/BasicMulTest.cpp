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

#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
// #include <cstdarg>
// Velox headers
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/expression/VectorFunction.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/exec/FilterProject.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"

// Custom headers
// #include "RewriteAction.h"
// #include "Mul2JoinAggRewriteAction.h"
#include "RuleManager.h"
#include "PlanState.h"
#include "Register.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class Mul2JoinAggRewriteActionTest : public HiveConnectorTestBase {
 public:
 Mul2JoinAggRewriteActionTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // Register hiveconnector for file splits.
    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);
  }

  ~Mul2JoinAggRewriteActionTest() {
    TearDown();
  }

  void SetUp() override {}

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}
  // Wait for all drivers to finish work.
  void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  /**
   * @brief A function to run logical plan.
   * 
   * @param numThreads The number of Velox executor threads.
   * @param numSplits The number of file splits.
   * @param myPlan The pointer to the planBuilder which builds the logical plan.
   * @param cataLog A class storing metadata and information related to UDFs and data sources.
  */
  void runPlan(
      int numThreads,
      int numKernelThreads,
      PlanBuilder& myPlan,
      CataLog &cataLog,
      std::string name,
      int sampleSize,
      int featureSize,
      int outputSize) {

    // Initializes executor.
    std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};
    // Initializes queryCtx.
    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};
    // Set queryCtx config.
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},// 100000000000000000  1000000
          {core::QueryConfig::kMaxOutputBatchRows, "1000"}});

    // Add hivesplits to the target plan node (data source node).
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();


    CursorParameters params;
    params.maxDrivers = numThreads;
    params.planNode = myPlan.planNode();
    params.queryCtx = queryCtx_;
    bool noMoreSplits = false;
    auto addSplits = [&noMoreSplits, &cataLog](exec::Task* task) {
    auto idFileAddrMap = cataLog.getIdAddressMap();
    std::vector<core::PlanNodeId> ids;
      if (!noMoreSplits) {
    for (const auto& entry : idFileAddrMap) {

      core::PlanNodeId key = entry.first;

      const std::vector<std::shared_ptr<TempFilePath>> fileAddr = entry.second;

      auto hiveSplits = makeHiveConnectorSplits(fileAddr);

      for (auto& split : hiveSplits) {

        task->addSplit(key, exec::Split(std::move(split)));

      }

      ids.push_back(key);
    }

    for (auto id: ids){
      task->noMoreSplits(id);
    }
      }
      noMoreSplits = true;
    };

    auto [cursor, actualResults] = readCursor(params, addSplits);
    waitForTaskCompletion(cursor->task().get());



    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;

    ss <<"number of velox driver:" 
        << numThreads << "," << "number of function threads:" << numKernelThreads << "," 
        << "sample size:" << sampleSize << "," << "feature size:" << featureSize << ","
        << "output size:" << outputSize << ",";
    


    int dataIdx = 0;
    int totalDataNum = 0;
    for (auto batchedData : actualResults) {
      batchedData = std::move(batchedData);
      int batchSize = batchedData->size();
      std::cout << fmt::format("[INFO] Batched Data: {}, Batch Size:{} \n", dataIdx, batchSize) << batchedData->toString() << std::endl;
      dataIdx += 1;
      totalDataNum += batchSize;
    }
    std::cout << fmt::format("[INFO] Total # of Batch: {}, Total # of Data: {}", dataIdx, totalDataNum) << std::endl;

    std::cout << "Time for FFNN with Input Data (sec): "
              << std::endl;

     ss << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs," <<"\n";

    std::ofstream outfile(name + "_output.txt", std::ios::app); // Open file in append mode
    if (outfile.is_open()) {
        outfile << ss.str(); // Write string to file
        outfile.close();    // Close file
    } else {
        std::cerr << "Error opening output file." << std::endl;
    }

  }

  struct DataFrame {
      std::vector<std::vector<float>> features;
      std::vector<float*> weights;
      std::vector<float*> bias;
      float* featuresFloat;
  };

/**
 * @brief A function generates random data source.
 * 
 * @param features The number of features (column count) in the data source.
 * @param samples The number of samples (row count) in the data source.
 * @param first_layer The output size of the first layer in the network.
 * 
 * @return DataFrame The structure used to denote the generated data.
*/
  DataFrame data_generate(
      int features, 
      int samples, 
      int first_layer){
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights + bias.
    // ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x weights + bias.
    int input_features_size = features;
    int num_samples = samples;

    int first_layer_output_size = first_layer;


    int input_total_size = input_features_size * num_samples;

    int weight_layer1_size = input_features_size * first_layer_output_size;

    // Seed the random number generator
    std::random_device rd;  
    // Initialize the Mersenne Twister engine
    std::mt19937 gen(rd());
    // Define the range
    std::uniform_real_distribution<float> distribution(0.0009, 0.0011);

    //Generate input
    std::vector<std::vector<float>> featureVectors;

    for (int i = 0; i < num_samples; i++) {

          std::vector<float> featureVector;

          for (int j = 0; j < input_features_size; j++) {

                  // featureVector.push_back(i*input_features_size+j);
                  featureVector.push_back(distribution(gen));

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

    //Generate weight
    float* weight_layer1 = new float[weight_layer1_size];

    for (int i = 0; i < weight_layer1_size; ++i) {

        weight_layer1[i] = 0.0001; 
        // weight_layer1[i] = i;

    }

    std::vector<float*> weights;
    weights.push_back(weight_layer1);


    // Create DataFrame
    DataFrame data;
    data.features = featureVectors;
    data.weights = weights;
    data.featuresFloat = floatArray;

    return data;
  }
  
  /**
   * @brief Registers a series of vector functions in the optimization namespace.
   * 
   * @param units1 Number of units in the first layer.
   * @param units2 Number of units in the second layer.
   * @param input_size1 Size of the input for the first layer.
   * @param input_size2 Size of the input for the second layer.
   * @param weightsFile_1 Pointer to the weights for the first layer.
   * @param weightsFile_2 Pointer to the weights for the second layer.
   * @param biasFile_1 Pointer to the bias for the first layer.
   * @param biasFile_2 Pointer to the bias for the second layer.
   * @param catalog Reference to a CataLog object to store metadata and information.
   * 
   * @return A string representing the composed vector function expression.
   */
  std::string registerFunctions(int threads, int units1, int input_size1, float* weightsFile_1, CataLog &catalog, bool isVerticalPartition) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "mat_mul0",
        MatrixMultiply_bench::signatures(),
        std::make_unique<MatrixMultiply_bench>(weightsFile_1, input_size1, units1, threads),
        {},
        true,
        catalog,
        isVerticalPartition
    );
  //   exec::registerVectorFunction(
  //   "mat_mul0",
  //   MatrixMultiply_bench::signatures(),
  //   std::make_unique<MatrixMultiply_bench>(weightsFile_1, input_size1, units1, threads)
  // );
      // Compose and return the vector function expression
     return "mat_mul0({})";

  }

  /**
   * @brief A test function to test the rewrite rule of Mul2JoinAggRewriteAction.
   * 
   * @param rewrite A boolean value indicating whether to perform a rewrite.
  */
  void testMul2JoinAggPlan(int samplesSize, int featuresSize, int outputSize, int veloxThreads, int kernelThreads, std::string name) {
    // Set data source config.
    int input_features_size = featuresSize;//597540
    int num_samples = samplesSize;
    // int input_features_size = featureSize;
    // int num_samples = sampleSize;
    int first_layer_output_size = outputSize;
    // int second_layer_output_size = 14588;
    // Initialize CataLog
    int velox_threads = veloxThreads;
    int core_function_threads = kernelThreads;
    CataLog cataLog;
    
    // Generate data source
    auto data = data_generate(input_features_size, num_samples, first_layer_output_size);
    // Create arrayVector for data source
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    // auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    
    std::vector<float> indices(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        indices[i] = static_cast<float>(i);
    }

    auto inputRowVector = maker.rowVector(
          {"v", "v_row"},
          {featureArrayVector,
           maker.flatVector(indices)});
    // Create file path
    auto file = TempFilePath::create();
    // Create file config
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    // Write the data source to a file, with the format defined by the rowVector
    writeToFile(file->path, {inputRowVector}, config);
    // In horizontal partition approach, it relies on Velox's batches to partition the data
    cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
    cataLog.setDataSourceStat({num_samples, input_features_size});
    cataLog.setUDFSchema("value", asRowType(inputRowVector->type()));
    // }
    // Build two dense layers UDFs using registerFunction in optimization namespace

        //basic plan
    bool isVerticalPartition = false;
    std::string compute = registerFunctions(
    core_function_threads,
    first_layer_output_size, 
    input_features_size, 
    data.weights[0], 
    cataLog,
    isVerticalPartition);

        // Initialize planNodeID
    core::PlanNodeId p0;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")});
    // Set original plan nodeId and file address of data source
    cataLog.setIdAddressMap(p0, {file});
    // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    // RuleManager ruleManager;
    // Create planState
    // PlanState planState(ruleManager);

    std::cout << "[INFO] Query Plan: \n" << myPlan.planNode()->toString(true,true) << std::endl;
    // Run the rewritten plan
    runPlan(velox_threads, core_function_threads, myPlan, cataLog, name, samplesSize, featuresSize, outputSize);

  }

 private:
  std::shared_ptr<MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  int sampleSize = std::atoi(argv[1]);//sampleSize
  int featuresSize = std::atoi(argv[2]);//featuresSize 
  int outputSize = std::atoi(argv[3]);//outputSize
  int velox_driver = std::atoi(argv[4]);//velox_driver
  int function_threads = std::atoi(argv[5]);//function_threads
  std::string name = argv[6];//type of implementation

  Mul2JoinAggRewriteActionTest demo;

  demo.testMul2JoinAggPlan(sampleSize, featuresSize, outputSize, velox_driver, function_threads, name);


}