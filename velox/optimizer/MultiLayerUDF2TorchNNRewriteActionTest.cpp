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
#include "RewriteAction.h"
#include "MultiLayerUDF2TorchNNRewriteAction.h"
#include "RuleManager.h"
#include "PlanState.h"
#include "CataLog.h"
#include "Register.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class MultiLayerUDF2TorchNNRewriteActionTest : public HiveConnectorTestBase {
 public:
 MultiLayerUDF2TorchNNRewriteActionTest() {
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

  ~MultiLayerUDF2TorchNNRewriteActionTest() {
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
      int numSplits,
      PlanBuilder& myPlan,
      CataLog &cataLog) {

    // Initializes executor.
    std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};
    // Initializes queryCtx.
    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};
    // Set queryCtx config.
    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},// 100000000000000000
          {core::QueryConfig::kMaxOutputBatchRows, "10000"}});

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

    ss << numSplits << "," << numThreads << ",";

    int dataIdx = 0;
    for (auto batchedData : actualResults) {
      std::cout << fmt::format("[INFO] Batched Data: {} \n", dataIdx) << batchedData->toString() << std::endl;
      dataIdx += 1;
    }

    std::cout << "Time for FFNN with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;
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
 * @param second_layer The output size of the second layer in the network.
 * @param third_layer The output size of the third layer in the network.
 * 
 * @return DataFrame The structure used to denote the generated data.
*/
  DataFrame data_generate_multi(
      int features, 
      int samples, 
      int first_layer, 
      int second_layer,
      int third_layer){
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights + bias.
    // ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x weights + bias.
    int input_features_size = features;
    int num_samples = samples;

    int first_layer_output_size = first_layer;
    int second_layer_output_size = second_layer;
    int third_layer_output_size = third_layer;

    int input_total_size = input_features_size * num_samples;

    int weight_layer1_size = input_features_size * first_layer_output_size;
    int weight_layer2_size = first_layer_output_size * second_layer_output_size;
    int weight_layer3_size = second_layer_output_size * third_layer_output_size;

    int bias_layer1_size = first_layer_output_size;
    int bias_layer2_size = second_layer_output_size;
    int bias_layer3_size = third_layer_output_size;
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

                  featureVector.push_back(i*input_features_size+j);

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

        weight_layer1[i] = 0.000001; 

    }
    float* weight_layer2 = new float[weight_layer2_size];

    for (int i = 0; i < weight_layer2_size; ++i) {

        weight_layer2[i] = 0.000001; 

    }

    float* weight_layer3 = new float[weight_layer3_size];

    for (int i = 0; i < weight_layer3_size; ++i) {

        weight_layer3[i] = 0.000001; 

    }

    std::vector<float*> weights;
    weights.push_back(weight_layer1);
    weights.push_back(weight_layer2);
    weights.push_back(weight_layer3);

    //Generate bias
    float* bias_layer1 = new float[bias_layer1_size];

    for (int i = 0; i < bias_layer1_size; ++i) {

        bias_layer1[i] = 0.00001; 

    }
    float* bias_layer2 = new float[bias_layer2_size];

    for (int i = 0; i < bias_layer2_size; ++i) {

        bias_layer2[i] = 0.00001; 

    }

    float* bias_layer3 = new float[bias_layer3_size];

    for (int i = 0; i < bias_layer3_size; ++i) {

        bias_layer3[i] = 0.00001; 

    }

    std::vector<float*> bias;
    bias.push_back(bias_layer1);
    bias.push_back(bias_layer2);
    bias.push_back(bias_layer3);

    // Create DataFrame
    DataFrame data;
    data.features = featureVectors;
    data.weights = weights;
    data.bias = bias;
    data.featuresFloat = floatArray;

    return data;
  }

  /**
   * @brief A test function to test the rewrite rule of TwoLayerUDF2TorchNN.
   * 
   * @param rewrite A boolean value indicating whether to perform a rewrite.
  */
  void testMultiLayerUDF2TorchNNPlan(bool rewrite) {
    // Set data source config.
    int layer_number = 3;
    int input_features_size = 800;//597540
    int num_samples = 1000;
    int first_layer_output_size = 1000;
    int second_layer_output_size = 1024;
    int third_layer_output_size = 14588;
    // Set splits number
    int num_splits = 4;
    CataLog cataLog;
    // Generate data source
    auto data = data_generate_multi(input_features_size, num_samples, first_layer_output_size, second_layer_output_size, third_layer_output_size);
    // Create arrayVector for data source
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    // Create file path
    auto file = TempFilePath::create();
    // Create file config
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    // Write the data source to a file, with the format defined by the rowVector
    writeToFile(file->path, {inputRowVector}, config);
    // Build third dense layers UDFs
    std::string compute =  NNBuilder()
                        .denseLayer(first_layer_output_size, input_features_size, data.weights[0], data.bias[0], NNBuilder::RELU)
                        .denseLayer(second_layer_output_size, first_layer_output_size, data.weights[1], data.bias[1], NNBuilder::RELU)
                        .denseLayer(third_layer_output_size, second_layer_output_size, data.weights[2], data.bias[2], NNBuilder::SOFTMAX)
                        .build();


    // Initialize planNodeID
    core::PlanNodeId p0;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")}) 
                .planBuild();
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // catalog
    cataLog.setIdAddressMap(p0, {file});
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);
    // Run rewriten rule
    if (rewrite) {
      // Get possible actions for this plan
      planState.getPossibleActions(planNode, cataLog);
      // Print possible actions
      for (const auto& entry : planState.actionsPair) {
        std::cout << entry.first << ": " << entry.second << std::endl;
      }
      // Choose one action from possible actions (Now we only pick the first one, later it would be choosen by MCTS)
      auto it = planState.actionsPair.begin();
      std::pair<std::string, std::string> testAction = std::make_pair("softmax8(mat_add7(mat_mul6(relu5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0(ROW[\"v\"])))))))))", "MultiLayerUDF2TorchNNRewriteAction");
      // Take one rewritten action
      planState.takeAction(planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, {testAction}, cataLog);
      // Update the planState (getPossibleAction after apply one action)
      planState.update(myPlan, cataLog);

    }
    std::cout << "Query Plan: \n" << myPlan.planNode()->toString(true, true) << std::endl;
    // Run the rewritten plan
    runPlan(8, 8, myPlan, cataLog);
  }

 private:
  std::shared_ptr<MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};


  VectorMaker maker{pool_.get()};
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  MultiLayerUDF2TorchNNRewriteActionTest demo;

  bool rewrite = true;

  if (argc > 1) {
    if (strcmp(argv[1], "N") == 0) {
      rewrite = false;
    }
  }

  if (rewrite) {
    std::cout
        << "================= Run UDF-Centric FFNN model w/ Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testMultiLayerUDF2TorchNNPlan(true);

  } else {
    std::cout
        << "================= Run UDF-Centric FFNN model w/o Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testMultiLayerUDF2TorchNNPlan(false);
  }

  std::cout
      << "--" << std::endl
      << "[Usage] " << std::endl
      << "./_build/release/velox/optimizer/torch2twolayer_test Y  //run FFNN model with rewriting rule 2"
      << std::endl
      << "./_build/release/velox/optimizer/torch2twolayer_test N  //run FFNN model with rewriting rule 2"
      << std::endl
      << "By default: Y is used" << std::endl;
}
