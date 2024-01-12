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
// #include <boost/interprocess/sync/interprocess_semaphore.hpp>
// #include <fcntl.h>
// #include <folly/init/Init.h>
// #include <stdlib.h>
// #include <torch/torch.h>
// #include <unistd.h>
// #include <cmath>
// #include <cstdlib>
// #include <cstring>
// #include <iostream>
// #include <memory>
// #include <random>
// #include <string>

// #include <iostream>
// #include <folly/init/Init.h>
// #include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
// #include "velox/functions/prestosql/registration/RegistrationFunctions.h"
// #include "velox/functions/Macros.h"
// #include "velox/functions/Registerer.h"
// #include "velox/parse/Expressions.h"
// #include "velox/parse/ExpressionsParser.h"
// #include "velox/parse/TypeResolver.h"
// #include "velox/type/Type.h"
// #include "velox/expression/VectorFunction.h"
// #include "velox/exec/tests/utils/PlanBuilder.h"
// #include "velox/vector/tests/utils/VectorMaker.h"
// #include "velox/exec/tests/utils/AssertQueryBuilder.h"
// #include "velox/exec/tests/utils/HiveConnectorTestBase.h"
// #include <boost/interprocess/sync/interprocess_semaphore.hpp>
// #include "velox/exec/tests/utils/TempDirectoryPath.h"
// #include "velox/common/memory/MemoryArbitrator.h"
// #include "velox/vector/fuzzer/VectorFuzzer.h"

// #include "velox/ml_functions/NNBuilder.h"
// #include <fstream>
// #include <sstream>
// #include <random>

// #include "velox/exec/FilterProject.h"
// #include "velox/common/file/FileSystems.h"
// #include "velox/dwio/dwrf/reader/DwrfReader.h"
// #include "RewriteAction.h"
// #include "TorchNN2TwoLayerUDFRewriteAction.h"
// #include "RuleManager.h"
// #include "PlanState.h"

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
#include "TorchNN2TwoLayerUDFRewriteAction.h"
#include "RuleManager.h"
#include "PlanState.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class TorchNN2TwoLayerUDFRewriteActionTest : public HiveConnectorTestBase {
 public:
  TorchNN2TwoLayerUDFRewriteActionTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
  }

  ~TorchNN2TwoLayerUDFRewriteActionTest() {
    TearDown();
  }

  void SetUp() override {}

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  void runPlan(
      std::string filePath,
      int numThreads,
      int numSplits,
      PlanBuilder& myPlan,
      core::PlanNodeId p0) {
    auto hiveSplits = makeHiveConnectorSplits(
        filePath, numSplits, dwio::common::FileFormat::DWRF);

    std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};

    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
         {core::QueryConfig::kMaxOutputBatchRows, "10000"}});

    auto task = exec::Task::create(
        "0",
        myPlan.planFragment(),
        0,
        queryCtx_,
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if (result)
            std::cout << result->toString() << std::endl;
          return exec::BlockingReason::kNotBlocked;
        });

    std::cout << "Hive splits:" << std::endl;

    for (auto& split : hiveSplits) {
      // std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
    }
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();



    task->start(numThreads);

    task->noMoreSplits(p0);

    // Start task with 2 as maximum drivers and wait for execution to finish

    waitForFinishedDrivers(task);

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;

    ss << numSplits << "," << numThreads << ",";

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


  void testTorchNN2TwoLayerUDFPlan(bool rewrite) {

    int input_features_size = 800;//597540
    int num_samples = 1000;
    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;
    int num_splits = 4;

    auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);

    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});

    auto file = TempFilePath::create();
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    writeToFile(file->path, {inputRowVector}, config);

    std::vector<int> dimensions;
    dimensions.push_back(input_features_size);
    dimensions.push_back(first_layer_output_size);
    dimensions.push_back(second_layer_output_size);

    float* weights[2] = {data.weights[0], data.weights[1]};
    float* bias[2] = {data.bias[0], data.bias[1]};

    exec::registerVectorFunction(
      "torchDNN0",
      TorchDNN::signatures(),
      std::make_unique<TorchDNN>(weights, bias, dimensions)
    );
    // create a plan for decision forest using UDF-centric style
    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(asRowType(inputRowVector->type()))
                .capturePlanNodeId(p0)
                .project({"torchDNN0(v)"})
                .planBuild();

    auto planNode = myPlan.planNode();

    RuleManager ruleManager;
    PlanState planState(ruleManager);

    if (rewrite) {
      // Create a rewrite action for this plan
      planState.getPossibleActions(planNode);
      // Print possible actions
      for (const auto& entry : planState.actionsPair) {
        std::cout << entry.first << ": " << entry.second << std::endl;
      }
      auto it = planState.actionsPair.begin();
      std::string testAction  = it->first;

      planState.takeAction(planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, {testAction});
      planState.update(myPlan);
      // Apply the rewrite action
    }

    // Run the rewritten plan
    runPlan(file->path, 8, 8, myPlan, p0);
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::addDefaultLeafMemoryPool();

  VectorMaker maker{pool_.get()};
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);

  TorchNN2TwoLayerUDFRewriteActionTest demo;

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

    demo.testTorchNN2TwoLayerUDFPlan(true);

  } else {
    std::cout
        << "================= Run UDF-Centric FFNN model w/o Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testTorchNN2TwoLayerUDFPlan(false);
  }

  std::cout
      << "--" << std::endl
      << "[Usage] " << std::endl
      << "./_build/release/velox/optimizer/torch2twolayer_test Y  //run FFNN model with rewriting rule 1"
      << std::endl
      << "./_build/release/velox/optimizer/torch2twolayer_test N  //run FFNN model with rewriting rule 1"
      << std::endl
      << "By default: Y is used" << std::endl;
}
