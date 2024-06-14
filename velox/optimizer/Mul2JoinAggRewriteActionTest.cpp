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
#include "Mul2JoinAggRewriteAction.h"
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
        {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
          {core::QueryConfig::kMaxOutputBatchRows, "1000"}});

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    // Create task for logical plan.
    // auto task = exec::Task::create(
    //     "0",
    //     myPlan.planFragment(),
    //     0,
    //     queryCtx_,
    //     [](RowVectorPtr result, ContinueFuture* /*unused*/) {
    //       if (result) {
    //         std::cout << "=============================\n";
    //         std::cout << result->toString() << " size: " << result->size() << std::endl;
    //         std::cout << result->toString(0, result->size()) << std::endl;
    //       }
    //       return exec::BlockingReason::kNotBlocked;
    //     });
    // // Get optimized idFileAddr map from cataLog
    // auto idFileAddrMap = cataLog.getIdAddressMap();

    // std::vector<core::PlanNodeId> ids;

    // std::cout << "Hive splits:" << std::endl;
    // // Create hivesplits for each entry in idFileAddr map, add splits to task
    // for (const auto& entry : idFileAddrMap) {

    //   core::PlanNodeId key = entry.first;

    //   const std::vector<std::shared_ptr<TempFilePath>> fileAddr = entry.second;

    //   auto hiveSplits = makeHiveConnectorSplits(fileAddr);

    //   for (auto& split : hiveSplits) {

    //     task->addSplit(key, exec::Split(std::move(split)));
    //   }

    //   ids.push_back(key);
    // }

    // // Add hivesplits to the target plan node (data source node).
    // std::chrono::steady_clock::time_point begin =
    //     std::chrono::steady_clock::now();


    // // Start the task by setting the number of drivers.
    // task->start(numThreads);
    // // Wait for no more splits.
    // for (auto id: ids){

    //   task->noMoreSplits(id);
    // }

    // // Wait for all drivers to finish.
    // waitForFinishedDrivers(task);

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

      const std::vector<std::string> fileAddr = entry.second;
      auto fileFormat = cataLog.getIdFileFormat(key);

      auto hiveSplits = makeHiveConnectorSplits(fileAddr, fileFormat);

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
    int totalDataNum = 0;
    for (auto batchedData : actualResults) {
      int batchSize = batchedData->size();
      std::cout << fmt::format("[INFO] Batched Data: {}, Batch Size:{} \n", dataIdx, batchSize) << batchedData->toString() << std::endl;
      dataIdx += 1;
      totalDataNum += batchSize;
    }

    std::cout << fmt::format("[INFO] Total # of Batch: {}, Total # of Data: {}\n", dataIdx, totalDataNum);

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
 * 
 * @return DataFrame The structure used to denote the generated data.
*/
  DataFrame data_generate(
      int features, 
      int samples, 
      int first_layer, 
      int second_layer){
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights + bias.
    // ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x weights + bias.
    int input_features_size = features;
    int num_samples = samples;

    int first_layer_output_size = first_layer;
    int second_layer_output_size = second_layer;

    int input_total_size = input_features_size * num_samples;

    int weight_layer1_size = input_features_size * first_layer_output_size;
    int weight_layer2_size = first_layer_output_size * second_layer_output_size;

    int bias_layer1_size = num_samples * first_layer_output_size;
    int bias_layer2_size = num_samples * second_layer_output_size;
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
                  // featureVector.push_back((i*input_features_size+j)/input_total_size);
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

        weight_layer1[i] = 0.00001; 

    }
    float* weight_layer2 = new float[weight_layer2_size];

    for (int i = 0; i < weight_layer2_size; ++i) {

        weight_layer2[i] = 0.00001; 

    }

    std::vector<float*> weights;
    weights.push_back(weight_layer1);
    weights.push_back(weight_layer2);

    //Generate bias
    float* bias_layer1 = new float[bias_layer1_size];

    for (int i = 0; i < bias_layer1_size; ++i) {

        bias_layer1[i] = 0; 

    }
    float* bias_layer2 = new float[bias_layer2_size];

    for (int i = 0; i < bias_layer2_size; ++i) {

        bias_layer2[i] = 0; 

    }

    std::vector<float*> bias;
    bias.push_back(bias_layer1);
    bias.push_back(bias_layer2);

    // Create DataFrame
    DataFrame data;
    data.features = featureVectors;
    data.weights = weights;
    data.bias = bias;
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
  std::string registerFunctions(int units1, int units2, int input_size1, int input_size2, float* weightsFile_1, float* weightsFile_2, float* biasFile_1, float* biasFile_2, CataLog &catalog, bool isVerticalPartition = false) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "mat_mul0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_1, input_size1, units1),
        {},
        true,
        catalog,
        isVerticalPartition
    );
    // Register matrix addition function for the first layer
    optimization::registerVectorFunction(
        "mat_add0",
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(biasFile_1, units1),
        {},
        true,
        catalog
    );
    // Register ReLU activation function for the first layer
    optimization::registerVectorFunction(
        "relu0",
        Relu::signatures(),
        std::make_unique<Relu>(),
        {},
        true,
        catalog
     );
    // Register matrix multiplication function for the second layer
    optimization::registerVectorFunction(
        "mat_mul1",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_2, input_size2, units2),
        {},
        true,
        catalog,
        isVerticalPartition
    );
    // Register matrix addition function for the second layer
    optimization::registerVectorFunction(
        "mat_add1",
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(biasFile_2, units2),
        {},
        true,
        catalog
    );
    // Register softmax activation function for the second layer
    optimization::registerVectorFunction(
        "softmax0",
        Softmax::signatures(),
        std::make_unique<Softmax>(),
        {},
        true,
        catalog
     );
      // Compose and return the vector function expression
     return "softmax0(mat_add1(mat_mul1(relu0(mat_add0(mat_mul0({}))))))";
    // return "mat_mul0({})";
    // return "relu0(mat_add0(mat_mul0({})))";
  }
  /**
   * @brief A test function to test the rewrite rule of Mul2JoinAggRewriteAction.
   * 
   * @param rewrite A boolean value indicating whether to perform a rewrite.
  */
  void testMul2JoinAggPlan(bool rewrite) {
    // Set data source config.
    int input_features_size = 1000;//597540
    int num_samples = 5000;
    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;
    // Set splits number
    int num_splits = 4;
    // Initialize CataLog
    CataLog cataLog;
    // Generate data source
    auto data = data_generate(input_features_size, num_samples, first_layer_output_size, second_layer_output_size);
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
    //  Check the input size against the blocking threshold in cataLog.
    //  If yes, preblock the input vector, store it, and add information in cataLog.
    //  If not, set dataSource in cataLog.
    if (input_features_size > cataLog.getBlockingThreshold()) {
      // If input size is larger than blocking threshold, preblock and store in cataLog
      std::vector<std::vector<float>> valuesBlock = optimization::create_input_block(input_features_size*num_samples, data.features, cataLog.getDefaultBlocksNum());
      optimization::FileStructure values = optimization::block_to_files(valuesBlock, cataLog.getDefaultBlocksNum(), 0);
      // Set data source blocks in cataLog
      cataLog.setDataSourceBlocks(values.schema, values.paths);
      // Set data source statistics in cataLog
      cataLog.setDataSourceStat({num_samples, input_features_size});
    }
    else {
      // If input size is not larger than blocking threshold, set dataSource in cataLog
      cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
      cataLog.setDataSourceStat({num_samples, input_features_size});
    }
    // Build two dense layers UDFs using registerFunction in optimization namespace
    bool isVerticalPartition = true;
    std::string compute = registerFunctions(
      first_layer_output_size, 
      second_layer_output_size, 
      input_features_size, 
      first_layer_output_size, 
      data.weights[0], 
      data.weights[1],  
      data.bias[0], 
      data.bias[1],
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
                .project({fmt::format(compute, "v")}) 
                .planBuild();
    // Set original plan nodeId and file address of data source
    cataLog.setIdAddressMap(p0, {file});
    // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
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
      // auto it = planState.actionsPair.begin();
      std::pair<std::string, std::string> testAction = std::make_pair("mat_mul0", "Mul2JoinAggRewriteAction");
      std::cout << "[INFO] Taken action: " << testAction << std::endl;
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
  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};
};

// DEFINE_int32(feature_size, 3000, "Feature size");
// DEFINE_int32(num_sample, 1000, "Number of samples");
// DEFINE_bool(rewrite, true, "Whether apply rewrite rule");

int main(int argc, char** argv) {
  // gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  Mul2JoinAggRewriteActionTest demo;

  // bool rewrite = FLAGS_rewrite;
  // int numSamples = FLAGS_num_sample;
  // int featureSize = FLAGS_feature_size;

  // std::cout << "numsample: " << numSamples << std::endl;

  // demo.testMul2JoinAggPlan(rewrite);


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

    demo.testMul2JoinAggPlan(true);

  } else {
    std::cout
        << "================= Run UDF-Centric FFNN model w/o Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testMul2JoinAggPlan(false);
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
