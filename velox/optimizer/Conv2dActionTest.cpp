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
#include "RewriteAction.h"
#include "Mul2JoinAggRewriteAction.h"
#include "RuleManager.h"
#include "PlanState.h"
#include "Register.h"
#include "ConvHelper.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class Conv2dActionTest : public HiveConnectorTestBase {
 public:
 Conv2dActionTest() {
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

  ~Conv2dActionTest() {
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
    constexpr int64_t KB = 1024L;
    constexpr int64_t MB = 1024L * KB;
    constexpr int64_t GB = 1024L * MB;
    std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};
    // Initializes queryCtx.
    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};
    // Set queryCtx config.
    // std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 50 * MB)};

    // queryCtx_->testingOverrideMemoryPool(rootPool);
    // queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});
    // queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "true"}});

    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, "100000000000000000"},// 100000000000000000  1000000
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

      const std::vector<std::string> fileAddr = entry.second;
      auto fileFormat = cataLog.getIdFileFormat(key);
      std::cout << "path size:"<< fileAddr.size() << std::endl;
      auto hiveSplits = makeHiveConnectorSplits(fileAddr, fileFormat);
      std::cout << "hive size:"<< hiveSplits.size() << std::endl;
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
    // for (auto batchedData : actualResults) {
    //   std::cout << fmt::format("[INFO] Batched Data: {} \n", dataIdx) << batchedData->toString(0, batchedData->size()) << std::endl;
    //   dataIdx += 1;
    // }

    std::cout << "Time for FFNN with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;
  }

  void runPlan_splits(
      int numThreads,
      int numSplits,
      PlanBuilder& myPlan,
      CataLog &cataLog) {

    // Initializes executor.
    constexpr int64_t KB = 1024L;
    constexpr int64_t MB = 1024L * KB;
    constexpr int64_t GB = 1024L * MB;
    std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};
    // Initializes queryCtx.
    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};
    // Set queryCtx config.
    std::shared_ptr<memory::MemoryPool> rootPool{memory::MemoryManager::getInstance()->addRootPool("root", 50 * MB)};

    queryCtx_->testingOverrideMemoryPool(rootPool);
    // queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});
    queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "true"}});

    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, "100000000000000000"},// 100000000000000000  1000000
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

      const std::vector<std::string> fileAddr = entry.second;

      // auto hiveSplits = makeHiveConnectorSplits(fileAddr[0], numSplits, dwio::common::FileFormat::DWRF);
      auto hiveSplits =  makeHiveConnectorSplits(fileAddr[0], cataLog.getDefaultSplits(), dwio::common::FileFormat::DWRF);
      std::cout << "hive size:"<< hiveSplits.size() << std::endl;
      for (auto& split : hiveSplits) {
        std::cout << split->toString() << std::endl;
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
    // for (auto batchedData : actualResults) {
    //   std::cout << fmt::format("[INFO] Batched Data: {} \n", dataIdx) << batchedData->toString(0, batchedData->size()) << std::endl;
    //   dataIdx += 1;
    // }

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
      int filters){

    int input_features_size = features;
    int num_samples = samples;

    // int first_layer_output_size = first_layer;

    int input_total_size = input_features_size * num_samples;

    int weight_layer1_size = first_layer;

    int bias_layer1_size = filters;
    // Seed the random number generator
    std::random_device rd;  
    // Initialize the Mersenne Twister engine
    std::mt19937 gen(rd());
    // Define the range
    std::uniform_real_distribution<float> distribution(0, 1);

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

        weight_layer1[i] = 1.0; 
        // weight_layer1[i] = i;

    }

    std::vector<float*> weights;
    weights.push_back(weight_layer1);

    //Generate bias
    float* bias_layer1 = new float[bias_layer1_size];

    for (int i = 0; i < bias_layer1_size; ++i) {

        bias_layer1[i] = 0.01; 

    }

    std::vector<float*> bias;
    bias.push_back(bias_layer1);

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
  std::string registerFunctions(int* dims, float* weightsFile_1, float* biasFile_1, int num_filters, int side, CataLog &catalog) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "conv2d0",
        Convolute::signatures(),
        std::make_unique<Convolute>(weightsFile_1, dims),
        {},
        true,
        catalog
    );

    optimization::registerVectorFunction(
        "vec_scal_add0",
        VectorScalarAddition::signatures(),
        std::make_unique<VectorScalarAddition>(biasFile_1, num_filters),        
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
    int height = dims[4] - dims[1] + 1;
    int width = dims[5] - dims[2] + 1;
    optimization::registerVectorFunction(
        "max_pool0",
        MaxPool::signatures(),
        std::make_unique<MaxPool>(side, height, width),
        {},
        true,
        catalog
    );

      // Compose and return the vector function expression
     return "max_pool0(relu0(vec_scal_add0(conv2d0({}))))";
    // return "mat_mul0({})";
    // return "relu0(mat_add0(mat_mul0({})))";
  }

  std::string registerFunctions(int* dims, float* weightsFile_1, float* biasFile_1, CataLog &catalog) {
    optimization::registerVectorFunction(
    "torchcnn0",
    TorchCNN::signatures(),
    std::make_unique<TorchCNN>(weightsFile_1, biasFile_1, dims),
    {},
    true,
    catalog
    );

      // Compose and return the vector function expression
     return "torchcnn0({})";
    // return "mat_mul0({})";
    // return "relu0(mat_add0(mat_mul0({})))";
  }

  std::string registerFunctions(int* dims, float* weightsFile_1, int input_size1, int units1, int side, CataLog &catalog) {
    optimization::registerVectorFunction(
        "mat_mul0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_1, input_size1, units1),
        {},
        true,
        catalog
    );
    // Register ReLU activation function for the first layer
    // optimization::registerVectorFunction(
    //     "relu0",
    //     Relu::signatures(),
    //     std::make_unique<Relu>(),
    //     {},
    //     true,
    //     catalog
    //  );
    // int height = dims[4] - dims[1] + 1;
    // int width = dims[5] - dims[2] + 1;
    // optimization::registerVectorFunction(
    //     "max_pool0",
    //     MaxPool::signatures(),
    //     std::make_unique<MaxPool>(side, height, width),
    //     {},
    //     true,
    //     catalog
    // );

      // Compose and return the vector function expression
    //  return "max_pool0(relu0(mat_mul0({})))";
    return "mat_mul0({})";
    // return "relu0(mat_add0(mat_mul0({})))";
  }

  void saveToFile(const std::string& filename, const ImageMatrix& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    for (int i = 0; i < matrix.num_row; ++i) {
        for (int j = 0; j < matrix.num_col; ++j) {
            file << matrix.data[i * matrix.num_col + j] << " ";
        }
        file << std::endl;
    }
    file.close();
}

  /**
   * @brief A test function to test the rewrite rule of Mul2JoinAggRewriteAction.
   * 
   * @param rewrite A boolean value indicating whether to perform a rewrite.
  */
  void testConv2dActionPlan(bool rewrite, int flag, int samples) {
    // Set data source config.
    int cnn_filters = 64;
    int cnn_filter_dims[] = {1,1,64}; // height * width * channels
    // int cnn_filters = 2048;
    // int cnn_filter_dims[] = {1,1,3};
    int weights_size = cnn_filter_dims[0] * cnn_filter_dims[1] * cnn_filter_dims[2] * cnn_filters;

    int input_dims[] = {112,112,64}; 
    // int input_dims[] = {2500,2500,3};
    int input_size = input_dims[0] * input_dims[1] * input_dims[2];
    int num_samples = samples;
    int max_pool_size = 2;

    int dims[] = {cnn_filters, cnn_filter_dims[0], cnn_filter_dims[1], cnn_filter_dims[2], input_dims[0], input_dims[1], max_pool_size};
   
    


    // Initialize CataLog
    CataLog cataLog;
    
    // cataLog.setDefaultBlocksSize(256);
    // cataLog.setBlockingThreshold(1);
    // Generate data source
    auto data = data_generate(input_size, num_samples, weights_size, cnn_filters);
    
    // ImageMatrix kernel = convert2Matrix(cnn_filter_dims, cnn_filters, data.weights[0], data.bias[0]);
    // ImageMatrix images = convert2Matrix(cnn_filter_dims, input_dims, num_samples, data.featuresFloat);


    if (flag == 1) {
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

    cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
    cataLog.setDataSourceStat({num_samples, input_size, input_dims[0], input_dims[1], input_dims[2]});
    // }
    // Build two dense layers UDFs using registerFunction in optimization namespace

    std::string compute = registerFunctions(
      dims, 
      data.weights[0], 
      data.bias[0],
      cnn_filters,
      dims[6], //max_pool size 
      cataLog);



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
    // cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    runPlan(8, 8, myPlan, cataLog);
    }
    else if(flag == 2) {
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

    cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
    cataLog.setDataSourceStat({num_samples, input_size, input_dims[0], input_dims[1], input_dims[2]});

    std::string compute = registerFunctions(dims, data.weights[0], data.bias[0], cataLog);

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
    // cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    runPlan(8, 8, myPlan, cataLog);

    }

    else if(flag == 3) {
    auto blocks = convert2Blocks(input_dims[0], input_dims[1], input_dims[2], num_samples, cnn_filter_dims[0], cnn_filter_dims[2], 1, 0, 32);
    ImageMatrix kernel = convert2Matrix(cnn_filter_dims, cnn_filters, data.weights[0], data.bias[0]);
    // ImageMatrix images = convert2Matrix(cnn_filter_dims, input_dims, num_samples, data.featuresFloat);
    // auto values = convert2Matrix(input_dims[0], input_dims[1], input_dims[2], num_samples, cnn_filter_dims[0], cnn_filter_dims[2], 1, 0, 1);


    // saveToFile("kernel.txt", kernel);
    // saveToFile("image.txt", images);

    int kernel_col = kernel.getColSize();
    int kernel_row = kernel.getRowSize();

    // int image_col = images.getColSize();
    // int image_row = images.getRowSize();



    // std::vector<std::vector<float>> featureVectors;

    // for (int i = 0; i < image_row; i++) {

    //       std::vector<float> featureVector;

    //       for (int j = 0; j < image_col; j++) {

    //               featureVector.push_back(images.data[i*image_col + j]);
    //       }

    //       featureVectors.push_back(featureVector);

    //   }

    //   auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    // auto featureArrayVector = maker.arrayVector<float>(values, REAL());
    RowTypePtr schema;
    std::vector<std::shared_ptr<TempFilePath>> paths;
    for (auto& block : blocks) {
      std::vector<std::vector<float>> result;
      result.push_back(block);
      auto featureArrayVector = maker.arrayVector<float>(result, REAL());
      auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
      schema = asRowType(inputRowVector->type());
      auto file = TempFilePath::create();
      auto config = std::make_shared<facebook::velox::dwrf::Config>();
      writeToFile(file->path, {inputRowVector}, config);
      paths.push_back(file);
    }
    // auto featureArrayVector = maker.arrayVector<float>(blocks, REAL());
    // auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    
    // Create file path
    auto file = TempFilePath::create();
    // Create file config
    uint64_t kSizeKB = 1024UL;
    uint32_t rows = num_samples / 8;
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    // config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 1000 * kSizeKB);
    // config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, rows);
    // Write the data source to a file, with the format defined by the rowVector
    // writeToFile(file->path, {inputRowVector}, config);

    // cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
    cataLog.setDataSourceStat({num_samples, input_size, input_dims[0], input_dims[1], input_dims[2]});

    std::string compute = registerFunctions(dims, kernel.data, kernel_row, kernel_col, dims[6], cataLog);

        // Initialize planNodeID
    core::PlanNodeId p0;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(schema)
                .capturePlanNodeId(p0)
                .project({fmt::format(compute, "v")}) 
                .planBuild();
    // Set original plan nodeId and file address of data source
    // cataLog.setIdAddressMap(p0, {file});
    cataLog.setIdAddressMap(p0, paths);
    // Set vector name and nodeId of data source
    // cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    // runPlan_splits(8, 392, myPlan, cataLog);
    runPlan(8, 8, myPlan, cataLog);

    // std::cout << image_col << "," << image_row << std::endl;
    }

    else  {
    // auto blocks = convert2Blocks(input_dims[0], input_dims[1], input_dims[2], num_samples, cnn_filter_dims[0], cnn_filter_dims[2], 1, 0, 32);
    ImageMatrix kernel = convert2Matrix(cnn_filter_dims, cnn_filters, data.weights[0], data.bias[0]);
    // ImageMatrix images = convert2Matrix(cnn_filter_dims, input_dims, num_samples, data.featuresFloat);
    auto values = convert2Matrix(input_dims[0], input_dims[1], input_dims[2], num_samples, cnn_filter_dims[0], cnn_filter_dims[2], 1, 0, 12544);


    // saveToFile("kernel.txt", kernel);
    // saveToFile("image.txt", images);

    int kernel_col = kernel.getColSize();
    int kernel_row = kernel.getRowSize();

    // int image_col = images.getColSize();
    // int image_row = images.getRowSize();



    // std::vector<std::vector<float>> featureVectors;

    // for (int i = 0; i < image_row; i++) {

    //       std::vector<float> featureVector;

    //       for (int j = 0; j < image_col; j++) {

    //               featureVector.push_back(images.data[i*image_col + j]);
    //       }

    //       featureVectors.push_back(featureVector);

    //   }

    //   auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
    auto featureArrayVector = maker.arrayVector<float>(values, REAL());
    // auto featureArrayVector = maker.arrayVector<float>(blocks, REAL());
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    
    // Create file path
    auto file = TempFilePath::create();
    // Create file config
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    // Write the data source to a file, with the format defined by the rowVector
    writeToFile(file->path, {inputRowVector}, config);

    cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
    cataLog.setDataSourceStat({num_samples, input_size, input_dims[0], input_dims[1], input_dims[2]});

    std::string compute = registerFunctions(dims, kernel.data, kernel_row, kernel_col, dims[6], cataLog);

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
    // cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    runPlan(8, 8, myPlan, cataLog);

    // std::cout << image_col << "," << image_row << std::endl;
    }
  }

  


 private:
  std::shared_ptr<MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  int number1 = std::atoi(argv[1]);//actions
  int number2 = std::atoi(argv[2]);//sample size

  Conv2dActionTest demo;

  bool rewrite = true;

  if (argc > 1) {
    // if (strcmp(argv[1], "N") == 0) {
      if (strcmp(argv[3], "N") == 0) {
      rewrite = false;
    }
  }

  if (rewrite) {
    std::cout
        << "================= Run UDF-Centric CNN model w/ Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testConv2dActionPlan(true, number1, number2);

  } else {
    std::cout
        << "================= Run UDF-Centric CNN model w/o Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testConv2dActionPlan(false, number1, number2);
  }

  std::cout
      << "--" << std::endl
      << "[Usage] " << std::endl
      << "./_build/release/velox/optimizer/torch2twolayer_test Y  //run CNN model with rewriting rule 2"
      << std::endl
      << "./_build/release/velox/optimizer/torch2twolayer_test N  //run CNN model with rewriting rule 2"
      << std::endl
      << "By default: Y is used" << std::endl;
}
