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
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

// Custom headers
#include <json/json.h>
#include "velox/cost_model/CostEstimator.h"
#include "velox/optimizer/Helper.h"
#include "velox/optimizer/Mul2JoinAggRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/TwoLayerUDF2TorchNNRewriteAction.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

#define BUFFER_SIZE 1024

Json::Value receiveJsonFromSocket(int clientSocket) {
  char messageBuffer[BUFFER_SIZE];
  memset(messageBuffer, 0, BUFFER_SIZE);
  recv(clientSocket, messageBuffer, BUFFER_SIZE, 0);
  Json::CharReaderBuilder jsonReader;
  Json::Value receivedJsonMessage;
  std::istringstream jsonStream(messageBuffer);
  Json::parseFromStream(jsonReader, jsonStream, &receivedJsonMessage, nullptr);
  return receivedJsonMessage;
}

void sendJsonBySocket(Json::Value jsonMessage, int clientSocket) {
  std::string jsonMessageStr = jsonMessage.toStyledString();
  send(clientSocket, jsonMessageStr.c_str(), jsonMessageStr.length(), 0);
}

void sendAcknowledgment(int clientSocket) {
  const  char* ack_message = "ACK";
  send(clientSocket, ack_message, strlen(ack_message), 0);
}

class IntegratedMCTSTest : public HiveConnectorTestBase {
 public:
  IntegratedMCTSTest() {
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
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
  }

  ~IntegratedMCTSTest() {
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
   * @param filePath The file path for the data source file to be split.
   * @param numThreads The number of Velox executor threads.
   * @param numSplits The number of file splits.
   * @param myPlan The pointer to the planBuilder which builds the logical plan.
   * @param p0 The planNodeID for the plan node that needs to add file splits.
   */
  // TODO: needs to integrated with runPlanWithCataLog
  float runPlan(
      std::string filePath,
      int numThreads,
      int numSplits,
      PlanBuilder& myPlan,
      core::PlanNodeId p0,
      int repeatRun = 1) {
    float totalElapsedTime = 0;
    for (int i = 0; i < repeatRun; i++) {
      // Create hivesplits for file.
      auto hiveSplits = makeHiveConnectorSplits(
          filePath, numSplits, dwio::common::FileFormat::DWRF);
      // Initializes executor.
      std::shared_ptr<folly::Executor> executor_{
          std::make_shared<folly::CPUThreadPoolExecutor>(
              std::thread::hardware_concurrency())};
      // Initializes queryCtx.
      std::shared_ptr<core::QueryCtx> queryCtx_{
          std::make_shared<core::QueryCtx>(executor_.get())};
      // Set queryCtx config.
      queryCtx_->testingOverrideConfigUnsafe(
          {{core::QueryConfig::kPreferredOutputBatchBytes, "100000000"},
           {core::QueryConfig::kMaxOutputBatchRows, "1000000"}});
      // Create task for logical plan.
      auto task = exec::Task::create(
          "0",
          myPlan.planFragment(),
          0,
          queryCtx_,
          [](RowVectorPtr result, ContinueFuture* /*unused*/) {
            if (result)
              // std::cout << result->toString() << std::endl;
              return exec::BlockingReason::kNotBlocked;
          });

      // std::cout << "Hive splits:" << std::endl;
      // Add hivesplits to the target plan node (data source node).
      for (auto& split : hiveSplits) {
        // std::cout << split->toString() << std::endl;
        task->addSplit(p0, exec::Split(std::move(split)));
      }
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      // Start the task by setting the number of drivers.
      task->start(numThreads);
      // Add all splits.
      task->noMoreSplits(p0);
      // Wait for all drivers to finish.
      waitForFinishedDrivers(task);

      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      auto elapsedTime =
          (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()) /
          1000000.0;
      totalElapsedTime += elapsedTime;
    }
    std::stringstream ss;

    ss << numSplits << "," << numThreads << ",";

    // std::cout << "Time for FFNN with Input Data (sec): " << std::endl;

    // std::cout << ss.str() << totalElapsedTime/repeatRun << " secs" <<
    // std::endl;
    return totalElapsedTime / repeatRun;
  }

  /**
   * @brief A function to run logical plan.
   *
   * @param numThreads The number of Velox executor threads.
   * @param numSplits The number of file splits.
   * @param myPlan The pointer to the planBuilder which builds the logical plan.
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
   */
  float runPlanWithCataLog(
      int numThreads,
      int numSplits,
      PlanBuilder& myPlan,
      CataLog& cataLog,
      int repeatRun = 1) {
    float totalElapsedTime = 0;
    for (int i = 0; i < repeatRun; i++) {
      // Initializes executor.
      std::shared_ptr<folly::Executor> executor_{
          std::make_shared<folly::CPUThreadPoolExecutor>(
              std::thread::hardware_concurrency())};
      // Initializes queryCtx.
      std::shared_ptr<core::QueryCtx> queryCtx_{
          std::make_shared<core::QueryCtx>(executor_.get())};
      // Set queryCtx config.
      queryCtx_->testingOverrideConfigUnsafe(
          {{core::QueryConfig::kPreferredOutputBatchBytes, "100000000"},
           {core::QueryConfig::kMaxOutputBatchRows, "1000000"}});


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

            const std::vector<std::shared_ptr<TempFilePath>> fileAddr =
                entry.second;

            auto hiveSplits = makeHiveConnectorSplits(fileAddr);

            for (auto& split : hiveSplits) {
              task->addSplit(key, exec::Split(std::move(split)));
            }

            ids.push_back(key);
          }

          for (auto id : ids) {
            task->noMoreSplits(id);
          }
        }
        noMoreSplits = true;
      };

      auto [cursor, actualResults] = readCursor(params, addSplits);
      waitForTaskCompletion(cursor->task().get());

      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();

      auto elapsedTime =
          (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()) /
          1000000.0;
      totalElapsedTime += elapsedTime;
    }

    return totalElapsedTime / repeatRun;
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
  DataFrame
  data_generate(int features, int samples, int first_layer, int second_layer) {
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights
    // + bias. ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x
    // weights + bias.
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

    // Generate input
    std::vector<std::vector<float>> featureVectors;

    for (int i = 0; i < num_samples; i++) {
      std::vector<float> featureVector;

      for (int j = 0; j < input_features_size; j++) {
        featureVector.push_back(
            (i * input_features_size + j) / input_total_size);
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

    // Generate weight
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

    // Generate bias
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

    // Create DataFrame
    DataFrame data;
    data.features = featureVectors;
    data.weights = weights;
    data.bias = bias;
    data.featuresFloat = floatArray;

    return data;
  }

  /**
   * @brief Registers a series of vector functions in the optimization
   * namespace.
   *
   * @param units1 Number of units in the first layer.
   * @param units2 Number of units in the second layer.
   * @param input_size1 Size of the input for the first layer.
   * @param input_size2 Size of the input for the second layer.
   * @param weightsFile_1 Pointer to the weights for the first layer.
   * @param weightsFile_2 Pointer to the weights for the second layer.
   * @param biasFile_1 Pointer to the bias for the first layer.
   * @param biasFile_2 Pointer to the bias for the second layer.
   * @param catalog Reference to a CataLog object to store metadata and
   * information.
   *
   * @return A string representing the composed vector function expression.
   */
  std::string registerFunctions(
      int units1,
      int units2,
      int input_size1,
      int input_size2,
      float* weightsFile_1,
      float* weightsFile_2,
      float* biasFile_1,
      float* biasFile_2,
      CataLog& catalog) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "mat_mul0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_1, input_size1, units1),
        {},
        true,
        catalog);
    // Register matrix addition function for the first layer
    optimization::registerVectorFunction(
        "mat_add0",
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(biasFile_1, units1),
        {},
        true,
        catalog);
    // Register ReLU activation function for the first layer
    optimization::registerVectorFunction(
        "relu0",
        Relu::signatures(),
        std::make_unique<Relu>(),
        {},
        true,
        catalog);
    // Register matrix multiplication function for the second layer
    optimization::registerVectorFunction(
        "mat_mul1",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_2, input_size2, units2),
        {},
        true,
        catalog);
    // Register matrix addition function for the second layer
    optimization::registerVectorFunction(
        "mat_add1",
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(biasFile_2, units2),
        {},
        true,
        catalog);
    // Register softmax activation function for the second layer
    optimization::registerVectorFunction(
        "softmax0",
        Softmax::signatures(),
        std::make_unique<Softmax>(),
        {},
        true,
        catalog);
    // Compose and return the vector function expression
    return "softmax0(mat_add1(mat_mul1(relu0(mat_add0(mat_mul0({}))))))";
  }

  /**
   * @brief A test function to test the rewrite rule of TwoLayerUDF2TorchNN.
   *
   * @param rewrite A boolean value indicating whether to perform a rewrite.
   */
  void testTwoLayerUDF2TorchNNPlan(
      bool rewrite,
      int repeatRun,
      int featureSize,
      int numSamples) {
    // Set data source config.
    // int input_features_size = 800; // 597540
    int input_features_size = featureSize; // 597540
    // int num_samples = 1000;
    int num_samples = numSamples;
    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;
    CataLog cataLog;
    // Set splits number
    int num_splits = 4;
    // Generate data source
    auto data = data_generate(
        input_features_size,
        num_samples,
        first_layer_output_size,
        second_layer_output_size);
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
    // Build two dense layers UDFs
    std::string compute = NNBuilder()
                              .denseLayer(
                                  first_layer_output_size,
                                  input_features_size,
                                  data.weights[0],
                                  data.bias[0],
                                  NNBuilder::RELU)
                              .denseLayer(
                                  second_layer_output_size,
                                  first_layer_output_size,
                                  data.weights[1],
                                  data.bias[1],
                                  NNBuilder::SOFTMAX)
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
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);
    // Run rewriten rule
    if (rewrite) {
      // Get possible actions for this plan
      planState.getPossibleActions(planNode, cataLog);
      // Print possible actions
      // for (const auto& entry : planState.actionsPair) {
      //   std::cout << "[INFO] print action pair\n";
      //   std::cout << entry.first << ": " << entry.second << std::endl;
      // }
      // Choose one action from possible actions (Now we only pick the first
      // one, later it would be choosen by MCTS)
      auto it = planState.actionsPair.begin();
      std::pair<std::string, std::string> testAction = *it;
      // Take one rewritten action
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);
      // Update the planState (getPossibleAction after apply one action)
      planState.update(myPlan, cataLog);
    }

    // Run the rewritten plan
    // std::cout << "Plan: " << myPlan.planNode()->toString(true, true) <<
    // std::endl;
    float averageExectuionTime =
        runPlan(file->path, 8, 8, myPlan, p0, repeatRun);
    std::cout << averageExectuionTime;
  }

  /**
   * @brief A test function to test the rewrite rule of
   * Mul2JoinAggRewriteAction.
   *
   * @param rewrite A boolean value indicating whether to perform a rewrite.
   */
  void testMul2JoinAggPlan(
      bool rewrite,
      int repeatRun,
      int featureSize,
      int numSamples) {
    // Set data source config.
    int input_features_size = featureSize; // 597540
    int num_samples = numSamples;
    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;
    // Set splits number
    int num_splits = 4;
    // Initialize CataLog
    CataLog cataLog;
    // Generate data source
    auto data = data_generate(
        input_features_size,
        num_samples,
        first_layer_output_size,
        second_layer_output_size);
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
    //  If yes, preblock the input vector, store it, and add information in
    //  cataLog. If not, set dataSource in cataLog.
    if (input_features_size > cataLog.getBlockingThreshold()) {
      // If input size is larger than blocking threshold, preblock and store in
      // cataLog
      std::vector<std::vector<float>> valuesBlock =
          optimization::create_input_block(
              input_features_size * num_samples,
              data.features,
              cataLog.getDefaultBlocksNum());
      optimization::FileStructure values = optimization::block_to_files(
          valuesBlock, cataLog.getDefaultBlocksNum(), 0);
      // Set data source blocks in cataLog
      cataLog.setDataSourceBlocks(values.schema, values.paths);
      // Set data source statistics in cataLog
      cataLog.setDataSourceStat({num_samples, input_features_size});
    } else {
      // If input size is not larger than blocking threshold, set dataSource in
      // cataLog
      cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
      cataLog.setDataSourceStat({num_samples, input_features_size});
    }
    // Build two dense layers UDFs using registerFunction in optimization
    // namespace
    std::string compute = registerFunctions(
        first_layer_output_size,
        second_layer_output_size,
        input_features_size,
        first_layer_output_size,
        data.weights[0],
        data.weights[1],
        data.bias[0],
        data.bias[1],
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
    cataLog.setVectorIdMap(p0, "v");
    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // std::cout<<"rule size" << ruleManager.rules.size() << std::endl;
    // auto it = ruleManager.rules.find("TwoLayerUDF2TorchNNRewriteAction");
    // ruleManager.rules.erase(it);
    // std::cout<<"rule size" << ruleManager.rules.size() << std::endl;
    // Create planState
    PlanState planState(ruleManager);
    // Run rewriten rule
    if (rewrite) {
      // Get possible actions for this plan
      planState.getPossibleActions(planNode, cataLog);
      // std::cout << "action size" << planState.actionsPair.size() <<
      // std::endl; Print possible actions for (const auto& entry :
      // planState.actionsPair) {
      //   std::cout << entry.first << ": " << entry.second << std::endl;
      // }
      // Choose one action from possible actions (Now we only pick the first
      // one, later it would be choosen by MCTS)
      auto it = planState.actionsPair.begin();
      std::pair<std::string, std::string> testAction = *it;
      // Take one rewritten action
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);
      // Update the planState (getPossibleAction after apply one action)
      planState.update(myPlan, cataLog);
    }

    // Run the rewritten plan
    // std::cout << myPlan.planNode()->toString(true, true) << std::endl;
    float averageExectuionTime =
        runPlanWithCataLog(8, 8, myPlan, cataLog, repeatRun);
    std::cout << averageExectuionTime;
  }

  void testIntegratedMCTS(int featureSize, int numSamples, int repeatRun) {
    // Set data source config.
    // int input_features_size = 800; // 597540
    int input_features_size = featureSize; // 597540
    // int num_samples = 1000;
    int num_samples = numSamples;
    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;
    CataLog cataLog;
    // Set splits number
    int num_splits = 4;
    // Generate data source
    auto data = data_generate(
        input_features_size,
        num_samples,
        first_layer_output_size,
        second_layer_output_size);
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

    if (input_features_size > cataLog.getBlockingThreshold()) {
      // If input size is larger than blocking threshold, preblock and store in
      // cataLog
      std::vector<std::vector<float>> valuesBlock =
          optimization::create_input_block(
              input_features_size * num_samples,
              data.features,
              cataLog.getDefaultBlocksNum());
      optimization::FileStructure values = optimization::block_to_files(
          valuesBlock, cataLog.getDefaultBlocksNum(), 0);
      // Set data source blocks in cataLog
      cataLog.setDataSourceBlocks(values.schema, values.paths);
      // Set data source statistics in cataLog
      cataLog.setDataSourceStat({num_samples, input_features_size});
    } else {
      // If input size is not larger than blocking threshold, set dataSource in
      // cataLog
      cataLog.setDataSource(asRowType(inputRowVector->type()), {file});
      // Set data source statistics in cataLog
      cataLog.setDataSourceStat({num_samples, input_features_size});
    }

    std::string compute = registerFunctions(
        first_layer_output_size,
        second_layer_output_size,
        input_features_size,
        first_layer_output_size,
        data.weights[0],
        data.weights[1],
        data.bias[0],
        data.bias[1],
        cataLog);

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
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);
    // Run rewriten rule

    // Set up socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
      std::cerr << "Error creating socket\n";
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serverAddr.sin_port = htons(12345);

    if (connect(
            clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) ==
        -1) {
      std::cerr << "Error connecting to server\n";
      close(clientSocket);
    }

    std::cout << "Start optimization." << std::endl;

    // send start message to start MCTS optimization
    // start flag and initial query plan
    Json::Value startJsonMessage;
    startJsonMessage["mctsAction"] = "start";
    startJsonMessage["queryPlan"] = planNode->toString(true, true);
    sendJsonBySocket(startJsonMessage, clientSocket);
    bool optimizationIsFinished = false;

    while (!optimizationIsFinished) {
      planNode = myPlan.planNode();
      // received json message from MCTS
      Json::Value receivedJsonMessage = receiveJsonFromSocket(clientSocket);
      std::string mctsAction = receivedJsonMessage["mctsAction"].asString();
      std::cout << "===================================" << std::endl;
      std::cout << "Received message with mcts action: " << mctsAction
                << std::endl;
      std::cout << "JSON Message: " << receivedJsonMessage << std::endl;
      if (mctsAction == "resetPlan") {
        // if it is root node, it needs to start with original plan
        // the p0 will be increased after capturePlanNodeId is called
        // so it is required to clean the old IdAddressMap and VectorIdMap
        // before reset the myPlan
        cataLog.clearIdAddressMap();
        cataLog.clearVectorIdMap();
        myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                     .tableScan(asRowType(inputRowVector->type()))
                     .capturePlanNodeId(p0)
                     .project({fmt::format(compute, "v")});
        cataLog.setIdAddressMap(p0, {file});
        cataLog.setVectorIdMap(p0, "v");
        planNode = myPlan.planNode();
        // send acknowledgement for synchronization 
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getQueryPlan") {
        Json::Value jsonMessage;
        jsonMessage["communicateFlag"] = true;
        jsonMessage["mctsAction"] = "recQueryPlan";
        jsonMessage["queryPlan"] = myPlan.planNode()->toString(true, true);
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "getActionSpace") {
        planState.getPossibleActions(planNode, cataLog);
        Json::Value jsonMessage;
        jsonMessage["actionSpace"] = Json::arrayValue;
        for (const auto& entry : planState.actionsPair) {
          // std::cout << "[ACTION SBACE] " << entry.first << ", " <<
          // entry.second << std::endl;
          Json::Value jsonEntry;
          jsonEntry["expression"] = entry.first;
          jsonEntry["action"] = entry.second;
          jsonMessage["actionSpace"].append(jsonEntry);
        }
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "takeAction") {
        std::pair<std::string, std::string> targetAction;
        targetAction.first = receivedJsonMessage["targetString"].asString();
        targetAction.second = receivedJsonMessage["targetAction"].asString();

        std::cout << "[INFO] take action: " << targetAction << std::endl;
        if (targetAction.first != "None") {
          // None action is selected
          planState.takeAction(
              planNode,
              nullptr,
              maker,
              myPlan,
              pool_,
              planNodeIdGenerator,
              {targetAction},
              cataLog);
          planState.update(myPlan, cataLog);
        }
        std::cout << "[INFO] current my query plan"
                  << myPlan.planNode()->toString(true, true) << std::endl;
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getCost") {
        Json::Value jsonMessage;
        if (receivedJsonMessage["costMode"] == "offline") {
          float executeTime = runPlanWithCataLog(8, 8, myPlan, cataLog, 1);
          jsonMessage["reward"] = executeTime;
          std::cout << "[INFO] get Cost(offline): "
                    << " time: " << executeTime << std::endl;
        } else if (receivedJsonMessage["costMode"] == "online") {
          std::shared_ptr<Catalog> catalog =
              std::make_shared<Catalog>(Catalog("db-catalog"));

          std::shared_ptr<OutputStat> stat =
              std::make_shared<OutputStat>(OutputStat(1, 2));

          Source src1 = Source(p0, Source::Type::FILE, std::move(stat));

          catalog->addSource(std::make_shared<Source>(src1));

          CostModel* cm = new SimpleCostModel(catalog);
          CostEstimator* ce =
              new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

          planNode = myPlan.planNode();
          CostEstimate cost = ce->estimateCost(planNode);
          jsonMessage["reward"] = cost.cost + 1;
          std::cout << "[INFO] get Cost(online): " << cost.cost << std::endl;
        }
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);

      } else if (mctsAction == "finished") {
        // finished
        // nothing to do
      }

      optimizationIsFinished =
          receivedJsonMessage["optimizationIsFinished"].asBool();
      std::cout << "[INFO] reached end of the loop, current opt flag: "
                << optimizationIsFinished << std::endl;
    };

    // Run the rewritten plan
    std::cout << "[INFO] MCTS finished, run the optimized query plan"
              << std::endl;
    std::cout << "[INFO] Optimized query plan"
              << myPlan.planNode()->toString(true, true) << std::endl;
    runPlanWithCataLog(8, 8, myPlan, cataLog, 1);
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::addDefaultLeafMemoryPool();

  VectorMaker maker{pool_.get()};
};

DEFINE_string(mode, "mcts", "Mode: mcts or benchmark");
DEFINE_bool(rewrite, true, "Whether  rewrite");
DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(feature_size, 1000, "FFNN Feature size");
DEFINE_int32(num_sample, 1000, "Number of samples");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  std::string mode = FLAGS_mode;
  bool rewrite = FLAGS_rewrite;
  int repeatRun = FLAGS_num_repeat;
  int featureSize = FLAGS_feature_size;
  int numSample = FLAGS_num_sample;
  IntegratedMCTSTest demo;
  if (mode == "mcts") {
    demo.testIntegratedMCTS(featureSize, numSample, repeatRun);
  } else if (mode == "benchmark_udf2torchdnn") {
    demo.testTwoLayerUDF2TorchNNPlan(
        rewrite, repeatRun, featureSize, numSample);
  } else if (mode == "benchmark_mul2joinagg") {
    demo.testMul2JoinAggPlan(rewrite, repeatRun, featureSize, numSample);
  }
}
