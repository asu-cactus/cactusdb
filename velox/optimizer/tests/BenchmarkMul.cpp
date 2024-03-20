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


class BenchmarkTest : public HiveConnectorTestBase {
 public:
  BenchmarkTest() {
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

  ~BenchmarkTest() {
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
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
   */
  float runPlanWithCataLog(
      int numThreads,
      int numSplits,
      PlanBuilder& myPlan,
      CataLog& cataLog,
      int repeatRun = 1,
      int verbose = 1) {
    float totalElapsedTime = 0;
    std::vector<RowVectorPtr> finalResult;
    int dataIdx;
    int totalDataNum;

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
          {
            {core::QueryConfig::kPreferredOutputBatchBytes, "10000000"},
           {core::QueryConfig::kMaxOutputBatchRows, "1000000"},
           {core::QueryConfig::kPreferredOutputBatchRows, "1000"}});


      // Add hivesplits to the target plan node (data source node).
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      CursorParameters params;
      params.maxDrivers = numThreads;
      params.planNode = myPlan.planNode();
      params.queryCtx = queryCtx_;
      bool noMoreSplits = false;
      auto addSplits = [&noMoreSplits, &cataLog, &numSplits](exec::Task* task) {
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
      
      if (i == repeatRun - 1) {
        finalResult = actualResults;
        dataIdx = 0;
        totalDataNum = 0;
        for (auto batchedData : finalResult) {
          batchedData = std::move(batchedData);
          int batchSize = batchedData->size();
          if (verbose == 1) {
            // std::cout << fmt::format("[INFO] Batched Data: {}, Batch Size:{} \n", dataIdx, batchSize) << batchedData->toString() << std::endl;
          } else if (verbose == 2) {
            // std::cout << fmt::format("[INFO] Batched Data: {}, Batch Size:{} \n", dataIdx, batchSize) << batchedData->toString() << "\n" << batchedData->toString(0, batchedData->size()) << std::endl;
          }
          dataIdx += 1;
          totalDataNum += batchSize;
        }
        finalResult = std::move(finalResult);
      }
    }

    // std::cout << fmt::format("[INFO] Total # of Batch: {}, Total # of Data: {}", dataIdx, totalDataNum) << std::endl;
    // std::cout << "DEBUG, REACHED here";
    // finalResult = std::move(finalResult);


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
            // featureVector.push_back(1);
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
      // weight_layer1[i] = i;
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
      CataLog& catalog,
      bool isVerticalPartition,
      int numThreads) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "mat_mul0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_1, input_size1, units1, numThreads),
        {},
        true,
        catalog,
        isVerticalPartition);
    
    return "mat_mul0({})";

  }


  /**
   * @brief A test function to test the rewrite rule of
   * Mul2JoinAggRewriteAction.
   *
   * @param rewrite A boolean value indicating whether to perform a rewrite.
   */
  void testSingleRewrite(
      bool rewrite,
      int repeatRun,
      int featureSize,
      int outputSize,
      int numSamples,
      int numDriver,
      int numThreads,
      int numBlocks,
      bool splitToDisk,
      std::string benchmarkMode,
      int verbose) {
    // Set data source config.
    int input_features_size = featureSize; // 597540
    int num_samples = numSamples;
    int first_layer_output_size = outputSize;
    int second_layer_output_size = 14588;
    // Set splits number
    // Initialize CataLog
    CataLog cataLog;
    int blockSize = outputSize / numBlocks;
    // std::cout << "blockSize:" << blockSize << std::endl;
    // cataLog.setDefaultBlocksSize(256);
    cataLog.setDefaultBlocksSize(blockSize);
    cataLog.setBlockingThreshold(1);
    cataLog.setDefaultBlocksNum(numBlocks);
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
    cataLog.setUDFSchema("value", asRowType(inputRowVector->type()));
    // Build two dense layers UDFs using registerFunction in optimization
    // namespace
    bool isVerticalPartition = true;
    if (benchmarkMode == "mul2joinAggHorizontal") {
      isVerticalPartition = false;
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
        cataLog,
        isVerticalPartition,
        numThreads);

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
    if (splitToDisk) {

      std::vector<std::shared_ptr<TempFilePath>> paths;
          // Calculate the number of elements in each part (except the last one)
      size_t partSize = numSamples / (numBlocks - 1);
    // Calculate the number of elements in the last part
      size_t lastPartSize = numSamples - partSize * (numBlocks - 1);
      for (size_t i = 0; i < numBlocks - 1; ++i) {
          std::vector<std::vector<float>> result(data.features.begin() + i * partSize, data.features.begin() + (i + 1) * partSize);
          auto featureArrayVector = maker.arrayVector<float>(result, REAL());
          auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
          auto file = TempFilePath::create();
          auto config = std::make_shared<facebook::velox::dwrf::Config>();
          writeToFile(file->path, {inputRowVector}, config);
          paths.push_back(file);
      }
      std::vector<std::vector<float>> result(data.features.end() - lastPartSize, data.features.end());
      auto featureArrayVector = maker.arrayVector<float>(result, REAL());
      auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
      auto file = TempFilePath::create();
      auto config = std::make_shared<facebook::velox::dwrf::Config>();
      writeToFile(file->path, {inputRowVector}, config);
      paths.push_back(file);

      cataLog.setIdAddressMap(p0, paths);
      cataLog.setVectorIdMap(p0, "v");
    }
    else {

      cataLog.setIdAddressMap(p0, {file});
      // Set vector name and nodeId of data source
      cataLog.setVectorIdMap(p0, "v");
    }
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
      std::pair<std::string, std::string> testAction;
      if (benchmarkMode == "mul2joinAgg") {
        testAction = std::make_pair("mat_mul0", "Mul2JoinAggRewriteAction");
        cataLog.setVerticalMulThreads(numThreads);
      } else if (benchmarkMode == "mul2joinAggHorizontal") {
        testAction = std::make_pair("mat_mul0", "Mul2JoinAggHorizontalRewriteAction");
        cataLog.setHorizontalMulThreads(numThreads);
      } else {
         throw std::runtime_error(fmt::format("Non-supported benchmark mode: {}", benchmarkMode));
      }

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
    float averageExectuionTime =
        runPlanWithCataLog(numDriver, numDriver, myPlan, cataLog, repeatRun, verbose);
    std::cout << averageExectuionTime;
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};
};

DEFINE_string(mode, "mcts", "Mode: mcts or benchmark");
DEFINE_bool(rewrite, true, "Whether  rewrite");
DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(feature_size, 1000, "FFNN Feature size");
DEFINE_int32(num_sample, 1000, "Number of samples");
DEFINE_int32(output_size, 1024, "output size");
DEFINE_int32(num_driver, 8, "Number of drivers");
DEFINE_int32(num_function_threads, 8, "Number of core function threads");
DEFINE_int32(num_blocks, 4, "Number of blocks in partition");
DEFINE_bool(split_disk, false, "Whether  split to disk");
DEFINE_int32(verbose, 1, "Verbose");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  std::string mode = FLAGS_mode;
  bool rewrite = FLAGS_rewrite;
  int repeatRun = FLAGS_num_repeat;
  int featureSize = FLAGS_feature_size;
  int outputSize = FLAGS_output_size;
  int numSample = FLAGS_num_sample;
  int numDriver = FLAGS_num_driver;
  int numThreads = FLAGS_num_function_threads;
  int numBlocks = FLAGS_num_blocks;
  bool splitToDisk = FLAGS_split_disk;
  int verbose = FLAGS_verbose;
  BenchmarkTest demo;
  // available single benchmark mode: mul2joinAgg, mul2joinAggHorizontal
  demo.testSingleRewrite(
          rewrite, repeatRun, featureSize, outputSize, numSample, numDriver, numThreads, numBlocks, splitToDisk, mode, verbose);
}