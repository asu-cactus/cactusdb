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

#include <omp.h>


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

  void writeCSV(const std::vector<float>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Iterate over each element in the vector
    for (auto it = data.begin(); it != data.end(); ++it) {
        file << *it; // Write the element to the file

        // Add comma if it's not the last element in the vector
        if (std::next(it) != data.end()) {
            file << ",";
        }
    }

    file << "\n"; // End of row

    file.close();
}

  float runPlanWithCataLog(
      int numThreads,
      int numEigen,
      int numTorch,
      std::string batchSize,
      PlanBuilder& myPlan,
      CataLog& cataLog,
      int repeatRun,
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

      // writeCSV(emptyMatrix, "/home/ubuntu/velox/velox/optimizer/tests/output.csv");

      CostModel* cm = new SimpleCostModel(cataLog);
      CostEstimator* ce =
          new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

      auto planNode = myPlan.planNode();
      CostEstimate cost = ce->estimateCost(planNode);

      // std::cout << "estimated cost:" << cost.cost << std::endl;
      // std::cout << "estimated eigen cost:" << cost.eigenCost << std::endl;
      // std::cout << "estimated torch cost:" << cost.torchCost << std::endl;

      std::vector<float> costMatrix;
      costMatrix.push_back(cost.eigenCost);
      costMatrix.push_back(cost.torchCost);

      writeCSV(costMatrix, "/home/ubuntu/velox/velox/optimizer/tests/cost.csv");

      openblas_set_num_threads(numEigen);//eigen
      torch::set_num_threads(numTorch);//torch
      // omp_set_num_threads(4);//omp for xgboost

      queryCtx_->testingOverrideConfigUnsafe(
          {
            {core::QueryConfig::kPreferredOutputBatchBytes, "25000000000"},
           {core::QueryConfig::kMaxOutputBatchRows, batchSize}});

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

    return totalElapsedTime / repeatRun;
  }


  struct DataFrame {
    std::vector<std::vector<float>> features;
    std::vector<float*> weights;
    std::vector<float*> bias;
    float* featuresFloat;
    std::string weightsFile;
      ~DataFrame() {
      // Deallocate memory for each float* in weights and bias vectors
      for (float* ptr : weights) {
          delete[] ptr;
      }
      for (float* ptr : bias) {
          delete[] ptr;
      }

      // Deallocate memory for featuresFloat
      delete[] featuresFloat;
  }
};

void saveFloatArray(const char* filename, float* array, long size) {
    ofstream outfile(filename);
    if (outfile.is_open()) {
        // Convert size to string and save it first
        outfile << size << "\n";
        
        // Save the array elements
        for (long i = 0; i < size; ++i) {
            outfile << array[i] << "\n";
        }
        outfile.close();
    } else {
        cerr << "Unable to open file " << filename << endl;
    }
}

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
  data_generate(int samples, std::vector<int> layers) {
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights
    // + bias. ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x
    // weights + bias.
    int dims = layers.size();
    int input_features_size = layers[0];
    int num_samples = samples;

    int input_total_size = input_features_size * num_samples;
    std::vector<float*> weights;
    std::vector<float*> bias;

    for (int i=0;i<(dims-1);i++) {
      long weight_size = layers[i] * static_cast<long>(layers[i+1]);
      int bias_size = num_samples * layers[i+1];

      float* weight_layer = new float[weight_size];
      for (long j = 0; j <weight_size; ++j) {
        weight_layer[j] = 0.000001;
      }
      weights.push_back(weight_layer);

      float* bias_layer = new float[bias_size];
      for (long k = 0; k <weight_size; ++k) {
        weight_layer[k] = 0.000001;
      }
      bias.push_back(bias_layer);
    }

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

    // Create DataFrame
    DataFrame data;
    data.features = featureVectors;
    data.weightsFile = "/home/ubuntu/velox/weights.txt";
    data.weights = weights;
    data.featuresFloat = floatArray;
    data.bias = bias;

    return data;
  }

  DataFrame
  data_generate_blocks(int features, int samples, int first_layer, int second_layer, CataLog& cataLog) {
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights
    // + bias. ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x
    // weights + bias.
    int input_features_size = features;
    int num_samples = samples;

    int first_layer_output_size = first_layer;
    int second_layer_output_size = second_layer;

    int input_total_size = input_features_size * num_samples;

    long weight_layer1_size = input_features_size * static_cast<long>(first_layer_output_size);
    long weight_layer2_size = first_layer_output_size * second_layer_output_size;
    int weight_test = input_features_size * first_layer_output_size;
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

    optimization::FileStructure weightsFileStructure;
    int blocksSize = cataLog.getDefaultBlocksSize();
    // std::cout << "blocksSize:" << blocksSize << std::endl;
    weightsFileStructure = create_blocks_to_files(input_features_size, first_layer_output_size, blocksSize);
    std::string nameSuffix = "_vertical";
    cataLog.add("mat_mul0", weightsFileStructure.schema, weightsFileStructure.paths, 1, nameSuffix);

    // Create DataFrame

    DataFrame data;
    data.features = featureVectors;

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
      std::vector<int> layers,
      // std::string weightsFile_1,
      std::vector<float*> weights,
      std::vector<float*> bias,
      CataLog& catalog,
      bool isVerticalPartition) {
    // Register for the first layer
    optimization::registerVectorFunction(
        "mat_mul0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weights[0], layers[0], layers[1]),
        {},
        true,
        catalog,
        isVerticalPartition);

    optimization::registerVectorFunction(
        "mat_add0",
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(bias[0], layers[1]),
        {},
        true,
        catalog,
        isVerticalPartition);

    optimization::registerVectorFunction(
        "relu0",
        Relu::signatures(),
        std::make_unique<Relu>(),
        {},
        true,
        catalog,
        isVerticalPartition);

    optimization::registerVectorFunction(
        "torchDNN",
        TorchDNN::signatures(),
        std::make_unique<TorchDNN>(weights, bias, layers),
        {},
        true,
        catalog,
        isVerticalPartition);
    
    return "torchDNN(relu0(mat_add0(mat_mul0({}))))";

  }


  /**
   * @brief A test function to test the rewrite rule of
   * Mul2JoinAggRewriteAction.
   *
   * @param rewrite A boolean value indicating whether to perform a rewrite.
   */
  void testSingleRewrite(
      int repeatRun,
      int featureSize,
      int outputSize,
      int numSamples,
      int numDriver,
      int numEigen,
      int numTorch,
      int numBlocks,
      int option,
      std::string batchSize,
      int verbose) {
    // Set data source config.
    int input_features_size = featureSize; // 597540
    int num_samples = numSamples;
    int first_layer_output_size = outputSize;
    int second_layer_output_size = 14588;
    int third_layer_output_size = 4096;
    int fourth_layer_output_size = 2048;
    int fith_layer_output_size = 1024;
    // Set splits number
    // Initialize CataLog

    CataLog cataLog;
    int blockSize = outputSize / numBlocks;
    // std::cout << "blockSize:" << blockSize << std::endl;
    // cataLog.setDefaultBlocksSize(256);
    cataLog.setDefaultBlocksSize(blockSize);
    cataLog.setDefaultBlocksNum(numBlocks);
    cataLog.setBlockingThreshold(1);
    
    // Generate data source
    if (option == 1) {
      std::vector<int> dims = {input_features_size, input_features_size, first_layer_output_size};
      auto data = data_generate(num_samples, dims);
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    // Create file path
    auto file = TempFilePath::create();
    auto config = std::make_shared<facebook::velox::dwrf::Config>();
    writeToFile(file->path, {inputRowVector}, config);

    bool isVerticalPartition = false;
    std::string compute = registerFunctions(
        dims,
        data.weights,
        data.bias,
        cataLog,
        isVerticalPartition);


    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .project({fmt::format(compute, "v")})
                      .planBuild();

    cataLog.setIdAddressMap(p0, {file});
      // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");

    // Add source to catalog
    std::shared_ptr<OutputStat> stat =
    std::make_shared<OutputStat>(OutputStat(numSamples, featureSize));
    Source src = Source(p0, Source::Type::FILE, std::move(stat));
    cataLog.addSource(std::make_shared<Source>(src));

      float averageExectuionTime =
        runPlanWithCataLog(numDriver, numEigen, numTorch, batchSize, myPlan, cataLog, repeatRun, verbose);
      std::cout << averageExectuionTime;

    }

    else if(option == 2) {
      std::vector<int> dims = {input_features_size, input_features_size, first_layer_output_size, second_layer_output_size};
      auto data = data_generate(
        num_samples,
        dims);
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    // Create file path
    auto file = TempFilePath::create();
    
    auto config = std::make_shared<facebook::velox::dwrf::Config>();

    writeToFile(file->path, {inputRowVector}, config);

    bool isVerticalPartition = false;
    std::string compute = registerFunctions(
        dims,
        data.weights,
        data.bias,
        cataLog,
        isVerticalPartition);


    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .project({fmt::format(compute, "v")})
                      .planBuild();

    cataLog.setIdAddressMap(p0, {file});
      // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");

        // Add source to catalog
    std::shared_ptr<OutputStat> stat =
    std::make_shared<OutputStat>(OutputStat(numSamples, featureSize));
    Source src = Source(p0, Source::Type::FILE, std::move(stat));
    cataLog.addSource(std::make_shared<Source>(src));

      float averageExectuionTime =
        runPlanWithCataLog(numDriver, numEigen, numTorch, batchSize, myPlan, cataLog, repeatRun, verbose);
      std::cout << averageExectuionTime;

    }

    else if(option == 3) {
      std::vector<int> dims = {input_features_size, input_features_size, first_layer_output_size, second_layer_output_size, third_layer_output_size};
      auto data = data_generate(
        num_samples,
        dims);
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    // Create file path

    auto file = TempFilePath::create();
    
    auto config = std::make_shared<facebook::velox::dwrf::Config>();

    writeToFile(file->path, {inputRowVector}, config);

    bool isVerticalPartition = false;
    std::string compute = registerFunctions(
        dims,
        data.weights,
        data.bias,
        cataLog,
        isVerticalPartition);


    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .project({fmt::format(compute, "v")})
                      .planBuild();

    cataLog.setIdAddressMap(p0, {file});
      // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");
        // Add source to catalog
    std::shared_ptr<OutputStat> stat =
    std::make_shared<OutputStat>(OutputStat(numSamples, featureSize));
    Source src = Source(p0, Source::Type::FILE, std::move(stat));
    cataLog.addSource(std::make_shared<Source>(src));

      float averageExectuionTime =
        runPlanWithCataLog(numDriver, numEigen, numTorch, batchSize, myPlan, cataLog, repeatRun, verbose);
      std::cout << averageExectuionTime;

    }

    else if(option == 4) {
      std::vector<int> dims = {input_features_size, input_features_size, first_layer_output_size, second_layer_output_size, third_layer_output_size, fourth_layer_output_size};
      auto data = data_generate(
        num_samples,
        dims);
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    // Create file path

    auto file = TempFilePath::create();
    
    auto config = std::make_shared<facebook::velox::dwrf::Config>();

    writeToFile(file->path, {inputRowVector}, config);

    bool isVerticalPartition = false;
    std::string compute = registerFunctions(
        dims,
        data.weights,
        data.bias,
        cataLog,
        isVerticalPartition);


    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .project({fmt::format(compute, "v")})
                      .planBuild();

    cataLog.setIdAddressMap(p0, {file});
      // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");
        // Add source to catalog
    std::shared_ptr<OutputStat> stat =
    std::make_shared<OutputStat>(OutputStat(numSamples, featureSize));
    Source src = Source(p0, Source::Type::FILE, std::move(stat));
    cataLog.addSource(std::make_shared<Source>(src));

      float averageExectuionTime =
        runPlanWithCataLog(numDriver, numEigen, numTorch, batchSize, myPlan, cataLog, repeatRun, verbose);
      std::cout << averageExectuionTime;

    }

    else {
      std::vector<int> dims = {input_features_size, input_features_size, first_layer_output_size, second_layer_output_size, third_layer_output_size, fourth_layer_output_size, fith_layer_output_size};
      auto data = data_generate(
        num_samples,
        dims);
    auto featureArrayVector = maker.arrayVector<float>(data.features, REAL());
    // Create rowVector for data source
    auto inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
    // Create file path

    auto file = TempFilePath::create();
    
    auto config = std::make_shared<facebook::velox::dwrf::Config>();

    writeToFile(file->path, {inputRowVector}, config);

    bool isVerticalPartition = false;
    std::string compute = registerFunctions(
        dims,
        data.weights,
        data.bias,
        cataLog,
        isVerticalPartition);


    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .project({fmt::format(compute, "v")})
                      .planBuild();

    cataLog.setIdAddressMap(p0, {file});
      // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");
        // Add source to catalog
    std::shared_ptr<OutputStat> stat =
    std::make_shared<OutputStat>(OutputStat(numSamples, featureSize));
    Source src = Source(p0, Source::Type::FILE, std::move(stat));
    cataLog.addSource(std::make_shared<Source>(src));

      float averageExectuionTime =
        runPlanWithCataLog(numDriver, numEigen, numTorch, batchSize, myPlan, cataLog, repeatRun, verbose);
      std::cout << averageExectuionTime;

    }
    


    




    
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};
};

DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(feature_size, 1000, "FFNN Feature size");
DEFINE_int32(num_sample, 1000, "Number of samples");
DEFINE_int32(output_size, 10240, "output size");
DEFINE_int32(num_driver, 8, "Number of drivers");
DEFINE_int32(num_eigen, 8, "Number of drivers");
DEFINE_int32(num_torch, 8, "Number of drivers");
DEFINE_int32(num_function_threads, 8, "Number of core function threads");
DEFINE_int32(num_blocks, 4, "Number of blocks in partition");
DEFINE_int32(option, 1, "option of layers");
DEFINE_string(batch_size, "1000", "batch size of ouput");
DEFINE_int32(verbose, 1, "Verbose");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  int repeatRun = FLAGS_num_repeat;
  int featureSize = FLAGS_feature_size;
  int outputSize = FLAGS_output_size;
  int numSample = FLAGS_num_sample;
  int numDriver = FLAGS_num_driver;
  int numEigen = FLAGS_num_eigen;
  int numTorch = FLAGS_num_torch;
  int numBlocks = FLAGS_num_blocks;
  int option = FLAGS_option;
  std::string batchSize = FLAGS_batch_size;
  int verbose = FLAGS_verbose;
  BenchmarkTest demo;
  // available single benchmark mode: mul2joinAgg, mul2joinAggHorizontal
  demo.testSingleRewrite(
          repeatRun, featureSize, outputSize, numSample, numDriver, numEigen, numTorch, numBlocks, option, batchSize, verbose);
}