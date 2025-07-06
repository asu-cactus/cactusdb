/*
 * Copyright (c) 2025 ASU Cactus Lab.
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
#include <torch/torch.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include "velox/optimizer/Helper.h"

// Velox headers
#include <H5Cpp.h>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/FraudDetectionFunctions.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

// Custom headers
#include <json/json.h>
#include "velox/cost_model/CostEstimator.h"
#include "velox/cost_model/Stat.h"
#include "velox/optimizer/Mul2JoinAggRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/TwoLayerUDF2TorchNNRewriteAction.h"
#include "velox/optimizer/tests/BenchmarkQueryTemplates.h"
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include "velox/optimizer/tests/ModelRegister.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class IntegratedMCTSTest : public HiveConnectorTestBase {
 public:
  IntegratedMCTSTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    Type::registerSerDe();
    common::Filter::registerSerDe();
    connector::hive::HiveTableHandle::registerSerDe();
    connector::hive::LocationHandle::registerSerDe();
    connector::hive::HiveColumnHandle::registerSerDe();
    connector::hive::HiveInsertTableHandle::registerSerDe();
    registerPartitionFunctionSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
    filesystems::registerLocalFileSystem();
    // Register hiveconnector for file splits.
    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    tempDirPath_ = exec::test::TempDirectoryPath::create();
    rootPool_ = memory::MemoryManager::getInstance()->addRootPool(
        "ProfileQueryGenerator");
    pool_ = rootPool_->addLeafChild("ProfileQueryGenerator");
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

  int cacheQueryPlanAndCateLog(PlanBuilder& planBuilder, CataLog& cataLog) {
    int queryPlanCacheId = queryPlanCacheId_++;
    auto serializedPlan = planBuilder.planNode()->serialize();
    queryPlanCaches_[queryPlanCacheId] = serializedPlan;
    cataLogIdAddressMapCaches_[queryPlanCacheId] = cataLog.getIdAddressMap();

    return queryPlanCacheId;
  }

  void resetQueryPlanAndQueryPlanFromCache(
      PlanBuilder& planBuilder,
      CataLog& cataLog,
      int queryPlanCacheId) {
    auto it1 = queryPlanCaches_.find(queryPlanCacheId);
    if (it1 != queryPlanCaches_.end()) {
      auto serializedPlan = it1->second;
      auto deserlizedUpdatedPlanNode =
          ISerializable::deserialize<core::PlanNode>(
              serializedPlan, pool_.get());
      planBuilder.setRoot(deserlizedUpdatedPlanNode);
    } else {
      throw std::runtime_error(
          fmt::format(
              "[ERROR]queryPlanCacheId: {} was not found queryPlanCaches.",
              queryPlanCacheId));
    }

    auto it2 = cataLogIdAddressMapCaches_.find(queryPlanCacheId);
    if (it2 != cataLogIdAddressMapCaches_.end()) {
      cataLog.setIdAddressMap(it2->second);
    } else {
      throw std::runtime_error(
          fmt::format(
              "[ERROR]queryPlanCacheId: {} was not found in cataLogIdAddressMapCaches.",
              queryPlanCacheId));
    }
  }

  // Function from ParquetTestBase.h
  std::unique_ptr<dwio::common::FileSink> createSink(
      const std::string& filePath) {
    auto sink = dwio::common::FileSink::create(
        fmt::format("file:{}", filePath), {.pool = pool_.get()});
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

  struct DataFrame {
    std::vector<std::vector<float>> features;
    std::vector<float*> weights;
    std::vector<float*> bias;
    float* featuresFloat;
    std::vector<std::shared_ptr<TempFilePath>> feature_paths;
  };

  /**
   * @brief Registers a series of vector functions in the optimization
   * namespace.
   *
   * @param units1 Number of units in the first layer.
   * @param units2 Number of units in the second layer.
   * @param input_size Size of the input for the first layer.
   * @param weights1 Pointer to the weights for the first layer.
   * @param weights2 Pointer to the weights for the second layer.
   * @param bias1 Pointer to the bias for the first layer.
   * @param bias2 Pointer to the bias for the second layer.
   * @param catalog Reference to a CataLog object to store metadata and
   * information.
   *
   * @return A string representing the composed vector function expression.
   */

  std::vector<std::vector<float>>
  loadFeaturesFromCSV(std::string filePath, int numSamples, int numFeature) {
    int size = numSamples * numFeature;

    std::cout << "Loading tensor of size " << size << " from " << filePath
              << std::endl;

    std::ifstream file(filePath.c_str());

    std::vector<std::vector<float>> inputArrayVector;

    int index = 0;

    std::string line;

    while (numSamples--) { // Read a line from the file

      std::vector<float> curRow(numFeature);

      std::getline(file, line);

      std::istringstream iss(
          line); // Create an input string stream from the line

      std::string numberStr;

      int colIndex = 0;

      while (std::getline(
          iss, numberStr, ',')) { // Read each number separated by comma
        //
        float number = std::stof(numberStr); // Convert the string to float

        if (colIndex < numFeature)

          curRow[colIndex] = number;

        colIndex++;
      }

      inputArrayVector.push_back(std::move(curRow));
    }

    file.close();

    return inputArrayVector;
  }

  std::vector<std::shared_ptr<TempFilePath>> splitDataToFiles(
      std::vector<std::vector<float>> data,
      int numSplit = 4,
      bool createIndex = false) {
    std::vector<std::shared_ptr<TempFilePath>> paths;

    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    size_t numSamples = data.size();
    size_t partitionSize = ceil(numSamples / numSplit);
    for (size_t i = 0; i < numSplit; i++) {
      auto startIdx = data.begin() + i * partitionSize;
      auto endIdx = data.begin() + (i + 1) * partitionSize;
      endIdx = (endIdx < data.end()) ? endIdx : data.end();
      size_t numSampleInPartition = (i + 1) * partitionSize <= numSamples
          ? partitionSize
          : numSamples - i * partitionSize;

      std::vector<std::vector<float>> partialData(startIdx, endIdx);
      auto featureArrayVector = maker.arrayVector<float>(partialData, REAL());
      RowVectorPtr inputRowVector;
      if (createIndex) {
        std::vector<int> indexes = randomGenerator.genIntRange(
            i * partitionSize, i * partitionSize + numSampleInPartition);
        auto indexFlatVector = maker.flatVector<int>(indexes);
        inputRowVector = maker.rowVector(
            {"idx", "v"},
            {std::move(indexFlatVector), std::move(featureArrayVector)});
      } else {
        inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
      }
      auto file = TempFilePath::create();
      auto config = std::make_shared<facebook::velox::dwrf::Config>();
      writeToFile(file->path, {inputRowVector}, config);
      paths.push_back(file);
    }

    return paths;
  }

  std::string process_mem_usage() {
    using std::ifstream;
    using std::ios_base;
    using std::string;

    double vm_usage = 0.0;
    double resident_set = 0.0;

    // Read data from /proc/self/stat
    ifstream stat_stream("/proc/self/stat", ios_base::in);
    if (!stat_stream) {
      std::cerr << "Error opening /proc/self/stat" << std::endl;
      return "";
    }

    // Extract relevant fields
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
        tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
        stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
        starttime >> vsize >> rss;

    stat_stream.close();

    // Get page size in KB
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024 / 1024 / 1024;

    // Calculate memory usage
    vm_usage = vsize / 1024.0 / 1024.0 / 1024.0;
    resident_set = rss * page_size_kb;
    std::cout << fmt::format(
                     " vm_usage: {:.2f} , resident_set: {:.2f}",
                     vm_usage,
                     resident_set)
              << std::endl;
    return "";
  }

  void runProfile(
      std::string workload,
      std::string queryTemplate,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      int numThreads,
      int repeatRun,
      int verbose,
      bool rewrite,
      int dataBatchSize = 256,
      std::string dataPath = "") {
    PlanBuilder myPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;
    std::string computationStr;

    if (workload == "ml") {
      if (queryTemplate == "ml-q1" || queryTemplate == "ml-q2" ||
          queryTemplate == "ml-q3") {
        if (queryTemplate == "ml-q1") {
          // register ml-q1 models
          registerTwoTowerFunc(cataLog, pool_, false /*isVerticalPartition*/);
          registerMLTrendingModelFunctions(cataLog, pool_);
        } else if (queryTemplate == "ml-q2") {
          registerMLTrendingModelFunctions(cataLog, pool_);
          registerMLInterestMovieModelFunctions(cataLog, pool_);
          registerMLMovieTagEncoderModelFunctions(cataLog, pool_);
          registerMLDLRMModelFunctions(cataLog, pool_);
        } else if (queryTemplate == "ml-q3") {
          registerMLQ3UserMovieInterestModelFunctions(cataLog, pool_);
          registerMLQ3UserMovieRatingModelFunctions(cataLog, pool_);
          registerMLMovieTagEncoderModelFunctions(cataLog, pool_);
          registerMLMovieTagEncoderModelFunctions1(cataLog, pool_);
        }

        // use original movielens dataset and pre-defined query plan
        myPlan = setupMovielensDBQuery(
            queryTemplate, cataLog, pool_, planNodeIdGenerator);

      } else {
        // use profile query plan
        checkValidProfileQueryGenerationSetting(
            numberOfTuples, dummyFeatureSizes, queryTemplate);
        generateDummyData(
            workload,
            numberOfTuples,
            dummyFeatureSizes,
            cataLog,
            pool_,
            rootPool_,
            dataBatchSize,
            dataPath);
        myPlan = setupProfileQueryPlan(
            workload,
            queryTemplate,
            modelGroupId_,
            cataLog,
            pool_,
            planNodeIdGenerator);
      }

    } else {
      throw std::runtime_error(
          fmt::format("Non-supported workload: {}", workload));
    }

    std::cout << "[INFO] Original Query Plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;

    if (rewrite) {
      // Randomly rewrite the query plan to generate various query plans.
      myPlan = rewriteQuery(
          cataLog, pool_, myPlan, planNodeIdGenerator, verbose, "random");
    }

    std::cout << "[INFO] Executed Query Plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;
    auto serializedPlan = myPlan.planNode()->serialize();
    std::string queryOutPutPath =
        "/home/velox/velox/optimizer/tests/serializedQueryPlan.json";
    augmentSerializedPlan(serializedPlan, cataLog);
    writeStringToFile(folly::toJson(serializedPlan), queryOutPutPath);

    std::cout << "[INFO] IdAddressMap: \n";
    for (auto entry : cataLog.getIdAddressMap()) {
      std::cout << entry.first << ": # Files: " << entry.second.size() << " "
                << entry.second << std::endl;
    }

    float executeTime = runPlanWithCataLog(
        pool_, numThreads, myPlan, cataLog, repeatRun, verbose);

    std::string latencyOutputPath =
        "/home/velox/velox/optimizer/tests/executionLatency.txt";
    writeStringToFile(std::to_string(executeTime), latencyOutputPath);

    std::cout << "[INFO] Execution time: " << executeTime << std::endl;

    /*
    Comment it out for now, since it is not used in the current version of
    training data collections.
    // Collect optimal rule
    int initQueryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    auto planNode = myPlan.planNode();
    planState.getPossibleActions(planNode, cataLog);
    std::map<std::string, std::vector<std::string>> actionsPair =
        planState.actionsPair;
    // Use a min-heap to store the rule latency
    std::priority_queue<
        std::pair<float, std::string>,
        std::vector<std::pair<float, std::string>>,
        std::greater<std::pair<float, std::string>>>
        minHeapOfRuleLatency;
    // Enumerate plan
    for (auto& action : actionsPair) {
      std::string targetExpr = action.first;
      std::vector<std::string> applicableRules = action.second;
      for (auto& ruleName : applicableRules) {
        std::pair<std::string, std::string> selectedAction =
            std::make_pair(targetExpr, ruleName);

        // Apply the rule and get the new plan
        planState.takeAction(
            planNode,
            nullptr,
            maker,
            myPlan,
            pool_,
            planNodeIdGenerator,
            {selectedAction},
            cataLog);
        float latency = runPlanWithCataLog(
            pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
        // Store the current rule latency, the rule with the minimum latency
        // will be stored as the top in the min-heap
        minHeapOfRuleLatency.push(std::make_pair(latency, ruleName));

        // Rest the query plan and catalog to the original state
        resetQueryPlanAndQueryPlanFromCache(
            myPlan, cataLog, initQueryPlanCacheId);
        planNode = myPlan.planNode();
        planState.clearTransformedExpr();
        planState.getPossibleActions(planNode, cataLog);
      }
    }

    // Obtain the optimal rule name
    auto [minLatency, optimalRuleName] = minHeapOfRuleLatency.top();

    std::string optimalRuleOutputPath =
        "/home/velox/velox/optimizer/tests/optimalRule.txt";
    writeStringToFile(optimalRuleName, optimalRuleOutputPath);

    std::cout << "[INFO] Optimal Rule: " << optimalRuleName << std::endl;
    */

    return;
  }

  void benchmarkQueryFromTemplate(
      std::string workload,
      std::string queryTemplate,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      int numThreads,
      int repeatRun,
      int verbose,
      bool rewrite,
      int dataBatchSize = 256,
      std::string dataPath = "") {
    PlanBuilder queryPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;

    // During the benchmark, we are going to use the real movielens & TPCx-AI
    // datasets

    queryPlan = setupProfileQueryPlanFromTemplate(
        workload,
        queryTemplate,
        modelGroupId_,
        cataLog,
        pool_,
        planNodeIdGenerator);

    float executeTime = runPlanWithCataLog(
        pool_, numThreads, queryPlan, cataLog, repeatRun, verbose);

    std::string latencyOutputPath =
        "/home/cactusdb/velox/optimizer/tests/executionLatency.txt";
    writeStringToFile(std::to_string(executeTime), latencyOutputPath);

    auto serializedPlan = queryPlan.planNode()->serialize();
    std::string queryOutPutPath =
        "/home/cactusdb/velox/optimizer/tests/serializedQueryPlan.json";
    augmentSerializedPlan(serializedPlan, cataLog);
    writeStringToFile(folly::toJson(serializedPlan), queryOutPutPath);

    auto queryPlanStr = queryPlan.planNode()->toString(true, true);
    std::string queryPlanStrOutputPath =
        "/home/cactusdb/velox/optimizer/tests/queryPlanStr.txt";
    writeStringToFile(queryPlanStr, queryPlanStrOutputPath);

    std::cout << "[INFO] Execution time: " << executeTime << std::endl;
  }

  void benchmarkQueryFromTemplate1(
      std::string workload,
      std::string queryTemplate,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      int numThreads,
      int repeatRun,
      int verbose,
      bool rewrite,
      int dataBatchSize = 256,
      std::string dataPath = "") {
    PlanBuilder queryPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;

    // During the benchmark, we are going to use the real movielens & TPCx-AI
    // datasets

    queryPlan = setupProfileQueryPlanFromTemplate1(
        workload,
        queryTemplate,
        modelGroupId_,
        cataLog,
        pool_,
        planNodeIdGenerator);

    float executeTime = runPlanWithCataLog(
        pool_, numThreads, queryPlan, cataLog, repeatRun, verbose);

    std::cout << "[INFO] Executed Query Plan: \n"
              << queryPlan.planNode()->toString(true, true) << std::endl;

    std::string latencyOutputPath =
        "/home/cactusdb/velox/optimizer/tests/executionLatency.txt";
    writeStringToFile(std::to_string(executeTime), latencyOutputPath);

    auto serializedPlan = queryPlan.planNode()->serialize();
    std::string queryOutPutPath =
        "/home/cactusdb/velox/optimizer/tests/serializedQueryPlan.json";
    augmentSerializedPlan(serializedPlan, cataLog);
    writeStringToFile(folly::toJson(serializedPlan), queryOutPutPath);

    auto queryPlanStr = queryPlan.planNode()->toString(true, true);
    std::string queryPlanStrOutputPath =
        "/home/cactusdb/velox/optimizer/tests/queryPlanStr.txt";
    writeStringToFile(queryPlanStr, queryPlanStrOutputPath);

    std::cout << "[INFO] Execution time: " << executeTime << std::endl;
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::MemoryManager::getInstance()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};
  std::shared_ptr<TempDirectoryPath> tempDirPath_;
  std::map<int, std::map<core::PlanNodeId, std::vector<std::string>>>
      cataLogIdAddressMapCaches_;

  VectorMaker maker{pool_.get()};
  static inline int queryPlanCacheId_ = 0;
  std::map<int, folly::dynamic> queryPlanCaches_;
  static inline int modelGroupId_ = 0;
};

DEFINE_string(workload, "ml", "workload: ml, movielens");
DEFINE_string(
    query_template,
    "user",
    "Query template: user, movie, movie_relevance_tag");
DEFINE_string(model, "ffnn", "Model: ffnn, df, two-tower, llm");
DEFINE_bool(rewrite, true, "Whether randomly rewrite the query");
DEFINE_int32(num_repeat, 1, "Number of repeat run");
DEFINE_int32(user_feature_size, 256, "User ffnn feature size");
DEFINE_int32(movie_feature_size, 256, "Movie ffnn feature size");
DEFINE_int32(num_user, 1000, "Number of user");
DEFINE_int32(num_movie, 1000, "Number of movie");
DEFINE_int32(num_tag, 1000, "Number of tag");
DEFINE_int32(num_driver, 8, "Number of drivers");
DEFINE_int32(verbose, 2, "Verbose");
DEFINE_int32(data_batch_size, 256, "Data batch size");
DEFINE_string(data_path, "", "Data path to store the generated data");

int main(int argc, char** argv) {
  memory::MemoryManager::initialize({});
  folly::init(&argc, &argv, false);
  std::string workload = FLAGS_workload;
  std::string queryTemplate = FLAGS_query_template;
  std::string model = FLAGS_model;

  bool rewrite = FLAGS_rewrite;
  int repeatRun = FLAGS_num_repeat;
  int userFeatureSize = FLAGS_user_feature_size;
  int movieFeatureSize = FLAGS_movie_feature_size;
  int numUser = FLAGS_num_user;
  int numMovie = FLAGS_num_movie;
  int numTag = FLAGS_num_tag;
  int numDriver = FLAGS_num_driver;
  int verbose = FLAGS_verbose;
  int dataBatchSize = FLAGS_data_batch_size;
  std::string dataPath = FLAGS_data_path;
  IntegratedMCTSTest demo;

  std::vector<int> numberOfTuples;
  std::vector<int> dummyFeatureSizes;

  if (workload == "ml") {
    // TODO: refactor: change to another name
    numberOfTuples.push_back(numUser);
    numberOfTuples.push_back(numMovie);
    numberOfTuples.push_back(numTag);
    dummyFeatureSizes.push_back(userFeatureSize);
    dummyFeatureSizes.push_back(movieFeatureSize);
    std::cout << "numberOfTuples: " << numberOfTuples << std::endl;
    std::cout << "dummyFeatureSizes: " << dummyFeatureSizes << std::endl;

    demo.runProfile(
        workload,
        queryTemplate,
        numberOfTuples,
        dummyFeatureSizes,
        numDriver,
        repeatRun,
        verbose,
        rewrite,
        dataBatchSize,
        dataPath);
  } else if (workload == "movielens" || workload == "tpcxai") {
    numberOfTuples.push_back(numUser);
    numberOfTuples.push_back(numMovie);
    numberOfTuples.push_back(numTag);
    dummyFeatureSizes.push_back(userFeatureSize);
    dummyFeatureSizes.push_back(movieFeatureSize);
    std::cout << "numberOfTuples: " << numberOfTuples << std::endl;
    std::cout << "dummyFeatureSizes: " << dummyFeatureSizes << std::endl;

    demo.benchmarkQueryFromTemplate(
        workload,
        queryTemplate,
        numberOfTuples,
        dummyFeatureSizes,
        numDriver,
        repeatRun,
        verbose,
        rewrite,
        dataBatchSize,
        dataPath);
  } else if (workload == "movielens1" || workload == "tpcxai1") {
    numberOfTuples.push_back(numUser);
    numberOfTuples.push_back(numMovie);
    numberOfTuples.push_back(numTag);
    dummyFeatureSizes.push_back(userFeatureSize);
    dummyFeatureSizes.push_back(movieFeatureSize);
    std::cout << "numberOfTuples: " << numberOfTuples << std::endl;
    std::cout << "dummyFeatureSizes: " << dummyFeatureSizes << std::endl;

    demo.benchmarkQueryFromTemplate1(
        workload,
        queryTemplate,
        numberOfTuples,
        dummyFeatureSizes,
        numDriver,
        repeatRun,
        verbose,
        rewrite,
        dataBatchSize,
        dataPath);
  } else {
    throw std::runtime_error(
        fmt::format("Non-supported workload: {}", workload));
  }
}
