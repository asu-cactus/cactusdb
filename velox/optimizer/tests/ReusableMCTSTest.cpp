#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>

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
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/ChatGPT.h"
#include "velox/ml_functions/ComplexLayer.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/Dropout.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/Encoder.h"
#include "velox/ml_functions/FraudDetectionFunctions.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/ml_functions/SequencePooling.h"
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
#include "velox/optimizer/Helper.h"
#include "velox/optimizer/Mul2JoinAggRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/TwoLayerUDF2TorchNNRewriteAction.h"
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include "velox/optimizer/tests/ModelRegister.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class ReusableMCTSTest : public HiveConnectorTestBase {
 public:
  ReusableMCTSTest() {
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
  }

  ~ReusableMCTSTest() {
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
      throw std::runtime_error(fmt::format(
          "[ERROR]queryPlanCacheId: {} was not found queryPlanCaches.",
          queryPlanCacheId));
    }

    auto it2 = cataLogIdAddressMapCaches_.find(queryPlanCacheId);
    if (it2 != cataLogIdAddressMapCaches_.end()) {
      cataLog.setIdAddressMap(it2->second);
    } else {
      throw std::runtime_error(fmt::format(
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

  std::vector<std::vector<int>> readModelStructureFromFile(
      const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
      std::cerr << "Error opening file!" << std::endl;
      return {};
    }

    std::vector<std::vector<int>> all_lines;
    std::string line;

    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::vector<int> values;
      int num;

      // Read each integer in the line
      while (iss >> num) {
        values.push_back(num);
      }

      // Store the line of integers in the main vector
      all_lines.push_back(values);
    }

    file.close();
    return all_lines;
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

  std::string registerNNModel(
      std::vector<int> units,
      CataLog& catalog,
      bool hasArgmax = false) {
    // use input size as random seed
    RandomGenerator randomGenerator = RandomGenerator(-1, 1, units[0]);
    int modelGroupId = modelGroupId_++;
    int functionId = 0;
    int numberOfLayers = units.size() - 1;

    optimization::registerVectorFunction(
        "relu",
        Relu::signatures(),
        std::make_unique<Relu>(),
        {},
        true,
        catalog);
    optimization::registerVectorFunction(
        "softmax",
        Softmax::signatures(),
        std::make_unique<Softmax>(),
        {},
        true,
        catalog);
    optimization::registerVectorFunction(
        "argmax",
        Argmax::signatures(),
        std::make_unique<Argmax>(),
        {},
        true,
        catalog);

    std::string modelComputationStr = "{}";
    int lastSize = units[0];

    for (int i = 1; i < units.size(); i++) {
      int layerSize = units[i];
      std::vector<std::vector<float>> weights =
          randomGenerator.genFloat2dVector(lastSize, layerSize);
      std::vector<std::vector<float>> bias =
          randomGenerator.genFloat2dVector(1, layerSize);
      std::string matMulName =
          fmt::format("mat_mul{}_{}", modelGroupId, functionId++);
      optimization::registerVectorFunction(
          matMulName,
          MatrixMultiply::signatures(),
          std::make_unique<MatrixMultiply>(
              std::move(flattenVectorToPointer(weights)), lastSize, layerSize),
          {},
          true,
          catalog);
      std::string matVectorAddName =
          fmt::format("mat_vector_add{}_{}", modelGroupId, functionId++);
      optimization::registerVectorFunction(
          matVectorAddName,
          MatrixVectorAddition::signatures(),
          std::make_unique<MatrixVectorAddition>(
              std::move(flattenVectorToPointer(bias)), layerSize),
          {},
          true,
          catalog);
      modelComputationStr = matVectorAddName + "(" + matMulName + "(" +
          modelComputationStr + "))";
      if (i != units.size() - 1) {
        modelComputationStr = "relu(" + modelComputationStr + ")";
      } else {
        modelComputationStr = "softmax(" + modelComputationStr + ")";
      }
      lastSize = layerSize;
    }
    if (hasArgmax) {
      modelComputationStr = "argmax(" + modelComputationStr + ")";
    }
    return modelComputationStr;
  }

  PlanBuilder setupProfileQueryPlan(
      std::string mode,
      std::string queryTemplate,
      CataLog& cataLog,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      int randomSeed = -1) {
    // bool generateFilter = stringToBool(getEnvVar("CD_PROFILE_W_FILTER"));
    bool generateFilter = true;

    unsigned timestampSeed =
        std::chrono::system_clock::now().time_since_epoch().count();
    if (randomSeed != -1) {
      timestampSeed = randomSeed;
    }
    RandomGenerator randomGenerator = RandomGenerator(-1, 1, timestampSeed);
    randomGenerator.setIntRange(0, 1);
    PlanBuilder myPlan;

    std::vector<std::vector<int>> userModelStructures =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
    std::vector<std::vector<int>> movieModelStructures =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/movie_dummy_model_structure.txt");
    std::vector<std::vector<int>> tagModelStructures =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/tag_dummy_model_structure.txt");

    if (mode == "ml") {
      RowTypePtr userDataRowType = cataLog.getRegisteredDataSrcSchema("user");
      RowTypePtr movieDataRowType = cataLog.getRegisteredDataSrcSchema("movie");
      RowTypePtr movieRelevanceTagRowType =
          cataLog.getRegisteredDataSrcSchema("movie_relevance_tag");

      dwio::common::FileFormat userFileFormat =
          cataLog.getRegisteredDataSrcFormat("user");
      dwio::common::FileFormat movieFileFormat =
          cataLog.getRegisteredDataSrcFormat("movie");
      dwio::common::FileFormat movieRelevanceTagFileFormat =
          cataLog.getRegisteredDataSrcFormat("movie_relevance_tag");

      std::vector<std::string> userFilePaths =
          cataLog.getRegisteredDataSrcFiles("user");
      std::vector<std::string> movieFilePaths =
          cataLog.getRegisteredDataSrcFiles("movie");
      std::vector<std::string> movieRelevanceTagFilePaths =
          cataLog.getRegisteredDataSrcFiles("movie_relevance_tag");

      std::pair<int, int> userStats = cataLog.getRegisteredDataSrcStats("user");
      int userNumRows = userStats.first;
      int userNumCols = userStats.second;
      std::pair<int, int> movieStats =
          cataLog.getRegisteredDataSrcStats("movie");
      int movieNumRows = movieStats.first;
      int movieNumCols = movieStats.second;
      std::pair<int, int> movieRelevanceTagStats =
          cataLog.getRegisteredDataSrcStats("movie_relevance_tag");
      int movieRelevanceTagNumRows = movieRelevanceTagStats.first;
      int movieRelevanceTagNumCols = movieRelevanceTagStats.second;

      if (queryTemplate == "" || queryTemplate == "user") {
        std::string userModel1ComputExpr = registerNNModel(
            userModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readUserDataPlanNodeId;
        myPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .tableScan(userDataRowType, {}, "")
                     .capturePlanNodeId(readUserDataPlanNodeId)
                     .project(
                         {"u_user_id",
                          "u_age",
                          "u_gender",
                          "u_occupation",
                          "u_zipcode",
                          "u_features"})
                     .project(
                         {"u_user_id",
                          "u_age",
                          "u_gender",
                          "u_occupation",
                          "u_zipcode",
                          fmt::format(userModel1ComputExpr, "u_features")});
        if (generateFilter) {
          std::vector<std::string> filterExpr =
              sampleUserMovieFilterExpr("user", randomSeed);
          for (auto expr : filterExpr) {
            myPlan = myPlan.filter(expr);
          }
          // myPlan = myPlan.filter(filterExpr);
        }

        cataLog.setIdAddressMap(
            readUserDataPlanNodeId,
            userFilePaths,
            dwio::common::FileFormat::DWRF);
        cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
        std::shared_ptr<OutputStat> userStats =
            std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
        Source userSrc =
            Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
        cataLog.addSource(std::make_shared<Source>(userSrc));
      } else if (queryTemplate == "movie") {
        std::string movieModel1ComputExpr = registerNNModel(
            movieModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readMovieDataPlanNodeId;
        myPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .tableScan(movieDataRowType, {}, "")
                     .capturePlanNodeId(readMovieDataPlanNodeId)
                     .project(
                         {"m_movie_id",
                          "m_title",
                          "m_genres",
                          "m_spoken_languages",
                          "m_popularity",
                          "m_vote_average",
                          "m_vote_count",
                          "m_features"})
                     .project(
                         {"m_movie_id",
                          "m_title",
                          "m_genres",
                          "m_spoken_languages",
                          "m_popularity",
                          "m_vote_average",
                          "m_vote_count",
                          "m_features",
                          fmt::format(movieModel1ComputExpr, "m_features")});

        if (generateFilter) {
          std::vector<std::string> filterExpr =
              sampleUserMovieFilterExpr("movie", randomSeed);
          for (auto expr : filterExpr) {
            myPlan = myPlan.filter(expr);
          }
          // myPlan = myPlan.filter(filterExpr);
        }

        cataLog.setIdAddressMap(
            readMovieDataPlanNodeId,
            movieFilePaths,
            dwio::common::FileFormat::DWRF);

        cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
        std::shared_ptr<OutputStat> movieStats = std::make_shared<OutputStat>(
            OutputStat(movieNumRows, movieNumCols));
        Source movieSrc =
            Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
        cataLog.addSource(std::make_shared<Source>(movieSrc));
      } else if (queryTemplate == "movie_tag") {
        std::string tagModel1ComputExpr = registerNNModel(
            tagModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readMovieRelevanceTagDataPlanNodeId;
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(movieRelevanceTagRowType, {}, "")
                .capturePlanNodeId(readMovieRelevanceTagDataPlanNodeId)
                .project(
                    {"mt_movie_id",
                     "mt_relevance_score",
                     fmt::format(tagModel1ComputExpr, "mt_relevance_score")});

        cataLog.setIdAddressMap(
            readMovieRelevanceTagDataPlanNodeId,
            movieRelevanceTagFilePaths,
            dwio::common::FileFormat::DWRF);

        cataLog.addNodeIdRelationName(
            readMovieRelevanceTagDataPlanNodeId, "movie_relevance_tag");
        std::shared_ptr<OutputStat> movieRelevanceTagStats =
            std::make_shared<OutputStat>(
                OutputStat(movieRelevanceTagNumRows, movieRelevanceTagNumCols));
        Source movieRelevanceTagSrc = Source(
            readMovieRelevanceTagDataPlanNodeId,
            Source::Type::FILE,
            movieRelevanceTagStats);
        cataLog.addSource(std::make_shared<Source>(movieRelevanceTagSrc));
      } else if (
          queryTemplate == "movie_user" || queryTemplate == "user_movie") {
        std::string userModel1ComputExpr = registerNNModel(
            userModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readUserDataPlanNodeId;
        auto userPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                            .tableScan(userDataRowType, {}, "")
                            .capturePlanNodeId(readUserDataPlanNodeId)
                            .project(
                                {"u_user_id",
                                 "u_age",
                                 "u_gender",
                                 "u_occupation",
                                 "u_zipcode",
                                 "u_features"});

        std::string movieModel1ComputExpr = registerNNModel(
            movieModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readMovieDataPlanNodeId;
        auto moviePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                             .tableScan(movieDataRowType, {}, "")
                             .capturePlanNodeId(readMovieDataPlanNodeId)
                             .project(
                                 {"m_movie_id",
                                  "m_title",
                                  "m_genres",
                                  "m_spoken_languages",
                                  "m_popularity",
                                  "m_vote_average",
                                  "m_vote_count",
                                  "m_features"});
        RandomGenerator numModelGenerator =
            RandomGenerator(-1, 1, timestampSeed);
        numModelGenerator.setIntRange(0, 2);
        int numModel = numModelGenerator.genRandomIntValue();
        myPlan = userPlan.nestedLoopJoin(
            moviePlan.planNode(),
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features"});

        if (numModel == 1) {
          myPlan = myPlan.project(
              {"u_user_id",
               "u_age",
               "u_gender",
               "u_occupation",
               "u_zipcode",
               "u_features",
               "m_movie_id",
               "m_title",
               "m_genres",
               "m_spoken_languages",
               "m_popularity",
               "m_vote_average",
               "m_vote_count",
               "m_features",
               fmt::format(userModel1ComputExpr, "u_features")});
        } else if (numModel == 2) {
          myPlan = myPlan.project(
              {"u_user_id",
               "u_age",
               "u_gender",
               "u_occupation",
               "u_zipcode",
               "u_features",
               "m_movie_id",
               "m_title",
               "m_genres",
               "m_spoken_languages",
               "m_popularity",
               "m_vote_average",
               "m_vote_count",
               "m_features",
               fmt::format(userModel1ComputExpr, "u_features"),
               fmt::format(movieModel1ComputExpr, "m_features")});
        }

        if (generateFilter) {
          std::vector<std::string> filterExpr =
              sampleUserMovieFilterExpr("movie_user", randomSeed);
          for (auto expr : filterExpr) {
            myPlan = myPlan.filter(expr);
          }
          // myPlan = myPlan.filter(filterExpr);
        }

        cataLog.setIdAddressMap(
            readUserDataPlanNodeId,
            userFilePaths,
            dwio::common::FileFormat::DWRF);
        cataLog.setIdAddressMap(
            readMovieDataPlanNodeId,
            movieFilePaths,
            dwio::common::FileFormat::DWRF);
        cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
        cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");

        std::shared_ptr<OutputStat> userStats =
            std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
        Source userSrc =
            Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
        cataLog.addSource(std::make_shared<Source>(userSrc));
        std::shared_ptr<OutputStat> movieStats = std::make_shared<OutputStat>(
            OutputStat(movieNumRows, movieNumCols));
        Source movieSrc =
            Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
        cataLog.addSource(std::make_shared<Source>(movieSrc));
      } else if (
          queryTemplate == "movie_user_tag" ||
          queryTemplate == "user_movie_tag") {
        std::string userModel1ComputExpr = registerNNModel(
            userModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readUserDataPlanNodeId;
        auto userPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                            .tableScan(userDataRowType, {}, "")
                            .capturePlanNodeId(readUserDataPlanNodeId)
                            .project(
                                {"u_user_id",
                                 "u_age",
                                 "u_gender",
                                 "u_occupation",
                                 "u_zipcode",
                                 "u_features"});

        std::string movieModel1ComputExpr = registerNNModel(
            movieModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readMovieDataPlanNodeId;
        auto moviePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                             .tableScan(movieDataRowType, {}, "")
                             .capturePlanNodeId(readMovieDataPlanNodeId)
                             .project({
                                 "m_movie_id",
                                 "m_title",
                                 "m_genres",
                                 "m_spoken_languages",
                                 "m_popularity",
                                 "m_vote_average",
                                 "m_vote_count",
                                 "m_features",
                             });

        std::string tagModel1ComputExpr = registerNNModel(
            tagModelStructures[0],
            cataLog,
            randomGenerator.genRandomIntValue());

        core::PlanNodeId readMovieRelevanceTagDataPlanNodeId;
        auto tagPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(movieRelevanceTagRowType, {}, "")
                .capturePlanNodeId(readMovieRelevanceTagDataPlanNodeId)
                .project({
                    "mt_movie_id",
                    "mt_relevance_score",
                });

        auto movieTagPlan = tagPlan.hashJoin(
            {"mt_movie_id"},
            {"m_movie_id"},
            moviePlan.planNode(),
            "",
            {"m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             "mt_relevance_score"});

        RandomGenerator numModelGenerator =
            RandomGenerator(-1, 1, timestampSeed);
        numModelGenerator.setIntRange(0, 3);
        int numModel = numModelGenerator.genRandomIntValue();
        myPlan = userPlan.nestedLoopJoin(
            movieTagPlan.planNode(),
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             "mt_relevance_score"});

        if (numModel == 1) {
          myPlan = myPlan.project(
              {"u_user_id",
               "u_age",
               "u_gender",
               "u_occupation",
               "u_zipcode",
               "u_features",
               "m_movie_id",
               "m_title",
               "m_genres",
               "m_spoken_languages",
               "m_popularity",
               "m_vote_average",
               "m_vote_count",
               "m_features",
               "mt_relevance_score",
               fmt::format(userModel1ComputExpr, "u_features")});
        } else if (numModel == 2) {
          myPlan = myPlan.project(
              {"u_user_id",
               "u_age",
               "u_gender",
               "u_occupation",
               "u_zipcode",
               "u_features",
               "m_movie_id",
               "m_title",
               "m_genres",
               "m_spoken_languages",
               "m_popularity",
               "m_vote_average",
               "m_vote_count",
               "m_features",
               "mt_relevance_score",
               fmt::format(userModel1ComputExpr, "u_features"),
               fmt::format(movieModel1ComputExpr, "m_features")});
        } else if (numModel == 3) {
          myPlan = myPlan.project(
              {"u_user_id",
               "u_age",
               "u_gender",
               "u_occupation",
               "u_zipcode",
               "u_features",
               "m_movie_id",
               "m_title",
               "m_genres",
               "m_spoken_languages",
               "m_popularity",
               "m_vote_average",
               "m_vote_count",
               "m_features",
               "mt_relevance_score",
               fmt::format(userModel1ComputExpr, "u_features"),
               fmt::format(movieModel1ComputExpr, "m_features"),
               fmt::format(tagModel1ComputExpr, "mt_relevance_score")});
        }

        if (generateFilter) {
          std::vector<std::string> filterExpr =
              sampleUserMovieFilterExpr("movie_user", randomSeed);
          for (auto expr : filterExpr) {
            myPlan = myPlan.filter(expr);
          }
          // myPlan = myPlan.filter(filterExpr);
        }

        cataLog.setIdAddressMap(
            readUserDataPlanNodeId,
            userFilePaths,
            dwio::common::FileFormat::DWRF);
        cataLog.setIdAddressMap(
            readMovieDataPlanNodeId,
            movieFilePaths,
            dwio::common::FileFormat::DWRF);
        cataLog.setIdAddressMap(
            readMovieRelevanceTagDataPlanNodeId,
            movieRelevanceTagFilePaths,
            dwio::common::FileFormat::DWRF);

        cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
        cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
        cataLog.addNodeIdRelationName(
            readMovieRelevanceTagDataPlanNodeId, "movie_relevance_tag");
        std::shared_ptr<OutputStat> userStats =
            std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
        Source userSrc =
            Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
        cataLog.addSource(std::make_shared<Source>(userSrc));
        std::shared_ptr<OutputStat> movieStats = std::make_shared<OutputStat>(
            OutputStat(movieNumRows, movieNumCols));
        Source movieSrc =
            Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
        cataLog.addSource(std::make_shared<Source>(movieSrc));
        std::shared_ptr<OutputStat> movieRelevanceTagStats =
            std::make_shared<OutputStat>(
                OutputStat(movieRelevanceTagNumRows, movieRelevanceTagNumCols));
        Source movieRelevanceTagSrc = Source(
            readMovieRelevanceTagDataPlanNodeId,
            Source::Type::FILE,
            movieRelevanceTagStats);
        cataLog.addSource(std::make_shared<Source>(movieRelevanceTagSrc));
      } else {
        throw std::runtime_error(
            fmt::format("Non-supported queryTemplate: {}", queryTemplate));
      }
    } else {
      throw std::runtime_error(fmt::format("Non-supported model: {}", mode));
    }

    return myPlan;
  }

  void generateDummyData(
      std::string mode,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      CataLog& cataLog,
      int dataBatchSize = 256) {
    if (mode == "ml") {
      VectorMaker maker{pool_.get()};
      // it should contains the numbers: # of users, # of movies, # of relevance
      // tag
      assert(numberOfTuples.size() == 3 && dummyFeatureSizes.size() == 2);
      int numUsers = numberOfTuples[0];
      int numMovies = numberOfTuples[1];
      int numRelevanceTags = numberOfTuples[2];
      int userFeatureSize = dummyFeatureSizes[0];
      int movieFeatureSize = dummyFeatureSizes[1];

      std::string tableStatsPath =
          "/home/velox/velox/optimizer/tests/tableStats.txt";
      remove(tableStatsPath.c_str());

      std::vector<std::vector<int>> userModelStructures =
          readModelStructureFromFile(
              "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
      std::vector<std::vector<int>> movieModelStructures =
          readModelStructureFromFile(
              "/home/velox/velox/optimizer/tests/movie_dummy_model_structure.txt");
      std::vector<std::vector<int>> tagModelStructures =
          readModelStructureFromFile(
              "/home/velox/velox/optimizer/tests/tag_dummy_model_structure.txt");
      // Check: dummy feature size equals the input size of the model
      assert(userFeatureSize == userModelStructures[0][0]);
      assert(movieFeatureSize == movieModelStructures[0][0]);
      assert(numRelevanceTags == tagModelStructures[0][0]);

      RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
      RandomSampler randomSampler = RandomSampler(0);
      // sample user data
      std::vector<int> userIDs = randomGenerator.genIntRange(0, numUsers);
      std::vector<std::string> userGender =
          randomSampler.sampleFromSets<std::string>(numUsers, {"M", "F"});
      std::vector<int> userAge = randomGenerator.gen1DInt(numUsers, 10, 70);
      std::vector<int> userOccupation =
          randomGenerator.gen1DInt(numUsers, 0, 20);
      std::vector<std::string> zipCode = {
          "94043",
          "94301",
          "94305",
          "94306",
          "94309",
          "80212",
          "80213",
          "80219",
          "12301",
          "40201"};
      std::vector<std::string> userZipcode =
          randomSampler.sampleFromSets<std::string>(numUsers, zipCode);
      std::vector<std::vector<float>> userFeatures =
          randomGenerator.genFloat2dVector(numUsers, userFeatureSize);

      cataLog.addNumericalColMinMax("u_user_id", 0, numUsers);
      cataLog.addNumericalColMinMax("u_age", 10, 70);
      cataLog.addNumericalColMinMax("u_occupation", 0, 20);
      cataLog.addCategoricalColVals("u_zipcode", zipCode);

      auto userDataRowVector = maker.rowVector(
          {"u_user_id",
           "u_gender",
           "u_age",
           "u_occupation",
           "u_zipcode",
           "u_features"},
          {maker.flatVector<int>(userIDs, INTEGER()),
           maker.flatVector<std::string>(userGender, VARCHAR()),
           maker.flatVector<int>(userAge, INTEGER()),
           maker.flatVector<int>(userOccupation, INTEGER()),
           maker.flatVector<std::string>(userZipcode, VARCHAR()),
           maker.arrayVector<float>(userFeatures, REAL())});

      // output the histogram for the user data
      cataLog.outputHistogramForData(
          userDataRowVector, "user", 50, tableStatsPath);

      MovieTitleGenerator movieTitleGenerator = MovieTitleGenerator(0);
      std::vector<int> movieIDs = randomGenerator.genIntRange(0, numMovies);
      std::vector<std::string> movieTitle =
          movieTitleGenerator.generateBatchRandomTitles(numMovies);
      std::vector<std::string> genres = {
          "Action",
          "Adventure",
          "Animation",
          "Children",
          "Comedy",
          "Crime",
          "Documentary",
          "Drama",
          "Fantasy",
          "Film-Noir",
          "Horror",
          "Musical",
          "Mystery",
          "Romance",
          "Sci-Fi",
          "Thriller",
          "War",
          "Western"};
      std::vector<std::string> movieGenres =
          randomSampler.sampleFromSets<std::string>(numMovies, genres);
      std::vector<std::string> spokenLanguage = {
          "English",
          "French",
          "German",
          "Italian",
          "Japanese",
          "Korean",
          "Mandarin",
          "Spanish"};
      std::vector<std::string> movieSpokenLanguage =
          randomSampler.sampleFromSets<std::string>(numMovies, spokenLanguage);
      randomGenerator.setFloatRange(0, 10);
      std::vector<float> moviePopularity =
          randomGenerator.genFloat1dVector(numMovies);
      randomGenerator.setFloatRange(0, 5);
      std::vector<float> movieVoteAverage =
          randomGenerator.genFloat1dVector(numMovies);
      std::vector<int> movieVoteCount =
          randomGenerator.gen1DInt(numMovies, 0, 5000);
      std::vector<std::vector<float>> movieFeatures =
          randomGenerator.genFloat2dVector(numMovies, movieFeatureSize);

      cataLog.addNumericalColMinMax("m_movie_id", 0, numMovies);
      cataLog.addNumericalColMinMax("m_popularity", 0, 10);
      cataLog.addNumericalColMinMax("m_vote_average", 0, 5);
      cataLog.addNumericalColMinMax("m_vote_count", 0, 5000);
      cataLog.addCategoricalColVals("m_genres", genres);
      cataLog.addCategoricalColVals("m_spoken_languages", spokenLanguage);

      auto movieDataRowVector = maker.rowVector(
          {"m_movie_id",
           "m_title",
           "m_genres",
           "m_spoken_languages",
           "m_popularity",
           "m_vote_average",
           "m_vote_count",
           "m_features"},
          {maker.flatVector<int>(movieIDs, INTEGER()),
           maker.flatVector<std::string>(movieTitle, VARCHAR()),
           maker.flatVector<std::string>(movieGenres, VARCHAR()),
           maker.flatVector<std::string>(movieSpokenLanguage, VARCHAR()),
           maker.flatVector<float>(moviePopularity, REAL()),
           maker.flatVector<float>(movieVoteAverage, REAL()),
           maker.flatVector<int>(movieVoteCount, INTEGER()),
           maker.arrayVector<float>(movieFeatures, REAL())});

      // output the histogram for the movie data
      cataLog.outputHistogramForData(
          movieDataRowVector, "movie", 50, tableStatsPath);

      randomGenerator.setFloatRange(0, 1);
      std::vector<std::vector<float>> movieRelevanceTags =
          randomGenerator.genFloat2dVector(numMovies, numRelevanceTags);

      cataLog.addNumericalColMinMax("mt_movie_id", 0, numMovies);

      auto movieRelevanceTagRowVector = maker.rowVector(
          {"mt_movie_id", "mt_relevance_score"},
          {maker.flatVector<int>(movieIDs, INTEGER()),
           maker.arrayVector<float>(movieRelevanceTags, REAL())});

      // output the histogram for the movie relevance tag data
      cataLog.outputHistogramForData(
          movieRelevanceTagRowVector,
          "movie_relevance_tag",
          50,
          tableStatsPath);

      std::vector<std::shared_ptr<TempFilePath>> userFilePaths =
          splitRowVectorIntoBatchFiles(userDataRowVector, dataBatchSize);
      std::pair<int, int> userStats =
          std::make_pair(numUsers, userDataRowVector->childrenSize());
      cataLog.registerDataSrc(
          "user",
          userFilePaths,
          asRowType(userDataRowVector->type()),
          userStats);

      std::vector<std::shared_ptr<TempFilePath>> movieFilePaths =
          splitRowVectorIntoBatchFiles(movieDataRowVector, dataBatchSize);
      std::pair<int, int> movieStats =
          std::make_pair(numMovies, movieDataRowVector->childrenSize());
      cataLog.registerDataSrc(
          "movie",
          movieFilePaths,
          asRowType(movieDataRowVector->type()),
          movieStats);

      std::vector<std::shared_ptr<TempFilePath>> movieRelevanceTagFilePaths =
          splitRowVectorIntoBatchFiles(
              movieRelevanceTagRowVector, dataBatchSize);
      std::pair<int, int> movieRelevanceTagStats =
          std::make_pair(numMovies, movieRelevanceTagRowVector->childrenSize());
      cataLog.registerDataSrc(
          "movie_relevance_tag",
          movieRelevanceTagFilePaths,
          asRowType(movieRelevanceTagRowVector->type()),
          movieRelevanceTagStats);
    }
  }

  void vanillaMCTS(
      std::string mode,
      std::string queryTemplate,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      int numThreads,
      int repeatRun,
      int verbose,
      bool rewrite,
      int dataBatchSize = 256) {
    PlanBuilder myPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;
    std::string computationStr;

    generateDummyData(
        mode, numberOfTuples, dummyFeatureSizes, cataLog, dataBatchSize);

    if (mode == "ml") {
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

      const char* globalRandomSeedEnv = std::getenv("CD_GLOBAL_RANDOMSEED");
      int globalRandomSeed =
          globalRandomSeedEnv ? std::stoi(globalRandomSeedEnv) : 0;
      myPlan = setupProfileQueryPlan(
          mode, queryTemplate, cataLog, planNodeIdGenerator, globalRandomSeed);

      // } else if (model == "df") {
      // } else if (model == "two-tower") {
      // } else if (model == "llm") {
      // } else if (model == "fraud") {
      // } else if (model == "ml-q1") {
      // } else if (model == "ml-q2") {
      // } else if (model == "ml-q3") {
    } else {
      throw std::runtime_error(fmt::format("Non-supported model: {}", mode));
    }

    std::cout << "[INFO] Original Query Plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;

    float unOptimizedExecutionTime = runPlanWithCataLog(
        pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
    std::cout << "[INFO] Unoptimized Execution time: "
              << unOptimizedExecutionTime << std::endl;

    // Get the logical plan
    auto planNode = myPlan.planNode();

    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    planState.getPossibleActions(planNode, cataLog);

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
    // cache initial query plan and catalog
    int initQueryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
    // send start message to start MCTS optimization
    // start flag and initial query plan
    Json::Value startJsonMessage;
    startJsonMessage["mctsAction"] = "start";
    startJsonMessage["queryPlan"] = planNode->toString(true, true);
    std::cout << "json message: " << startJsonMessage << std::endl;
    sendJsonBySocket(startJsonMessage, clientSocket);
    bool optimizationIsFinished = false;

    while (!optimizationIsFinished) {
      planNode = myPlan.planNode();
      // received json message from MCTS
      Json::Value receivedJsonMessage = receiveJsonFromSocket(clientSocket);
      std::string mctsAction = receivedJsonMessage["mctsAction"].asString();
      LOG(INFO) << "===================================" << std::endl;
      LOG(INFO) << "Received message with mcts action: " << mctsAction
                << std::endl;
      if (mctsAction == "") {
        LOG(INFO) << "Un-captured error happened" << std::endl;
        return;
      }
      LOG(INFO) << "JSON Message: " << receivedJsonMessage << std::endl;
      if (mctsAction == "resetPlan") {
        // if it is root node, it needs to start with original plan
        // the p0 will be increased after capturePlanNodeId is called
        // so it is required to clean the old IdAddressMap and VectorIdMap
        // before reset the myPlan

        resetQueryPlanAndQueryPlanFromCache(
            myPlan, cataLog, initQueryPlanCacheId);
        planNode = myPlan.planNode();
        planState.clearTransformedExpr();
        planState.getPossibleActions(planNode, cataLog);
        // send acknowledgement for synchronization
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getQueryPlan") {
        Json::Value jsonMessage;
        jsonMessage["communicateFlag"] = true;
        jsonMessage["mctsAction"] = "recQueryPlan";
        jsonMessage["queryPlan"] =
            "\"" + myPlan.planNode()->toString(true, true) + "\"";
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "getActionSpace") {
        planState.getPossibleActions(planNode, cataLog);
        Json::Value jsonMessage;
        jsonMessage["actionSpace"] = Json::arrayValue;
        for (const auto& entry : planState.actionsPair) {
          // LOG(INFO) << "[ACTION SPACE] " << entry.first << s't'd
          // entry.second << std::endl;
          Json::Value jsonEntry;
          jsonEntry["expression"] = entry.first;
          jsonEntry["action"] = Json::arrayValue;
          for (auto action : entry.second) {
            jsonEntry["action"].append(Json::Value(action));
          }
          jsonMessage["actionSpace"].append(jsonEntry);
        }
        // cache current state
        int queryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
        jsonMessage["queryPlanCacheId"] = queryPlanCacheId;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "takeAction") {
        std::pair<std::string, std::string> targetAction;
        targetAction.first = receivedJsonMessage["targetString"].asString();
        targetAction.second = receivedJsonMessage["targetAction"].asString();

        LOG(INFO) << "[INFO] take action: " << targetAction << std::endl;
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
        LOG(INFO) << "[INFO] current my query plan"
                  << myPlan.planNode()->toString(true, true) << std::endl;
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "cacheState") {
        int queryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
        Json::Value jsonMessage;
        jsonMessage["queryPlanCacheId"] = queryPlanCacheId;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "resetState") {
        int queryPlanCacheId = receivedJsonMessage["queryPlanCacheId"].asInt();
        resetQueryPlanAndQueryPlanFromCache(myPlan, cataLog, queryPlanCacheId);
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getCost") {
        Json::Value jsonMessage;
        if (receivedJsonMessage["costMode"] == "offline") {
          float executeTime = runPlanWithCataLog(
              pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
          jsonMessage["reward"] = executeTime;
          LOG(INFO) << "[INFO] get Cost(offline): " << " time: " << executeTime
                    << std::endl;
        } else if (receivedJsonMessage["costMode"] == "online") {
          CostModel* cm = new SimpleCostModel(cataLog);
          CostEstimator* ce =
              new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

          planNode = myPlan.planNode();
          CostEstimate cost = ce->estimateCost(planNode);
          jsonMessage["reward"] = cost.cost;
          LOG(INFO) << "[INFO] get Cost(online): " << cost.cost << std::endl;
        }
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);

      } else if (mctsAction == "runPlan") {
        auto latency = runPlanWithCataLog(
            pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
        Json::Value jsonMessage;
        jsonMessage["latency"] = latency;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "finished") {
        // finished
        // nothing to do
      }

      optimizationIsFinished =
          receivedJsonMessage["optimizationIsFinished"].asBool();
      LOG(INFO) << "[INFO] reached end of the loop, current opt flag: "
                << optimizationIsFinished << std::endl;
    };

    return;
  }

  void reusableMCTS(
      std::string mode,
      std::string queryTemplate,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      // int featureSize,
      // int numSamples,
      int numThreads,
      int repeatRun,
      // int blockSize,
      int verbose,
      bool rewrite,
      int dataBatchSize = 256) {
    PlanBuilder myPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;
    std::string computationStr;

    if (mode == "ml") {
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
        generateDummyData(
            mode, numberOfTuples, dummyFeatureSizes, cataLog, dataBatchSize);

        const char* globalRandomSeedEnv = std::getenv("CD_GLOBAL_RANDOMSEED");
        int globalRandomSeed =
            globalRandomSeedEnv ? std::stoi(globalRandomSeedEnv) : 0;
        myPlan = setupProfileQueryPlan(
            mode,
            queryTemplate,
            cataLog,
            planNodeIdGenerator,
            globalRandomSeed);
      }

    } else {
      throw std::runtime_error(fmt::format("Non-supported model: {}", mode));
    }

    std::cout << "[INFO] Original Query Plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;

    outputAugmentedQueryPlan(cataLog, myPlan);
    // float unOptimizedExecutionTime = runPlanWithCataLog(
    //     pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
    // std::cout << "[INFO] Unoptimized Execution time: "
    //           << unOptimizedExecutionTime << std::endl;
    // return;

    /* if (rewrite) {
      myPlan = rewriteQuery(cataLog, pool_, myPlan, planNodeIdGenerator,
    verbose);
    } */

    // std::cout << "[INFO] Executed Query Plan: \n"
    //           << myPlan.planNode()->toString(true, true) << std::endl;
    // auto serializedPlan = myPlan.planNode()->serialize();
    // std::string queryOutPutPath =
    //     "/home/velox/velox/optimizer/tests/serializedQueryPlan.json";
    // augmentSerializedPlan(serializedPlan, cataLog);
    // writeStringToFile(folly::toJson(serializedPlan), queryOutPutPath);

    // float executeTime =
    //     runPlanWithCataLog(numThreads, myPlan, cataLog, repeatRun, verbose);

    // std::string latencyOutPutPath =
    //     "/home/velox/velox/optimizer/tests/executionLatency.txt";
    // writeStringToFile(std::to_string(executeTime), latencyOutPutPath);

    // std::cout << "[INFO] Execution time: " << executeTime << std::endl;

    // std::cout << "Success" << std::endl;

    // myPlan = setupProfileQueryPlan(
    //     model,
    //     computationStr,
    //     inputFilePaths,
    //     inputTempFiles,
    //     numSamples,
    //     featureSize,
    //     cataLog,
    //     planNodeIdGenerator);

    // Get the logical plan
    auto planNode = myPlan.planNode();

    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    planState.getPossibleActions(planNode, cataLog);

    // std::cout << "[INFO] All possible actions:" << std::endl;
    // for (auto entry : planState.actionsPair) {
    //   std::cout << entry.first << ": " << entry.second << std::endl;
    // }

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
    // cache initial query plan and catalog
    int initQueryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
    // send start message to start MCTS optimization
    // start flag and initial query plan
    Json::Value startJsonMessage;
    startJsonMessage["mctsAction"] = "start";
    std::cout << "json message: " << startJsonMessage << std::endl;
    sendJsonBySocket(startJsonMessage, clientSocket);
    bool optimizationIsFinished = false;

    while (!optimizationIsFinished) {
      planNode = myPlan.planNode();
      // received json message from MCTS
      Json::Value receivedJsonMessage = receiveJsonFromSocket(clientSocket);
      std::string mctsAction = receivedJsonMessage["mctsAction"].asString();
      LOG(INFO) << "===================================" << std::endl;
      LOG(INFO) << "Received message with mcts action: " << mctsAction
                << std::endl;
      if (mctsAction == "") {
        LOG(INFO) << "Un-captured error happened" << std::endl;
        return;
      }
      LOG(INFO) << "JSON Message: " << receivedJsonMessage << std::endl;
      if (mctsAction == "resetPlan") {
        // if it is root node, it needs to start with original plan
        // the p0 will be increased after capturePlanNodeId is called
        // so it is required to clean the old IdAddressMap and VectorIdMap
        // before reset the myPlan

        resetQueryPlanAndQueryPlanFromCache(
            myPlan, cataLog, initQueryPlanCacheId);
        planNode = myPlan.planNode();
        planState.clearTransformedExpr();
        planState.getPossibleActions(planNode, cataLog);
        // std::cout << "[INFO] All possible actions:" << std::endl;
        // for (auto entry : planState.actionsPair) {
        //   std::cout << entry.first << ": " << entry.second << std::endl;
        // }
        // send acknowledgement for synchronization
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getQueryPlan") {
        // Dump structured query plan to disk
        // default path:
        // /home/velox/velox/optimizer/tests/structuredQueryPlan.txt Json::Value
        outputAugmentedQueryPlan(cataLog, myPlan);
        outputStructuredQueryPlan(myPlan);
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getActionSpace") {
        planState.getPossibleActions(planNode, cataLog);
        Json::Value jsonMessage;
        jsonMessage["actionSpace"] = Json::arrayValue;
        for (const auto& entry : planState.actionsPair) {
          // LOG(INFO) << "[ACTION SPACE] " << entry.first << s't'd
          // entry.second << std::endl;
          Json::Value jsonEntry;
          jsonEntry["expression"] = entry.first;
          jsonEntry["action"] = Json::arrayValue;
          for (auto action : entry.second) {
            jsonEntry["action"].append(Json::Value(action));
          }
          jsonMessage["actionSpace"].append(jsonEntry);
        }
        // cache current state
        int queryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
        jsonMessage["queryPlanCacheId"] = queryPlanCacheId;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "takeAction") {
        std::string selectedHighLevelOptRule =
            receivedJsonMessage["targetAction"].asString();

        LOG(INFO) << "[INFO] selectedHighLevelOptRule: "
                  << selectedHighLevelOptRule << std::endl;
        if (selectedHighLevelOptRule != "None") {
          // Take action
          std::pair<std::string, std::string> targetAction;
          std::vector<std::string> applicableTargetExprs;

          for (const auto& entry : planState.actionsPair) {
            for (auto action : entry.second) {
              if (action == selectedHighLevelOptRule) {
                // add all applicable target expressions falling into the
                // selected high level optimization rule
                applicableTargetExprs.push_back(entry.first);
              }
            }
          }

          if (applicableTargetExprs.size() == 0) {
            throw std::runtime_error(
                "[INFO] No applicable target expressions found for the "
                "selected high level optimization rule: " +
                selectedHighLevelOptRule);
          }
          // cache current state
          int cachedQueryPlanCacheId =
              cacheQueryPlanAndCateLog(myPlan, cataLog);
          // enumerate plan
          targetAction.second = selectedHighLevelOptRule;

          std::string tempEnumeratePlanPaths =
              "/home/velox/velox/optimizer/tests/_tempOptimization";
          deleteFilesInFolder(tempEnumeratePlanPaths);

          for (int i = 0; i < applicableTargetExprs.size(); i++) {
            targetAction.first = applicableTargetExprs[i];
            planNode = myPlan.planNode();
            planState.takeAction(
                planNode,
                nullptr,
                maker,
                myPlan,
                pool_,
                planNodeIdGenerator,
                {targetAction},
                cataLog);
            std::string queryPlanOutputPath =
                fmt::format("{}/{}.json", tempEnumeratePlanPaths, i);
            outputAugmentedQueryPlan(cataLog, myPlan, queryPlanOutputPath);
            // std::cout << "[debug] Enumerated plan: " << i << std::endl
            //           << "path: " << queryPlanOutputPath << std::endl;
            // reset the plan
            resetQueryPlanAndQueryPlanFromCache(
                myPlan, cataLog, cachedQueryPlanCacheId);
          }

          sendAcknowledgment(clientSocket);
          // wait to receive another message indicating the cost estimation is
          // done
          Json::Value queryEstimationMessage =
              receiveJsonFromSocket(clientSocket);
          int selectedTargetExprIndex =
              queryEstimationMessage["selectedPlanIdx"].asInt();

          auto listPlanIds = queryEstimationMessage["listPlanIds"];
          auto listPlanLatencies = queryEstimationMessage["listPlanLatencies"];
          // reset the plan
          resetQueryPlanAndQueryPlanFromCache(
              myPlan, cataLog, cachedQueryPlanCacheId);

          targetAction.first = applicableTargetExprs[selectedTargetExprIndex];
          planNode = myPlan.planNode();
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
          outputAugmentedQueryPlan(cataLog, myPlan);
        }
        LOG(INFO) << "[INFO] current my query plan"
                  << myPlan.planNode()->toString(true, true) << std::endl;
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "cacheState") {
        int queryPlanCacheId = cacheQueryPlanAndCateLog(myPlan, cataLog);
        Json::Value jsonMessage;
        jsonMessage["queryPlanCacheId"] = queryPlanCacheId;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "resetState") {
        // reset the query plan and catalog to the cached state
        int queryPlanCacheId = receivedJsonMessage["queryPlanCacheId"].asInt();
        resetQueryPlanAndQueryPlanFromCache(myPlan, cataLog, queryPlanCacheId);
        outputAugmentedQueryPlan(cataLog, myPlan);
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getCost") {
        Json::Value jsonMessage;
        if (receivedJsonMessage["costMode"] == "offline") {
          float executeTime = runPlanWithCataLog(
              pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
          jsonMessage["reward"] = executeTime;
          LOG(INFO) << "[INFO] get Cost(offline): " << " time: " << executeTime
                    << std::endl;
        } else if (receivedJsonMessage["costMode"] == "online") {
          CostModel* cm = new SimpleCostModel(cataLog);
          CostEstimator* ce =
              new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

          planNode = myPlan.planNode();
          CostEstimate cost = ce->estimateCost(planNode);
          jsonMessage["reward"] = cost.cost;
          LOG(INFO) << "[INFO] get Cost(online): " << cost.cost << std::endl;
        }
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);

      } else if (mctsAction == "runPlan") {
        auto latency = runPlanWithCataLog(
            pool_, numThreads, myPlan, cataLog, repeatRun, verbose);
        Json::Value jsonMessage;
        jsonMessage["latency"] = latency;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "finished") {
        // finished
        // nothing to do
      }

      optimizationIsFinished =
          receivedJsonMessage["optimizationIsFinished"].asBool();
      LOG(INFO) << "[INFO] reached end of the loop, current opt flag: "
                << optimizationIsFinished << std::endl;
    };

    return;
  }

  void baselineOptimizer(
      std::string mctsType,
      std::string mode,
      std::string queryTemplate,
      std::vector<int> numberOfTuples,
      std::vector<int> dummyFeatureSizes,
      int numThreads,
      int repeatRun,
      int verbose,
      bool rewrite,
      int dataBatchSize = 256) {
    PlanBuilder myPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;
    std::string computationStr;

    if (mode == "ml") {
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
        generateDummyData(
            mode, numberOfTuples, dummyFeatureSizes, cataLog, dataBatchSize);
        const char* globalRandomSeedEnv = std::getenv("CD_GLOBAL_RANDOMSEED");
        int globalRandomSeed =
            globalRandomSeedEnv ? std::stoi(globalRandomSeedEnv) : 0;
        myPlan = setupProfileQueryPlan(
            mode,
            queryTemplate,
            cataLog,
            planNodeIdGenerator,
            globalRandomSeed);
      }

    } else {
      throw std::runtime_error(fmt::format("Non-supported model: {}", mode));
    }

    std::cout << "[INFO] Original Query Plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;

    std::chrono::steady_clock::time_point timeOptimizerStart =
        std::chrono::steady_clock::now();

    if (rewrite) {
      if (mctsType == "arbitrary") {
        arbitraryQueryRewrite(
            cataLog, pool_, myPlan, planNodeIdGenerator, verbose);
      } else if (mctsType == "heuristic") {
        heuristicQueryRewrite(
            cataLog, pool_, myPlan, planNodeIdGenerator, verbose);
      } else {
        throw std::runtime_error(
            fmt::format("Non-supported MCTS type: {}", mctsType));
      }
    }

    std::chrono::steady_clock::time_point timeOptimizerEnd =
        std::chrono::steady_clock::now();

    std::cout << "[INFO] Executed Query Plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;
    float executionTime = runPlanWithCataLog(
        pool_, numThreads, myPlan, cataLog, repeatRun, verbose);

    auto queryOptimizerElapsedTime =
        (std::chrono::duration_cast<std::chrono::microseconds>(
             timeOptimizerEnd - timeOptimizerStart)
             .count()) /
        1000000.0;
    std::cout << "[INFO] Arbitrary Query Optimizer Execution time: "
              << queryOptimizerElapsedTime << std::endl;
    std::cout << "[INFO] Arbitrary Query Optimized Plan Execution time: "
              << executionTime << std::endl;
  }

  void collectMovieLensStats(int numThreads, int repeatRun, int verbose) {
    std::string tableStatsPath =
        "/home/velox/velox/optimizer/tests/tableStats.txt";
    remove(tableStatsPath.c_str());

    PlanBuilder myPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    myPlan =
        setupMovielensDBQuery("user_only", cataLog, pool_, planNodeIdGenerator);
    std::vector<RowVectorPtr> finalResult;
    runPlanWithCataLog(
        pool_,
        numThreads,
        myPlan,
        cataLog,
        finalResult,
        1 /*repeatRun*/,
        verbose,
        true /*copy result*/);
    RowVectorPtr userDataRowVector = mergeRowVectors(finalResult, pool_);

    // output the histogram for the user data
    cataLog.outputHistogramForData(
        userDataRowVector, "user", 50, tableStatsPath);
    cataLog.clearIdAddressMap();

    myPlan = setupMovielensDBQuery(
        "movie_only", cataLog, pool_, planNodeIdGenerator);
    finalResult.clear();
    runPlanWithCataLog(
        pool_,
        numThreads,
        myPlan,
        cataLog,
        finalResult,
        1 /*repeatRun*/,
        verbose,
        true /*copy result*/);
    RowVectorPtr movieDataRowVector = mergeRowVectors(finalResult, pool_);

    // output the histogram for the user data
    cataLog.outputHistogramForData(
        movieDataRowVector, "movie", 50, tableStatsPath);
    cataLog.clearIdAddressMap();

    myPlan = setupMovielensDBQuery(
        "movie_rating_only", cataLog, pool_, planNodeIdGenerator);
    finalResult.clear();
    runPlanWithCataLog(
        pool_,
        numThreads,
        myPlan,
        cataLog,
        finalResult,
        1 /*repeatRun*/,
        verbose,
        true /*copy result*/);
    RowVectorPtr movieRatingDataRowVector = mergeRowVectors(finalResult, pool_);
    cataLog.outputHistogramForData(
        movieRatingDataRowVector, "movie_rating", 50, tableStatsPath);
    cataLog.clearIdAddressMap();

    myPlan = setupMovielensDBQuery(
        "movie_tag_only", cataLog, pool_, planNodeIdGenerator);
    finalResult.clear();
    runPlanWithCataLog(
        pool_,
        numThreads,
        myPlan,
        cataLog,
        finalResult,
        1 /*repeatRun*/,
        verbose,
        true /*copy result*/);
    RowVectorPtr movieTagDataRowVector = mergeRowVectors(finalResult, pool_);
    cataLog.outputHistogramForData(
        movieTagDataRowVector, "movie_relevance_tag", 50, tableStatsPath);

    // std::cout << "userDataRowVector: " << userDataRowVector->toString()
    //           << std::endl;
    // std::cout << "DONE" << std::endl;
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::MemoryManager::getInstance()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};
  std::shared_ptr<TempDirectoryPath> tempDirPath_;

  VectorMaker maker{pool_.get()};
  static inline int queryPlanCacheId_ = 0;
  std::map<int, folly::dynamic> queryPlanCaches_;
  std::map<int, std::map<core::PlanNodeId, std::vector<std::string>>>
      cataLogIdAddressMapCaches_;
  static inline int modelGroupId_ = 0;
};

DEFINE_string(mcts, "reusable", "MCTS type: reusable, vanilla");
DEFINE_string(mode, "ml", "Mode: ml");
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

int main(int argc, char** argv) {
  memory::MemoryManager::initialize({});
  folly::init(&argc, &argv, false);
  std::string mode = FLAGS_mode;
  std::string queryTemplate = FLAGS_query_template;
  std::string model = FLAGS_model;
  std::string mctsType = FLAGS_mcts;

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
  ReusableMCTSTest demo;

  std::vector<int> numberOfTuples;
  std::vector<int> dummyFeatureSizes;

  if (mode == "ml") {
    numberOfTuples.push_back(numUser);
    numberOfTuples.push_back(numMovie);
    numberOfTuples.push_back(numTag);
    dummyFeatureSizes.push_back(userFeatureSize);
    dummyFeatureSizes.push_back(movieFeatureSize);
  } else if (mode == "collect_ml_stats") {
    demo.collectMovieLensStats(numDriver, repeatRun, verbose);
    return 0;
  }

  std::cout << "numberOfTuples: " << numberOfTuples << std::endl;
  std::cout << "dummyFeatureSizes: " << dummyFeatureSizes << std::endl;

  if (mctsType == "vanilla") {
    demo.vanillaMCTS(
        mode,
        queryTemplate,
        numberOfTuples,
        dummyFeatureSizes,
        numDriver,
        repeatRun,
        verbose,
        rewrite,
        dataBatchSize);
  } else if (mctsType == "reusable") {
    demo.reusableMCTS(
        mode,
        queryTemplate,
        numberOfTuples,
        dummyFeatureSizes,
        numDriver,
        repeatRun,
        verbose,
        rewrite,
        dataBatchSize);
  } else if (mctsType == "arbitrary" || mctsType == "heuristic") {
    demo.baselineOptimizer(
        mctsType,
        mode,
        queryTemplate,
        numberOfTuples,
        dummyFeatureSizes,
        numDriver,
        repeatRun,
        verbose,
        rewrite,
        dataBatchSize);
  } else {
    throw std::runtime_error(
        fmt::format("Non-supported MCTS type: {}", mctsType));
  }
}
