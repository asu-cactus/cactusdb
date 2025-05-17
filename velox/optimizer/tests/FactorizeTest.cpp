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
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include "torch/torch.h"
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
#include "velox/optimizer/MLFactorizationRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
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

  void TwoTowerQuery(int numThreads, int repeatRun, int verbose) {
    PlanBuilder myPlan;
    CataLog cataLog;
    Timer timer;

    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // User-Tower

    // user_id
    int userIdNumEmbedding = 6040;

    // gender
    int genderNumEmbedding = 2;

    // age
    int ageNumEmbedding = 7;

    // occupation
    int occupationNumEmbedding = 21;

    int numSamples = 500;

    std::vector<std::vector<int>> userIndicesVector =
        randomGenerator.genLookUpIndices(numSamples, userIdNumEmbedding - 1);
    auto userIndicesArray =
        maker.arrayVector<int>(userIndicesVector, INTEGER());
    auto userIndicesArrayRowVector =
        maker.rowVector({"user_id"}, {userIndicesArray});

    std::vector<std::vector<int>> genderIndicesVector =
        randomGenerator.genLookUpIndices(numSamples, genderNumEmbedding - 1);
    auto genderIndicesArray =
        maker.arrayVector<int>(genderIndicesVector, INTEGER());
    auto genderIndicesArrayRowVector =
        maker.rowVector({"gender"}, {genderIndicesArray});

    std::vector<std::vector<int>> ageIndicesVector =
        randomGenerator.genLookUpIndices(numSamples, ageNumEmbedding - 1);
    auto ageIndicesArray = maker.arrayVector<int>(ageIndicesVector, INTEGER());
    auto ageIndicesArrayRowVector = maker.rowVector({"age"}, {ageIndicesArray});

    std::vector<std::vector<int>> occupationIndicesVector =
        randomGenerator.genLookUpIndices(
            numSamples, occupationNumEmbedding - 1);
    auto occupationIndicesArray =
        maker.arrayVector<int>(occupationIndicesVector, INTEGER());
    auto occupationIndicesArrayRowVector =
        maker.rowVector({"occupation"}, {occupationIndicesArray});

    randomGenerator.setFloatRange(0, 1);
    std::vector<std::vector<float>> userMeanRatingVector =
        randomGenerator.genFloat2dVector(numSamples, 1);
    auto userMeanRatingArray =
        maker.arrayVector<float>(userMeanRatingVector, REAL());

    int movieIdNumEmbedding = 3706;

    // genres
    int genresNumEmbedding = 1000;

    std::vector<std::vector<int>> movieIndicesVector =
        randomGenerator.genLookUpIndices(numSamples, movieIdNumEmbedding - 1);
    auto movieIndicesArray =
        maker.arrayVector<int>(movieIndicesVector, INTEGER());
    auto movieIndicesArrayRowVector =
        maker.rowVector({"movie_id"}, {movieIndicesArray});

    std::vector<std::vector<int>> genresIndicesVector =
        randomGenerator.genLookUpIndices(numSamples, genresNumEmbedding - 1, 6);
    auto genresIndicesArray =
        maker.arrayVector<int>(genresIndicesVector, INTEGER());
    auto genresIndicesArrayRowVector =
        maker.rowVector({"genres"}, {genresIndicesArray});

    randomGenerator.setFloatRange(0, 1);
    std::vector<std::vector<float>> itemMeanRatingVector =
        randomGenerator.genFloat2dVector(numSamples, 1);
    auto itemMeanRatingArray =
        maker.arrayVector<float>(itemMeanRatingVector, REAL());

    auto userRowVector = maker.rowVector(
        {"u_user_id", "u_gender", "u_age", "u_occupation", "u_rating"},
        {userIndicesArray,
         genderIndicesArray,
         ageIndicesArray,
         occupationIndicesArray,
         userMeanRatingArray});

    auto movieRowVector = maker.rowVector(
        {"m_movie_id", "m_genres", "m_rating"},
        {movieIndicesArray, genresIndicesArray, itemMeanRatingArray});

    registerTwoTowerFunc(cataLog, pool_, false /*isVerticalPartition*/);

    PlanNodeId userRowDataPlanNode, movieRowDataPlanNode;
    std::cout << "got here" << std::endl;
    myPlan =
        exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({userRowVector})
            .capturePlanNodeId(userRowDataPlanNode)
            .project({
                "u_user_id",
                "user_id_embedding(user_id_encoder(u_user_id)) as u_user_id_embed",
                "gender_embedding(u_gender) as u_gender",
                "age_embedding(u_age) as u_age",
                "occupation_embedding(u_occupation) as u_occupation",
                "u_rating as u_user_mean_rating",
            })
            .project(
                {"u_user_id",
                 "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features"})
            .nestedLoopJoin(
                exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({movieRowVector})
                    .capturePlanNodeId(movieRowDataPlanNode)
                    .project(
                        {"m_movie_id",
                         "movie_id_embedding(m_movie_id) as m_movie_id_embed",
                         "sequence_pooling(genres_embedding(m_genres)) as m_genres_embed",
                         "m_rating as m_movie_mean_rating"})
                    .project(
                        {"m_movie_id",
                         "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features"})
                    .planNode(),
                {"u_user_id",
                 "m_movie_id",
                 "user_tower_features",
                 "movie_tower_features"})
            .project(
                {"u_user_id",
                 "m_movie_id",
                 "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                 "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
            .project(
                {"u_user_id",
                 "m_movie_id",
                 "cosine_similarity(user_nn_out, movie_nn_out)"});

    std::cout << "got here2" << std::endl;
    printMap(cataLog.getIdAddressMap());

    if (verbose >= 2) {
      std::cout << "[INFO] Plan: \n"
                << myPlan.planNode()->toString(true, true) << std::endl;
    }

    float executionTime = runPlanWithCataLog(
        pool_, numThreads, myPlan, cataLog, repeatRun, verbose);

    std::cout << "[INFO] Execution time: " << executionTime << " seconds"
              << std::endl;

    RuleManager ruleManager;
    ruleManager.rules.clear();
    ruleManager.rules.emplace(
        "MLDecompositionPushdownRewriteAction",
        std::make_shared<MLDecompositionPushdownRewriteAction>());
    // Create planState
    PlanState planState(ruleManager);

    timer.tic();
    planState.getPossibleActions(myPlan.planNode(), cataLog);
    std::cout << "[INFO] Data Flow Analysis time: " << timer.toc() << std::endl;
    planState.showAllActions();

    // // Below is generating the plan
    // std::chrono::steady_clock::time_point begin =
    //     std::chrono::steady_clock::now();
    // auto userPlan = exec::test::PlanBuilder(pool_.get())
    //                     .values({userIndicesArrayRowVector})
    //                     .project({"user_id_embedding(user_id)"})
    //                     .planNode();

    // auto userEmbedding =
    //     exec::test::AssertQueryBuilder(userPlan).copyResults(pool_.get());

    // auto genderPlan = exec::test::PlanBuilder(pool_.get())
    //                       .values({genderIndicesArrayRowVector})
    //                       .project({"gender_embedding(gender)"})
    //                       .planNode();

    // auto genderEmbedding =
    //     exec::test::AssertQueryBuilder(genderPlan).copyResults(pool_.get());

    // auto agePlan = exec::test::PlanBuilder(pool_.get())
    //                    .values({ageIndicesArrayRowVector})
    //                    .project({"age_embedding(age)"})
    //                    .planNode();

    // auto ageEmbedding =
    //     exec::test::AssertQueryBuilder(agePlan).copyResults(pool_.get());

    // auto occupationPlan = exec::test::PlanBuilder(pool_.get())
    //                           .values({occupationIndicesArrayRowVector})
    //                           .project({"occupation_embedding(occupation)"})
    //                           .planNode();

    // auto occupationEmbedding =
    //     exec::test::AssertQueryBuilder(occupationPlan).copyResults(pool_.get());

    // // std::cout << "[INFO] Results: \n" << results->toString() << std::endl;
    // //   std::cout << "[INFO] Results: \n"
    // //             << userEmbedding->toString(0, userEmbedding->size()) <<
    // //             std::endl;

    // auto in1 = maker.rowVector(
    //     {"in1", "in2"},
    //     {userEmbedding->childAt(0), genderEmbedding->childAt(0)});

    // auto concatPlan1 = exec::test::PlanBuilder(pool_.get())
    //                        .values({in1})
    //                        .project({"concat(in1, in2)"})
    //                        .planNode();

    // auto out1 =
    //     exec::test::AssertQueryBuilder(concatPlan1).copyResults(pool_.get());

    // auto in2 = maker.rowVector(
    //     {"in1", "in2"}, {out1->childAt(0), ageEmbedding->childAt(0)});

    // auto concatPlan2 = exec::test::PlanBuilder(pool_.get())
    //                        .values({in2})
    //                        .project({"concat(in1, in2)"})
    //                        .planNode();
    // auto out2 =
    //     exec::test::AssertQueryBuilder(concatPlan2).copyResults(pool_.get());

    // auto in3 = maker.rowVector(
    //     {"in1", "in2"}, {out2->childAt(0), occupationEmbedding->childAt(0)});

    // auto concatPlan3 = exec::test::PlanBuilder(pool_.get())
    //                        .values({in3})
    //                        .project({"concat(in1, in2)"})
    //                        .planNode();

    // auto out3 =
    //     exec::test::AssertQueryBuilder(concatPlan3).copyResults(pool_.get());

    // auto in4 = maker.rowVector(
    //     {"in1", "in2"}, {out3->childAt(0), userMeanRatingArray});

    // auto concatPlan4 = exec::test::PlanBuilder(pool_.get())
    //                        .values({in4})
    //                        .project({"concat(in1, in2) as user_nn_in"})
    //                        .planNode();

    // auto out4 =
    //     exec::test::AssertQueryBuilder(concatPlan4).copyResults(pool_.get());

    // //   std::cout << "[INFO] user DNN input: \n"
    // //             << out4->toString(0, out4->size()) << std::endl;

    // auto userNNPlan =
    //     exec::test::PlanBuilder(pool_.get())
    //         .values({out4})
    //         .project(
    //             {"relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_nn_in))))))))))))
    //             as user_nn_out"})
    //         .planNode();
    // auto userNNOut =
    //     exec::test::AssertQueryBuilder(userNNPlan).copyResults(pool_.get());
    // //   std::cout << "[INFO] user DNN output: \n"
    // //             << userNNOut->toString(0, userNNOut->size()) << std::endl;

    // // Item-Tower

    // auto itemPlan = exec::test::PlanBuilder(pool_.get())
    //                     .values({movieIndicesArrayRowVector})
    //                     .project({"movie_id_embedding(movie_id)"})
    //                     .planNode();

    // auto itemEmbedding =
    //     exec::test::AssertQueryBuilder(itemPlan).copyResults(pool_.get());

    // //   std::cout << "[INFO] genresIndicesArrayRowVector 1: \n"
    // //             << genresIndicesArrayRowVector->toString(
    // //                    0, genresIndicesArrayRowVector->size())
    // //             << std::endl;

    // auto genresPlan =
    //     exec::test::PlanBuilder(pool_.get())
    //         .values({genresIndicesArrayRowVector})
    //         .project({"sequence_pooling(genres_embedding(genres))"})
    //         .planNode();

    // auto genresEmbedding =
    //     exec::test::AssertQueryBuilder(genresPlan).copyResults(pool_.get());

    // auto in2_1 = maker.rowVector(
    //     {"in1", "in2"},
    //     {itemEmbedding->childAt(0), genresEmbedding->childAt(0)});

    // auto concatPlan2_1 = exec::test::PlanBuilder(pool_.get())
    //                          .values({in2_1})
    //                          .project({"concat(in1, in2)"})
    //                          .planNode();

    // auto out2_1 =
    //     exec::test::AssertQueryBuilder(concatPlan2_1).copyResults(pool_.get());

    // auto in2_2 = maker.rowVector(
    //     {"in1", "in2"}, {out2_1->childAt(0), itemMeanRatingArray});

    // auto concatPlan2_2 = exec::test::PlanBuilder(pool_.get())
    //                          .values({in2_2})
    //                          .project({"concat(in1, in2) as item_nn_in"})
    //                          .planNode();
    // auto out2_2 =
    //     exec::test::AssertQueryBuilder(concatPlan2_2).copyResults(pool_.get());

    // //   std::cout << "[INFO] item dnn input: \n"
    // //             << out2_2->toString(0, out2_2->size()) << std::endl;

    // auto itemNNPlan =
    //     exec::test::PlanBuilder(pool_.get())
    //         .values({out2_2})
    //         .project(
    //             {"relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(item_nn_in))))))))))))
    //             as item_nn_out"})
    //         .planNode();
    // auto itemNNOut =
    //     exec::test::AssertQueryBuilder(itemNNPlan).copyResults(pool_.get());
    // //   std::cout << "[INFO] item NN output: \n"
    // //             << itemNNOut->toString(0, itemNNOut->size()) << std::endl;

    // auto finalInputRowVector = maker.rowVector(
    //     {"in1", "in2"}, {userNNOut->childAt(0), itemNNOut->childAt(0)});
    // auto finalStagePlan = exec::test::PlanBuilder(pool_.get())
    //                           .values({finalInputRowVector})
    //                           .project({"cosine_similarity(in1, in2)"})
    //                           .planNode();
    // auto scores =
    //     exec::test::AssertQueryBuilder(finalStagePlan).copyResults(pool_.get());
    // std::chrono::steady_clock::time_point end =
    //     std::chrono::steady_clock::now();
    // std::cout << "Time for Two Tower Model (sec) = "
    //           << (std::chrono::duration_cast<std::chrono::microseconds>(
    //                   end - begin)
    //                   .count()) /
    //         1000000.0
    //           << std::endl;
    // std::cout << "[INFO] final score: \n"
    //           << scores->toString(0, scores->size()) << std::endl;
  }

  void LLMQuery(int numThreads, int repeatRun, int verbose) {
    PlanBuilder myPlan;
    CataLog cataLog;
    Timer timer;
    // Initialize

    std::vector<std::string> userDataPaths =
        getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/user");
    std::vector<std::string> movieDataPaths =
        getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/movie");
    auto userDataRowType =
        ROW({"user_id", "description"}, {INTEGER(), VARCHAR()});

    auto movieDataRowType = ROW({"id", "description"}, {INTEGER(), VARCHAR()});

    core::PlanNodeId readUserDataPlanNodeId;
    core::PlanNodeId readMoviewDataPlanNodeId;
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    exec::registerVectorFunction(
        "chatgpt_server", ChatGPT::signatures(), std::make_unique<ChatGPT>());

    exec::registerVectorFunction(
        "chatgpt_recommender",
        ChatGPTRecommender::signatures(),
        std::make_unique<ChatGPTRecommender>());

    myPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(userDataRowType, {}, "")
            .capturePlanNodeId(readUserDataPlanNodeId)
            .project(
                {"CAST(user_id AS VARCHAR) as user_id",
                 "description AS user_description"})
            .nestedLoopJoin(
                PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(movieDataRowType, {}, "")
                    .capturePlanNodeId(readMoviewDataPlanNodeId)
                    .project(
                        {"CAST(id AS VARCHAR) AS movie_id",
                         "description AS movie_description"})
                    .planNode(),
                {"user_id",
                 "movie_id",
                 "user_description",
                 "movie_description"})
            .project(
                {"user_id",
                 "movie_id",
                 "chatgpt_server(user_description, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description",
                 "chatgpt_server(movie_description, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description"})
            .project(
                {"user_id",
                 "movie_id",
                 "chatgpt_recommender(user_description, movie_description, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.')"});

    RuleManager ruleManager;
    ruleManager.rules.clear();
    ruleManager.rules.emplace(
        "MLDecompositionPushdownRewriteAction",
        std::make_shared<MLDecompositionPushdownRewriteAction>());
    // Create planState
    PlanState planState(ruleManager);

    timer.tic();
    planState.getPossibleActions(myPlan.planNode(), cataLog);
    std::cout << "[INFO] Data Flow Analysis time: " << timer.toc() << std::endl;
    planState.showAllActions();

    if (verbose >= 2) {
      std::cout << "[Query Plan: ]" << myPlan.planNode()->toString(true, true)
                << std::endl;
    }
  }

  void factorizationTest(
      // std::string mode,
      // std::string queryTemplate,
      // std::vector<int> numberOfTuples,
      // std::vector<int> dummyFeatureSizes,
      int numThreads,
      int repeatRun,
      int verbose
      // bool rewrite,
      // int dataBatchSize = 256
  ) {
    PlanBuilder myPlan;
    CataLog cataLog;
    Timer timer;
    timer.tic();
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;

    // load serialized model
    std::string serializedModelPath =
        "/home/velox/temp/test_onnx_serialized.txt";
    std::string modelDataPath = "/home/velox/temp/test_onnx_ffnn.h5";
    std::vector<std::string> serializedGraphExprs =
        readFileByLines(serializedModelPath);
    std::string modelDependDataPath =
        "/home/velox/temp/test_onnx_dependencies.txt";
    std::unordered_map<int, std::vector<int>> modelDependData =
        loadONNXGroupDependencyData(modelDependDataPath);

    // std::cout << "debug: modelDependData: " << std::endl;
    // printMap(modelDependData);

    // maintain a map of modelGroupId to modelExpr, the modelExpr is a pair of
    // left expr and right expr. That is used to easily set the model input
    // the later nodes take intermediate group output will store the whole expr
    // to the left expr.
    std::map<int, std::pair<std::string, std::string>> modelGroupIdToExprMap;
    int groupId = 0;

    for (auto expr : serializedGraphExprs) {
      std::vector<std::string> splittedExpr =
          optimization::splitString(expr, '|');
      std::string leftExpr = splittedExpr[0];
      std::string rightExpr = splittedExpr[1];
      std::string outputName = splittedExpr[2];
      // std::cout << "debug: " << splittedExpr << std::endl;
      std::pair<std::string, std::string> modelExprPair;
      if (rightExpr.find("output") == std::string::npos) {
        // the case of model groups take inputs from the query plan
        modelExprPair = std::make_pair(leftExpr, rightExpr + outputName);
      } else {
        modelExprPair = std::make_pair(leftExpr + rightExpr + outputName, "");
      }
      modelGroupIdToExprMap[groupId++] = modelExprPair;

      std::vector<std::string> parsedSingleExprs;
      std::vector<std::string> matchedExprs;
      // parse the target string into a std::vecotor<DLKernel(string)>
      parseDLExpressions(leftExpr + rightExpr, parsedSingleExprs, matchedExprs);
      // reverse the ordering
      std::reverse(parsedSingleExprs.begin(), parsedSingleExprs.end());

      // std::cout << "debug: parsedSingleExprs" << parsedSingleExprs <<
      // std::endl;
      for (auto mlFuncName : parsedSingleExprs) {
        // std::cout << "debug: mlFuncName: " << mlFuncName << std::endl;
        // std::vector<std::string> splittedMLFunc =
        // optimization::splitString(mlFunc, '-');
        // std::cout << "splittedMLFunc: " << splittedMLFunc << std::endl;
        // std::string mlFuncName = splittedMLFunc[0];
        // std::cout << "mlFuncName: " << mlFuncName << std::endl;
        if (mlFuncName.find("relu") != std::string::npos) {
          optimization::registerVectorFunction(
              "relu",
              Relu::signatures(),
              std::make_unique<Relu>(),
              {},
              true,
              cataLog);
        } else if (mlFuncName.find("mat_mul") != std::string::npos) {
          // checkOrAbort(2, splittedMLFunc.size(), "ParseMLFunc");
          // std::string mlFuncParamName = splittedMLFunc[1];
          // std::cout << "mlFuncParamName: " << mlFuncParamName << std::endl;
          std::string weightName = fmt::format("{}_weight", mlFuncName);

          // load weight matrix and dims
          std::vector<std::vector<float>> matMulWeight =
              loadHDF5Array(modelDataPath, weightName);
          std::vector<std::vector<float>> matMulDims =
              loadHDF5Array(modelDataPath, fmt::format("{}_shape", mlFuncName));
          checkOrAbort(matMulDims.size(), 2, "ReadMatMulDims");

          // register function
          optimization::registerVectorFunction(
              mlFuncName,
              MatrixMultiply::signatures(),
              std::make_unique<MatrixMultiply>(
                  std::move(flattenVectorToPointer(matMulWeight)),
                  matMulDims[0][0],
                  matMulDims[1][0]),
              {},
              true,
              cataLog);
        } else if (mlFuncName.find("mat_add") != std::string::npos) {
          std::string weightName = fmt::format("{}_weight", mlFuncName);

          // load weight matrix and dims
          std::vector<std::vector<float>> matAddWeight =
              loadHDF5Array(modelDataPath, weightName);
          std::vector<std::vector<float>> matAddDims =
              loadHDF5Array(modelDataPath, fmt::format("{}_shape", mlFuncName));
          checkOrAbort(matAddDims.size(), 1, "ReadMatAddDims");

          // register func
          optimization::registerVectorFunction(
              mlFuncName,
              MatrixVectorAddition::signatures(),
              std::make_unique<MatrixVectorAddition>(
                  std::move(flattenVectorToPointer(matAddWeight)),
                  matAddDims[0][0]),
              {},
              true,
              cataLog);
        }

        // 1: bit-vector -> factorization ()
        // 2: query plan statistics : number of rows (data-flow analysis),
        // number of features (data-flow analysis) feature reduction (retrieved
        // from ML Func metadata) Velox card. estimation (hyperloglog)
        //
      }
    }

    // std::cout << "debug: modelGroupIdToExprMap: " << modelGroupIdToExprMap
    // << std::endl;

    // Init query plan
    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    std::vector<int> input1JK = randomGenerator.gen1DInt(5000, 0, 20);
    std::vector<std::vector<float>> input1 =
        randomGenerator.genFloat2dVector(5000, 10);
    std::vector<int> input2JK = randomGenerator.gen1DInt(2000, 0, 20);
    std::vector<std::vector<float>> input2 =
        randomGenerator.genFloat2dVector(2000, 5);

    auto input0Vector = maker.arrayVector<float>(input1, REAL());
    auto input0JKVector = maker.flatVector<int>(input1JK);
    auto input1Vector = maker.arrayVector<float>(input2, REAL());
    auto input1JKVector = maker.flatVector<int>(input2JK);
    auto input0RowVector =
        maker.rowVector({"input0jk", "input0"}, {input0JKVector, input0Vector});
    auto input1RowVector =
        maker.rowVector({"input1jk", "input1"}, {input1JKVector, input1Vector});

    myPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                 .values({input0RowVector})
                 .hashJoin(
                     {"input0jk"},
                     {"input1jk"},
                     PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({input1RowVector})
                         .planNode(),
                     "",
                     {"input0jk", "input0", "input1jk", "input1"});

    std::vector<std::string> queryProjectionExprs;
    for (int i = 0; i < modelGroupIdToExprMap.size(); i++) {
      std::string computationExpr;
      auto it = modelDependData.find(i);
      if (it != modelDependData.end()) {
        // std::cout << "got here1: " << queryProjectionExprs << std::endl;
        myPlan.project(queryProjectionExprs);
        // std::cout << "got here2" << std::endl;
        queryProjectionExprs.clear();
        computationExpr = modelGroupIdToExprMap[i].first;

      } else {
        computationExpr = fmt::format(
            "{}input{},input{}{}",
            modelGroupIdToExprMap[i].first,
            i,
            i + 1,
            modelGroupIdToExprMap[i].second);
      }
      queryProjectionExprs.push_back(computationExpr);
    }
    if (queryProjectionExprs.size() > 0) {
      myPlan.project(queryProjectionExprs);
    }

    std::cout << "[DEBUG] model loading + query init: " << timer.toc()
              << std::endl;

    // Add model into query plan:
    if (verbose >= 2) {
      std::cout << "[Query Plan: ]" << myPlan.planNode()->toString(true, true)
                << std::endl;
    }

    float unOptimizedExecutionTime = runPlanWithCataLog(
        pool_, numThreads, myPlan, cataLog, repeatRun, verbose);

    std::cout << "[INFO] Total Execution time: " << unOptimizedExecutionTime
              << std::endl;

    RuleManager ruleManager;
    ruleManager.rules.clear();
    ruleManager.rules.emplace(
        "MLFactorizationRewriteAction",
        std::make_shared<MLFactorizationRewriteAction>());
    // Create planState
    PlanState planState(ruleManager);

    timer.tic();
    planState.getPossibleActions(myPlan.planNode(), cataLog);
    std::cout << "[INFO] Data Flow Analysis time: " << timer.toc() << std::endl;
    planState.showAllActions();

    std::cout << "[Debug]: factorizableOpSrcMap" << std::endl;
    printMap(cataLog.getfactorizableOpSrcMap());
    std::cout << "[Debug]: FactorizableSrcPushdownNodesMap" << std::endl;
    printMap(cataLog.getFactorizableSrcPushdownNodesMap());
    // std::cout << printUnorderedMap(cataLog.getfactorizableOpSrcMap()) <<
    // std::endl;

    auto sketchPlan1 =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({input0RowVector})
            .rowNumber({}, std::nullopt, true)
            .singleAggregation({}, {"approx_set(row_number, 0.01)"})
            .project({"cardinality(a0)"});

    auto sketchTime1 = runPlanWithCataLog(
        pool_, numThreads, sketchPlan1, cataLog, repeatRun, verbose);
    std::cout << "[INFO] Sketch1 Execution time: " << sketchTime1 << std::endl;

    auto sketchPlan2 =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({input1RowVector})
            .rowNumber({}, std::nullopt, true)
            .singleAggregation({}, {"approx_set(row_number, 0.01)"})
            .project({"cardinality(a0)"});

    auto sketchTime2 = runPlanWithCataLog(
        pool_, numThreads, sketchPlan2, cataLog, repeatRun, verbose);
    std::cout << "[INFO] Sketch2 Execution time: " << sketchTime2 << std::endl;

    auto sketchPlan3 =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({input0RowVector})
            .hashJoin(
                {"input0jk"},
                {"input1jk"},
                PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({input1RowVector})
                    .planNode(),
                "",
                {"input0jk", "input0", "input1jk", "input1"})
            .rowNumber({}, std::nullopt, true)
            .singleAggregation({}, {"approx_set(row_number, 0.01)"})
            .project({"cardinality(a0)"});
    auto sketchTime3 = runPlanWithCataLog(
        pool_, numThreads, sketchPlan3, cataLog, repeatRun, verbose);
    std::cout << "[INFO] Sketch3 Execution time: " << sketchTime3 << std::endl;

    return;
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

DEFINE_string(query, "ffnn", "ffnn, llm");
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
  std::string query = FLAGS_query;
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

  if (query == "ffnn") {
    demo.factorizationTest(numDriver, repeatRun, verbose);
  } else if (query == "llm") {
    demo.LLMQuery(numDriver, repeatRun, verbose);
  } else if (query == "two_tower") {
    demo.TwoTowerQuery(numDriver, repeatRun, verbose);
  } else {
    std::cerr << "Invalid query type: " << query << std::endl;
    return 1;
  }
}
