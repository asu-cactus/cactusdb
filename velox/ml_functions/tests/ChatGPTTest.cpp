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
#include <folly/init/Init.h>
#include "velox/common/base/Fs.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/QueryPlanner.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

std::vector<std::string> getFilePathsFromDir(const std::string& dirPath) {
  std::vector<std::string> filePaths;
  for (auto const& dirEntry : fs::directory_iterator(dirPath)) {
    if (!dirEntry.is_regular_file()) {
      continue;
    }
    // Ignore hidden files.
    if (dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    // auto dataFile = CustomTempFilePath::create(dirEntry.path());
    filePaths.push_back(dirEntry.path());
  }
  return filePaths;
}

class ChatGPTTest : public HiveConnectorTestBase {
 public:
  ChatGPTTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);
  }

  ~ChatGPTTest() {}

  void simpleTest();
  void LLMWithoutOptimization();
  void LLMWithOptimization();

  void SetUp() override {
    // TODO: not used for now
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};

  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};
};

void ChatGPTTest::simpleTest() {
  std::vector<std::string> inputVectors;
  inputVectors.push_back("give me a review of toothpaste");
  inputVectors.push_back("give me a review of toothbrush");
  auto inputFlatVector = maker.flatVector<std::string>(inputVectors);

  auto inputRowVector = maker.rowVector({"x"}, {inputFlatVector});

  exec::registerVectorFunction(
      "chatgpt_server", ChatGPT::signatures(), std::make_unique<ChatGPT>());

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"chatgpt_server(x)"})
                    .planNode();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for ChatGPT Simple Test (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void ChatGPTTest::LLMWithoutOptimization() {
  std::vector<std::string> userDataPaths =
      getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/user");
  std::vector<std::string> movieDataPaths =
      getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/movie");

  // auto userDataRowType =
  //     ROW({"UserID",  "Action",          "Adventure", "Animation", "Comedy",
  //          "Crime",   "Documentary",     "Drama",     "Family",    "Fantasy",
  //          "Foreign", "History",         "Horror",    "Music",     "Mystery",
  //          "Romance", "Science Fiction", "TV Movie",  "Thriller",  "War",
  //          "Western", "description"},
  //         {INTEGER(), REAL(), REAL(), REAL(), REAL(), REAL(),   REAL(),
  //         REAL(),
  //          REAL(),    REAL(), REAL(), REAL(), REAL(), REAL(),   REAL(),
  //          REAL(), REAL(),    REAL(), REAL(), REAL(), REAL(), VARCHAR()});
  auto userDataRowType =
      ROW({"user_id", "description"}, {INTEGER(), VARCHAR()});

  auto movieDataRowType =
      ROW({"id",
           //  "title",
           //  "genres",
           //  "spoken_languages",
           //  "popularity",
           //  "vote_average",
           //  "vote_count",
           //  "overview",
           "description"},
          {INTEGER(),
           //  VARCHAR(),
           //  VARCHAR(),
           //  VARCHAR(),
           //  REAL(),
           //  REAL(),
           //  INTEGER(),
           //  VARCHAR(),
           VARCHAR()});

  core::PlanNodeId readUserDataPlanNodeId;
  core::PlanNodeId readMoviewDataPlanNodeId;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  exec::registerVectorFunction(
      "chatgpt_server", ChatGPT::signatures(), std::make_unique<ChatGPT>());

  exec::registerVectorFunction(
      "chatgpt_recommender",
      ChatGPTRecommender::signatures(),
      std::make_unique<ChatGPTRecommender>());

  auto myPlan =
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
              {"user_id", "movie_id", "user_description", "movie_description"})
          .project(
              {"user_id",
               "movie_id",
               "CONCAT(user_id, user_description) AS user_description_processed",
               "CONCAT(movie_id, movie_description) AS movie_description_processed"})
          .project(
              {"user_id",
               "movie_id",
               "chatgpt_server(user_description_processed, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description",
               "chatgpt_server(movie_description_processed, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description"})
          .project(
              {"user_id",
               "movie_id",
               "chatgpt_recommender(user_description, movie_description, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.')"});

  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  // std::cout << "[DEBUG] QueryPlan: \n" << myPlan.planNode()->toString(true,
  // true) << std::endl;

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  CursorParameters params;
  params.maxDrivers = 8;
  params.planNode = myPlan.planNode();
  params.queryCtx = queryCtx_;
  bool noMoreSplits = false;

  auto addSplits = [&](exec::Task* task) {
    std::vector<core::PlanNodeId> ids;
    if (!noMoreSplits) {
      auto userHiveSplits = makeHiveConnectorSplits(
          userDataPaths, dwio::common::FileFormat::PARQUET);
      for (auto& split : userHiveSplits) {
        task->addSplit(readUserDataPlanNodeId, exec::Split(std::move(split)));
      }
      ids.push_back(readUserDataPlanNodeId);

      auto movieHiveSplits = makeHiveConnectorSplits(
          movieDataPaths, dwio::common::FileFormat::PARQUET);
      for (auto& split : movieHiveSplits) {
        task->addSplit(readMoviewDataPlanNodeId, exec::Split(std::move(split)));
      }
      ids.push_back(readMoviewDataPlanNodeId);

      for (auto id : ids) {
        task->noMoreSplits(id);
      }
    }
    noMoreSplits = true;
  };

  auto [cursor, actualResults] = readCursor(params, addSplits);
  waitForTaskCompletion(cursor->task().get());

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for ChatGPT w/o Optimization Test (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  int dataIdx = 0;
  int totalDataNum = 0;
  int verbose = 2;
  for (auto batchedData : actualResults) {
    batchedData = std::move(batchedData);
    int batchSize = batchedData->size();
    if (verbose == 2) {
      std::cout << fmt::format(
                       "[INFO] Batched Data: {}, Batch Size:{} \n",
                       dataIdx,
                       batchSize)
                << batchedData->toString() << std::endl;
    } else if (verbose == 3) {
      std::cout << fmt::format(
                       "[INFO] Batched Data: {}, Batch Size:{} \n",
                       dataIdx,
                       batchSize)
                << batchedData->toString() << "\n"
                << batchedData->toString(0, batchedData->size()) << std::endl;
    }
    dataIdx += 1;
    totalDataNum += batchSize;
  }

  std::cout << fmt::format(
                   "[INFO] Total # of Batch: {}, Total # of Data: {}",
                   dataIdx,
                   totalDataNum)
            << std::endl;
}

void ChatGPTTest::LLMWithOptimization() {
  std::vector<std::string> userDataPaths =
      getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/user");
  std::vector<std::string> movieDataPaths =
      getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/movie");

  // auto userDataRowType =
  //     ROW({"UserID",  "Action",          "Adventure", "Animation", "Comedy",
  //          "Crime",   "Documentary",     "Drama",     "Family",    "Fantasy",
  //          "Foreign", "History",         "Horror",    "Music",     "Mystery",
  //          "Romance", "Science Fiction", "TV Movie",  "Thriller",  "War",
  //          "Western", "description"},
  //         {INTEGER(), REAL(), REAL(), REAL(), REAL(), REAL(),   REAL(),
  //         REAL(),
  //          REAL(),    REAL(), REAL(), REAL(), REAL(), REAL(),   REAL(),
  //          REAL(), REAL(),    REAL(), REAL(), REAL(), REAL(), VARCHAR()});
  auto userDataRowType =
      ROW({"user_id", "description"}, {INTEGER(), VARCHAR()});

  auto movieDataRowType =
      ROW({"id",
           //  "title",
           //  "genres",
           //  "spoken_languages",
           //  "popularity",
           //  "vote_average",
           //  "vote_count",
           //  "overview",
           "description"},
          {INTEGER(),
           //  VARCHAR(),
           //  VARCHAR(),
           //  VARCHAR(),
           //  REAL(),
           //  REAL(),
           //  INTEGER(),
           //  VARCHAR(),
           VARCHAR()});

  core::PlanNodeId readUserDataPlanNodeId;
  core::PlanNodeId readMoviewDataPlanNodeId;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  exec::registerVectorFunction(
      "chatgpt_server", ChatGPT::signatures(), std::make_unique<ChatGPT>());
  exec::registerVectorFunction(
      "chatgpt_recommender",
      ChatGPTRecommender::signatures(),
      std::make_unique<ChatGPTRecommender>());

  auto myPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(userDataRowType, {}, "")
          .capturePlanNodeId(readUserDataPlanNodeId)
          .project(
              {"CAST(user_id AS VARCHAR) AS user_id",
               "description as user_description"})
          .project(
              {"user_id",
               "CONCAT(user_id, user_description) AS user_description_processed"})
          .project(
              {"user_id",
               "chatgpt_server(user_description_processed, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description"})
          .nestedLoopJoin(
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(movieDataRowType, {}, "")
                  .capturePlanNodeId(readMoviewDataPlanNodeId)
                  .project(
                      {"CAST(id AS VARCHAR) AS movie_id",
                       "description AS movie_description"})
                  .project(
                      {"movie_id",
                       "CONCAT(movie_id, movie_description) AS movie_description_processed"})
                  .project(
                      {"movie_id",
                       "chatgpt_server(movie_description_processed, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description"})
                  .planNode(),
              {"user_id", "movie_id", "user_description", "movie_description"})
          .project(
              {"user_id",
               "movie_id",
               "CONCAT(user_id, user_description) as user_description",
               "CONCAT(movie_id, movie_description) AS movie_description"})
          .project(
              {"user_id",
               "movie_id",
               "chatgpt_recommender(user_description, movie_description, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.')"});

  // std::cout << "[DEBUG] QueryPlan: \n" << myPlan.planNode()->toString(true,
  // true) << std::endl;

  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  CursorParameters params;
  params.maxDrivers = 8;
  params.planNode = myPlan.planNode();
  params.queryCtx = queryCtx_;
  bool noMoreSplits = false;

  auto addSplits = [&](exec::Task* task) {
    std::vector<core::PlanNodeId> ids;
    if (!noMoreSplits) {
      auto userHiveSplits = makeHiveConnectorSplits(
          userDataPaths, dwio::common::FileFormat::PARQUET);
      for (auto& split : userHiveSplits) {
        task->addSplit(readUserDataPlanNodeId, exec::Split(std::move(split)));
      }
      ids.push_back(readUserDataPlanNodeId);

      auto movieHiveSplits = makeHiveConnectorSplits(
          movieDataPaths, dwio::common::FileFormat::PARQUET);
      for (auto& split : movieHiveSplits) {
        task->addSplit(readMoviewDataPlanNodeId, exec::Split(std::move(split)));
      }
      ids.push_back(readMoviewDataPlanNodeId);

      for (auto id : ids) {
        task->noMoreSplits(id);
      }
    }
    noMoreSplits = true;
  };

  auto [cursor, actualResults] = readCursor(params, addSplits);
  waitForTaskCompletion(cursor->task().get());

  // auto results =
  //     exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for ChatGPT w/ Optimization Test (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  int dataIdx = 0;
  int totalDataNum = 0;
  int verbose = 2;
  for (auto batchedData : actualResults) {
    batchedData = std::move(batchedData);
    int batchSize = batchedData->size();
    if (verbose == 2) {
      std::cout << fmt::format(
                       "[INFO] Batched Data: {}, Batch Size:{} \n",
                       dataIdx,
                       batchSize)
                << batchedData->toString() << std::endl;
    } else if (verbose == 3) {
      std::cout << fmt::format(
                       "[INFO] Batched Data: {}, Batch Size:{} \n",
                       dataIdx,
                       batchSize)
                << batchedData->toString() << "\n"
                << batchedData->toString(0, batchedData->size()) << std::endl;
    }
    dataIdx += 1;
    totalDataNum += batchSize;
  }

  std::cout << fmt::format(
                   "[INFO] Total # of Batch: {}, Total # of Data: {}",
                   dataIdx,
                   totalDataNum)
            << std::endl;
}

/*
Please use export OPENAI_API_KEY=********  to set-up your openai key before
running the test and modify the corresponding data path
*/
int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  ChatGPTTest demo;
  // demo.simpleTest();
  demo.LLMWithOptimization();
  demo.LLMWithoutOptimization();
}
