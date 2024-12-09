#pragma once
#include <iostream>
#include <string>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/ml_functions/tests/MLTestUtility.h"

#define BUFFER_SIZE 1024

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

facebook::velox::RowVectorPtr copyRowVector(
    const facebook::velox::RowVectorPtr& rowVector,
    std::shared_ptr<memory::MemoryPool>& pool) {
  // Vector to hold the deep copies of child vectors.
  std::vector<facebook::velox::VectorPtr> childCopies;

  // Iterate through the children of the RowVector.
  for (const auto& child : rowVector->children()) {
    if (child) {
      // Allocate a new vector and copy the data into it.
      auto childCopy = facebook::velox::BaseVector::create(
          child->type(), child->size(), pool.get());

      // Copy the data from the source vector to the new vector.
      childCopy->copy(child.get(), 0, 0, child->size());

      childCopies.push_back(childCopy);
    } else {
      // If the child is null, push a null vector.
      childCopies.push_back(nullptr);
    }
  }

  // Create a new RowVector with the copied children.
  return std::make_shared<facebook::velox::RowVector>(
      pool.get(), // Memory pool for allocation.
      rowVector->type(), // The RowType of the original RowVector.
      nullptr, // Nulls buffer (nullptr for no nulls).
      rowVector->size(), // Number of rows in the RowVector.
      std::move(childCopies)); // Null count (if available).
};

/**
 * @brief A function to run logical plan.
 *  
 * @param pool The memory pool for the Velox executor.
 * @param numThreads The number of Velox executor threads.
 * @param numSplits The number of file splits.
 * @param myPlan The pointer to the planBuilder which builds the logical plan.
 * @param cataLog A class storing metadata and information related to UDFs and
 * data sources.
 */
float runPlanWithCataLog(
    std::shared_ptr<memory::MemoryPool>& pool,
    int numThreads,
    PlanBuilder& myPlan,
    CataLog& cataLog,
    std::vector<RowVectorPtr>& finalResult,
    int repeatRun = 1,
    int verbose = 1,
    bool copyResult = false) {
  float totalElapsedTime = 0;
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
        {{core::QueryConfig::kPreferredOutputBatchBytes, "10000000"},
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
    auto addSplits = [&noMoreSplits, &cataLog](exec::Task* task) {
      auto idFileAddrMap = cataLog.getIdAddressMap();
      std::vector<core::PlanNodeId> ids;
      if (!noMoreSplits) {
        for (const auto& entry : idFileAddrMap) {
          core::PlanNodeId key = entry.first;
          const std::vector<std::string> fileAddr = entry.second;
          // check file exists
          for (const auto& addr : fileAddr) {
            if (!fs::exists(addr)) {
              LOG(ERROR) << "[ERROR] File not exists: " << addr << std::endl;
              return;
            }
          }
          auto fileFormat = cataLog.getIdFileFormat(key);
          auto hiveSplits = HiveConnectorTestBase::makeHiveConnectorSplits(
              fileAddr, fileFormat);

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
      dataIdx = 0;
      totalDataNum = 0;
      for (auto batchedData : actualResults) {
        int batchSize = batchedData->size();
        if (verbose == 3) {
          std::cout << fmt::format(
                           "[INFO] Batched Data: {}, Batch Size:{} \n",
                           dataIdx,
                           batchSize)
                    << batchedData->toString() << std::endl;
        } else if (verbose == 4) {
          std::cout << fmt::format(
                           "[INFO] Batched Data: {}, Batch Size:{} \n",
                           dataIdx,
                           batchSize)
                    << batchedData->toString() << "\n"
                    << batchedData->toString(0, batchedData->size())
                    << std::endl;
        }
        dataIdx += 1;
        totalDataNum += batchSize;
        // finalResult.push_back(batchedData);
        if (copyResult) {
          finalResult.push_back(copyRowVector(batchedData, pool));
        }
      }
    }
  }
  if (verbose >= 1) {
    std::cout << fmt::format(
                     "[INFO] Total # of Batch: {}, Total # of Data: {}",
                     dataIdx,
                     totalDataNum)
              << std::endl;
  }

  return totalElapsedTime / repeatRun;
}

float runPlanWithCataLog(
    std::shared_ptr<memory::MemoryPool>& pool,
    int numThreads,
    PlanBuilder& myPlan,
    CataLog& cataLog,
    int repeatRun = 1,
    int verbose = 1) {
  std::vector<RowVectorPtr> finalResult;
  return runPlanWithCataLog(
      pool,
      numThreads,
      myPlan,
      cataLog,
      finalResult,
      repeatRun,
      verbose,
      false /* copyResult */);
}

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
  const char* ack_message = "ACK";
  send(clientSocket, ack_message, strlen(ack_message), 0);
}

std::vector<std::vector<float>> loadHDF5Array(
    const std::string& filename,
    const std::string& datasetName) {
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File not found: " + filename);
  }
  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  // Get the number of dimensions
  int rank = dataspace.getSimpleExtentNdims();
  // std::cout << "Rank: " << rank << std::endl;

  // Allocate space for the dimensions
  std::vector<hsize_t> dims(rank);

  // Get the dataset dimensions
  dataspace.getSimpleExtentDims(dims.data(), nullptr);

  size_t rows;
  size_t cols;

  if (rank == 1) {
    rows = dims[0];
    cols = 1;
  } else if (rank == 2) {
    rows = dims[0];
    cols = dims[1];
  } else {
    throw std::runtime_error("Unsupported rank: " + std::to_string(rank));
  }

  // Read data into a 1D vector
  std::vector<float> flatData(rows * cols);
  dataset.read(flatData.data(), H5::PredType::NATIVE_FLOAT);

  // Convert to 2D vector
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result[i][j] = flatData[i * cols + j];
    }
  }

  // Close the dataset and file
  dataset.close();
  file.close();

  return result;
};

PlanBuilder setupMovielensDBQuery(
    std::string queryType,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
  std::string queryOptType =
      getEnvVar("CD_VELOX_QUERY_OPT_TYPE"); // env used for ablation study of
                                            // rewrite rules
  PlanBuilder queryPlan;

  auto movieTagDataRowType =
      ROW({"mt_movie_id", "mt_relevance_score"}, {INTEGER(), ARRAY(REAL())});
  auto movieDataRowType =
      ROW({"m_movie_id",
           "m_title",
           "m_genres",
           "m_spoken_languages",
           "m_popularity",
           "m_vote_average",
           "m_vote_count",
           "m_overview"},
          {INTEGER(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           REAL(),
           REAL(),
           INTEGER(),
           VARCHAR()});
  auto userDataRowType =
      ROW({"u_user_id", "u_gender", "u_age", "u_occupation", "u_zipcode"},
          {INTEGER(), VARCHAR(), INTEGER(), INTEGER(), VARCHAR()});
  auto ratingDataRowType =
      ROW({"r_user_id", "r_movie_id", "r_rating", "r_timestamp"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()});

  std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

  if (dataDirPrefix == "") {
    // use default value:
    dataDirPrefix = "/home/velox/resources/data/parquet/movielens/final/";
  }

  std::vector<std::string> movieTagDataPaths =
      getFilePathsFromDir(dataDirPrefix + "movie_tag_relevance");
  std::vector<std::string> movieDataPaths =
      getFilePathsFromDir(dataDirPrefix + "movie");
  std::vector<std::string> userDataPaths =
      getFilePathsFromDir(dataDirPrefix + "user");
  std::vector<std::string> ratingDataPaths =
      getFilePathsFromDir(dataDirPrefix + "rating");

  int movieTagNumRows, movieTagNumCols, movieNumRows, movieNumCols, userNumRows,
      userNumCols, ratingNumRows, ratingNumCols;

  readDataStats(
      dataDirPrefix + "movie_tag_relevance_stats.txt",
      movieTagNumRows,
      movieTagNumCols);
  readDataStats(dataDirPrefix + "movie_stats.txt", movieNumRows, movieNumCols);
  readDataStats(dataDirPrefix + "user_stats.txt", userNumRows, userNumCols);
  readDataStats(
      dataDirPrefix + "rating_stats.txt", ratingNumRows, ratingNumCols);
  if (queryType.find("user_only") != std::string::npos) {
    PlanNodeId readUserDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(userDataRowType, {}, "")
                    .capturePlanNodeId(readUserDataPlanNodeId)
                    .project(
                        {"u_user_id",
                         "u_gender",
                         "u_age",
                         "u_occupation",
                         "u_zipcode"});
    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("movie_only") != std::string::npos) {
    PlanNodeId readMovieDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
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
                         "m_overview"});
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("movie_rating_only") != std::string::npos) {
    PlanNodeId readRatingDataPlanNodeId;
    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(ratingDataRowType, {}, "")
            .capturePlanNodeId(readRatingDataPlanNodeId)
            .project({"r_user_id", "r_movie_id", "r_rating", "r_timestamp"});
    cataLog.setIdAddressMap(
        readRatingDataPlanNodeId,
        ratingDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("movie_tag_only") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(movieTagDataRowType, {}, "")
                    .capturePlanNodeId(readMovieTagDataPlanNodeId)
                    .project({"mt_movie_id", "mt_relevance_score"});
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("q1") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    PlanNodeId readRatingDataPlanNodeId1;
    PlanNodeId readRatingDataPlanNodeId2;
    if (queryOptType.empty() || queryOptType == "" ||
        queryOptType == "fusion") {
      // default query
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "movie_description_array"});

      queryPlan =
          readUserAvgRatingPlan
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.planNode(),
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "movie_description_array"})
              .filter("m_genres LIKE '\%Action\%'")
              .project(
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres, m_movie_mean_rating) as movie_tower_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
    } else if (queryOptType.find("ffnn_pushdown") != std::string::npos) {
      // ffnn pushdown
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .project(
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .filter("m_genres LIKE '\%Action\%'")
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating"});

      queryPlan =
          readUserAvgRatingPlan
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.planNode(),
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres, m_movie_mean_rating) as movie_tower_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});

    } else if (
        queryOptType.find("ffnn_pushdown_n_reorder") != std::string::npos) {
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .filter("m_genres LIKE '\%Action\%'")
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .project(
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating"});

      queryPlan =
          readUserAvgRatingPlan
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.planNode(),
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres, m_movie_mean_rating) as movie_tower_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
    } else if (
        queryOptType.find("decomposition_pushdown") != std::string::npos) {
      // optimized query
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "movie_description_array"});

      queryPlan =
          readUserAvgRatingPlan
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   //  "user_id_embedding(user_id_encoder(array_constructor(u_user_id)))
                   //  as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features"})
              .project(
                  {"u_user_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out"})
              // .orderBy({"u_user_id"}, false)
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.orderBy({"m_movie_id"}, false)
                      .project(
                          {"m_movie_id",
                           "movie_description_array",
                           "m_genres AS m_genres1",
                           "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                           "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres",
                           "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
                      .project({
                          "m_movie_id",
                          "movie_description_array",
                          "m_genres1",
                          "concat(m_movie_id_embed, m_genres, m_movie_mean_rating) as movie_tower_features",
                      })
                      .project(
                          {"m_movie_id",
                           "movie_description_array",
                           "m_genres1",
                           "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
                      .planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "user_nn_out",
                   "movie_nn_out",
                   "m_genres1",
                   "movie_description_array"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "user_nn_out",
                   "movie_nn_out",
                   "m_genres1",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .filter("m_genres1 LIKE '\%Action\%'")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)",
                   "trending_prediction",
                   "m_genres1"});

    } else if (queryOptType.find("optimized") != std::string::npos) {
      // optimized query
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .filter("m_genres LIKE '\%Action\%'")
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction",
              })
              .filter("trending_prediction = 1")
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating"});

      queryPlan =
          readUserAvgRatingPlan
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features"})
              .project(
                  {"u_user_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out"})
              .nestedLoopJoin(
                  readMovieAvgRatingPlan
                      // .orderBy({"m_movie_id"}, false)
                      .project(
                          {"m_movie_id",
                           "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                           "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres",
                           "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
                      .project({
                          "m_movie_id",
                          "concat(m_movie_id_embed, m_genres, m_movie_mean_rating) as movie_tower_features",
                      })
                      .project(
                          {"m_movie_id",
                           "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
                      .planNode(),
                  {"u_user_id", "m_movie_id", "user_nn_out", "movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
    } else {
      // NON-SUPPORTED
      throw std::invalid_argument(
          "Unsupported query optimization type: " + queryOptType);
    }

    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readRatingDataPlanNodeId1,
        ratingDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readRatingDataPlanNodeId2,
        ratingDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("q2") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    if (queryOptType.empty() || queryOptType == "" ||
        queryOptType == "mlq2-fusion" || queryOptType == "mlq2-mul2join") {
      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .project(
                          {"m_movie_id",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count"})
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 5, false)
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "u_user_id",
                  "u_age",
                  "u_occupation",
                  "gender_encoder(u_gender) as u_gender_encoded",
                  "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_score",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features",
                  "llm_ffnn_interest_scaler(transform(array_constructor(u_age, u_occupation), x-> CAST(X as REAL)))  AS u_interest_features",
              })
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "m_trending_features",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                   "u_final_interest_features",
                   "mt_relevance_score"})
              .filter("trending_prediction = 1")
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "mt_relevance_score"})
              .filter("user_interest_prediction = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                   "gender_embedding(u_gender_encoded) as u_gender_embed",
                   "mt_relevance_score"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out",
                   "concat(u_age_embed, u_occupation_embed, u_gender_embed) as categorical_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "concat(bottom_mlp_out, categorical_features) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"});
    }
    if (queryOptType == "decomposition_pushdown") {
      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .project(
                          {"m_movie_id",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count"})
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
              })
              .project({
                  "m_movie_id",
                  "mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 5, false)
                      .project(
                          {"u_user_id",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                           "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                           "gender_embedding(gender_encoder(u_gender)) as u_gender_embed"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "gender_encoder(u_gender) as u_gender_encoded",
                   "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features",
                   "llm_ffnn_interest_scaler(transform(array_constructor(u_age, u_occupation), x-> CAST(X as REAL)))  AS u_interest_features",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "m_trending_features",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                   "u_final_interest_features",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .filter("trending_prediction = 1")
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .filter("user_interest_prediction = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "concat(bottom_mlp_out, u_age_embed, u_occupation_embed, u_gender_embed) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"});

    } else if (queryOptType.find("optimized") != std::string::npos) {
      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .project(
                          {"m_movie_id",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count",
                           "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features"})
                      .project({
                          "m_movie_id",
                          "m_popularity",
                          "m_vote_average",
                          "m_vote_count",
                          "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                      })
                      .filter("trending_prediction = 1")
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
              })
              .project(
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out"});
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 5, false)
                      .project(
                          {"u_user_id",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                           "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                           "gender_embedding(gender_encoder(u_gender)) as u_gender_embed"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "llm_ffnn_interest_scaler(transform(array_constructor(u_age, u_occupation), x-> CAST(X as REAL)))  AS u_interest_features",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "m_movie_id",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed",
                   "bottom_mlp_out"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .filter("user_interest_prediction = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "concat(bottom_mlp_out, u_age_embed, u_occupation_embed, u_gender_embed) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"});
    }
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("q3") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readMovieTagDataPlanNodeId2;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    if (queryOptType.empty() || queryOptType == "" ||
        queryOptType == "mlq3-fusion" || queryOptType == "mlq3-mul2join") {
      auto movieTagQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId2)
              .project(
                  {"mt_movie_id AS mt_movie_id1",
                   "relu(mat_vector_add20_4(mat_mul20_3(relu(mat_vector_add20_2(mat_mul20_1(mt_relevance_score)))))) AS mt_relevance_ir1"});

      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .limit(0, 1000, false)
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average",
                   "m_genres"})
              .filter("m_genres LIKE '\%Adventure\%'");
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 50, false)
                      .project(
                          {"u_user_id",
                           "CAST (u_age AS REAL) AS u_age",
                           "if (u_gender = 'M', 1.0, 0.0) AS u_gender",
                           "CAST (u_occupation AS REAL) AS u_occupation"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_score",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "model_features",
                   "mt_relevance_score",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_score",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features",
                   "user_movie_interest_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir",
                   "user_movie_rating_pred",
                   "model_features",
                   "user_movie_interest_pred"})
              .nestedLoopJoin(
                  movieTagQueryPlan.planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id1",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_ir",
                   "mt_movie_id1",
                   "mt_relevance_ir1",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id1",
                   "mt_movie_id",
                   "m_genres",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred"})
              .filter("user_movie_interest_pred = 1")
              .filter("user_movie_rating_pred = 5");

    } else if (queryOptType.find("mlq3-pushdown") != std::string::npos) {
      auto movieTagQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId2)
              .project(
                  {"mt_movie_id AS mt_movie_id1",
                   "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir1"});

      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .limit(0, 1000, false)
                      .filter("m_genres LIKE '\%Adventure\%'")
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average"})
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir",
                  "m_popularity",
                  "m_vote_average",
              })
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_ir",
                  "m_popularity",
                  "m_vote_average",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 50, false)
                      .project(
                          {"u_user_id",
                           "CAST (u_age AS REAL) AS u_age",
                           "if (u_gender = 'M', 1.0, 0.0) AS u_gender",
                           "CAST (u_occupation AS REAL) AS u_occupation"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "m_popularity",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "model_features",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .filter("user_movie_interest_pred = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features"})
              .filter("user_movie_rating_pred = 5")
              .nestedLoopJoin(
                  movieTagQueryPlan.planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "mt_movie_id1",
                   "mt_relevance_ir1"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim"});

    } else if (queryOptType.find("mlq3-optimized") != std::string::npos) {
      auto movieTagQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId2)
              .project(
                  {"mt_movie_id AS mt_movie_id1",
                   "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir1"});

      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .limit(0, 1000, false)
                      .filter("m_genres LIKE '\%Adventure\%'")
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average"})
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir",
                  "m_popularity",
                  "m_vote_average",
              })
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_ir",
                  "m_popularity",
                  "m_vote_average",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 50, false)
                      .project(
                          {"u_user_id",
                           "CAST (u_age AS REAL) AS u_age",
                           "if (u_gender = 'M', 1.0, 0.0) AS u_gender",
                           "CAST (u_occupation AS REAL) AS u_occupation"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "m_popularity",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "model_features",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .filter("user_movie_interest_pred = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features"})
              .filter("user_movie_rating_pred = 5")
              .nestedLoopJoin(
                  movieTagQueryPlan.planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "mt_movie_id1",
                   "mt_relevance_ir1"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim"});
    }
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId2,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);
  }

  return queryPlan;
};

std::vector<std::string> sampleUserMovieFilterExpr(std::string filterTable) {
  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  RandomSampler randomSampler = RandomSampler(timestampSeed);

  std::vector<std::string> userGenderFilterExprs = {
      "u_gender = 'M'",
      "u_gender = 'F'",
  };

  std::vector<std::string> userAgeFilterExprs = {
      "u_age > 30",
      "u_age < 50",
      "u_age >= 20",
      "u_age <= 60",
      "u_age = 40",
      "u_age > 15",
      "u_age < 25",
      "u_age >= 55",
      "u_age <= 35",
      "u_age = 65",
  };

  std::vector<std::string> userOccupationFilterExprs = {
      "u_occupation = 10",
      "u_occupation < 5",
      "u_occupation > 15",
      "u_occupation = 0",
      "u_occupation >= 7",
      "u_occupation <= 3",
      "u_occupation = 12",
      "u_occupation < 20",
      "u_occupation > 8",
      "u_occupation = 18",
  };

  std::vector<std::string> userZipCodeFilterExprs = {
      "u_zipcode = '94043'",
      "u_zipcode = '94301'",
      "u_zipcode = '94305'",
      "u_zipcode = '94306'",
      "u_zipcode = '94309'",
      "u_zipcode = '80212'",
      "u_zipcode = '80213'",
      "u_zipcode = '80219'",
      "u_zipcode = '12301'",
      "u_zipcode = '40201'",
  };

  std::vector<std::string> movieGenresFilterExprs = {
      "m_genres = 'Action'",
      "m_genres = 'Comedy'",
      "m_genres = 'Drama'",
      "m_genres = 'Horror'",
      "m_genres = 'Sci-Fi'",
      "m_genres = 'Romance'",
      "m_genres = 'Adventure'",
      "m_genres = 'Thriller'",
      "m_genres = 'Fantasy'",
      "m_genres = 'Documentary'",
  };

  std::vector<std::string> movieSpokenLanguageFilterExprs = {
      "m_spoken_languages = 'English'",
      "m_spoken_languages = 'French'",
      "m_spoken_languages = 'German'",
      "m_spoken_languages = 'Japanese'",
      "m_spoken_languages = 'Spanish'",
      "m_spoken_languages = 'Italian'",
      "m_spoken_languages = 'Korean'",
      "m_spoken_languages = 'Mandarin'",
  };

  std::vector<std::string> moviePopularityFilterExprs = {
      "m_popularity > 5.0",
      "m_popularity < 3.0",
      "m_popularity >= 7.5",
      "m_popularity <= 2.5",
      "m_popularity = 8.0",
      "m_popularity > 4.0",
      "m_popularity < 6.0",
      "m_popularity >= 9.0",
      "m_popularity <= 1.0",
      "m_popularity = 3.5",

  };

  std::vector<std::string> movieVoteAverageFilterExprs = {
      "m_vote_average > 3.0",
      "m_vote_average < 2.0",
      "m_vote_average >= 4.5",
      "m_vote_average <= 1.5",
      "m_vote_average = 5.0",
      "m_vote_average > 1.0",
      "m_vote_average < 4.0",
      "m_vote_average >= 2.5",
      "m_vote_average <= 3.5",
      "m_vote_average = 3.0",
  };

  std::vector<std::string> movieVoteCountFilterExprs = {
      "m_vote_count > 500",
      "m_vote_count < 100",
      "m_vote_count >= 750",
      "m_vote_count <= 250",
      "m_vote_count = 300",
      "m_vote_count > 200",
      "m_vote_count < 600",
      "m_vote_count >= 1000",
      "m_vote_count <= 50",
      "m_vote_count = 150",
  };

  std::vector<std::vector<std::string>> predefinedUserFilterExprs = {
      userGenderFilterExprs,
      userAgeFilterExprs,
      userOccupationFilterExprs,
      userZipCodeFilterExprs,
  };

  std::vector<std::vector<std::string>> predefinedMovieFilterExprs = {
      movieGenresFilterExprs,
      movieSpokenLanguageFilterExprs,
      moviePopularityFilterExprs,
      movieVoteAverageFilterExprs,
      movieVoteCountFilterExprs,
  };

  std::vector<std::vector<std::string>> combinedFilterExprSets;

  std::vector<std::string> sampledFilterExprs;
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, timestampSeed);
  std::vector<std::vector<std::string>> sampleFilterPool;
  if (filterTable == "user") {
    randomGenerator.setIntRange(1, 3);
    sampleFilterPool = predefinedUserFilterExprs;
  } else if (filterTable == "movie") {
    randomGenerator.setIntRange(1, 3);
    sampleFilterPool = predefinedMovieFilterExprs;
  } else if (filterTable == "movie_user") {
    randomGenerator.setIntRange(1, 3);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        predefinedUserFilterExprs.begin(),
        predefinedUserFilterExprs.end());
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        predefinedMovieFilterExprs.begin(),
        predefinedMovieFilterExprs.end());
    sampleFilterPool = combinedFilterExprSets;
  } else {
    throw std::invalid_argument(
        "Invalid table for sampling filter expression: " + filterTable);
  }

  std::set<std::vector<std::string>> usedExprSets;
  while (sampledFilterExprs.size() < randomGenerator.genRandomIntValue()) {
    auto pickedExprSets = randomSampler.sampleFromSets(1, sampleFilterPool)[0];
    if (usedExprSets.find(pickedExprSets) == usedExprSets.end()) {
      auto sampledFilterExpr =
          randomSampler.sampleFromSets(1, pickedExprSets)[0];
      sampledFilterExprs.push_back(sampledFilterExpr);
      usedExprSets.insert(pickedExprSets);
    }
  }

  return sampledFilterExprs;
};

// Function to write a string to a file
void writeStringToFile(const std::string& str, const std::string& filename) {
  // Open the file in write mode
  std::ofstream outfile(filename);

  // Check if the file opened successfully
  if (outfile.is_open()) {
    // Write the string to the file
    outfile << str;

    // Close the file
    outfile.close();
  } else {
    std::cerr << "Error: Could not open the file for writing." << std::endl;
  }
};

void outputAugmentedQueryPlan(
    CataLog& cataLog,
    PlanBuilder& plan,
    std::string outputPath = "") {
  auto serializedPlan = plan.planNode()->serialize();
  if (outputPath == "") {
    outputPath = "/home/velox/velox/optimizer/tests/serializedQueryPlan.json";
  }
  augmentSerializedPlan(serializedPlan, cataLog);
  writeStringToFile(folly::toJson(serializedPlan), outputPath);
};

void outputStructuredQueryPlan(PlanBuilder& plan) {
  auto structuredPlan = plan.planNode()->toString(true, true);
  writeStringToFile(
      structuredPlan,
      "/home/velox/velox/optimizer/tests/structuredQueryPlan.txt");
}

void deleteFilesInFolder(const std::string& folderPath) {
  try {
    // Check if the folder exists and is a directory
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
      std::cerr << "Error: Folder does not exist or is not a directory."
                << std::endl;
      return;
    }

    // Iterate through the files in the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
      if (fs::is_regular_file(entry.path())) {
        // Delete the file
        fs::remove(entry.path());
        // std::cout << "Deleted file: " << entry.path() << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception occurred: " << e.what() << std::endl;
  }
};
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& rowVectors,
    std::shared_ptr<memory::MemoryPool>& pool) {
  if (rowVectors.empty()) {
    return nullptr; // Nothing to merge
  }

  // Check schema consistency and get the number of columns.
  size_t numColumns = rowVectors[0]->childrenSize();
  for (const auto& rowVector : rowVectors) {
    VELOX_CHECK_EQ(rowVector->childrenSize(), numColumns, "Schema mismatch.");
  }

  // Total number of rows in the merged RowVector.
  vector_size_t totalRows = 0;
  for (const auto& rowVector : rowVectors) {
    totalRows += rowVector->size();
  }

  // Prepare new child vectors for the merged RowVector.
  std::vector<VectorPtr> mergedColumns(numColumns);
  for (size_t colIdx = 0; colIdx < numColumns; ++colIdx) {
    // Determine the type of the column.
    auto columnType = rowVectors[0]->childAt(colIdx)->type();

    // Create a new FlatVector to hold all data for this column.
    auto flatVector = BaseVector::create(columnType, totalRows, pool.get());

    // Fill the new FlatVector with data from each RowVector.
    vector_size_t offset = 0;
    for (const auto& rowVector : rowVectors) {
      auto sourceColumn = rowVector->childAt(colIdx);
      auto numRows = sourceColumn->size();

      flatVector->copy(sourceColumn.get(), offset, 0, numRows);
      offset += numRows;
    }

    mergedColumns[colIdx] = flatVector;
  }

  // Create the merged RowVector.
  return std::make_shared<RowVector>(
      pool.get(),
      rowVectors[0]->type(), // Use the type of the first RowVector.
      BufferPtr(nullptr), // No nulls buffer for the whole RowVector.
      totalRows,
      std::move(mergedColumns));
}
