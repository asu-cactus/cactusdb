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

#pragma once
#include <H5Cpp.h>
#include <json/json.h>
#include <iostream>
#include <sstream>
#include <string>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/optimizer/CataLog.h"
#include "velox/optimizer/Helper.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include "velox/optimizer/tests/ModelRegister.h"

using namespace optimization;
using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

PlanBuilder setupProfileQueryPlanFromTemplate(
    std::string workload,
    std::string queryTemplate,
    int& modelGroupId_,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
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
  randomGenerator.setIntRange(10, 3000);
  PlanBuilder queryPlan;

  if (workload == "movielens") {
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

    int movieTagNumRows, movieTagNumCols, movieNumRows, movieNumCols,
        userNumRows, userNumCols, ratingNumRows, ratingNumCols;

    readDataStats(
        dataDirPrefix + "movie_tag_relevance_stats.txt",
        movieTagNumRows,
        movieTagNumCols);
    readDataStats(
        dataDirPrefix + "movie_stats.txt", movieNumRows, movieNumCols);
    readDataStats(dataDirPrefix + "user_stats.txt", userNumRows, userNumCols);
    readDataStats(
        dataDirPrefix + "rating_stats.txt", ratingNumRows, ratingNumCols);

    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    PlanNodeId readRatingDataPlanNodeId1;
    PlanNodeId readRatingDataPlanNodeId2;

    if (queryTemplate == "user_only") {
      std::unordered_map<std::string, int> genderMapping;
      genderMapping["F"] = 0;
      genderMapping["M"] = 1;

      optimization::registerVectorFunction(
          "gender_encoder",
          StringEncoder::signatures(),
          std::make_unique<StringEncoder>(std::move(genderMapping)),
          {},
          true,
          cataLog);

      int modelGroupId_ = 0;
      auto modelStr =
          registerNNModel({3, 128, 3}, cataLog, modelGroupId_, false);

      std::cout << "[INFO] modelStr: " << modelStr << std::endl;

      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .project(
                  {"u_user_id",
                   "u_age",
                   "gender_encoder(u_gender) as u_gender_encoded",
                   "u_occupation",
                   "u_zipcode"})
              .project({
                  "u_user_id",
                  "transform(concat(array_constructor(u_age), u_gender_encoded, array_constructor(u_occupation)), x-> CAST(x AS REAL)) as u_features" // ARRAY(REAL)
              })
              .project({"u_user_id", fmt::format(modelStr, "u_features")});

      // select u_user_id, dnn(features) as pred from users;
      // select dnn(features) as pred from users;
      // Set data files for the data source nodes, it is okay if the node
      // is actually not used
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      // cataLog.setIdAddressMap(
      //     readMovieDataPlanNodeId,
      //     movieDataPaths,
      //     dwio::common::FileFormat::PARQUET);
      // cataLog.setIdAddressMap(
      //     readRatingDataPlanNodeId1,
      //     ratingDataPaths,
      //     dwio::common::FileFormat::PARQUET);
      // cataLog.setIdAddressMap(
      //     readRatingDataPlanNodeId2,
      //     ratingDataPaths,
      //     dwio::common::FileFormat::PARQUET);

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      // cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      // cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1,
      // "movie_rating");
      // cataLog.addNodeIdRelationName(readRatingDataPlanNodeId2,
      // "movie_rating"); cataLog.addNodeIdRelationName(
      //     readMovieTagDataPlanNodeId, "movie_relevance_tag");

      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));

      // std::shared_ptr<OutputStat> movieStats =
      //     std::make_shared<OutputStat>(OutputStat(movieNumRows,
      //     movieNumCols));
      // Source movieSrc =
      //     Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      // cataLog.addSource(std::make_shared<Source>(movieSrc));

      // std::shared_ptr<OutputStat> ratingStats = std::make_shared<OutputStat>(
      //     OutputStat(ratingNumRows, ratingNumCols));
      // Source ratingSrc1 =
      //     Source(readRatingDataPlanNodeId1, Source::Type::FILE, ratingStats);
      // cataLog.addSource(std::make_shared<Source>(ratingSrc1));
      // Source ratingSrc2 =
      //     Source(readRatingDataPlanNodeId2, Source::Type::FILE, ratingStats);
      // cataLog.addSource(std::make_shared<Source>(ratingSrc2));
      // std::shared_ptr<OutputStat> movieTagStats =
      // std::make_shared<OutputStat>(
      //     OutputStat(movieTagNumRows, movieTagNumCols));
      // Source movieTagSrc =
      //     Source(readMovieTagDataPlanNodeId, Source::Type::FILE,
      //     movieTagStats);
      // cataLog.addSource(std::make_shared<Source>(movieTagSrc));
    } else if (queryTemplate == "template5") {
        // gender_encoder
        registerGenderEncoder(cataLog, pool_);
        // user_age_minmax_scaler
        normalizeAge(cataLog,pool_, "q5_user_age_minmax_scaler.txt");
        // user_occupation_minmax_scaler
        normalizeOccupation(cataLog,pool_, "q5_user_occupation_minmax_scaler.txt");


      std::vector<std::vector<int>> userModelStructures = readModelStructureFromFile(
          "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
       auto modelStr =
          registerNNModel(userModelStructures[0], cataLog, modelGroupId_, false);

    //   int hidden1 = randomGenerator.genRandomIntValue();
    //   int hidden2 = randomGenerator.genRandomIntValue();
    //   std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2 << std::endl;
    //   auto modelStr =
    //       registerNNModel({3, hidden1, hidden2, 3706}, cataLog, modelGroupId_, false);

    //   auto modelStr =
    //       registerNNModel({3, 1024, 2048, 3706}, cataLog, modelGroupId_, false);

      std::cout << "[INFO] modelStr: " << modelStr << std::endl;

      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_gender",                   
                   "u_occupation",
                   "u_zipcode",
                   "gender_encoder(u_gender) as u_gender_encoded",
                    "user_age_minmax_scaler(transform(array_constructor(u_age), x-> CAST(x AS REAL))) as u_age_encoded",
                    "user_occupation_minmax_scaler(transform(array_constructor(u_occupation), x-> CAST(x AS REAL))) as u_occupation_encoded"})
              .project({
                  "u_user_id",
                  "u_age",
                   "u_gender",                   
                   "u_occupation",
                   "u_zipcode",
                   "transform(concat(u_gender_encoded,u_age_encoded,u_occupation_encoded), x-> CAST(x AS REAL)) as features",

              })
              .project({
                    "u_user_id", 
                    "u_age",
                    "u_gender",                   
                    "u_occupation",
                    "u_zipcode", 
                    fmt::format(modelStr, "features")});
        std::vector<std::string> filterExpr = sampleUserMovieFilterExpr("user", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }

      // select u_user_id, dnn(features) as pred from users;
      // select dnn(features) as pred from users;
      // Set data files for the data source nodes, it is okay if the node
      // is actually not used
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");

      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));
    } else if (queryTemplate == "template6") {
        std::string svdModelPath = "/home/velox/resources/model/tpcxai_sf1/final/velox/movielens_template6_svd.h5";
        std::vector<std::vector<float>> bu = loadHDF5Array(svdModelPath, "bu");
        std::vector<std::vector<float>> bi = loadHDF5Array(svdModelPath, "bi");
        std::vector<std::vector<float>> pu = loadHDF5Array(svdModelPath, "pu");
        std::vector<std::vector<float>> qi = loadHDF5Array(svdModelPath, "qi");
        optimization::registerVectorFunction(
            "svd",
            SVD::signatures(),
            std::make_unique<SVD>(
                std::move(flattenVectorToPointer(bu)),
                std::move(flattenVectorToPointer(bi)),
                std::move(flattenVectorToPointer(pu)),
                std::move(flattenVectorToPointer(qi)),
                7071,
                6818,
                100),
            {},
            true,
            cataLog);
    }

    // TODO
  } else if (workload == "tpcxai") {
    // TODO
  } else {
    throw std::runtime_error(
        "Unsupported workload: " + workload +
        ". Currently only movielens is supported.");
  }

  return queryPlan;
}


PlanBuilder setupProfileQueryPlanFromTemplate1(
    std::string workload,
    std::string queryTemplate,
    int& modelGroupId_,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
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
  PlanBuilder queryPlan;

  if (workload == "movielens1") {
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

    int movieTagNumRows, movieTagNumCols, movieNumRows, movieNumCols,
        userNumRows, userNumCols, ratingNumRows, ratingNumCols;

    readDataStats(
        dataDirPrefix + "movie_tag_relevance_stats.txt",
        movieTagNumRows,
        movieTagNumCols);
    readDataStats(
        dataDirPrefix + "movie_stats.txt", movieNumRows, movieNumCols);
    readDataStats(dataDirPrefix + "user_stats.txt", userNumRows, userNumCols);
    readDataStats(
        dataDirPrefix + "rating_stats.txt", ratingNumRows, ratingNumCols);

    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    PlanNodeId readRatingDataPlanNodeId1;
    PlanNodeId readRatingDataPlanNodeId2;

    if (queryTemplate == "user_only") {
      std::unordered_map<std::string, int> genderMapping;
      genderMapping["F"] = 0;
      genderMapping["M"] = 1;

      optimization::registerVectorFunction(
          "gender_encoder",
          StringEncoder::signatures(),
          std::make_unique<StringEncoder>(std::move(genderMapping)),
          {},
          true,
          cataLog);

      int modelGroupId_ = 0;
      auto modelStr =
          registerNNModel({3, 128, 3}, cataLog, modelGroupId_, false);

      std::cout << "[INFO] modelStr: " << modelStr << std::endl;

      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .project(
                  {"u_user_id",
                   "u_age",
                   "gender_encoder(u_gender) as u_gender_encoded",
                   "u_occupation",
                   "u_zipcode"})
              .project({
                  "u_user_id",
                  "transform(concat(array_constructor(u_age), u_gender_encoded, array_constructor(u_occupation)), x-> CAST(x AS REAL)) as u_features" // ARRAY(REAL)
              })
              .project({"u_user_id", fmt::format(modelStr, "u_features")});

      // select u_user_id, dnn(features) as pred from users;
      // select dnn(features) as pred from users;
      // Set data files for the data source nodes, it is okay if the node
      // is actually not used
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      // cataLog.setIdAddressMap(
      //     readMovieDataPlanNodeId,
      //     movieDataPaths,
      //     dwio::common::FileFormat::PARQUET);
      // cataLog.setIdAddressMap(
      //     readRatingDataPlanNodeId1,
      //     ratingDataPaths,
      //     dwio::common::FileFormat::PARQUET);
      // cataLog.setIdAddressMap(
      //     readRatingDataPlanNodeId2,
      //     ratingDataPaths,
      //     dwio::common::FileFormat::PARQUET);

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      // cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      // cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1,
      // "movie_rating");
      // cataLog.addNodeIdRelationName(readRatingDataPlanNodeId2,
      // "movie_rating"); cataLog.addNodeIdRelationName(
      //     readMovieTagDataPlanNodeId, "movie_relevance_tag");

      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));

      // std::shared_ptr<OutputStat> movieStats =
      //     std::make_shared<OutputStat>(OutputStat(movieNumRows,
      //     movieNumCols));
      // Source movieSrc =
      //     Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      // cataLog.addSource(std::make_shared<Source>(movieSrc));

      // std::shared_ptr<OutputStat> ratingStats = std::make_shared<OutputStat>(
      //     OutputStat(ratingNumRows, ratingNumCols));
      // Source ratingSrc1 =
      //     Source(readRatingDataPlanNodeId1, Source::Type::FILE, ratingStats);
      // cataLog.addSource(std::make_shared<Source>(ratingSrc1));
      // Source ratingSrc2 =
      //     Source(readRatingDataPlanNodeId2, Source::Type::FILE, ratingStats);
      // cataLog.addSource(std::make_shared<Source>(ratingSrc2));
      // std::shared_ptr<OutputStat> movieTagStats =
      // std::make_shared<OutputStat>(
      //     OutputStat(movieTagNumRows, movieTagNumCols));
      // Source movieTagSrc =
      //     Source(readMovieTagDataPlanNodeId, Source::Type::FILE,
      //     movieTagStats);
      // cataLog.addSource(std::make_shared<Source>(movieTagSrc));
    } else if (queryTemplate == "movie_rating_pivot") {

      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(ratingDataRowType, {}, "")
              .capturePlanNodeId(readRatingDataPlanNodeId1)
              .project(
                  {"r_user_id",
                   "r_movie_id",
                   "r_rating",
                   "r_timestamp"})
              .partialAggregation(
                {"r_user_id"}, {"map_agg(r_movie_id, r_rating)"})
              .finalAggregation();
      cataLog.setIdAddressMap(
          readRatingDataPlanNodeId1,
          ratingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(
          readRatingDataPlanNodeId1, "movie_rating");
      std::shared_ptr<OutputStat> ratingStats =
          std::make_shared<OutputStat>(OutputStat(ratingNumRows, ratingNumCols));
      Source ratingSrc1 =
          Source(readRatingDataPlanNodeId1, Source::Type::FILE, ratingStats);
      cataLog.addSource(std::make_shared<Source>(ratingSrc1));
    }

    // TODO
  } else if (workload == "tpcxai") {
    // TODO
  } else {
    throw std::runtime_error(
        "Unsupported workload: " + workload +
        ". Currently only movielens is supported.");
  }

  return queryPlan;
}