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
#include "velox/optimizer/DecisionForestUDF2RelationRewriteAction.h"
#include "velox/optimizer/RewriteAction.h"
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

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");

      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));
    } else if (queryTemplate == "template1") {
      registerTwoTowerFunc(cataLog, pool_);
      registerMLTrendingModelFunctions(cataLog, pool_);
      PlanNodeId readMovieTagDataPlanNodeId;
      PlanNodeId readUserDataPlanNodeId;
      PlanNodeId readMovieDataPlanNodeId;
      PlanNodeId readRatingDataPlanNodeId1;
      PlanNodeId readRatingDataPlanNodeId2;
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
                   "gender_embedding(gender_encoder(u_gender)) as u_gender_encoded",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_encoded",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_encoded",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres_embed",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "m_genres",
                   "m_spoken_languages"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender_encoded, u_age_encoded, u_occupation_encoded, u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "m_genres",
                   "m_spoken_languages"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "m_genres",
                   "m_spoken_languages",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "m_genres",
                   "m_spoken_languages",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("template1", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
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

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));

      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      std::shared_ptr<OutputStat> movieStats =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      cataLog.addSource(std::make_shared<Source>(movieSrc));

      cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1, "movie_rating");
      cataLog.addNodeIdRelationName(readRatingDataPlanNodeId2, "movie_rating");
      std::shared_ptr<OutputStat> ratingStats = std::make_shared<OutputStat>(
          OutputStat(ratingNumRows, ratingNumCols));
      Source ratingSrc1 =
          Source(readRatingDataPlanNodeId1, Source::Type::FILE, ratingStats);
      cataLog.addSource(std::make_shared<Source>(ratingSrc1));
      Source ratingSrc2 =
          Source(readRatingDataPlanNodeId2, Source::Type::FILE, ratingStats);
      cataLog.addSource(std::make_shared<Source>(ratingSrc2));
    } else if (queryTemplate == "template2") {
      registerMLTrendingModelFunctions(cataLog, pool_);
      registerMLInterestMovieModelFunctions(cataLog, pool_);
      registerMLMovieTagEncoderModelFunctions(cataLog, pool_);
      registerMLDLRMModelFunctions(cataLog, pool_);
      PlanNodeId readMovieTagDataPlanNodeId;
      PlanNodeId readUserDataPlanNodeId;
      PlanNodeId readMovieDataPlanNodeId;
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
                           "m_genres",
                           "m_spoken_languages"})
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_genres",
                   "m_spoken_languages",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "m_genres",
                  "m_spoken_languages",
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
                   "m_vote_count",
                   "m_genres",
                   "m_spoken_languages"})
              .project({
                  "u_user_id",
                  "u_age",
                  "u_occupation",
                  "gender_encoder(u_gender) as u_gender_encoded",
                  "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_score",
                  "m_genres",
                  "m_spoken_languages",
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
                   "m_genres",
                   "m_spoken_languages",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                   "u_final_interest_features",
                   "mt_relevance_score"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "mt_relevance_score",
                   "trending_prediction"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                   "gender_embedding(u_gender_encoded) as u_gender_embed",
                   "mt_relevance_score",
                   "trending_prediction",
                   "user_interest_prediction"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "trending_prediction",
                   "user_interest_prediction",
                   "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out",
                   "concat(u_age_embed, u_occupation_embed, u_gender_embed) as categorical_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "trending_prediction",
                   "user_interest_prediction",
                   "concat(bottom_mlp_out, categorical_features) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "trending_prediction",
                   "user_interest_prediction",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"})
              .filter("trending_prediction = 1")
              .filter("user_interest_prediction = 1");
      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("template2", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
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

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));

      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      std::shared_ptr<OutputStat> movieStats =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      cataLog.addSource(std::make_shared<Source>(movieSrc));

      cataLog.addNodeIdRelationName(
          readMovieTagDataPlanNodeId, "movie_relevance_tag");
      std::shared_ptr<OutputStat> movieTagStats = std::make_shared<OutputStat>(
          OutputStat(movieTagNumRows, movieTagNumCols));
      Source movieTagSrc =
          Source(readMovieTagDataPlanNodeId, Source::Type::FILE, movieTagStats);
      cataLog.addSource(std::make_shared<Source>(movieTagSrc));
    } else if (queryTemplate == "template3") {
      PlanNodeId readMovieTagDataPlanNodeId;
      PlanNodeId readMovieTagDataPlanNodeId2;
      PlanNodeId readUserDataPlanNodeId;
      PlanNodeId readMovieDataPlanNodeId;
      registerMLQ3UserMovieInterestModelFunctions(cataLog, pool_);
      registerMLQ3UserMovieRatingModelFunctions(cataLog, pool_);
      registerMLMovieTagEncoderModelFunctions(cataLog, pool_);
      registerMLMovieTagEncoderModelFunctions1(cataLog, pool_);
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
                   "m_genres",
                   "m_spoken_languages"});
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
                   "m_spoken_languages",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "mt_relevance_score",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "model_features",
                   "mt_relevance_score",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "mt_relevance_score",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features",
                   "user_movie_interest_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "m_spoken_languages",
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
                   "m_spoken_languages",
                   "mt_relevance_ir",
                   "mt_relevance_ir1",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id1",
                   "mt_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred",
                   "cosine_sim"})
              .filter("user_movie_interest_pred = 1")
              .filter("user_movie_rating_pred = 5");

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("template3", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
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

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));

      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      std::shared_ptr<OutputStat> movieStats =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      cataLog.addSource(std::make_shared<Source>(movieSrc));

      cataLog.addNodeIdRelationName(
          readMovieTagDataPlanNodeId, "movie_relevance_tag");
      std::shared_ptr<OutputStat> movieTagStats = std::make_shared<OutputStat>(
          OutputStat(movieTagNumRows, movieTagNumCols));
      Source movieTagSrc =
          Source(readMovieTagDataPlanNodeId, Source::Type::FILE, movieTagStats);
      cataLog.addSource(std::make_shared<Source>(movieTagSrc));

      cataLog.addNodeIdRelationName(
          readMovieTagDataPlanNodeId2, "movie_relevance_tag");
      std::shared_ptr<OutputStat> movieTagStats2 = std::make_shared<OutputStat>(
          OutputStat(movieTagNumRows, movieTagNumCols));
      Source movieTagSrc2 = Source(
          readMovieTagDataPlanNodeId2, Source::Type::FILE, movieTagStats2);
      cataLog.addSource(std::make_shared<Source>(movieTagSrc2));
    } else if (queryTemplate == "template5") {
      // gender_encoder
      registerGenderEncoder(cataLog, pool_);
      // user_age_minmax_scaler
      registerMovielensAgeMinMaxScaler(
          cataLog, pool_, "q4_user_age_minmax_scaler.txt");
      // user_occupation_minmax_scaler
      registerMovielensOccupationMinMaxScaler(
          cataLog, pool_, "q4_user_occupation_minmax_scaler.txt");

      std::vector<std::vector<int>> userModelStructures =
          readModelStructureFromFile(
              "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
      auto modelStr = registerNNModel(
          userModelStructures[0], cataLog, modelGroupId_, false);

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
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   fmt::format(modelStr, "features")});
      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("user", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
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
      std::string svdModelPath =
          "/home/velox/resources/model/movielens/final/velox/movielens_template6_svd.h5";
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
              pu.size(),
              qi.size(),
              pu[0].size()),
          {},
          true,
          cataLog);

      queryPlan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .nestedLoopJoin(
                          PlanBuilder(planNodeIdGenerator)
                              .tableScan(movieDataRowType, {}, "")
                              .capturePlanNodeId(readMovieDataPlanNodeId)
                              .planNode(),
                          {// what columns to project from the join
                           "u_user_id",
                           "m_movie_id",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "u_zipcode",
                           "m_genres"})
                      // .project({
                      //     "CAST (u_user_id AS INTEGER) AS u_user_id",
                      //     "CAST (m_movie_id as INTEGER) AS m_movie_id",
                      //     "u_age",
                      //     "u_gender",
                      //     "u_occupation",
                      //     "u_zipcode",
                      //     "m_genres"})
                      .project(
                          {"u_user_id",
                           "m_movie_id",
                           "svd(u_user_id, m_movie_id) as pred",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "u_zipcode",
                           "m_genres"});

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("user_movie_genres", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }
      // — user side
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addSource(std::make_shared<Source>(Source(
          readUserDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(userNumRows, userNumCols))));

      // — movie side
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      cataLog.addSource(std::make_shared<Source>(Source(
          readMovieDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(movieNumRows, movieNumCols))));

    } else if (queryTemplate == "template7") {
      // Embedding
      VectorMaker maker{pool_.get()};
      constexpr int numUserEmbeddings = 6041;
      constexpr int numMovieEmbedding = 3707;
      constexpr int embeddingDims = 32;

      std::vector<std::vector<float>> userEmbedWeights =
          randomGenerator.genFloat2dVector(numUserEmbeddings, embeddingDims);
      auto userWeightsVector =
          maker.arrayVector<float>(userEmbedWeights, REAL());
      exec::registerVectorFunction(
          "user_embedding",
          Embedding::signatures(),
          std::make_unique<Embedding>(
              userWeightsVector->elements()->values()->asMutable<float>(),
              numUserEmbeddings,
              embeddingDims));

      std::vector<std::vector<float>> movieEmbedWeights =
          randomGenerator.genFloat2dVector(numMovieEmbedding, embeddingDims);
      auto movieWeightsVector =
          maker.arrayVector<float>(movieEmbedWeights, REAL());
      exec::registerVectorFunction(
          "movie_embedding",
          Embedding::signatures(),
          std::make_unique<Embedding>(
              movieWeightsVector->elements()->values()->asMutable<float>(),
              numMovieEmbedding,
              embeddingDims));

      // Cosine Similarity
      exec::registerVectorFunction(
          "cosine_similarity",
          CosineSimilarity::signatures(),
          std::make_unique<CosineSimilarity>(embeddingDims));

      // Query Plan
      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator)
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .planNode(),
                  {// what columns to project from the join
                   "u_user_id",
                   "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres",
                   "user_embedding(array_constructor(u_user_id)) as user_embed",
                   "movie_embedding(array_constructor(m_movie_id)) as movie_embed"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres",
                   "cosine_similarity(user_embed, movie_embed) as pred"});

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("user_movie_genres", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }
      // — user side
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addSource(std::make_shared<Source>(Source(
          readUserDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(userNumRows, userNumCols))));

      // — movie side
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      cataLog.addSource(std::make_shared<Source>(Source(
          readMovieDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(movieNumRows, movieNumCols))));
    } else if (queryTemplate == "template10") {
      // Embedding
      VectorMaker maker{pool_.get()};
      constexpr int numEmbeddings = 3707; // 3707
      constexpr int embeddingDims = 256; // 256

      RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
      randomGenerator.setIntRange(0, numEmbeddings - 1);

      std::vector<std::vector<float>> weights =
          randomGenerator.genFloat2dVector(numEmbeddings, embeddingDims);
      auto weightsVector = maker.arrayVector<float>(weights, REAL());

      exec::registerVectorFunction(
          "embedding",
          Embedding::signatures(),
          std::make_unique<Embedding>(
              weightsVector->elements()->values()->asMutable<float>(),
              numEmbeddings,
              embeddingDims));

      // Feature processor
      // gender_encoder
      registerGenderEncoder(cataLog, pool_);
      // user_age_minmax_scaler
      registerMovielensAgeMinMaxScaler(
          cataLog, pool_, "q4_user_age_minmax_scaler.txt");
      // user_occupation_minmax_scaler
      registerMovielensOccupationMinMaxScaler(
          cataLog, pool_, "q4_user_occupation_minmax_scaler.txt");
      // Feed forward Neural Network
      int hidden1 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << std::endl;
      auto modelStr = registerNNModel(
          {embeddingDims + 3, hidden1, 1}, cataLog, modelGroupId_, false);

      // Query Plan
      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator)
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .planNode(),
                  {// what columns to project from the join
                   "u_user_id",
                   "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "embedding(array_constructor(m_movie_id)) as embed",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres",
                   "gender_encoder(u_gender) as gender_encoded",
                   "user_age_minmax_scaler(transform(array_constructor(u_age), x-> CAST(x AS REAL))) as age_encoded",
                   "user_occupation_minmax_scaler(transform(array_constructor(u_occupation), x-> CAST(x AS REAL))) as occupation_encoded"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres",
                   "transform(concat(gender_encoded,age_encoded,occupation_encoded,embed), x-> CAST(x AS REAL)) as features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres",
                   fmt::format(modelStr, "features")});

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("user_movie_genres", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }
      // — user side
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addSource(std::make_shared<Source>(Source(
          readUserDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(userNumRows, userNumCols))));

      // — movie side
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      cataLog.addSource(std::make_shared<Source>(Source(
          readMovieDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(movieNumRows, movieNumCols))));
    } else {
      throw std::runtime_error(
          "Unsupported query template for movielens workload: " +
          queryTemplate);
    }

  } else if (workload == "tpcxai") {
    std::string queryOptType =
        getEnvVar("CD_VELOX_QUERY_OPT_TYPE"); // env used for ablation study of

    auto finicialAccountDataRowType =
        ROW({"fa_customer_sk", "transaction_limit"}, {BIGINT(), DOUBLE()});
    auto finicialTransactionsDataRowType =
        ROW({"amount",
             "iban",
             "sender_id",
             "receiver_id",
             "transaction_id",
             "time"},
            {DOUBLE(), VARCHAR(), BIGINT(), VARCHAR(), BIGINT(), VARCHAR()});
    auto orderDataRowType =
        ROW({"o_order_id", "o_customer_sk", "weekday", "date", "store"},
            {BIGINT(), BIGINT(), VARCHAR(), VARCHAR(), BIGINT()});
    auto lineitemDataRowType =
        ROW({"li_order_id", "li_product_id", "quantity", "price"},
            {BIGINT(), BIGINT(), BIGINT(), DOUBLE()});
    auto productDataRowType =
        ROW({"p_product_id", "name", "department"},
            {BIGINT(), VARCHAR(), VARCHAR()});
    auto storeDeptDataRowType =
        ROW({"store", "department", "num_of_week"},
            {BIGINT(), VARCHAR(), BIGINT()});
    auto productRatingRowType =
        ROW({"user_id", "product_id"}, {BIGINT(), BIGINT()});
    auto customerDataRowType =
        ROW({"c_customer_sk",
             "c_customer_id",
             "c_current_addr_sk",
             "c_first_name",
             "c_last_name",
             "c_preferred_cust_flag",
             "c_birth_day",
             "c_birth_month",
             "c_birth_year",
             "c_birth_country",
             "c_login",
             "c_email_address"},
            {INTEGER(),
             VARCHAR(),
             INTEGER(),
             VARCHAR(),
             VARCHAR(),
             VARCHAR(),
             INTEGER(),
             INTEGER(),
             INTEGER(),
             VARCHAR(),
             VARCHAR(),
             VARCHAR()});
    auto orderReturnDataRowType =
        ROW({"or_order_id", "or_product_id", "or_return_quantity"},
            {INTEGER(), INTEGER(), INTEGER()});
    auto reviewDataRowType = ROW({"id", "text"}, {INTEGER(), VARCHAR()});

    std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

    if (dataDirPrefix == "") {
      // use default value:
      dataDirPrefix =
          "/home/velox/resources/data/parquet/tpcxai_sf1/final/serving/";
    }

    std::vector<std::string> finicialAccountDataPaths =
        getFilePathsFromDir(dataDirPrefix + "financial_account");
    std::vector<std::string> financialTransactionsDataPaths =
        getFilePathsFromDir(dataDirPrefix + "financial_transactions");
    std::vector<std::string> orderDataPaths =
        getFilePathsFromDir(dataDirPrefix + "order");
    std::vector<std::string> lineitemDataPaths =
        getFilePathsFromDir(dataDirPrefix + "lineitem");
    std::vector<std::string> productDataPaths =
        getFilePathsFromDir(dataDirPrefix + "product");
    std::vector<std::string> storeDeptDataPaths =
        getFilePathsFromDir(dataDirPrefix + "store_dept");
    std::vector<std::string> productRatingDataPaths =
        getFilePathsFromDir(dataDirPrefix + "product_rating");
    std::vector<std::string> customerDataPaths =
        getFilePathsFromDir(dataDirPrefix + "customer");
    std::vector<std::string> orderReturnDataPaths =
        getFilePathsFromDir(dataDirPrefix + "order_returns");
    std::vector<std::string> reviewDataPaths =
        getFilePathsFromDir(dataDirPrefix + "review");

    int finicialAccountNumRows, finicialAccountNumCols,
        finicialTransactionsNumRows, finicialTransactionsNumCols, orderNumRows,
        orderNumCols, lineitemNumRows, lineitemNumCols, productNumRows,
        productNumCols, storeDeptNumRows, storeDeptNumCols,
        productRatingNumRows, productRatingNumCols, customerNumRows,
        customerNumCols, orderReturnNumRows, orderReturnNumCols, reviewNumRows,
        reviewNumCols;

    readDataStats(
        dataDirPrefix + "financial_account_stats.txt",
        finicialAccountNumRows,
        finicialAccountNumCols);
    readDataStats(
        dataDirPrefix + "financial_transactions_stats.txt",
        finicialTransactionsNumRows,
        finicialTransactionsNumCols);
    readDataStats(
        dataDirPrefix + "order_stats.txt", orderNumRows, orderNumCols);
    readDataStats(
        dataDirPrefix + "lineitem_stats.txt", lineitemNumRows, lineitemNumCols);
    readDataStats(
        dataDirPrefix + "product_stats.txt", productNumRows, productNumCols);
    readDataStats(
        dataDirPrefix + "store_dept_stats.txt",
        storeDeptNumRows,
        storeDeptNumCols);
    readDataStats(
        dataDirPrefix + "product_rating_stats.txt",
        productRatingNumRows,
        productRatingNumCols);
    readDataStats(
        dataDirPrefix + "customer_stats.txt", customerNumRows, customerNumCols);
    readDataStats(
        dataDirPrefix + "order_returns_stats.txt",
        orderReturnNumRows,
        orderReturnNumCols);
    readDataStats(
        dataDirPrefix + "review_stats.txt", reviewNumRows, reviewNumCols);

    PlanNodeId readProductRatingDataPlanNodeId;
    PlanNodeId readOrderDataPlanNodeId;
    PlanNodeId readLineitemDataPlanNodeId;
    PlanNodeId readProductDataPlanNodeId;
    PlanNodeId readCustomerDataPlanNodeId;
    PlanNodeId readFinancialAccountDataPlanNodeId;
    PlanNodeId readFinancialTransactionsDataPlanNodeId;
    PlanNodeId readStoreDeptDataPlanNodeId;
    PlanNodeId readOrderReturnDataPlanNodeId;

    if (queryTemplate == "template1") {
      PlanNodeId readStoreDataPlanNodeId;
      PlanNodeId readOrderDataPlanNodeId;

      queryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(storeDeptDataRowType, {}, "")
              .capturePlanNodeId(readStoreDataPlanNodeId)
              .localPartition({"s_store"})
              .project({"s_store", "s_features as store_feature"})
              .filter("is_popular_store(store_feature) = 1")
              .project(
                  {"s_store",
                   "mat_vector_add_1(mat_mul_12(store_feature)) as dnn_part2"})
              .hashJoin(
                  {"s_store"},
                  {"o_store"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(orderDataRowType, {}, "")
                      .capturePlanNodeId(readOrderDataPlanNodeId)
                      .localPartition({"o_store"})
                      .filter("o_weekday != 'Sunday'")
                      .project(
                          {"o_order_id",
                           "o_customer_sk",
                           "o_store",
                           "o_date",
                           "o_weekday"})
                      .project(
                          {"o_order_id",
                           "o_store",
                           "mat_mul_11(concat(customer_id_embedding(convert_int_array(o_customer_sk)), get_order_features(o_date, o_weekday))) as dnn_part1"})
                      .planNode(),
                  "",
                  {"o_order_id", "dnn_part1", "dnn_part2"})
              .project(
                  {"o_order_id",
                   "get_max_index(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(vector_addition(dnn_part1, dnn_part2))))))))) AS predicted_trip_type"});
      cataLog.setIdAddressMap(
          readStoreDataPlanNodeId,
          storeDeptDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readOrderDataPlanNodeId,
          orderDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readStoreDataPlanNodeId, "store");
      cataLog.addNodeIdRelationName(readOrderDataPlanNodeId, "order");
      cataLog.addSource(std::make_shared<Source>(Source(
          readStoreDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(storeDeptNumRows, storeDeptNumCols)))));
      cataLog.addSource(std::make_shared<Source>(Source(
          readOrderDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(orderNumRows, orderNumCols))));
    } else if (queryTemplate == "template2") {
      PlanNodeId readFinancialTransactionsDataPlanNodeId;
      queryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(finicialTransactionsDataRowType, {}, "")
              .capturePlanNodeId(readFinancialTransactionsDataPlanNodeId)
              .project(
                  {"transaction_id",
                   "t_sender",
                   "t_amount",
                   "date_to_timestamp(t_time) as t_timestamp"})
              .filter("is_working_day(t_timestamp) = 1")
              .project(
                  {"transaction_id",
                   "t_sender",
                   "t_timestamp",
                   "get_transaction_features(t_amount, t_timestamp) as transaction_feature"})
              .filter("xgboost_fraud_transaction(transaction_feature) >= 0.5")
              .project(
                  {"transaction_id",
                   "t_sender",
                   "t_timestamp",
                   "mat_mul_12(transaction_feature) AS dnn_part12"})
              .hashJoin(
                  {"t_sender"},
                  {"c_customer_sk"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(customerDataRowType, {}, "")
                      .capturePlanNodeId(readCustomerDataPlanNodeId)
                      //  .values(batchesCustomer)
                      .project(
                          {"c_customer_sk",
                           "c_address_num",
                           "c_cust_flag",
                           "c_birth_day",
                           "c_birth_month",
                           "c_birth_year",
                           "c_birth_country"})
                      .hashJoin(
                          {"c_customer_sk"},
                          {"fa_customer_sk"},
                          PlanBuilder(planNodeIdGenerator, pool_.get())
                              .tableScan(finicialAccountDataRowType, {}, "")
                              .capturePlanNodeId(
                                  readFinancialAccountDataPlanNodeId)
                              //  .values(batchesAccount)
                              .project(
                                  {"fa_customer_sk", "fa_transaction_limit"})
                              .planNode(),
                          "",
                          {"c_customer_sk",
                           "c_address_num",
                           "c_cust_flag",
                           "c_birth_day",
                           "c_birth_month",
                           "c_birth_year",
                           "c_birth_country",
                           "fa_transaction_limit"})
                      .project(
                          {"c_customer_sk",
                           "c_birth_year",
                           "mat_vector_add_1(mat_mul_11(get_customer_features(c_address_num, c_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, fa_transaction_limit))) as dnn_part11"})
                      .planNode(),
                  "",
                  {"transaction_id",
                   "t_timestamp",
                   "dnn_part12",
                   "c_birth_year",
                   "dnn_part11"})
              .filter("age_during_transaction(t_timestamp, c_birth_year) >= 18")
              .project(
                  {"transaction_id",
                   "get_binary_class(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(vector_addition(dnn_part11, dnn_part12))))))))) AS fraud_type"});
      cataLog.setIdAddressMap(
          readFinancialTransactionsDataPlanNodeId,
          financialTransactionsDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readCustomerDataPlanNodeId,
          customerDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(
          readFinancialTransactionsDataPlanNodeId, "financial_transaction");
      cataLog.addNodeIdRelationName(readCustomerDataPlanNodeId, "customer");
      cataLog.addSource(std::make_shared<Source>(Source(
          readFinancialTransactionsDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(OutputStat(
              finicialTransactionsNumRows, finicialTransactionsNumCols)))));
      cataLog.addSource(std::make_shared<Source>(Source(
          readCustomerDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(customerNumRows, customerNumCols))));
    } else if (queryTemplate == "template3") {
      PlanNodeId readProductDataPlanNodeId;
      PlanNodeId readProductRatingDataPlanNodeId;
      PlanNodeId readProductRatingDataPlanNodeId1;
      PlanNodeId readCustomerDataPlanNodeId;

      auto productPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(productDataRowType, {}, "")
              .capturePlanNodeId(readProductDataPlanNodeId)
              .localPartition({"p_product_id"})
              .project({"p_product_id", "p_dept"})
              .hashJoin(
                  {"p_product_id"},
                  {"r_product_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(productRatingRowType, {}, "")
                      .capturePlanNodeId(readProductRatingDataPlanNodeId)
                      .localPartition({"r_product_id"})
                      .project({"r_product_id", "r_rating"})
                      .singleAggregation(
                          {"r_product_id"},
                          {"avg(r_rating) as avg_product_rating"})
                      .planNode(),
                  "",
                  {"p_product_id", "p_dept", "avg_product_rating"})
              .project(
                  {"p_product_id",
                   "concat(embedding_product(convert_int_array(p_product_id)), embedding_dept(convert_int_array(p_dept)), get_product_rating(CAST(avg_product_rating AS REAL))) as product_feature"})
              .project(
                  {"p_product_id",
                   "relu(batch_norm3_product(mat_vector_add_3_product(mat_mul_3_product(relu(batch_norm2_product(mat_vector_add_2_product(mat_mul_2_product(relu(batch_norm1_product(mat_vector_add_1_product(mat_mul_1_product(product_feature)))))))))))) AS product_encoding"});

      queryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(customerDataRowType, {}, "")
              .capturePlanNodeId(readCustomerDataPlanNodeId)
              .localPartition({"c_customer_sk"})
              .project(
                  {"c_customer_sk",
                   "c_address_num",
                   "get_age(c_birth_year) as age",
                   "c_birth_country",
                   "c_cust_flag"})
              .hashJoin(
                  {"c_customer_sk"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(productRatingRowType, {}, "")
                      .capturePlanNodeId(readProductRatingDataPlanNodeId1)
                      .localPartition({"r_user_id"})
                      .project({"r_user_id", "r_rating"})
                      .singleAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as avg_customer_rating"})
                      .filter("avg_customer_rating >= 4.0")
                      .planNode(),
                  "",
                  {"c_customer_sk",
                   "c_address_num",
                   "age",
                   "c_birth_country",
                   "c_cust_flag",
                   "avg_customer_rating"})
              .project(
                  {"c_customer_sk",
                   "avg_customer_rating",
                   "concat(embedding_customer(convert_int_array(c_customer_sk)), embedding_addr(convert_int_array(c_address_num)), embedding_age(convert_int_array(age)), embedding_country(convert_int_array(c_birth_country)), get_customer_extra_feature(c_cust_flag, CAST(avg_customer_rating AS REAL))) as customer_feature"})
              .project(
                  {"c_customer_sk",
                   "avg_customer_rating",
                   "relu(batch_norm3_customer(mat_vector_add_3_customer(mat_mul_3_customer(relu(batch_norm2_customer(mat_vector_add_2_customer(mat_mul_2_customer(relu(batch_norm1_customer(mat_vector_add_1_customer(mat_mul_1_customer(customer_feature)))))))))))) AS customer_encoding"})
              .nestedLoopJoin(
                  productPlan.planNode(),
                  {"c_customer_sk",
                   "avg_customer_rating",
                   "customer_encoding",
                   "p_product_id",
                   "product_encoding"})
              .project(
                  {"c_customer_sk",
                   "p_product_id",
                   "avg_customer_rating",
                   "vector_addition(customer_encoding, product_encoding) as final_encoding"});
      cataLog.setIdAddressMap(
          readProductDataPlanNodeId,
          productDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readProductRatingDataPlanNodeId,
          productRatingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readProductRatingDataPlanNodeId1,
          productRatingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readCustomerDataPlanNodeId,
          customerDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");
      cataLog.addNodeIdRelationName(
          readProductRatingDataPlanNodeId, "product_rating");
      cataLog.addNodeIdRelationName(
          readProductRatingDataPlanNodeId1, "product_rating");
      cataLog.addNodeIdRelationName(readCustomerDataPlanNodeId, "customer");
      Source productSrc = Source(
          readProductDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(productNumRows, productNumCols)));
      Source productRatingSrc = Source(
          readProductRatingDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(productRatingNumRows, productRatingNumCols)));
      Source productRatingSrc1 = Source(
          readProductRatingDataPlanNodeId1,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(productRatingNumRows, productRatingNumCols)));
      Source customerSrc = Source(
          readCustomerDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(customerNumRows, customerNumCols)));

    } else if (queryTemplate == "template4") { // uc7
      std::string svdModelPath =
          "/home/velox/resources/model/tpcxai_sf1/final/velox/tpcxai_template4_svd.h5";
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
              pu.size(),
              qi.size(),
              pu[0].size()),
          {},
          true,
          cataLog);

      queryPlan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(productRatingRowType, {}, "")
                      .capturePlanNodeId(readProductRatingDataPlanNodeId)
                      .hashJoin(
                          {"product_id"},
                          {"p_product_id"},
                          PlanBuilder(planNodeIdGenerator, pool_.get())
                              .tableScan(productDataRowType, {}, "")
                              .capturePlanNodeId(readProductDataPlanNodeId)
                              .planNode(),
                          "",
                          {
                              "product_id",
                              "user_id",
                              "department",
                          })
                      .project(
                          {"CAST (user_id AS INTEGER) AS user_id",
                           "CAST (product_id as INTEGER) AS product_id",
                           "department"})
                      .hashJoin(
                          {"user_id"},
                          {"c_customer_sk"},
                          PlanBuilder(planNodeIdGenerator, pool_.get())
                              .tableScan(customerDataRowType, {}, "")
                              .capturePlanNodeId(readCustomerDataPlanNodeId)
                              .planNode(),
                          "",
                          {"product_id",
                           "user_id",
                           "department",
                           "c_birth_day",
                           "c_birth_country"})
                      .project(
                          {"user_id",
                           "product_id",
                           "department",
                           "c_birth_day",
                           "c_birth_country",
                           "svd(user_id, product_id) as pred"});
      if (generateFilter) {
        std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr(
            "department_birthDay_birthCountry", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }

      cataLog.setIdAddressMap(
          readProductRatingDataPlanNodeId,
          productRatingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readProductDataPlanNodeId,
          productDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readCustomerDataPlanNodeId,
          customerDataPaths,
          dwio::common::FileFormat::PARQUET);

      cataLog.addNodeIdRelationName(
          readProductRatingDataPlanNodeId, "product_rating");
      cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");
      cataLog.addNodeIdRelationName(readCustomerDataPlanNodeId, "customer");

      Source productRatingSrc = Source(
          readProductRatingDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(productRatingNumRows, productRatingNumCols)));
      Source productSrc = Source(
          readProductDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(productNumRows, productNumCols)));
      Source customerSrc = Source(
          readCustomerDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(customerNumRows, customerNumCols)));

      cataLog.addSource(std::make_shared<Source>(productRatingSrc));
      cataLog.addSource(std::make_shared<Source>(productSrc));
      cataLog.addSource(std::make_shared<Source>(customerSrc));
    } else if (queryTemplate == "template6") { // uc8
      // Register model
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2
                << std::endl;
      auto modelStr = registerNNModel(
          {4, hidden1, hidden2, 384}, cataLog, modelGroupId_, false);
      // Register functions: department_encoder
      registerTPCxAIDepartmentEncoder(cataLog, pool_);

      // Query Plan
      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(orderDataRowType, {}, "")
              .capturePlanNodeId(readOrderDataPlanNodeId)
              .project({
                  "o_order_id",
                  "store",
                  "CAST (date AS TIMESTAMP) AS date",
              })
              .project({
                  "o_order_id",
                  "store",
                  "date",
                  "day_of_week(date) as weekday",
              })
              .hashJoin(
                  {"o_order_id"},
                  {"li_order_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(lineitemDataRowType, {}, "")
                      .capturePlanNodeId(readLineitemDataPlanNodeId)
                      .planNode(),
                  "",
                  {"li_product_id",
                   // "li_order_id",
                   "o_order_id",
                   "quantity",
                   "price",
                   "date",
                   "weekday"})
              .hashJoin(
                  {"li_product_id"},
                  {"p_product_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(productDataRowType, {}, "")
                      .capturePlanNodeId(readProductDataPlanNodeId)
                      .planNode(),
                  "",
                  {
                      "o_order_id",
                      // "li_order_id",
                      "quantity",
                      "price",
                      "date",
                      "weekday",
                      "department",
                  })
              .partialAggregation(
                  {"o_order_id", "date", "department", "quantity"},
                  {"sum(quantity) as scan_count",
                   "min(weekday) as weekday",
                   "avg(price) as price"})
              .finalAggregation()
              .project(
                  {"o_order_id",
                   "date",
                   "department",
                   "weekday",
                   "quantity",
                   "price",
                   "array_constructor(quantity, scan_count, weekday) as features",
                   "department_encoder(department) as department_encoded"})
              .project(
                  {"o_order_id",
                   "date",
                   "department",
                   "weekday",
                   "quantity",
                   "price",
                   "transform(concat(features, department_encoded), x-> CAST(x as REAL)) as features"})
              .project(
                  {"o_order_id",
                   "date",
                   "department",
                   "weekday",
                   "quantity",
                   "price",
                   fmt::format(modelStr, "features")});
      if (generateFilter) {
        std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr(
            "orderTime_department_weekday_price_quantity", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }

      cataLog.setIdAddressMap(
          readOrderDataPlanNodeId,
          orderDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readLineitemDataPlanNodeId,
          lineitemDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readProductDataPlanNodeId,
          productDataPaths,
          dwio::common::FileFormat::PARQUET);

      cataLog.addNodeIdRelationName(readOrderDataPlanNodeId, "order");
      cataLog.addNodeIdRelationName(readLineitemDataPlanNodeId, "lineitem");
      cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");

      Source orderSrc = Source(
          readOrderDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(OutputStat(orderNumRows, orderNumCols)));
      Source lineitemSrc = Source(
          readLineitemDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(lineitemNumRows, lineitemNumCols)));
      Source productSrc = Source(
          readProductDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(productNumRows, productNumCols)));

      cataLog.addSource(std::make_shared<Source>(orderSrc));
      cataLog.addSource(std::make_shared<Source>(lineitemSrc));
      cataLog.addSource(std::make_shared<Source>(productSrc));

    } else if (queryTemplate == "template7") { // uc10
      // Register model
      int hidden1 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << std::endl;
      auto modelStr =
          registerNNModel({2, hidden1, 1}, cataLog, modelGroupId_, false);

      queryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(finicialAccountDataRowType, {}, "")
              .capturePlanNodeId(readFinancialAccountDataPlanNodeId)
              .hashJoin(
                  {"fa_customer_sk"},
                  {"sender_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(finicialTransactionsDataRowType, {}, "")
                      .capturePlanNodeId(
                          readFinancialTransactionsDataPlanNodeId)
                      .project(
                          {"transaction_id",
                           "sender_id",
                           "CAST(time AS TIMESTAMP) as time",
                           "CAST(hour(CAST(time AS TIMESTAMP)) as DOUBLE) as business_hour",
                           "amount"})
                      .planNode(),
                  "",
                  {"transaction_id",
                   "fa_customer_sk",
                   "business_hour",
                   "amount",
                   "transaction_limit",
                   "time"})
              .project(
                  {"transaction_id",
                   "CAST(fa_customer_sk AS INTEGER) as fa_customer_sk",
                   "business_hour",
                   "amount",
                   "transaction_limit",
                   "time"})
              .hashJoin(
                  {"fa_customer_sk"},
                  {"c_customer_sk"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(customerDataRowType, {}, "")
                      .capturePlanNodeId(readCustomerDataPlanNodeId)
                      .project({"c_customer_sk", "c_birth_day"})
                      .planNode(),
                  "",
                  {"transaction_id",
                   "amount",
                   "time",
                   "c_birth_day",
                   "business_hour",
                   "transaction_limit"})
              .project(
                  {"transaction_id",
                   "amount",
                   "time",
                   "c_birth_day",
                   "amount / transaction_limit as amount_norm",
                   "business_hour / 23.0 as business_hour_norm"})
              .project(
                  {"transaction_id",
                   "amount",
                   "time",
                   "c_birth_day",
                   "transform(array_constructor(amount_norm, business_hour_norm), x-> CAST(X as REAL)) as features"})
              .project(
                  {"transaction_id",
                   "amount",
                   "time",
                   "c_birth_day",
                   fmt::format(modelStr, "features")});

      if (generateFilter) {
        std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr(
            "transactionTime_amount_birthDay", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }
      cataLog.setIdAddressMap(
          readFinancialAccountDataPlanNodeId,
          finicialAccountDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readFinancialTransactionsDataPlanNodeId,
          financialTransactionsDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readCustomerDataPlanNodeId,
          customerDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(
          readFinancialAccountDataPlanNodeId, "financial_account");
      cataLog.addNodeIdRelationName(
          readFinancialTransactionsDataPlanNodeId, "financial_transactions");
      cataLog.addNodeIdRelationName(readCustomerDataPlanNodeId, "customer");
      Source financialAccountSrc = Source(
          readFinancialAccountDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(finicialAccountNumRows, finicialAccountNumCols)));
      Source financialTransactionsSrc = Source(
          readFinancialTransactionsDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(OutputStat(
              finicialTransactionsNumRows, finicialTransactionsNumCols)));
      Source customerSrc = Source(
          readCustomerDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(customerNumRows, customerNumCols)));
      cataLog.addSource(std::make_shared<Source>(financialAccountSrc));
      cataLog.addSource(std::make_shared<Source>(financialTransactionsSrc));
      cataLog.addSource(std::make_shared<Source>(customerSrc));
    } else if (queryTemplate == "template8") { // uc3
      // Register functions: department_encoder
      registerTPCxAIDepartmentEncoder(cataLog, pool_);

      // Register model
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();

      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2
                << std::endl;
      auto modelStr = registerNNModel(
          {3, hidden1, hidden2, 1}, cataLog, modelGroupId_, false);

      // Query Plan
      queryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(storeDeptDataRowType, {}, "")
              .capturePlanNodeId(readStoreDeptDataPlanNodeId)
              .project({
                  "CAST(store as INTEGER) store",
                  "department",
                  "num_of_week",
                  "department_encoder(department) as department_encoded",
                  "CAST(num_of_week / 156 AS REAL) as num_of_week_norm",
              })
              .project(
                  {"store",
                   "department",
                   "num_of_week",
                   "transform(concat(store, department_encoded), x-> CAST(x as REAL))  as features1",
                   "array_constructor(num_of_week_norm) as features2"})
              .project(
                  {"store",
                   "department",
                   "num_of_week",
                   "concat(features1, features2) as features"})
              .project(
                  {"store",
                   "department",
                   "num_of_week",
                   fmt::format(modelStr, "features")});
      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleTPCxAIFilterExpr("department_numWeek_store", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }
      cataLog.setIdAddressMap(
          readStoreDeptDataPlanNodeId,
          storeDeptDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readStoreDeptDataPlanNodeId, "store_dept");
      Source storeDeptSrc = Source(
          readStoreDeptDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              OutputStat(storeDeptNumRows, storeDeptNumCols)));
      cataLog.addSource(std::make_shared<Source>(storeDeptSrc));
    } else if (queryTemplate == "template9") {
      // Register model
      int hidden1 = randomGenerator.genRandomIntValue();

      std::cout << "[INFO] hidden units: " << hidden1 << std::endl;
      auto modelStr =
          registerNNModel({2, hidden1, 30}, cataLog, modelGroupId_, false);
      auto makeGroupsBuilder = [&]() {
        auto plan =
            PlanBuilder(planNodeIdGenerator)
                .tableScan(orderDataRowType, {}, "")
                .capturePlanNodeId(readOrderDataPlanNodeId)
                .hashJoin(
                    {"o_order_id"},
                    {"li_order_id"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(lineitemDataRowType, {}, "")
                        .capturePlanNodeId(readLineitemDataPlanNodeId)
                        .planNode(),
                    /*extraFilter=*/{},
                    /*outputCols=*/
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "weekday",
                     "store",
                     "li_product_id",
                     "quantity",
                     "price"},
                    JoinType::kInner)
                .hashJoin(
                    {"li_product_id"},
                    {"p_product_id"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(productDataRowType, {}, "")
                        .capturePlanNodeId(readProductDataPlanNodeId)
                        .planNode(),
                    /*extraFilter=*/{},
                    /*outputCols=*/
                    {
                        "o_customer_sk",
                        "o_order_id",
                        "date",
                        "weekday",
                        "store", // from order table
                        "li_product_id",
                        "quantity",
                        "price", // from lineitem table
                        "name",
                        "department" // from product table
                    },
                    JoinType::kInner)
                .project(
                    {"o_customer_sk",
                     "date",
                     "weekday", // from order table
                     "store AS store_id", // from order table
                     "CAST(o_order_id AS INTEGER) AS o_order_id",
                     "quantity",
                     "price", // from lineitem table
                     "li_product_id AS product_id", // from lineitem table
                     "name",
                     "department"}) // from product table
                .hashJoin(
                    {"o_order_id"},
                    {"or_order_id"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(orderReturnDataRowType, {}, "")
                        .capturePlanNodeId(readOrderReturnDataPlanNodeId)
                        .planNode(),
                    /*extraFilter=*/{},
                    /*outputCols=*/
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "store_id",
                     "product_id",
                     "department",
                     "quantity",
                     "price",
                     "or_return_quantity"},
                    JoinType::kInner)
                .project(
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "store_id",
                     "product_id",
                     "department",
                     "year(parse_datetime(date, 'yyyy-MM-dd HH:mm:ss')) AS year_",
                     "quantity",
                     "price",
                     "or_return_quantity",
                     "(cast(or_return_quantity as DOUBLE) * price) as rq_p",
                     "(cast(quantity as DOUBLE) * price) as q_p"})
                .partialAggregation(
                    /*groupKeys=*/
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "store_id",
                     "product_id",
                     "department"},
                    /*aggregates=*/
                    {"min(year_) as invoice_year",
                     "sum(rq_p) as num",
                     "sum(q_p) as den"})
                .finalAggregation()
                .project(
                    {"o_customer_sk",
                     "o_order_id",
                     "invoice_year",
                     "(num / den) AS ratio",
                     "date",
                     "store_id",
                     "product_id",
                     "department"});
        if (generateFilter) {
          std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr(
              "orderTime_store_product_department", timestampSeed);
          for (auto expr : filterExpr) {
            plan = plan.filter(expr);
          }
        }
        // order
        cataLog.setIdAddressMap(
            readOrderDataPlanNodeId,
            orderDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readOrderDataPlanNodeId, "order");
        cataLog.addSource(std::make_shared<Source>(Source(
            readOrderDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(orderNumRows, orderNumCols))));

        // lineitem
        cataLog.setIdAddressMap(
            readLineitemDataPlanNodeId,
            lineitemDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readLineitemDataPlanNodeId, "lineitem");
        cataLog.addSource(std::make_shared<Source>(Source(
            readLineitemDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(lineitemNumRows, lineitemNumCols))));

        // Product
        cataLog.setIdAddressMap(
            readProductDataPlanNodeId,
            productDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");
        cataLog.addSource(std::make_shared<Source>(Source(
            readProductDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(productNumRows, productNumCols))));

        // order_return
        cataLog.setIdAddressMap(
            readOrderReturnDataPlanNodeId,
            orderReturnDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(
            readOrderReturnDataPlanNodeId, "order_returns");
        cataLog.addSource(std::make_shared<Source>(Source(
            readOrderReturnDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(
                orderReturnNumRows, orderReturnNumCols))));
        return plan;
      };

      auto ratioBuilder =
          makeGroupsBuilder()
              .partialAggregation(
                  /*groupKeys=*/{"o_customer_sk"},
                  /*aggregates=*/{"avg(ratio) as avg_return_ratio"})
              .finalAggregation()
              .project({"o_customer_sk", "avg_return_ratio"});

      auto frequencyBuilder =
          makeGroupsBuilder()
              .partialAggregation(
                  {"o_customer_sk", "invoice_year"}, {"count(1) as num_return"})
              .finalAggregation()
              .project({"o_customer_sk", "invoice_year", "num_return"})
              .partialAggregation(
                  {"o_customer_sk"}, {"avg(num_return) as avg_num_return"})
              .finalAggregation()
              .project({"o_customer_sk AS freq_customer_sk", "avg_num_return"});

      auto frequencyPlanNode = frequencyBuilder.planNode();

      queryPlan =
          ratioBuilder
              .hashJoin(
                  {"o_customer_sk"},
                  {"freq_customer_sk"},
                  frequencyPlanNode,
                  /*extraFilter=*/{},
                  /*outputCols=*/
                  {"o_customer_sk", "avg_return_ratio", "avg_num_return"},
                  JoinType::kInner)
              .project(
                  {"o_customer_sk",
                   "CAST(avg_return_ratio AS REAL) as avg_return_ratio",
                   "CAST(avg_num_return AS REAL) as avg_num_return"})
              .project(
                  {"o_customer_sk",
                   "array_constructor(avg_return_ratio, avg_num_return) as features"})
              .project({"o_customer_sk", fmt::format(modelStr, "features")});

    } else {
      throw std::runtime_error(
          "Unsupported query template for tpcxai workload : " + queryTemplate);
    }

  } else {
    throw std::runtime_error(
        "Unsupported workload: " + workload +
        ". Currently, movielens and tpcxai are supported.");
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
  randomGenerator.setIntRange(10, 3000);
  PlanBuilder queryPlan;
  VectorMaker maker{pool_.get()};

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

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");

      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));

    }

    // TODO
    // template 4
    else if (queryTemplate == "template4") {
      registerGenderEncoder(cataLog, pool_);
      // gender_encoder
      registerMovielensAgeMinMaxScaler(
          cataLog, pool_, "q4_user_age_minmax_scaler.txt");
      // user_occupation_minmax_scaler
      registerMovielensOccupationMinMaxScaler(
          cataLog, pool_, "q4_user_occupation_minmax_scaler.txt");
      // user_occupation_minmax_scaler
      registerMovielensGenerOneHotEncoder(cataLog, pool_);
      // genres_encode
      // registerNNfunctions
      //  Register model
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();
      int hidden3 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2 << ", "
                << hidden3 << std::endl;
      // auto modelStr = registerNNModel({4, hidden1, hidden2, 384}, cataLog,
      // modelGroupId_, false);

      int modelGroupId_ = 0;
      std::string ffnnstring = registerNNModel(
          {21, hidden1, hidden2, hidden3, 1},
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());
      // std::cout << ffnnstring << "\n";
      queryPlan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .nestedLoopJoin(
                          PlanBuilder(planNodeIdGenerator)
                              .tableScan(movieDataRowType, {}, "")
                              .capturePlanNodeId(readMovieDataPlanNodeId)
                              .planNode(),
                          {// what columns to project from the join
                           "u_user_id",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "m_movie_id",
                           "m_genres"});

      // Filter here
      if (generateFilter) {
        std::vector<std::string> filterExpr = sampleUserMovieFilterExpr(
            "age_gender_occupation_genre", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      queryPlan =
          queryPlan
              .project(
                  {"gender_encoder(u_gender) as u_gender_encoded",
                   "user_age_minmax_scaler(transform(array_constructor(u_age), x-> CAST(x AS REAL))) as u_age_encoded",
                   "user_occupation_minmax_scaler(transform(array_constructor(u_occupation), x-> CAST(x AS REAL))) as u_occupation_encoded",
                   "genres_encoder(m_genres) as m_genres_encoded"})
              .project(
                  {"transform(concat(u_gender_encoded,u_age_encoded,u_occupation_encoded,m_genres_encoded), x-> CAST(x AS REAL)) as features"}
                  // ARRAY(REAL)
                  )
              .project({fmt::format(ffnnstring, "features")});

      // — user side
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addSource(std::make_shared<Source>(Source(
          readUserDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(userNumRows, userNumCols))));

      // — movie side
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      cataLog.addSource(std::make_shared<Source>(Source(
          readMovieDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(movieNumRows, movieNumCols))));
    }

    else if (queryTemplate == "template9") {
      registerMovielensPopularityMinMaxScaler(cataLog, pool_);
      registerMovielensVoteAverageMinMaxScaler(cataLog, pool_);
      registerMovielensVoteCountMinMaxScaler(cataLog, pool_);
      registerMovielensRatingMinMaxScaler(cataLog, pool_);
      registerMovielensGenerOneHotEncoder(cataLog, pool_);
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2
                << std::endl;
      int modelGroupId = 0;
      auto dnnString = registerNNModel(
          {4, hidden1, hidden2, 384}, cataLog, modelGroupId_, false);
      queryPlan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .hashJoin(
                          {"m_movie_id"},
                          {"r_movie_id"},
                          PlanBuilder(planNodeIdGenerator)
                              .tableScan(ratingDataRowType, {}, "")
                              .capturePlanNodeId(readRatingDataPlanNodeId1)
                              .project({"r_movie_id", "r_rating"})
                              .planNode(),
                          "",
                          {"m_movie_id",
                           "m_genres",
                           "m_title",
                           "m_spoken_languages",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count",
                           "r_rating"},
                          /*joinType=*/core::JoinType::kInner);

      // filter expressions
      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("genre_rating", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      queryPlan =
          queryPlan
              .project(
                  {"movie_popularity_minmax_scaler(transform(array_constructor(m_popularity),    x-> CAST(x AS REAL))) as m_popularity_norm",
                   "movie_vote_average_minmax_scaler(transform(array_constructor(m_vote_average), x-> CAST(x AS REAL))) as m_vote_avg_norm",
                   "movie_vote_count_minmax_scaler(transform(array_constructor(m_vote_count),    x-> CAST(x AS REAL))) as m_vote_count_norm",
                   "rating_minmax_scaler(transform(array_constructor(r_rating),  x-> CAST(x AS REAL))) as r_rating_norm",
                   // one-hot encode genres
                   "genres_encoder(m_genres)  as m_genres_one_hot"})
              .project(
                  {"transform(concat(m_popularity_norm,m_vote_avg_norm,m_vote_count_norm,r_rating_norm,m_genres_one_hot),x -> CAST(x AS REAL)) as features"})
              .project({fmt::format(dnnString, "features")});

      // movie side
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      cataLog.addSource(std::make_shared<Source>(Source(
          readMovieDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(movieNumRows, movieNumCols))));

      // rating side
      cataLog.setIdAddressMap(
          readRatingDataPlanNodeId1,
          ratingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1, "ratings");
      cataLog.addSource(std::make_shared<Source>(Source(
          readRatingDataPlanNodeId1,
          Source::Type::FILE,
          std::make_shared<OutputStat>(ratingNumRows, ratingNumCols))));
    }

    else if (queryTemplate == "movie_rating_pivot") {
      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(ratingDataRowType, {}, "")
              .capturePlanNodeId(readRatingDataPlanNodeId1)
              .project({"r_user_id", "r_movie_id", "r_rating", "r_timestamp"})
              .partialAggregation(
                  {"r_user_id"}, {"map_agg(r_movie_id, r_rating)"})
              .finalAggregation();
      cataLog.setIdAddressMap(
          readRatingDataPlanNodeId1,
          ratingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1, "movie_rating");
      std::shared_ptr<OutputStat> ratingStats = std::make_shared<OutputStat>(
          OutputStat(ratingNumRows, ratingNumCols));
      Source ratingSrc1 =
          Source(readRatingDataPlanNodeId1, Source::Type::FILE, ratingStats);
      cataLog.addSource(std::make_shared<Source>(ratingSrc1));

    }

    else if (queryTemplate == "template8") {
      // register vector function
      registerMovielensRatingMapToArray(cataLog, pool_);

      int modelGroupId = 0;
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();
      int hidden3 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2 << ", "
                << hidden3 << std::endl;
      auto autoencoder = registerNNModel(
          /*layers=*/{3706, hidden1, hidden2, hidden3, 1},
          cataLog,
          modelGroupId,
          randomGenerator.genRandomIntValue());

      queryPlan = PlanBuilder(planNodeIdGenerator)
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .hashJoin(
                          {"u_user_id"},
                          {"r_user_id"},
                          PlanBuilder(planNodeIdGenerator)
                              .tableScan(ratingDataRowType, {}, "")
                              .capturePlanNodeId(readRatingDataPlanNodeId1)
                              .project(
                                  {"r_user_id",
                                   "r_timestamp",
                                   "r_movie_id",
                                   "r_rating"})
                              .planNode(),
                          {}, // extra filters
                          // join project columns
                          {"u_user_id",
                           "u_gender",
                           "u_age",
                           "u_occupation",
                           "u_zipcode",
                           "r_movie_id",
                           "r_rating"},
                          /*joinType=*/core::JoinType::kInner);

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("user", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      queryPlan =
          queryPlan
              .partialAggregation(
                  {"u_user_id"},
                  {"(map_agg(r_movie_id, r_rating)) as map_ratings"})
              .finalAggregation()
              .project({"u_user_id", "map_to_array(map_ratings) as mappings_"})
              .project({fmt::format(autoencoder, "mappings_")});
      ;

      // user_side
      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addSource(std::make_shared<Source>(Source(
          readUserDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(userNumRows, userNumCols))));

      // rating side
      cataLog.setIdAddressMap(
          readRatingDataPlanNodeId1,
          ratingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1, "ratings");
      cataLog.addSource(std::make_shared<Source>(Source(
          readRatingDataPlanNodeId1,
          Source::Type::FILE,
          std::make_shared<OutputStat>(ratingNumRows, ratingNumCols))));
    }

  }

  /*************************************tpcxai*************************************/

  else if (workload == "tpcxai1") {
    auto finicialAccountDataRowType =
        ROW({"fa_customer_sk", "transaction_limit"}, {BIGINT(), DOUBLE()});
    auto finicialTransactionsDataRowType =
        ROW({"amount",
             "iban",
             "sender_id",
             "receiver_id",
             "transaction_id",
             "time"},
            {DOUBLE(), VARCHAR(), BIGINT(), VARCHAR(), BIGINT(), VARCHAR()});
    auto orderDataRowType =
        ROW({"o_order_id", "o_customer_sk", "weekday", "date", "store"},
            {BIGINT(), BIGINT(), VARCHAR(), VARCHAR(), BIGINT()});
    auto lineitemDataRowType =
        ROW({"li_order_id", "li_product_id", "quantity", "price"},
            {BIGINT(), BIGINT(), BIGINT(), DOUBLE()});
    auto productDataRowType =
        ROW({"p_product_id", "name", "department"},
            {BIGINT(), VARCHAR(), VARCHAR()});
    auto storeDeptDataRowType =
        ROW({"store", "department", "num_of_week"},
            {BIGINT(), VARCHAR(), BIGINT()});
    auto productRatingRowType =
        ROW({"user_id", "product_id"}, {BIGINT(), BIGINT()});
    auto customerDataRowType =
        ROW({"c_customer_sk",
             "c_customer_id",
             "c_current_addr_sk",
             "c_first_name",
             "c_last_name",
             "c_preferred_cust_flag",
             "c_birth_day",
             "c_birth_month",
             "c_birth_year",
             "c_birth_country",
             "c_login",
             "c_email_address"},
            {INTEGER(),
             VARCHAR(),
             INTEGER(),
             VARCHAR(),
             VARCHAR(),
             VARCHAR(),
             INTEGER(),
             INTEGER(),
             INTEGER(),
             VARCHAR(),
             VARCHAR(),
             VARCHAR()});
    auto orderReturnDataRowType =
        ROW({"or_order_id", "or_product_id", "or_return_quantity"},
            {BIGINT(), BIGINT(), INTEGER()});
    auto reviewDataRowType = ROW({"id", "text"}, {INTEGER(), VARCHAR()});

    std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

    if (dataDirPrefix == "") {
      // use default value:
      dataDirPrefix =
          "/home/velox/resources/data/parquet/tpcxai_sf1/final/serving/";
    }

    std::vector<std::string> finicialAccountDataPaths =
        getFilePathsFromDir(dataDirPrefix + "financial_account");
    std::vector<std::string> financialTransactionsDataPaths =
        getFilePathsFromDir(dataDirPrefix + "financial_transactions");
    std::vector<std::string> orderDataPaths =
        getFilePathsFromDir(dataDirPrefix + "order");
    std::vector<std::string> lineitemDataPaths =
        getFilePathsFromDir(dataDirPrefix + "lineitem");
    std::vector<std::string> productDataPaths =
        getFilePathsFromDir(dataDirPrefix + "product");
    std::vector<std::string> storeDeptDataPaths =
        getFilePathsFromDir(dataDirPrefix + "store_dept");
    std::vector<std::string> productRatingDataPaths =
        getFilePathsFromDir(dataDirPrefix + "product_rating");
    std::vector<std::string> customerDataPaths =
        getFilePathsFromDir(dataDirPrefix + "customer");
    std::vector<std::string> orderReturnDataPaths =
        getFilePathsFromDir(dataDirPrefix + "order_returns");
    std::vector<std::string> reviewDataPaths =
        getFilePathsFromDir(dataDirPrefix + "review");

    int finicialAccountNumRows, finicialAccountNumCols,
        finicialTransactionsNumRows, finicialTransactionsNumCols, orderNumRows,
        orderNumCols, lineitemNumRows, lineitemNumCols, productNumRows,
        productNumCols, storeDeptNumRows, storeDeptNumCols,
        productRatingNumRows, productRatingNumCols, customerNumRows,
        customerNumCols, orderReturnNumRows, orderReturnNumCols, reviewNumRows,
        reviewNumCols;

    readDataStats(
        dataDirPrefix + "financial_account_stats.txt",
        finicialAccountNumRows,
        finicialAccountNumCols);
    readDataStats(
        dataDirPrefix + "financial_transactions_stats.txt",
        finicialTransactionsNumRows,
        finicialTransactionsNumCols);
    readDataStats(
        dataDirPrefix + "order_stats.txt", orderNumRows, orderNumCols);
    readDataStats(
        dataDirPrefix + "lineitem_stats.txt", lineitemNumRows, lineitemNumCols);
    readDataStats(
        dataDirPrefix + "product_stats.txt", productNumRows, productNumCols);
    readDataStats(
        dataDirPrefix + "store_dept_stats.txt",
        storeDeptNumRows,
        storeDeptNumCols);
    readDataStats(
        dataDirPrefix + "product_rating_stats.txt",
        productRatingNumRows,
        productRatingNumCols);
    readDataStats(
        dataDirPrefix + "customer_stats.txt", customerNumRows, customerNumCols);
    readDataStats(
        dataDirPrefix + "order_returns_stats.txt",
        orderReturnNumRows,
        orderReturnNumCols);
    readDataStats(
        dataDirPrefix + "review_stats.txt", reviewNumRows, reviewNumCols);

    PlanNodeId readOrderDataPlanNodeId;
    PlanNodeId readOrderReturnDataPlanNodeId;
    PlanNodeId readLineitemDataPlanNodeId;
    PlanNodeId readProductDataPlanNodeId;
    PlanNodeId readReviewDataPlanNodeId;
    PlanNodeId readCustomerDataPlanNodeId;
    PlanNodeId readProductRatingPlanNodeId;

    if (queryTemplate == "template5") {
      // registerTPCxAIHFTokenizer
      registerTPCxAIHFTokenizer(cataLog, pool_);
      registerTPCxAITFFeatureExtractor(cataLog, pool_);

      int modelGroupId_ = 0;
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();
      // int hidden3 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2 << ", "
                << std::endl;
      auto modelStr = registerNNModel(
          {50265, hidden1, hidden2, 1}, cataLog, modelGroupId_, false);

      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(reviewDataRowType, {}, "")
              .capturePlanNodeId(readReviewDataPlanNodeId)
              .project(
                  {"id", "extract_tf_features(hf_tokenizer(text)) as feature"});

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleTPCxAIFilterExpr("idReview", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }
      queryPlan = queryPlan.project({fmt::format(modelStr, "feature")});

      // review Data read
      cataLog.setIdAddressMap(
          readReviewDataPlanNodeId,
          reviewDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readReviewDataPlanNodeId, "review");
      cataLog.addSource(std::make_shared<Source>(Source(
          readReviewDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(reviewNumRows, reviewNumCols))));
    }

    else if (queryTemplate == "template9") {
      // Register model
      int hidden1 = randomGenerator.genRandomIntValue();

      std::cout << "[INFO] hidden units: " << hidden1 << std::endl;
      auto modelStr =
          registerNNModel({2, hidden1, 30}, cataLog, modelGroupId_, false);
      auto makeGroupsBuilder = [&]() {
        auto plan =
            PlanBuilder(planNodeIdGenerator)
                .tableScan(orderDataRowType, {}, "")
                .capturePlanNodeId(readOrderDataPlanNodeId)
                .hashJoin(
                    {"o_order_id"},
                    {"li_order_id"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(lineitemDataRowType, {}, "")
                        .capturePlanNodeId(readLineitemDataPlanNodeId)
                        .planNode(),
                    /*extraFilter=*/{},
                    /*outputCols=*/
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "weekday",
                     "store",
                     "li_product_id",
                     "quantity",
                     "price"},
                    JoinType::kInner)
                .hashJoin(
                    {"li_product_id"},
                    {"p_product_id"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(productDataRowType, {}, "")
                        .capturePlanNodeId(readProductDataPlanNodeId)
                        .planNode(),
                    /*extraFilter=*/{},
                    /*outputCols=*/
                    {
                        "o_customer_sk",
                        "o_order_id",
                        "date",
                        "weekday",
                        "store", // from order table
                        "li_product_id",
                        "quantity",
                        "price", // from lineitem table
                        "name",
                        "department" // from product table
                    },
                    JoinType::kInner)
                .project(
                    {"o_customer_sk",
                     "date",
                     "weekday", // from order table
                     "store AS store_id", // from order table
                     "CAST(o_order_id AS INTEGER) AS o_order_id",
                     "quantity",
                     "price", // from lineitem table
                     "li_product_id AS product_id", // from lineitem table
                     "name",
                     "department"}) // from product table
                .hashJoin(
                    {"o_order_id"},
                    {"or_order_id"},
                    PlanBuilder(planNodeIdGenerator)
                        .tableScan(orderReturnDataRowType, {}, "")
                        .capturePlanNodeId(readOrderReturnDataPlanNodeId)
                        .planNode(),
                    /*extraFilter=*/{},
                    /*outputCols=*/
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "store_id",
                     "product_id",
                     "department",
                     "quantity",
                     "price",
                     "or_return_quantity"},
                    JoinType::kInner)
                .project(
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "store_id",
                     "product_id",
                     "department",
                     "year(parse_datetime(date, 'yyyy-MM-dd HH:mm:ss')) AS year_",
                     "quantity",
                     "price",
                     "or_return_quantity",
                     "(cast(or_return_quantity as DOUBLE) * price) as rq_p",
                     "(cast(quantity as DOUBLE) * price) as q_p"})
                .partialAggregation(
                    /*groupKeys=*/
                    {"o_customer_sk",
                     "o_order_id",
                     "date",
                     "store_id",
                     "product_id",
                     "department"},
                    /*aggregates=*/
                    {"min(year_) as invoice_year",
                     "sum(rq_p) as num",
                     "sum(q_p) as den"})
                .finalAggregation()
                .project(
                    {"o_customer_sk",
                     "o_order_id",
                     "invoice_year",
                     "(num / den) AS ratio",
                     "date",
                     "store_id",
                     "product_id",
                     "department"});
        if (generateFilter) {
          std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr(
              "orderTime_store_product_department", timestampSeed);
          for (auto expr : filterExpr) {
            plan = plan.filter(expr);
          }
        }
        // order
        cataLog.setIdAddressMap(
            readOrderDataPlanNodeId,
            orderDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readOrderDataPlanNodeId, "order");
        cataLog.addSource(std::make_shared<Source>(Source(
            readOrderDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(orderNumRows, orderNumCols))));

        // lineitem
        cataLog.setIdAddressMap(
            readLineitemDataPlanNodeId,
            lineitemDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readLineitemDataPlanNodeId, "lineitem");
        cataLog.addSource(std::make_shared<Source>(Source(
            readLineitemDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(lineitemNumRows, lineitemNumCols))));

        // Product
        cataLog.setIdAddressMap(
            readProductDataPlanNodeId,
            productDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");
        cataLog.addSource(std::make_shared<Source>(Source(
            readProductDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(productNumRows, productNumCols))));

        // order_return
        cataLog.setIdAddressMap(
            readOrderReturnDataPlanNodeId,
            orderReturnDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(
            readOrderReturnDataPlanNodeId, "order_returns");
        cataLog.addSource(std::make_shared<Source>(Source(
            readOrderReturnDataPlanNodeId,
            Source::Type::FILE,
            std::make_shared<OutputStat>(
                orderReturnNumRows, orderReturnNumCols))));
        return plan;
      };

    }

    else if (queryTemplate == "template10") {
      // department_encode
      registerTPCxAIDepartmentEncoder(cataLog, pool_);
      int hidden1 = randomGenerator.genRandomIntValue();
      int hidden2 = randomGenerator.genRandomIntValue();
      int hidden3 = randomGenerator.genRandomIntValue();
      std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2 << ", "
                << hidden3 << std::endl;
      int modelGroupId_ = 0;
      auto modelStr = registerNNModel(
          {4, hidden1, hidden2, hidden3, 1}, cataLog, modelGroupId_, false);

      queryPlan =
          PlanBuilder(planNodeIdGenerator)
              .tableScan(productRatingRowType, {}, "")
              .capturePlanNodeId(readProductRatingPlanNodeId)
              .hashJoin(
                  {"user_id"},
                  {"c_customer_sk"},
                  PlanBuilder(planNodeIdGenerator)
                      .tableScan(customerDataRowType, {}, "")
                      .capturePlanNodeId(readCustomerDataPlanNodeId)
                      .project(
                          {"cast(c_customer_sk as BIGINT) as c_customer_sk",
                           "c_birth_day",
                           "c_birth_month",
                           "c_birth_year",
                           "c_birth_country"})
                      .planNode(),
                  "",
                  {"user_id",
                   "product_id",
                   "c_customer_sk",
                   "c_birth_day",
                   "c_birth_month",
                   "c_birth_year",
                   "c_birth_country"},
                  JoinType::kInner)
              .hashJoin(
                  {"product_id"},
                  {"p_product_id"},
                  PlanBuilder(planNodeIdGenerator)
                      .tableScan(
                          productDataRowType, // { p_product_id BIGINT, name
                                              // VARCHAR, department VARCHAR }
                          {},
                          "")
                      .capturePlanNodeId(readProductDataPlanNodeId)
                      .planNode(),
                  "",
                  {"c_birth_day",
                   "c_birth_month",
                   "c_birth_year",
                   "c_birth_country",
                   "department"},
                  JoinType::kInner);

      if (generateFilter) {
        std::vector<std::string> filterExpr = sampleTPCxAIFilterExpr(
            "department_birthDay_birthCountry", timestampSeed);
        for (auto expr : filterExpr) {
          queryPlan = queryPlan.filter(expr);
        }
      }

      queryPlan =
          queryPlan
              .project(
                  {"department_encoder(department) department_",
                   "(1922.0 -   cast(c_birth_year as double))/(79.0) AS birth_year",
                   "(12.0   -   cast(c_birth_month as double))/(11.0) AS birth_month",
                   "(31.0   -   cast(c_birth_day as double))/(30.0) AS birth_day"})
              .project(
                  {"transform(concat(department_,array_constructor(birth_year),array_constructor(birth_month),array_constructor(birth_day)),x-> CAST(x AS REAL)) as features"})
              .project({fmt::format(modelStr, "features")});

      cataLog.setIdAddressMap(
          readCustomerDataPlanNodeId,
          customerDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readCustomerDataPlanNodeId, "customer");
      cataLog.addSource(std::make_shared<Source>(Source(
          readCustomerDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(customerNumRows, customerNumCols))));

      // rating
      cataLog.setIdAddressMap(
          readProductRatingPlanNodeId,
          productRatingDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readProductRatingPlanNodeId, "rating");
      cataLog.addSource(std::make_shared<Source>(Source(
          readProductRatingPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              productRatingNumRows, productRatingNumCols))));

      // product
      cataLog.setIdAddressMap(
          readProductDataPlanNodeId,
          productDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");
      cataLog.addSource(std::make_shared<Source>(Source(
          readProductDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(productNumRows, productNumCols))));
    }

  } 
  else if(workload == "imbridge"){  
    if(queryTemplate == "expedia"){

        // std::cout << "We are here!!!\n";
        //expedia starts here
        //std::string dataDir = getenv("CD_DATA_DIR_PREFIX");
        auto R1_hotelsType = ROW(
            {"prop_id",
            "prop_country_id",
            "prop_starrating",
            "prop_review_score",
            "prop_brand_bool",
            "count_clicks",
            "avg_bookings_usd",
            "stdev_bookings_usd",
            "count_bookings"},
            {VARCHAR(),   // prop_id
            VARCHAR(),   // prop_country_id
            BIGINT(),    // prop_starrating
            DOUBLE(),    // prop_review_score
            BIGINT(),    // prop_brand_bool
            BIGINT(),    // count_clicks
            DOUBLE(),    // avg_bookings_usd
            DOUBLE(),    // stdev_bookings_usd
            BIGINT()});  // count_bookings

        auto R2_searchesType = ROW(
            {"srch_id",
            "year",
            "month",
            "weekofyear",
            "time",
            "site_id",
            "visitor_location_country_id",
            "srch_destination_id",
            "srch_length_of_stay",
            "srch_booking_window",
            "srch_adults_count",
            "srch_children_count",
            "srch_room_count",
            "srch_saturday_night_bool",
            "random_bool"},
            {VARCHAR(), // srch_id
            VARCHAR(), // year
            VARCHAR(), // month
            VARCHAR(), // weekofyear
            VARCHAR(), // time
            VARCHAR(), // site_id
            VARCHAR(), // visitor_location_country_id
            VARCHAR(), // srch_destination_id
            BIGINT(),  // srch_length_of_stay
            BIGINT(),  // srch_booking_window
            BIGINT(),  // srch_adults_count
            BIGINT(),  // srch_children_count
            BIGINT(),  // srch_room_count
            BIGINT(),  // srch_saturday_night_bool
            BIGINT()   // random_bool
            });

        auto S_listingsExtType = ROW(
            {"srch_id",
            "prop_id",
            "position",
            "prop_location_score1",
            "prop_location_score2",
            "prop_log_historical_price",
            "price_usd",
            "promotion_flag",
            "orig_destination_distance"},
            {VARCHAR(), // srch_id
            VARCHAR(), // prop_id
            VARCHAR(), // position
            DOUBLE(),  // prop_location_score1
            DOUBLE(),  // prop_location_score2
            DOUBLE(),  // prop_log_historical_price
            DOUBLE(),  // price_usd
            BIGINT(),  // promotion_flag
            DOUBLE()   // orig_destination_distance
            });
        
    std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

    if (dataDirPrefix == "") {
      // use default value:
      dataDirPrefix =
          "/home/cactusdb/resources/data/parquet/expedia/";
    }
    std::vector<std::string> R1_hotelsDataPaths =
        getFilePathsFromDir(dataDirPrefix + "R1_hotels");
    std::vector<std::string> R2_searchesDataPath =
        getFilePathsFromDir(dataDirPrefix + "R2_searches");
    std::vector<std::string> S_listings_extensionDataPath =
        getFilePathsFromDir(dataDirPrefix + "S_listings_extension");

    int R1_hotelsNumRows, R1_hotelsNumCols, R2_searchesNumRows, R2_searchesNumCols, S_listings_extensionNumRows, S_listings_extensionNumCols;

    readDataStats(
        dataDirPrefix + "R1_hotels_stats.txt",
        R1_hotelsNumRows,
        R1_hotelsNumCols);

    readDataStats(
        dataDirPrefix + "R2_searches_stats.txt",
        R2_searchesNumRows,
        R2_searchesNumCols);

    readDataStats(
        dataDirPrefix + "S_listings_extension_stats.txt",
        S_listings_extensionNumRows,
        S_listings_extensionNumCols);

    PlanNodeId readR1_HotelsDataPlanNodeId;
    PlanNodeId readR2_searchesDataPlanNodeId;
    PlanNodeId readS_listings_extensionDataPlanNodeId;

    std::cout << "OKAY UPTO QUERYPLAN!!" << "\n";

    std::vector<std::pair<std::string, std::string>> encoderSpecs = {
    {"one_hot_prop_starrating", "/home/cactusdb/resources/model/expedia/onehotencoders/prop_starrating.csv"},
    {"one_hot_prop_brand_bool", "/home/cactusdb/resources/model/expedia/onehotencoders/prop_brand_bool.csv"},
    {"one_hot_count_clicks", "/home/cactusdb/resources/model/expedia/onehotencoders/count_clicks.csv"},
    {"one_hot_count_bookings", "/home/cactusdb/resources/model/expedia/onehotencoders/count_bookings.csv"},
    {"one_hot_srch_length_of_stay", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_length_of_stay.csv"},
    {"one_hot_srch_booking_window", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_booking_window.csv"},
    {"one_hot_srch_adults_count", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_adults_count.csv"},
    {"one_hot_srch_children_count", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_children_count.csv"},
    {"one_hot_srch_room_count", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_room_count.csv"},
    {"one_hot_srch_saturday_night_bool", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_saturday_night_bool.csv"},
    {"one_hot_random_bool", "/home/cactusdb/resources/model/expedia/onehotencoders/random_bool.csv"}
    };

    std::vector<std::pair<std::string, std::string>> stringEncoderSpecs = {
    {"one_hot_position", "/home/cactusdb/resources/model/expedia/onehotencoders/position.csv"},
    {"one_hot_prop_country_id", "/home/cactusdb/resources/model/expedia/onehotencoders/prop_country_id.csv"},
    {"one_hot_year", "/home/cactusdb/resources/model/expedia/onehotencoders/year.csv"},
    {"one_hot_month", "/home/cactusdb/resources/model/expedia/onehotencoders/month.csv"},
    {"one_hot_weekofyear", "/home/cactusdb/resources/model/expedia/onehotencoders/weekofyear.csv"},
    {"one_hot_time", "/home/cactusdb/resources/model/expedia/onehotencoders/time.csv"},
    {"one_hot_site_id", "/home/cactusdb/resources/model/expedia/onehotencoders/site_id.csv"},
    {"one_hot_visitor_location_country_id", "/home/cactusdb/resources/model/expedia/onehotencoders/visitor_location_country_id.csv"},
    {"one_hot_srch_destination_id", "/home/cactusdb/resources/model/expedia/onehotencoders/srch_destination_id.csv"}
    };

    for (const auto& [functionName, filePath] : encoderSpecs) {
        registerOneHotInt(functionName, filePath, cataLog, pool_);
    }

    for (const auto& [functionName, filePath] : stringEncoderSpecs) {
    registerOneHotString(functionName, filePath, cataLog, pool_);
    }

    // registerTreePredictExpedia(cataLog, pool_);
    optimization::registerVectorFunction(
        "decision_tree_predict",
        TreePrediction::signatures(),
        std::make_unique<TreePrediction>(
            0,
            "/home/cactusdb/resources/model/expedia/tree/0.txt",
            3979,
        true),
        {},
        /*deterministic=*/true,
        cataLog);

    const std::string statsDir = "/home/cactusdb/resources/model/expedia/scaler_files";

        // List of features to register:
        const std::vector<std::string> features = {
        "prop_location_score1", //f
        "prop_location_score2", //f
        "prop_log_historical_price", //f
        "price_usd",
        "orig_destination_distance",
        "prop_review_score",
        "avg_bookings_usd",
        "stdev_bookings_usd",
        };

        for (auto& feature : features) {
            // Build full path to that feature’s stats file (e.g. "/…/v1.txt")
            const std::string filePath = fmt::format("{}/{}.txt", statsDir, feature);
            registerFeatureMinMaxScaler(cataLog, pool_, feature, filePath);
        }
    
    queryPlan = PlanBuilder(planNodeIdGenerator)
        .tableScan(S_listingsExtType,{}, "")
        .capturePlanNodeId(readS_listings_extensionDataPlanNodeId)
        .filter("prop_location_score1 > 1.0 and prop_location_score2 > 0.1 and prop_log_historical_price > 4.0")
        .hashJoin(
            {"srch_id"},
            {"searches_srch_id"},
            PlanBuilder(planNodeIdGenerator)
                .tableScan(
                R2_searchesType,{},"")
                .capturePlanNodeId(readR2_searchesDataPlanNodeId)
                .project({"srch_id as searches_srch_id","year","month","weekofyear","time","site_id",
                        "visitor_location_country_id","srch_destination_id","srch_length_of_stay","srch_booking_window",
                        "srch_adults_count","srch_children_count","srch_room_count","CAST(srch_saturday_night_bool AS INTEGER) as srch_saturday_night_bool","CAST(random_bool AS INTEGER) as random_bool"
                    })
                .filter("srch_booking_window > 10 and srch_length_of_stay > 1")
                .planNode(),
            {}, //extra filters
            {
                //projections
                "srch_id","prop_id","position","prop_location_score1","prop_location_score2","prop_log_historical_price","price_usd","orig_destination_distance",
                "year","month","weekofyear","time","site_id","visitor_location_country_id","srch_destination_id","srch_length_of_stay","srch_booking_window","srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool","random_bool"
            },
        JoinType::kInner  
        )
        .hashJoin(
            {"prop_id"},  
            {"hotels_prop_id"},
            PlanBuilder(planNodeIdGenerator)
                .tableScan(R1_hotelsType,{},"")
                .capturePlanNodeId(readR1_HotelsDataPlanNodeId)
                .project({
                    "prop_id as hotels_prop_id","prop_country_id","prop_starrating","prop_review_score","CAST(prop_brand_bool AS INTEGER) as prop_brand_bool","count_clicks","avg_bookings_usd","stdev_bookings_usd","count_bookings"
                })
                .filter("count_bookings > 5")
                .planNode(),
                {},
            { //projections
                //from the s_listings
                "srch_id","prop_id","position","prop_location_score1","prop_location_score2","prop_log_historical_price","price_usd","orig_destination_distance",
                //from searches
                "year","month","weekofyear","time","site_id","visitor_location_country_id","srch_destination_id","srch_length_of_stay","srch_booking_window","srch_adults_count","srch_children_count","srch_room_count","srch_saturday_night_bool","random_bool",
                //from the hotels
                "prop_country_id","prop_starrating","prop_review_score","prop_brand_bool","count_clicks","avg_bookings_usd","stdev_bookings_usd","count_bookings"
            },
            JoinType::kInner
        )
        .project({
        "prop_id","srch_id",

        "prop_location_score1_minmax_scaler(transform(array_constructor(prop_location_score1),    x -> CAST(x AS REAL))) as prop_location_score1",
        "prop_location_score2_minmax_scaler(transform(array_constructor(prop_location_score2),    x -> CAST(x AS REAL))) as prop_location_score2",
        "prop_log_historical_price_minmax_scaler(transform(array_constructor(prop_log_historical_price),    x -> CAST(x AS REAL))) as prop_log_historical_price",
        "price_usd_minmax_scaler(transform(array_constructor(price_usd),    x -> CAST(x AS REAL))) as price_usd",
        "orig_destination_distance_minmax_scaler(transform(array_constructor(orig_destination_distance),    x -> CAST(x AS REAL))) as orig_destination_distance",
        "prop_review_score_minmax_scaler(transform(array_constructor(prop_review_score),    x -> CAST(x AS REAL))) as prop_review_score",
        "avg_bookings_usd_minmax_scaler(transform(array_constructor(avg_bookings_usd),    x -> CAST(x AS REAL))) as avg_bookings_usd",
        "stdev_bookings_usd_minmax_scaler(transform(array_constructor(stdev_bookings_usd),    x -> CAST(x AS REAL))) as stdev_bookings_usd",

        // One-hot UDFs with alias names
        "one_hot_prop_starrating(prop_starrating) as oh_prop_starrating",
        "one_hot_prop_brand_bool(prop_brand_bool) as oh_prop_brand_bool",
        "one_hot_count_clicks(count_clicks) as oh_count_clicks",
        "one_hot_count_bookings(count_bookings) as oh_count_bookings",
        "one_hot_srch_length_of_stay(srch_length_of_stay) as oh_srch_length_of_stay",
        "one_hot_srch_booking_window(srch_booking_window) as oh_srch_booking_window",
        "one_hot_srch_adults_count(srch_adults_count) as oh_srch_adults_count",
        "one_hot_srch_children_count(srch_children_count) as oh_srch_children_count",
        "one_hot_srch_room_count(srch_room_count) as oh_srch_room_count",
        "one_hot_srch_saturday_night_bool(srch_saturday_night_bool) as oh_srch_saturday_night_bool",
        "one_hot_random_bool(random_bool) as oh_random_bool",
        "one_hot_position(position) as oh_position",
        "one_hot_prop_country_id(prop_country_id) as oh_prop_country_id",
        "one_hot_year(year) as oh_year",
        "one_hot_month(month) as oh_month",
        "one_hot_weekofyear(weekofyear) as oh_weekofyear",
        "one_hot_time(time) as oh_time",
        "one_hot_site_id(site_id) as oh_site_id",
        "one_hot_visitor_location_country_id(visitor_location_country_id) as oh_visitor_location_country_id",
        "one_hot_srch_destination_id(srch_destination_id) as oh_srch_destination_id"
        })
        .project({

        "prop_id","srch_id",

        "transform(concat("

            // Numerical block
            "prop_location_score1, prop_location_score2, prop_log_historical_price, price_usd, "
                            "orig_destination_distance, prop_review_score, avg_bookings_usd, stdev_bookings_usd, "

            // Categorical block
            "oh_prop_starrating, oh_prop_brand_bool, oh_count_clicks, oh_count_bookings, "
            "oh_srch_length_of_stay, oh_srch_booking_window, oh_srch_adults_count, oh_srch_children_count, "
            "oh_srch_room_count, oh_srch_saturday_night_bool, oh_random_bool, "
            "oh_position, oh_prop_country_id, oh_year, oh_month, oh_weekofyear, "
            "oh_time, oh_site_id, oh_visitor_location_country_id, oh_srch_destination_id"

        "), x -> CAST(x AS REAL)) as u_features"
        })
        .project({
            "prop_id","srch_id",
            "decision_tree_predict(u_features) as decision_tree_result"
        });
        // .filter(
        // "prop_location_score1 > 1.0 "
        // " and prop_location_score2 > 0.1 "
        // " and prop_log_historical_price > 4.0 "
        // " and count_bookings > 5 "
        // " and srch_booking_window > 10 "
        // " and srch_length_of_stay > 1"
        // );

    cataLog.setIdAddressMap(
          readS_listings_extensionDataPlanNodeId,
          S_listings_extensionDataPath,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readS_listings_extensionDataPlanNodeId, "S_listings");
      cataLog.addSource(std::make_shared<Source>(Source(
          readS_listings_extensionDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(
              S_listings_extensionNumRows, S_listings_extensionNumCols))));

      // product
      cataLog.setIdAddressMap(
          readR1_HotelsDataPlanNodeId,
          R1_hotelsDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readR1_HotelsDataPlanNodeId, "R1_hotels");
      cataLog.addSource(std::make_shared<Source>(Source(
          readR1_HotelsDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(R1_hotelsNumRows, R1_hotelsNumCols))));

    cataLog.setIdAddressMap(
          readR2_searchesDataPlanNodeId,
          R2_searchesDataPath,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readR2_searchesDataPlanNodeId, "R2_searches");
      cataLog.addSource(std::make_shared<Source>(Source(
          readR2_searchesDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(R2_searchesNumRows, R2_searchesNumCols))));

    
    //let's try the optimization rule DecisionForestUDF2RelationRewriteAction
        auto planNode = queryPlan.planNode();
        // Create ruleManager
        RuleManager ruleManager;
        // Create planState
        PlanState planState(ruleManager);

        planState.getPossibleActions(planNode, cataLog);
        // Print possible actions
        std::cout << "Available Actions: " << std::endl;
        for (const auto& entry : planState.actionsPair) {
            std::cout << entry.first << ": " << entry.second << std::endl;
        }

        // 1) Define your list of (expression, rule) pairs:
        std::vector<std::pair<std::string, std::string>> actions = {
        // {R"(one_hot_count_bookings(ROW["count_bookings"]))",            "MLDecompositionPushdownRewriteAction"},
        // {R"(one_hot_count_clicks(ROW["count_clicks"]))",                "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_month(ROW["month"]))",                              "MLDecompositionPushdownRewriteAction"},
        // {R"(one_hot_prop_country_id(ROW["prop_country_id"]))",          "MLDecompositionPushdownRewriteAction"},
        // {R"(one_hot_prop_starrating(ROW["prop_starrating"]))",          "MLDecompositionPushdownRewriteAction"}
        {R"(one_hot_site_id(ROW["site_id"]))",                          "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_srch_adults_count(ROW["srch_adults_count"]))",      "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_srch_booking_window(ROW["srch_booking_window"]))",  "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_srch_children_count(ROW["srch_children_count"]))",  "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_srch_destination_id(ROW["srch_destination_id"]))",  "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_srch_length_of_stay(ROW["srch_length_of_stay"]))",  "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_srch_room_count(ROW["srch_room_count"]))",          "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_time(ROW["time"]))",                                "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_visitor_location_country_id(ROW["visitor_location_country_id"]))",
                                                                            "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_weekofyear(ROW["weekofyear"]))",                    "MLDecompositionPushdownRewriteAction"},
        {R"(one_hot_year(ROW["year"]))",                                "MLDecompositionPushdownRewriteAction"}
        };

        // 2) Apply each action one at a time and then update the plan:
        for (auto const& action : actions) {
            // a) Apply the single rewrite action
            planState.takeAction(
                planNode,
                nullptr,
                maker,
                queryPlan,
                pool_,
                planNodeIdGenerator,
                std::vector{action},
                cataLog
            );

            // b) Refresh planState to pick up the change
            planState.update(queryPlan, cataLog);

            planNode = queryPlan.planNode();
        }
        
              //expedia ends here
    }

    else if(queryTemplate == "creditcard"){
        //creditcard starts here 

        auto CreditCardType = ROW(
            {
                "time", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                "amount", "class"
            },
            {
                DOUBLE(), // Time
                DOUBLE(), // V1
                DOUBLE(), // V2
                DOUBLE(), // V3
                DOUBLE(), // V4
                DOUBLE(), // V5
                DOUBLE(), // V6
                DOUBLE(), // V7
                DOUBLE(), // V8
                DOUBLE(), // V9
                DOUBLE(), // V10
                DOUBLE(), // V11
                DOUBLE(), // V12
                DOUBLE(), // V13
                DOUBLE(), // V14
                DOUBLE(), // V15
                DOUBLE(), // V16
                DOUBLE(), // V17
                DOUBLE(), // V18
                DOUBLE(), // V19
                DOUBLE(), // V20
                DOUBLE(), // V21
                DOUBLE(), // V22
                DOUBLE(), // V23
                DOUBLE(), // V24
                DOUBLE(), // V25
                DOUBLE(), // V26
                DOUBLE(), // V27
                DOUBLE(), // V28
                DOUBLE(), // Amount
                BIGINT()  // Class
            });

            std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

        if (dataDirPrefix == "") {
        // use default value:
        dataDirPrefix =
            "/home/cactusdb/resources/data/parquet/creditcard/";
        }
        std::vector<std::string> creditcardDataPaths =
            getFilePathsFromDir(dataDirPrefix + "creditcard");
        
        int creditcardNumRows, creditcardNumCols;

        readDataStats(
        dataDirPrefix + "creditcard_stats.txt",
        creditcardNumRows,
        creditcardNumCols);

        PlanNodeId readcreditcardDataPlanNodeId;

        optimization::registerVectorFunction(
            "xgboost_predict",
            XGBoostPrediction::signatures(),
            std::make_unique<XGBoostPrediction>("/home/cactusdb/resources/model/creditcard/creditcard_xgb_model.json",
                 29),
                {},
        /*deterministic=*/true,
            cataLog);
        
        optimization::registerVectorFunction(
            "decision_forest_predict",
            ForestPrediction::signatures(),
            std::make_unique<ForestPrediction>("/home/cactusdb/resources/model/creditcard/tree_model",
                 29, true),
                {},
        /*deterministic=*/true,
            cataLog);

        const std::string statsDir = "/home/cactusdb/resources/model/creditcard/scaler_files";

        // List of features to register:
        const std::vector<std::string> features = {
        "v1","v2","v3","v4","v5","v6","v7","v8","v9",
        "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
        "v20","v21","v22","v23","v24","v25","v26","v27","v28",
        "amount"
        };

        for (auto& feature : features) {
            // Build full path to that feature’s stats file (e.g. "/…/v1.txt")
            const std::string filePath = fmt::format("{}/{}.txt", statsDir, feature);
            registerFeatureMinMaxScaler(cataLog, pool_, feature, filePath);
        }
    

        queryPlan = PlanBuilder(planNodeIdGenerator)
        .tableScan(CreditCardType,{}, "")
        .capturePlanNodeId(readcreditcardDataPlanNodeId)
        // .filter("v1 > 1.0 AND v2 < 0.27 AND v3 > 0.3")
        .project({
            "amount as pamount",

            "time", "v1_minmax_scaler(transform(array_constructor(v1),    x -> CAST(x AS REAL))) as v1",
                "v2_minmax_scaler(transform(array_constructor(v2),    x -> CAST(x AS REAL))) as v2",
                "v3_minmax_scaler(transform(array_constructor(v3),    x -> CAST(x AS REAL))) as v3",
                "v4_minmax_scaler(transform(array_constructor(v4),    x -> CAST(x AS REAL))) as v4",
                "v5_minmax_scaler(transform(array_constructor(v5),    x -> CAST(x AS REAL))) as v5",
                "v6_minmax_scaler(transform(array_constructor(v6),    x -> CAST(x AS REAL))) as v6",
                "v7_minmax_scaler(transform(array_constructor(v7),    x -> CAST(x AS REAL))) as v7",
                "v8_minmax_scaler(transform(array_constructor(v8),    x -> CAST(x AS REAL))) as v8",
                "v9_minmax_scaler(transform(array_constructor(v9),    x -> CAST(x AS REAL))) as v9",
                "v10_minmax_scaler(transform(array_constructor(v10),  x -> CAST(x AS REAL))) as v10",
                "v11_minmax_scaler(transform(array_constructor(v11),  x -> CAST(x AS REAL))) as v11",
                "v12_minmax_scaler(transform(array_constructor(v12),  x -> CAST(x AS REAL))) as v12",
                "v13_minmax_scaler(transform(array_constructor(v13),  x -> CAST(x AS REAL))) as v13",
                "v14_minmax_scaler(transform(array_constructor(v14),  x -> CAST(x AS REAL))) as v14",
                "v15_minmax_scaler(transform(array_constructor(v15),  x -> CAST(x AS REAL))) as v15",
                "v16_minmax_scaler(transform(array_constructor(v16),  x -> CAST(x AS REAL))) as v16",
                "v17_minmax_scaler(transform(array_constructor(v17),  x -> CAST(x AS REAL))) as v17",
                "v18_minmax_scaler(transform(array_constructor(v18),  x -> CAST(x AS REAL))) as v18",
                "v19_minmax_scaler(transform(array_constructor(v19),  x -> CAST(x AS REAL))) as v19",
                "v20_minmax_scaler(transform(array_constructor(v20),  x -> CAST(x AS REAL))) as v20",
                "v21_minmax_scaler(transform(array_constructor(v21),  x -> CAST(x AS REAL))) as v21",
                "v22_minmax_scaler(transform(array_constructor(v22),  x -> CAST(x AS REAL))) as v22",
                "v23_minmax_scaler(transform(array_constructor(v23),  x -> CAST(x AS REAL))) as v23",
                "v24_minmax_scaler(transform(array_constructor(v24),  x -> CAST(x AS REAL))) as v24",
                "v25_minmax_scaler(transform(array_constructor(v25),  x -> CAST(x AS REAL))) as v25",
                "v26_minmax_scaler(transform(array_constructor(v26),  x -> CAST(x AS REAL))) as v26",
                "v27_minmax_scaler(transform(array_constructor(v27),  x -> CAST(x AS REAL))) as v27",
                "v28_minmax_scaler(transform(array_constructor(v28),  x -> CAST(x AS REAL))) as v28",
                "amount_minmax_scaler(transform(array_constructor(amount), x -> CAST(x AS REAL))) as amount", "class"
        })
        .project({
            "concat(v1, v2, v3, v4, v5, v6, v7, v8, v9,"
            "v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,"
            "v20, v21, v22, v23, v24, v25, v26, v27, v28,"
            "amount) as u_features", "pamount"
        })
        .project({"pamount","xgboost_predict(u_features) as prediction_result"});


        cataLog.setIdAddressMap(
          readcreditcardDataPlanNodeId,
          creditcardDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readcreditcardDataPlanNodeId, "creditcard");
      cataLog.addSource(std::make_shared<Source>(Source(
          readcreditcardDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(creditcardNumRows, creditcardNumCols))));

    //     registerForesttoRelationalFunctions("/home/cactusdb/resources/model/creditcard/tree_model", 29);

    //     //let's try the optimization rule DecisionForestUDF2RelationRewriteAction
    //     auto planNode = queryPlan.planNode();
    //     // Create ruleManager
    //     RuleManager ruleManager;
    //     // Create planState
    //     PlanState planState(ruleManager);

    //     planState.getPossibleActions(planNode, cataLog);
    //     // Print possible actions
    //     std::cout << "Available Actions: " << std::endl;
    //     for (const auto& entry : planState.actionsPair) {
    //         std::cout << entry.first << ": " << entry.second << std::endl;
    //     }

    //     std::pair<std::string, std::string> testAction(
    //       "decision_forest_predict(ROW[\"u_features\"])",
    //       "DecisionForestUDF2RelationRewriteAction");

    //     // decision_forest_predict(ROW["u_features"]): DecisionForestUDF2RelationRewriteAction
    //     planState.takeAction(
    //       planNode,
    //       nullptr,
    //       maker,
    //       queryPlan,
    //       pool_,
    //       planNodeIdGenerator,
    //       {testAction},
    //       cataLog);
    //   // Update the planState (getPossibleAction after apply one action)
    //   planState.update(queryPlan, cataLog);

    //creditcard ends here 
    }

    else if(queryTemplate == "flights"){
        //flights starts here 
    //    auto R1_airlinesNames = std::vector<std::string>{"airlineid", "name1", "name2", "name4", "acountry", "active"};
        auto R1_airlinesType = ROW({"airlineid", "name1", "name2", "name4", "acountry", "active"}, {VARCHAR(), BIGINT(), VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR()});

        // auto R2_sairportsNames = std::vector<std::string>{"sairportid", "scity", "scountry", "slatitude", "slongitude", "stimezone", "sdst"};
        auto R2_sairportsType = ROW({"sairportid", "scity", "scountry", "slatitude", "slongitude", "stimezone", "sdst"}, {VARCHAR(), VARCHAR(), VARCHAR(), DOUBLE(), DOUBLE(), BIGINT(), VARCHAR()});

        // auto R3_dairportsNames = std::vector<std::string>{"dairportid", "dcity", "dcountry", "dlatitude", "dlongitude", "dtimezone", "ddst"};
        auto R3_dairportsType = ROW({"dairportid", "dcity", "dcountry", "dlatitude", "dlongitude", "dtimezone", "ddst"}, {VARCHAR(), VARCHAR(), VARCHAR(), DOUBLE(), DOUBLE(), BIGINT(), VARCHAR()});

        // auto S_routesNames = std::vector<std::string>{ "airlineid", "sairportid", "dairportid", "codeshare"};
        auto S_routesType = ROW({ "airlineid", "sairportid", "dairportid", "codeshare"}, { VARCHAR(), VARCHAR(), VARCHAR(), VARCHAR()});

        std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

        if (dataDirPrefix == "") {
        // use default value:
        dataDirPrefix =
            "/home/cactusdb/resources/data/parquet/flights/";
        }
        std::vector<std::string> R1_airlinesDataPaths =
            getFilePathsFromDir(dataDirPrefix + "R1_airlines");
        std::vector<std::string> R2_sairportsDataPaths =
            getFilePathsFromDir(dataDirPrefix + "R2_sairports");
        std::vector<std::string> R3_dairportsDataPaths =
            getFilePathsFromDir(dataDirPrefix + "R3_dairports");
        std::vector<std::string> S_routesDataPaths =
            getFilePathsFromDir(dataDirPrefix + "S_routes_100G");

        std::cout << "File Path Read!\n";

        int R1_airlinesNumRows, R1_airlinesNumCols, R2_sairportsNumRows, R2_sairportsNumCols, R3_dairportsNumRows, R3_dairportsNumCols, S_routesNumRows, S_routesNumCols;

        readDataStats(
        dataDirPrefix + "R1_airlines_stats.txt",
        R1_airlinesNumRows,
        R1_airlinesNumCols);
        readDataStats(
        dataDirPrefix + "R2_sairports_stats.txt",
        R2_sairportsNumRows,
        R2_sairportsNumCols);
        readDataStats(
        dataDirPrefix + "R3_dairports_stats.txt",
        R3_dairportsNumRows,
        R3_dairportsNumCols);
        readDataStats(
        dataDirPrefix + "S_routes_100G_stats.txt",
        S_routesNumRows,
        S_routesNumCols);

        std::cout << "Stats Read!\n";

        PlanNodeId readR1_airlinesDataPlanNodeId;
        PlanNodeId readR2_sairportsDataPlanNodeId;
        PlanNodeId readR3_dairportsDataPlanNodeId;
        PlanNodeId readS_routesDataPlanNodeId;

        std::vector<std::pair<std::string, std::string>> encoderSpecs = {
            {"one_hot_dtimezone", "/home/cactusdb/resources/model/flights/onehotencoders/dtimezone.csv"},
            {"one_hot_name1",     "/home/cactusdb/resources/model/flights/onehotencoders/name1.csv"},
            {"one_hot_stimezone", "/home/cactusdb/resources/model/flights/onehotencoders/stimezone.csv"}
        };

        std::vector<std::pair<std::string, std::string>> stringEncoderSpecs = {
            {"one_hot_acountry",                  "/home/cactusdb/resources/model/flights/onehotencoders/acountry.csv"},
            {"one_hot_active",                    "/home/cactusdb/resources/model/flights/onehotencoders/active.csv"},
            {"one_hot_dcity",                     "/home/cactusdb/resources/model/flights/onehotencoders/dcity.csv"},
            {"one_hot_dcountry",                  "/home/cactusdb/resources/model/flights/onehotencoders/dcountry.csv"},
            {"one_hot_ddst",                      "/home/cactusdb/resources/model/flights/onehotencoders/ddst.csv"},
            {"one_hot_name2",                     "/home/cactusdb/resources/model/flights/onehotencoders/name2.csv"},
            {"one_hot_name4",                     "/home/cactusdb/resources/model/flights/onehotencoders/name4.csv"},
            {"one_hot_scity",                     "/home/cactusdb/resources/model/flights/onehotencoders/scity.csv"},
            {"one_hot_scountry",                  "/home/cactusdb/resources/model/flights/onehotencoders/scountry.csv"},
            {"one_hot_sdst",                      "/home/cactusdb/resources/model/flights/onehotencoders/sdst.csv"}
        };

        for (const auto& [functionName, filePath] : encoderSpecs) {
            registerOneHotInt(functionName, filePath, cataLog, pool_);
        }

        for (const auto& [functionName, filePath] : stringEncoderSpecs) {
            registerOneHotString(functionName, filePath, cataLog, pool_);
        }

        const std::string statsDir = "/home/cactusdb/resources/model/flights/scaler_files";

        // List of features to register:
        const std::vector<std::string> features = {
        "dlatitude","dlongitude","slatitude","slongitude"
        };

        for (auto& feature : features) {
            // Build full path to that feature’s stats file (e.g. "/…/v1.txt")
            const std::string filePath = fmt::format("{}/{}.txt", statsDir, feature);
            registerFeatureMinMaxScaler(cataLog, pool_, feature, filePath);
        }

        optimization::registerVectorFunction(
            "decision_forest_predict",
            ForestPrediction::signatures(),
            std::make_unique<ForestPrediction>("/home/cactusdb/resources/model/flights/rf_dot_trees_custom",
                 6756, true),
                {},
        /*deterministic=*/true,
            cataLog);

        
        queryPlan = PlanBuilder(planNodeIdGenerator)
        // 1) Read S_routes
            .tableScan(S_routesType, {}, "")
            .capturePlanNodeId(readS_routesDataPlanNodeId)
            // .project({
            //     "airlineid", "sairportid", "dairportid", "codeshare"
            // });
        // 2) Join with R1_airlines on airlineid
        .hashJoin(
            /*leftKeys*/ {"airlineid"},
            /*rightKeys*/ {"r1_airlines_airlineid"},
            /*buildSide*/
            PlanBuilder(planNodeIdGenerator)
                .tableScan(R1_airlinesType, {}, "")
                .capturePlanNodeId(readR1_airlinesDataPlanNodeId)
                .project({
                    "airlineid as r1_airlines_airlineid",
                    "name1", "name2", "name4", "acountry", "active"
                })
                .filter("name2 = 't' AND name4 = 't' AND name1 > 2")
                .planNode(),
            /*leftFiltersRightFilters*/ {},
            /*outputColumns*/ {
                // from S_routes
                "airlineid",
                "sairportid",
                "dairportid",
                // from R1_airlines
                "name1", "name2", "name4", "acountry", "active"
            },
            JoinType::kInner
        )

        // 3) Join with R2_sairports on sairportid
        .hashJoin(
            {"sairportid"},
            {"r2_sairports_sairportid"},
            PlanBuilder(planNodeIdGenerator)
                .tableScan(R2_sairportsType, {}, "")
                .capturePlanNodeId(readR2_sairportsDataPlanNodeId)
                .project({
                    "sairportid as r2_sairports_sairportid",
                    "slatitude", "slongitude",
                    "scity", "scountry", "stimezone", "sdst"
                })
                .planNode(),
            {},
            {
                // carry forward
                "airlineid",
                "sairportid",
                "dairportid",
                "name1", "name2", "name4", "acountry", "active",
                // new from R2_sairports
                "slatitude", "slongitude", "scity", "scountry", "stimezone", "sdst"
            },
            JoinType::kInner
        )

        // 4) Join with R3_dairports on dairportid
        .hashJoin(
            {"dairportid"},
            {"r3_dairports_dairportid"},
            PlanBuilder(planNodeIdGenerator)
                .tableScan(R3_dairportsType, {}, "")
                .capturePlanNodeId(readR3_dairportsDataPlanNodeId)
                .project({
                    "dairportid as r3_dairports_dairportid",
                    "dlatitude", "dlongitude",
                    "dcity", "dcountry", "dtimezone", "ddst"
                })
                .planNode(),
            {},
            {
                // carry forward
                "airlineid",
                "sairportid",
                "dairportid",
                "name1", "name2", "name4", "acountry", "active",
                "slatitude", "slongitude", "scity", "scountry", "stimezone", "sdst",
                // new from R3_dairports
                "dlatitude", "dlongitude", "dcity", "dcountry", "dtimezone", "ddst"
            },
            JoinType::kInner
        )
        .project({

            "airlineid",
            "sairportid",
            "dairportid",

            "slatitude_minmax_scaler(transform(array_constructor(slatitude),    x -> CAST(x AS REAL))) as slatitude",
            "slongitude_minmax_scaler(transform(array_constructor(slongitude),    x -> CAST(x AS REAL))) as slongitude",
            "dlatitude_minmax_scaler(transform(array_constructor(dlatitude),    x -> CAST(x AS REAL))) as dlatitude",
            "dlongitude_minmax_scaler(transform(array_constructor(dlongitude),    x -> CAST(x AS REAL))) as dlongitude",


            "one_hot_dtimezone(dtimezone) as one_hot_dtimezone",
            "one_hot_name1(name1)     as one_hot_name1",
            "one_hot_stimezone(stimezone) as one_hot_stimezone",

            "one_hot_acountry(acountry)                          as one_hot_acountry",
            "one_hot_active(active)                              as one_hot_active",
            "one_hot_dcity(dcity)                                as one_hot_dcity",
            "one_hot_dcountry(dcountry)                          as one_hot_dcountry",
            "one_hot_ddst(ddst)                                  as one_hot_ddst",
            "one_hot_name2(name2)                                as one_hot_name2",
            "one_hot_name4(name4)                                as one_hot_name4",
            "one_hot_scity(scity)                                as one_hot_scity",
            "one_hot_scountry(scountry)                          as one_hot_scountry",
            "one_hot_sdst(sdst)                                  as one_hot_sdst"
        })
        .project({
            "airlineid",
            "sairportid",
            "dairportid",

            "transform( concat( slatitude,slongitude, dlatitude,dlongitude,"
            "one_hot_name1, one_hot_name2,one_hot_name4,one_hot_acountry,one_hot_active,one_hot_scity,"
            "one_hot_scountry,one_hot_stimezone,one_hot_sdst,one_hot_dcity,one_hot_dcountry,one_hot_dtimezone,one_hot_ddst), x -> CAST(x AS REAL)) as u_feature"
        })
        .project({
            "airlineid",
            "sairportid",
            "dairportid",

            "decision_forest_predict(u_feature)"
        });
        // .filter(
        //     "name2 = 't' AND name4 = 't' AND name1 > 2"
        // );

        //flights ends here
        cataLog.setIdAddressMap(
          readR1_airlinesDataPlanNodeId,
          R1_airlinesDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readR1_airlinesDataPlanNodeId, "r1_airlines");
      cataLog.addSource(std::make_shared<Source>(Source(
          readR1_airlinesDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(R1_airlinesNumRows, R1_airlinesNumCols))));
          
          cataLog.setIdAddressMap(
          readR2_sairportsDataPlanNodeId,
          R2_sairportsDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readR2_sairportsDataPlanNodeId, "r2_sairports");
      cataLog.addSource(std::make_shared<Source>(Source(
          readR2_sairportsDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(R2_sairportsNumRows, R2_sairportsNumCols))));

          cataLog.setIdAddressMap(
          readR3_dairportsDataPlanNodeId,
          R3_dairportsDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readR3_dairportsDataPlanNodeId, "r3_dairports");
      cataLog.addSource(std::make_shared<Source>(Source(
          readR3_dairportsDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(R3_dairportsNumRows, R3_dairportsNumCols))));

          cataLog.setIdAddressMap(
          readS_routesDataPlanNodeId,
          S_routesDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.addNodeIdRelationName(readS_routesDataPlanNodeId, "s_routes");
      cataLog.addSource(std::make_shared<Source>(Source(
          readS_routesDataPlanNodeId,
          Source::Type::FILE,
          std::make_shared<OutputStat>(S_routesNumRows, S_routesNumCols))));

        // return queryPlan;

        registerForesttoRelationalFunctions("/home/cactusdb/resources/model/flights/rf_dot_trees_custom", 6756);

        //let's try the optimization rule DecisionForestUDF2RelationRewriteAction
        auto planNode = queryPlan.planNode();
        // Create ruleManager
        RuleManager ruleManager;
        // Create planState
        PlanState planState(ruleManager);

        planState.getPossibleActions(planNode, cataLog);
        // Print possible actions
        std::cout << "Available Actions: " << std::endl;
        for (const auto& entry : planState.actionsPair) {
            std::cout << entry.first << ": " << entry.second << std::endl;
        }

       std::vector<std::pair<std::string, std::string>> actions = {
  {R"(one_hot_acountry(ROW["acountry"]))",       "MLDecompositionPushdownRewriteAction"},
  {R"(one_hot_active(ROW["active"]))",           "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_dcity(ROW["dcity"]))",             "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_dcountry(ROW["dcountry"]))",       "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_ddst(ROW["ddst"]))",               "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_dtimezone(ROW["dtimezone"]))",     "MLDecompositionPushdownRewriteAction"},
  {R"(one_hot_name1(ROW["name1"]))",             "MLDecompositionPushdownRewriteAction"},
  {R"(one_hot_name2(ROW["name2"]))",             "MLDecompositionPushdownRewriteAction"},
  {R"(one_hot_name4(ROW["name4"]))",             "MLDecompositionPushdownRewriteAction"}
//   {R"(one_hot_scity(ROW["scity"]))",             "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_scountry(ROW["scountry"]))",       "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_sdst(ROW["sdst"]))",               "MLDecompositionPushdownRewriteAction"},
//   {R"(one_hot_stimezone(ROW["stimezone"]))",     "MLDecompositionPushdownRewriteAction"}
};

        for (auto const& action : actions) {
                // a) Wrap the single action in a temporary vector and apply it
                planState.takeAction(
                    planNode,
                    nullptr,
                    maker,
                    queryPlan,
                    pool_,
                    planNodeIdGenerator,
                    std::vector{action},
                    cataLog
                );

                // b) Immediately refresh planState with any changes
                planState.update(queryPlan, cataLog);

                planNode = queryPlan.planNode();
        }

    }
    
  }
    else {
    throw std::runtime_error(
        "Unsupported workload: " + workload +
        ". Currently only movielens is supported.");
  }

  return queryPlan;
}