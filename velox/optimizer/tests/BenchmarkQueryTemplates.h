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
        if (generateFilter) {
            std::vector<std::string> filterExpr = sampleUserMovieFilterExpr("user", timestampSeed);
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
        std::string svdModelPath = "/home/velox/resources/model/movielens/final/velox/movielens_template6_svd.h5";
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
                { // what columns to project from the join
                    "u_user_id",
                    "m_movie_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "u_zipcode",
                   "m_genres"
                }
            )
            // .project({
            //     "CAST (u_user_id AS INTEGER) AS u_user_id",
            //     "CAST (m_movie_id as INTEGER) AS m_movie_id",
            //     "u_age",
            //     "u_gender",
            //     "u_occupation",
            //     "u_zipcode",
            //     "m_genres"})
            .project({
                "u_user_id", "m_movie_id", "svd(u_user_id, m_movie_id) as pred", 
                "u_age",
                "u_gender",
                "u_occupation",
                "u_zipcode",
                "m_genres"});

        if (generateFilter) {
            std::vector<std::string> filterExpr = sampleUserMovieFilterExpr("user_movie_genres", timestampSeed);
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
        cataLog.addSource(std::make_shared<Source>(
            Source(readUserDataPlanNodeId,
                Source::Type::FILE,
                std::make_shared<OutputStat>(userNumRows, userNumCols))));

        // — movie side
        cataLog.setIdAddressMap(
            readMovieDataPlanNodeId,
            movieDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
        cataLog.addSource(std::make_shared<Source>(
            Source(readMovieDataPlanNodeId,
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
        auto userWeightsVector = maker.arrayVector<float>(userEmbedWeights, REAL());
        exec::registerVectorFunction(
            "user_embedding",
            Embedding::signatures(),
            std::make_unique<Embedding>(
                userWeightsVector->elements()->values()->asMutable<float>(),
                numUserEmbeddings,
                embeddingDims));

        std::vector<std::vector<float>> movieEmbedWeights =
            randomGenerator.genFloat2dVector(numMovieEmbedding, embeddingDims);
        auto movieWeightsVector = maker.arrayVector<float>(movieEmbedWeights, REAL());
        exec::registerVectorFunction(
            "movie_embedding",
            Embedding::signatures(),
            std::make_unique<Embedding>(
                movieWeightsVector->elements()->values()->asMutable<float>(),
                numUserEmbeddings,
                embeddingDims));

        // Cosine Similarity
        exec::registerVectorFunction(
            "cosine_similarity",
            CosineSimilarity::signatures(),
            std::make_unique<CosineSimilarity>(embeddingDims));

        // Query Plan
       queryPlan = PlanBuilder(planNodeIdGenerator)
            .tableScan(userDataRowType, {}, "")
            .capturePlanNodeId(readUserDataPlanNodeId)
            .nestedLoopJoin(
                PlanBuilder(planNodeIdGenerator)
                    .tableScan(movieDataRowType, {}, "")
                    .capturePlanNodeId(readMovieDataPlanNodeId)
                    .planNode(),
                { // what columns to project from the join
                    "u_user_id", "m_movie_id",
                    "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres"
                })
            .project({
                "u_user_id", "m_movie_id",
                "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres",
                "user_embedding(array_constructor(u_user_id)) as user_embed",
                "movie_embedding(array_constructor(m_movie_id)) as movie_embed"})
            .project({ 
                "u_user_id", "m_movie_id",
                "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres",
                "cosine_similarity(user_embed, movie_embed) as pred"
            });

        if (generateFilter) {
            std::vector<std::string> filterExpr = sampleUserMovieFilterExpr("user_movie_genres", timestampSeed);
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
        cataLog.addSource(std::make_shared<Source>(
            Source(readUserDataPlanNodeId,
                Source::Type::FILE,
                std::make_shared<OutputStat>(userNumRows, userNumCols))));

        // — movie side
        cataLog.setIdAddressMap(
            readMovieDataPlanNodeId,
            movieDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
        cataLog.addSource(std::make_shared<Source>(
            Source(readMovieDataPlanNodeId,
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
        normalizeAge(cataLog,pool_, "q10_user_age_minmax_scaler.txt");
        // user_occupation_minmax_scaler
        normalizeOccupation(cataLog,pool_, "q10_user_occupation_minmax_scaler.txt");
        // Feed forward Neural Network
        int hidden1 = randomGenerator.genRandomIntValue();
        std::cout << "[INFO] hidden units: " << hidden1 << std::endl;
        auto modelStr = registerNNModel({embeddingDims+3, hidden1, 1}, cataLog, modelGroupId_, false);
    
        // Query Plan
        queryPlan = PlanBuilder(planNodeIdGenerator)
            .tableScan(userDataRowType, {}, "")
            .capturePlanNodeId(readUserDataPlanNodeId)
            .nestedLoopJoin(
                PlanBuilder(planNodeIdGenerator)
                    .tableScan(movieDataRowType, {}, "")
                    .capturePlanNodeId(readMovieDataPlanNodeId)
                    .planNode(),
                { // what columns to project from the join
                    "u_user_id", "m_movie_id",
                    "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres"
                })
            .project({
                "u_user_id", "m_movie_id", "embedding(array_constructor(m_movie_id)) as embed", 
                "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres",
                "gender_encoder(u_gender) as gender_encoded",
                "user_age_minmax_scaler(transform(array_constructor(u_age), x-> CAST(x AS REAL))) as age_encoded",
                "user_occupation_minmax_scaler(transform(array_constructor(u_occupation), x-> CAST(x AS REAL))) as occupation_encoded"})
            .project({
                "u_user_id", "m_movie_id", 
                "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres",
                "transform(concat(gender_encoded,age_encoded,occupation_encoded,embed), x-> CAST(x AS REAL)) as features"})
            .project({
                "u_user_id", "m_movie_id", 
                "u_age", "u_gender", "u_occupation", "u_zipcode", "m_genres",
                fmt::format(modelStr, "features")});

        if (generateFilter) {
            std::vector<std::string> filterExpr = sampleUserMovieFilterExpr("user_movie_genres", timestampSeed);
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
        cataLog.addSource(std::make_shared<Source>(
            Source(readUserDataPlanNodeId,
                Source::Type::FILE,
                std::make_shared<OutputStat>(userNumRows, userNumCols))));

        // — movie side
        cataLog.setIdAddressMap(
            readMovieDataPlanNodeId,
            movieDataPaths,
            dwio::common::FileFormat::PARQUET);
        cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
        cataLog.addSource(std::make_shared<Source>(
            Source(readMovieDataPlanNodeId,
                Source::Type::FILE,
                std::make_shared<OutputStat>(movieNumRows, movieNumCols))));
    } else {
        throw std::runtime_error("Unsupported query template for movielens workload: " + queryTemplate);
    }

  } else if (workload == "tpcxai") {
    std::string queryOptType =
        getEnvVar("CD_VELOX_QUERY_OPT_TYPE"); // env used for ablation study of

    auto finicialAccountDataRowType =
        ROW({"fa_customer_sk", "transaction_limit"}, {BIGINT(), DOUBLE()});
    auto finicialTransactionsDataRowType = ROW(
        {"amount", "iban", "sender_id", "receiver_id", "transaction_id", "time"},
        {DOUBLE(), VARCHAR(), BIGINT(), VARCHAR(), BIGINT(), VARCHAR()});
    auto orderDataRowType =
        ROW({"o_order_id", "o_customer_sk", "weekday", "date", "store"},
            {BIGINT(), BIGINT(), VARCHAR(), VARCHAR(), BIGINT()});
    auto lineitemDataRowType =
        ROW({"li_order_id", "li_product_id", "quantity", "price"},
            {BIGINT(), BIGINT(), BIGINT(), DOUBLE()});
    auto productDataRowType = ROW(
        {"p_product_id", "name", "department"}, {BIGINT(), VARCHAR(), VARCHAR()});
    auto storeDeptDataRowType = ROW(
        {"store", "department", "num_of_week"}, {BIGINT(), VARCHAR(), BIGINT()});
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
    std::vector<std::string> finicialTransactionsDataPaths =
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
        productNumCols, storeDeptNumRows, storeDeptNumCols, productRatingNumRows,
        productRatingNumCols, customerNumRows, customerNumCols,
        orderReturnNumRows, orderReturnNumCols, reviewNumRows, reviewNumCols;

    readDataStats(
        dataDirPrefix + "financial_account_stats.txt",
        finicialAccountNumRows,
        finicialAccountNumCols);
    readDataStats(
        dataDirPrefix + "financial_transactions_stats.txt",
        finicialTransactionsNumRows,
        finicialTransactionsNumCols);
    readDataStats(dataDirPrefix + "order_stats.txt", orderNumRows, orderNumCols);
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

    if (queryTemplate == "template4" ) { // uc7
        std::string svdModelPath = "/home/velox/resources/model/tpcxai_sf1/final/velox/tpcxai_template4_svd.h5";
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
                {"product_id",
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
                "department", "c_birth_day", "c_birth_country"
                })
            .project(
                {"user_id", "product_id", "svd(user_id, product_id) as pred"});
        // if (generateFilter) {
        //     std::vector<std::string> filterExpr = sampleOrderFilterExpr("order", timestampSeed);
        //     for (auto expr : filterExpr) {
        //         queryPlan = queryPlan.filter(expr);
        //     }
        // }

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
        
        cataLog.addNodeIdRelationName(readProductRatingDataPlanNodeId, "product_rating");
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
        std::cout << "[INFO] hidden units: " << hidden1 << ", " << hidden2 << std::endl;
        auto modelStr = registerNNModel({4, hidden1, hidden2, 384}, cataLog, modelGroupId_, false);
        // Register functions: department_encoder
        registerDepartmentEncoder(cataLog, pool_);

        // Query Plan
        queryPlan = PlanBuilder(planNodeIdGenerator)
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
                {"o_order_id",
                // "li_order_id",
                "quantity",
                "date",
                "department",
                "weekday"})
            .partialAggregation(
                {"o_order_id", "date", "department", "quantity"},
                {"sum(quantity) as scan_count", "min(weekday) as weekday"})
            .finalAggregation()
            .project(
                {"o_order_id",
                "date",
                "array_constructor(quantity, scan_count, weekday) as features",
                "department_encoder(department) as department_encoded"})
            .project(
                {"o_order_id",
                "date",
                "transform(concat(features, department_encoded), x-> CAST(x as REAL)) as features"})
            .project(
                {"o_order_id",
                "date",
                fmt::format(modelStr, "features")});
        // if (generateFilter) {
        //     std::vector<std::string> filterExpr = sampleOrderFilterExpr("order", timestampSeed);
        //     for (auto expr : filterExpr) {
        //         queryPlan = queryPlan.filter(expr);
        //     }
        // }

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

    } else {
        throw std::runtime_error("Unsupported query template for tpcxai workload : " + queryTemplate);
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