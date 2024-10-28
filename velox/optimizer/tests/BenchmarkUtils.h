#include <iostream>
#include <string>

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

  if (queryType.find("q1") != std::string::npos) {
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

    } if (queryOptType == "decomposition_pushdown" ) {
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
                          "gender_embedding(gender_encoder(u_gender)) as u_gender_embed"
                        })
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
                   "u_gender_embed"
                   })
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
                   "concat(bottom_mlp_out, u_age_embed, u_occupation_embed, u_gender_embed) as top_mlp_input"
                   })
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
                           "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features"
                           })
                      .project(
                          {"m_movie_id",
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
              .project({
                  "m_movie_id",
                  "mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out"
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
                          "gender_embedding(gender_encoder(u_gender)) as u_gender_embed"
                        })
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
                   "u_gender_embed"
                   })
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
                   "u_gender_embed"
                   })
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
  }

  return queryPlan;
};
