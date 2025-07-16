#pragma once

#include <json/json.h>
#include "velox/cost_model/CostEstimator.h"
#include "velox/ml_functions/SVD.h"
#include "velox/optimizer/Helper.h"
#include "velox/optimizer/Mul2JoinAggRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/TwoLayerUDF2TorchNNRewriteAction.h"
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include <fstream>
#include <sstream>

void registerTwoTowerFunc(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_,
    bool isVerticalPartition = false) {
  VectorMaker maker{pool_.get()};
  std::cout << "[INFO]: Register two tower model functions" << std::endl;
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  int embeddingDims = 32;

  // init user encoder
  std::unordered_map<int, int> userIdMapping;
  for (int i = 1; i < 6041; i++) {
    userIdMapping[i] = i - 1;
  }
  optimization::registerVectorFunction(
      "user_id_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(userIdMapping)),
      {},
      true,
      catalog,
      isVerticalPartition);

  // init movie encoder
  std::unordered_map<int, int> movieIdMapping;
  for (int i = 1; i < 3953; i++) {
    movieIdMapping[i] = i - 1;
  }

  optimization::registerVectorFunction(
      "movie_id_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(movieIdMapping)),
      {},
      true,
      catalog,
      isVerticalPartition);

  // init age encoder
  std::unordered_map<int, int> ageMapping;
  ageMapping[1] = 0;
  ageMapping[18] = 1;
  ageMapping[25] = 2;
  ageMapping[35] = 3;
  ageMapping[45] = 4;
  ageMapping[50] = 5;
  ageMapping[56] = 6;

  optimization::registerVectorFunction(
      "age_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(ageMapping)),
      {},
      true,
      catalog,
      isVerticalPartition);

  // init occupation  encoder
  std::unordered_map<int, int> occupationMapping;
  for (int i = 0; i < 21; i++) {
    occupationMapping[i] = i;
  }

  optimization::registerVectorFunction(
      "occupation_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(occupationMapping)),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::unordered_map<std::string, int> genderMapping;
  genderMapping["F"] = 0;
  genderMapping["M"] = 1;

  optimization::registerVectorFunction(
      "gender_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(genderMapping)),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::unordered_map<std::string, int> genresMapping = {
      {"Animation", 1},
      {"Children's", 2},
      {"Comedy", 3},
      {"Adventure", 4},
      {"Fantasy", 5},
      {"Romance", 6},
      {"Drama", 7},
      {"Action", 8},
      {"Crime", 9},
      {"Thriller", 10},
      {"Horror", 11},
      {"Sci-Fi", 12},
      {"Documentary", 13},
      {"War", 14},
      {"Musical", 15},
      {"Mystery", 16},
      {"Film-Noir", 17},
      {"Western", 18}};

  optimization::registerVectorFunction(
      "genres_encoder",
      StringVariadicEncoder::signatures(),
      std::make_unique<StringVariadicEncoder>(std::move(genresMapping)),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "convert_int_array",
      ConvertToIntArray::signatures(),
      std::make_unique<ConvertToIntArray>(),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "convert_float_array",
      ConvertToFloatArray::signatures(),
      std::make_unique<ConvertToFloatArray>(),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "convert_double_to_float_array",
      ConvertDoubleToFloatArray::signatures(),
      std::make_unique<ConvertDoubleToFloatArray>(),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "change_rating",
      ChangeRating::signatures(),
      std::make_unique<ChangeRating>(),
      {},
      true,
      catalog,
      isVerticalPartition);

  // User-Tower

  // user_id
  int userIdNumEmbedding = 6040;
  std::vector<std::vector<float>> userIdEmbeddingWeights =
      randomGenerator.genFloat2dVector(userIdNumEmbedding, embeddingDims);
  auto userIdEmbeddingWeightsVector =
      maker.arrayVector<float>(userIdEmbeddingWeights, REAL());

  // gender
  int genderNumEmbedding = 2;
  std::vector<std::vector<float>> genderEmbeddingWeights =
      randomGenerator.genFloat2dVector(genderNumEmbedding, embeddingDims);
  auto genderEmbeddingWeightsVector =
      maker.arrayVector<float>(genderEmbeddingWeights, REAL());

  // age
  int ageNumEmbedding = 7;
  std::vector<std::vector<float>> ageEmbeddingWeights =
      randomGenerator.genFloat2dVector(ageNumEmbedding, embeddingDims);
  auto ageEmbeddingWeightsVector =
      maker.arrayVector<float>(ageEmbeddingWeights, REAL());

  // occupation
  int occupationNumEmbedding = 21;
  std::vector<std::vector<float>> occupationEmbeddingWeights =
      randomGenerator.genFloat2dVector(occupationNumEmbedding, embeddingDims);
  auto occupationEmbeddingWeightsVector =
      maker.arrayVector<float>(occupationEmbeddingWeights, REAL());
  optimization::registerVectorFunction(
      "user_id_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(userIdEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          userIdNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "gender_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(genderEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          genderNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "age_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(ageEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          ageNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "occupation_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(occupationEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          occupationNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  randomGenerator.setFloatRange(-1, 1);
  std::vector<std::vector<float>> userNNweight1 =
      randomGenerator.genFloat2dVector(129, 300);
  auto userNNweight1Vector = maker.arrayVector<float>(userNNweight1, REAL());

  std::vector<std::vector<float>> userNNBias1 =
      randomGenerator.genFloat2dVector(300, 1);
  auto userNNBias1Vector = maker.arrayVector<float>(userNNBias1, REAL());

  std::vector<std::vector<float>> userNNweight2 =
      randomGenerator.genFloat2dVector(300, 300);
  auto userNNweight2Vector = maker.arrayVector<float>(userNNweight2, REAL());

  std::vector<std::vector<float>> userNNBias2 =
      randomGenerator.genFloat2dVector(300, 1);
  auto userNNBias2Vector = maker.arrayVector<float>(userNNBias2, REAL());

  std::vector<std::vector<float>> userNNweight3 =
      randomGenerator.genFloat2dVector(300, 128);
  auto userNNweight3Vector = maker.arrayVector<float>(userNNweight3, REAL());

  std::vector<std::vector<float>> userNNBias3 =
      randomGenerator.genFloat2dVector(128, 1);
  auto userNNBias3Vector = maker.arrayVector<float>(userNNBias3, REAL());

  optimization::registerVectorFunction(
      "mat_mul1_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              userNNweight1Vector->elements()->values()->asMutable<float>()),
          129,
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_vector_add1_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              userNNBias1Vector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_mul1_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              userNNweight2Vector->elements()->values()->asMutable<float>()),
          300,
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_vector_add1_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              userNNBias2Vector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_mul1_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              userNNweight3Vector->elements()->values()->asMutable<float>()),
          300,
          128),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_vector_add1_3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              userNNBias3Vector->elements()->values()->asMutable<float>()),
          128),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "relu",
      Relu::signatures(),
      std::make_unique<Relu>(),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<std::vector<float>> batchNorm1Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm1WeightVector =
      maker.arrayVector<float>(batchNorm1Weight, REAL());
  std::vector<std::vector<float>> batchNorm1Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm1BiasVector = maker.arrayVector<float>(batchNorm1Bias, REAL());

  optimization::registerVectorFunction(
      "batch_norm1_1",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          std::move(
              batchNorm1WeightVector->elements()->values()->asMutable<float>()),
          std::move(
              batchNorm1BiasVector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<std::vector<float>> batchNorm2Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2WeightVector =
      maker.arrayVector<float>(batchNorm2Weight, REAL());
  std::vector<std::vector<float>> batchNorm2Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2BiasVector = maker.arrayVector<float>(batchNorm2Bias, REAL());

  optimization::registerVectorFunction(
      "batch_norm1_2",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          std::move(
              batchNorm2WeightVector->elements()->values()->asMutable<float>()),
          std::move(
              batchNorm2BiasVector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<std::vector<float>> batchNorm3Weight =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm3WeightVector =
      maker.arrayVector<float>(batchNorm3Weight, REAL());
  std::vector<std::vector<float>> batchNorm3Bias =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm3BiasVector = maker.arrayVector<float>(batchNorm3Bias, REAL());

  optimization::registerVectorFunction(
      "batch_norm1_3",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          std::move(
              batchNorm3WeightVector->elements()->values()->asMutable<float>()),
          std::move(
              batchNorm3BiasVector->elements()->values()->asMutable<float>()),
          128),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<velox::dl::KernelType> userTowerKernelTypes = {
      velox::dl::KernelType::MatMul,
      velox::dl::KernelType::MatAdd,
      velox::dl::KernelType::BatchNorm,
      velox::dl::KernelType::ReLU,
      velox::dl::KernelType::MatMul,
      velox::dl::KernelType::MatAdd,
      velox::dl::KernelType::BatchNorm,
      velox::dl::KernelType::ReLU,
      velox::dl::KernelType::MatMul,
      velox::dl::KernelType::MatAdd,
      velox::dl::KernelType::BatchNorm,
      velox::dl::KernelType::ReLU};

  float* w1Weight = randomGenerator.genFloat1dArray(129 * 300);
  float* w1Bias = randomGenerator.genFloat1dArray(300);
  float* w2Weight = randomGenerator.genFloat1dArray(300 * 300);
  float* w2Bias = randomGenerator.genFloat1dArray(300);
  float* w3Weight = randomGenerator.genFloat1dArray(300 * 128);
  float* w3Bias = randomGenerator.genFloat1dArray(128);
  float* bn1Weight = randomGenerator.genFloat1dArray(300);
  float* bn1Bias = randomGenerator.genFloat1dArray(300);
  float* bn2Weight = randomGenerator.genFloat1dArray(300);
  float* bn2Bias = randomGenerator.genFloat1dArray(300);
  float* bn3Weight = randomGenerator.genFloat1dArray(300);
  float* bn3Bias = randomGenerator.genFloat1dArray(300);

  std::vector<float*> userTowerWeights = {
      w1Weight,
      w1Bias,
      bn1Weight,
      bn1Bias,
      w2Weight,
      w2Bias,
      bn2Weight,
      bn2Bias,
      w3Weight,
      w3Bias,
      bn3Weight,
      bn3Bias,
  };

  std::vector<int> userTowerdims = {129, 300, 300, 300, 300, 300, 300, 300,
                                    300, 300, 300, 300, 300, 300, 300, 300,
                                    300, 128, 128, 128, 128, 128, 128, 128};

  exec::registerVectorFunction(
      "user_torchNN",
      TorchDNNV2::signatures(),
      std::make_unique<TorchDNNV2>(
          userTowerKernelTypes, userTowerWeights, userTowerdims));

  int movieIdNumEmbedding = 3706;
  std::vector<std::vector<float>> movieIdEmbeddingWeights =
      randomGenerator.genFloat2dVector(movieIdNumEmbedding, embeddingDims);
  auto movieIdEmbeddingWeightsVector =
      maker.arrayVector<float>(movieIdEmbeddingWeights, REAL());

  // genres
  int genresNumEmbedding = 1000;
  std::vector<std::vector<float>> genresEmbeddingWeights =
      randomGenerator.genFloat2dVector(genresNumEmbedding, embeddingDims);
  auto genresEmbeddingWeightsVector =
      maker.arrayVector<float>(genresEmbeddingWeights, REAL());

  optimization::registerVectorFunction(
      "movie_id_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(movieIdEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          movieIdNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "genres_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(genderEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          genresNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "sequence_pooling",
      SequencePooling::signatures(),
      std::make_unique<SequencePooling>(std::string("MEAN"), embeddingDims),
      {},
      true,
      catalog,
      isVerticalPartition);

  randomGenerator.setFloatRange(-1, 1);
  std::vector<std::vector<float>> itemNNweight1 =
      randomGenerator.genFloat2dVector(65, 300);
  auto itemNNweight1Vector = maker.arrayVector<float>(itemNNweight1, REAL());

  std::vector<std::vector<float>> itemNNBias1 =
      randomGenerator.genFloat2dVector(300, 1);
  auto itemNNBias1Vector = maker.arrayVector<float>(itemNNBias1, REAL());

  std::vector<std::vector<float>> itemNNweight2 =
      randomGenerator.genFloat2dVector(300, 300);
  auto itemNNweight2Vector = maker.arrayVector<float>(itemNNweight2, REAL());

  std::vector<std::vector<float>> itemNNBias2 =
      randomGenerator.genFloat2dVector(300, 1);
  auto itemNNBias2Vector = maker.arrayVector<float>(itemNNBias2, REAL());

  std::vector<std::vector<float>> itemNNweight3 =
      randomGenerator.genFloat2dVector(300, 128);
  auto itemNNweight3Vector = maker.arrayVector<float>(itemNNweight3, REAL());

  std::vector<std::vector<float>> itemNNBias3 =
      randomGenerator.genFloat2dVector(128, 1);
  auto itemNNBias3Vector = maker.arrayVector<float>(itemNNBias3, REAL());

  optimization::registerVectorFunction(
      "mat_mul2_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              itemNNweight1Vector->elements()->values()->asMutable<float>()),
          65,
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_vector_add2_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              itemNNBias1Vector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_mul2_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              itemNNweight2Vector->elements()->values()->asMutable<float>()),
          300,
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_vector_add2_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              itemNNBias2Vector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_mul2_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              itemNNweight3Vector->elements()->values()->asMutable<float>()),
          300,
          128),
      {},
      true,
      catalog,
      isVerticalPartition);

  optimization::registerVectorFunction(
      "mat_vector_add2_3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              itemNNBias3Vector->elements()->values()->asMutable<float>()),
          128),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<std::vector<float>> batchNorm2_1Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_1WeightVector =
      maker.arrayVector<float>(batchNorm2_1Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_1Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_1BiasVector =
      maker.arrayVector<float>(batchNorm2_1Bias, REAL());

  optimization::registerVectorFunction(
      "batch_norm2_1",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          std::move(batchNorm2_1WeightVector->elements()
                        ->values()
                        ->asMutable<float>()),
          std::move(
              batchNorm2_1BiasVector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<std::vector<float>> batchNorm2_2Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_2WeightVector =
      maker.arrayVector<float>(batchNorm2_2Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_2Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_2BiasVector =
      maker.arrayVector<float>(batchNorm2_2Bias, REAL());

  optimization::registerVectorFunction(
      "batch_norm2_2",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          std::move(batchNorm2_2WeightVector->elements()
                        ->values()
                        ->asMutable<float>()),
          std::move(
              batchNorm2_2BiasVector->elements()->values()->asMutable<float>()),
          300),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<std::vector<float>> batchNorm2_3Weight =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm2_3WeightVector =
      maker.arrayVector<float>(batchNorm2_3Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_3Bias =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm2_3BiasVector =
      maker.arrayVector<float>(batchNorm2_3Bias, REAL());

  optimization::registerVectorFunction(
      "batch_norm2_3",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          std::move(batchNorm2_3WeightVector->elements()
                        ->values()
                        ->asMutable<float>()),
          std::move(
              batchNorm2_3BiasVector->elements()->values()->asMutable<float>()),
          128),
      {},
      true,
      catalog,
      isVerticalPartition);

  std::vector<velox::dl::KernelType> movieTowerKernelTypes = {
      velox::dl::KernelType::MatMul,
      velox::dl::KernelType::MatAdd,
      velox::dl::KernelType::BatchNorm,
      velox::dl::KernelType::ReLU,
      velox::dl::KernelType::MatMul,
      velox::dl::KernelType::MatAdd,
      velox::dl::KernelType::BatchNorm,
      velox::dl::KernelType::ReLU,
      velox::dl::KernelType::MatMul,
      velox::dl::KernelType::MatAdd,
      velox::dl::KernelType::BatchNorm,
      velox::dl::KernelType::ReLU};

  float* mw1Weight = randomGenerator.genFloat1dArray(65 * 300);
  float* mw1Bias = randomGenerator.genFloat1dArray(300);
  float* mw2Weight = randomGenerator.genFloat1dArray(300 * 300);
  float* mw2Bias = randomGenerator.genFloat1dArray(300);
  float* mw3Weight = randomGenerator.genFloat1dArray(300 * 128);
  float* mw3Bias = randomGenerator.genFloat1dArray(128);
  float* mbn1Weight = randomGenerator.genFloat1dArray(300);
  float* mbn1Bias = randomGenerator.genFloat1dArray(300);
  float* mbn2Weight = randomGenerator.genFloat1dArray(300);
  float* mbn2Bias = randomGenerator.genFloat1dArray(300);
  float* mbn3Weight = randomGenerator.genFloat1dArray(300);
  float* mbn3Bias = randomGenerator.genFloat1dArray(300);

  std::vector<float*> movieTowerWeights = {
      mw1Weight,
      mw1Bias,
      mbn1Weight,
      mbn1Bias,
      mw2Weight,
      mw2Bias,
      mbn2Weight,
      mbn2Bias,
      mw3Weight,
      mw3Bias,
      mbn3Weight,
      mbn3Bias,
  };
  std::vector<int> movieTowerdims = {65,  300, 300, 300, 300, 300, 300, 300,
                                     300, 300, 300, 300, 300, 300, 300, 300,
                                     300, 128, 128, 128, 128, 128, 128, 128};

  exec::registerVectorFunction(
      "movie_torchNN",
      TorchDNNV2::signatures(),
      std::make_unique<TorchDNNV2>(
          movieTowerKernelTypes, movieTowerWeights, movieTowerdims));

  optimization::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(128),
      {},
      true,
      catalog,
      isVerticalPartition);
}

void registerMLTrendingModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/movielens/final/velox/q1_ffnn_weights.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");
  std::vector<std::vector<float>> w3 = loadHDF5Array(ffnnModelPath, "w3");
  std::vector<std::vector<float>> b3 = loadHDF5Array(ffnnModelPath, "b3");

  optimization::registerVectorFunction(
      "mat_mul3_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 3, 128),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add3_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 128),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul3_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 128, 64),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add3_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 64),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul3_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w3)), 64, 2),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add3_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b3)), 2),
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
      "relu", Relu::signatures(), std::make_unique<Relu>(), {}, true, catalog);
  optimization::registerVectorFunction(
      "argmax",
      Argmax::signatures(),
      std::make_unique<Argmax>(),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "llm_ffnn_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(
          "/home/velox/resources/model/movielens/final/velox/q1_ffnn_minmax_scaler.txt"),
      {},
      true,
      catalog);
}

void registerMLDLRMModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  int embeddingDims = 128;
  // init age encoder
  std::unordered_map<int, int> ageMapping;
  ageMapping[1] = 0;
  ageMapping[18] = 1;
  ageMapping[25] = 2;
  ageMapping[35] = 3;
  ageMapping[45] = 4;
  ageMapping[50] = 5;
  ageMapping[56] = 6;

  optimization::registerVectorFunction(
      "age_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(ageMapping)),
      {},
      true,
      catalog);

  // init occupation  encoder
  std::unordered_map<int, int> occupationMapping;
  for (int i = 0; i < 21; i++) {
    occupationMapping[i] = i;
  }

  optimization::registerVectorFunction(
      "occupation_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(occupationMapping)),
      {},
      true,
      catalog);

  std::unordered_map<std::string, int> genderMapping;
  genderMapping["F"] = 0;
  genderMapping["M"] = 1;

  optimization::registerVectorFunction(
      "gender_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(genderMapping)),
      {},
      true,
      catalog);

  // gender
  int genderNumEmbedding = 2;
  std::vector<std::vector<float>> genderEmbeddingWeights =
      randomGenerator.genFloat2dVector(genderNumEmbedding, embeddingDims);
  auto genderEmbeddingWeightsVector =
      maker.arrayVector<float>(genderEmbeddingWeights, REAL());

  // age
  int ageNumEmbedding = 7;
  std::vector<std::vector<float>> ageEmbeddingWeights =
      randomGenerator.genFloat2dVector(ageNumEmbedding, embeddingDims);
  auto ageEmbeddingWeightsVector =
      maker.arrayVector<float>(ageEmbeddingWeights, REAL());

  // occupation
  int occupationNumEmbedding = 21;
  std::vector<std::vector<float>> occupationEmbeddingWeights =
      randomGenerator.genFloat2dVector(occupationNumEmbedding, embeddingDims);
  auto occupationEmbeddingWeightsVector =
      maker.arrayVector<float>(occupationEmbeddingWeights, REAL());

  optimization::registerVectorFunction(
      "gender_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(genderEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          genderNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "age_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(ageEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          ageNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "occupation_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          std::move(occupationEmbeddingWeightsVector->elements()
                        ->values()
                        ->asMutable<float>()),
          occupationNumEmbedding,
          embeddingDims),
      {},
      true,
      catalog);

  // bottom-mlp
};

void registerMLInterestMovieModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/movielens/final/velox/interest_ffnn_model_weights.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");
  std::vector<std::vector<float>> w3 = loadHDF5Array(ffnnModelPath, "w3");
  std::vector<std::vector<float>> b3 = loadHDF5Array(ffnnModelPath, "b3");
  std::vector<std::vector<float>> w4 = loadHDF5Array(ffnnModelPath, "w4");
  std::vector<std::vector<float>> b4 = loadHDF5Array(ffnnModelPath, "b4");

  optimization::registerVectorFunction(
      "mat_mul9_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 259, 128),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add9_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 128),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul9_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 128, 2),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add9_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 2),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_mul9_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w3)), 128, 64),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add9_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b3)), 64),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul9_7",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w4)), 64, 2),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add9_8",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b4)), 2),
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

  optimization::registerVectorFunction(
      "llm_ffnn_interest_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(
          "/home/velox/resources/model/movielens/final/velox/q2_ffnn_interest_scaler.txt"),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "convert_int_array",
      ConvertToIntArray::signatures(),
      std::make_unique<ConvertToIntArray>(),
      {},
      true,
      catalog);
}

void registerMLQ3UserMovieInterestModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/movielens/final/velox/q3_user_movie_interest_ffnn_weight.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");
  std::vector<std::vector<float>> w3 = loadHDF5Array(ffnnModelPath, "w3");
  std::vector<std::vector<float>> b3 = loadHDF5Array(ffnnModelPath, "b3");

  optimization::registerVectorFunction(
      "mat_mul15_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 5, 1024),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add15_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 1024),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul15_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 1024, 512),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add15_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 512),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_mul15_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w3)), 512, 2),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add15_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b3)), 2),
      {},
      true,
      catalog);
}

void registerMLQ3UserMovieRatingModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/movielens/final/velox/q3_user_movie_rating_ffnn_weight.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");
  std::vector<std::vector<float>> w3 = loadHDF5Array(ffnnModelPath, "w3");
  std::vector<std::vector<float>> b3 = loadHDF5Array(ffnnModelPath, "b3");

  optimization::registerVectorFunction(
      "mat_mul16_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 5, 512),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add16_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 512),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul16_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 512, 1024),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add16_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 1024),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_mul16_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w3)), 1024, 6),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add16_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b3)), 6),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "cosine_similarity_q3",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(256),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(), {}, true, catalog);

  optimization::registerVectorFunction(
      "argmax",
      Argmax::signatures(),
      std::make_unique<Argmax>(),
      {},
      true,
      catalog);
}

void registerMLMovieTagEncoderModelFunctions1(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnEncoderModelPath =
      "/home/velox/resources/model/movielens/final/velox/movie_tag_standalone_encoder_weight.h5";
  std::vector<std::vector<float>> w1 =
      loadHDF5Array(ffnnEncoderModelPath, "w1");
  std::vector<std::vector<float>> b1 =
      loadHDF5Array(ffnnEncoderModelPath, "b1");
  std::vector<std::vector<float>> w2 =
      loadHDF5Array(ffnnEncoderModelPath, "w2");
  std::vector<std::vector<float>> b2 =
      loadHDF5Array(ffnnEncoderModelPath, "b2");

  optimization::registerVectorFunction(
      "mat_mul20_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 140979, 2048),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add20_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 2048),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul20_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 2048, 256),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add20_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 256),
      {},
      true,
      catalog);
};

void registerMLMovieTagEncoderModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnEncoderModelPath =
      "/home/velox/resources/model/movielens/final/velox/movie_tag_standalone_encoder_weight.h5";
  std::vector<std::vector<float>> w1 =
      loadHDF5Array(ffnnEncoderModelPath, "w1");
  std::vector<std::vector<float>> b1 =
      loadHDF5Array(ffnnEncoderModelPath, "b1");
  std::vector<std::vector<float>> w2 =
      loadHDF5Array(ffnnEncoderModelPath, "w2");
  std::vector<std::vector<float>> b2 =
      loadHDF5Array(ffnnEncoderModelPath, "b2");

  optimization::registerVectorFunction(
      "mat_mul10_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 140979, 2048),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add10_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 2048),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul10_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 2048, 256),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add10_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 256),
      {},
      true,
      catalog);

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  randomGenerator.setFloatRange(-1, 1);
  std::vector<std::vector<float>> bottomMLPWeight1 =
      randomGenerator.genFloat2dVector(256, 128);
  auto bottomMLPWeight1Vector =
      maker.arrayVector<float>(bottomMLPWeight1, REAL());

  std::vector<std::vector<float>> bottomMLPBias1 =
      randomGenerator.genFloat2dVector(128, 1);
  auto bottomMLPBias1Vector = maker.arrayVector<float>(bottomMLPBias1, REAL());

  optimization::registerVectorFunction(
      "mat_mul11_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              bottomMLPWeight1Vector->elements()->values()->asMutable<float>()),
          256,
          128),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_vector_add11_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              bottomMLPBias1Vector->elements()->values()->asMutable<float>()),
          128),
      {},
      true,
      catalog);

  std::vector<std::vector<float>> topMLPWeight1 =
      randomGenerator.genFloat2dVector(512, 256);
  auto topMLPWeight1Vector = maker.arrayVector<float>(bottomMLPWeight1, REAL());

  std::vector<std::vector<float>> topMLPBias1 =
      randomGenerator.genFloat2dVector(256, 1);
  auto topMLPBias1Vector = maker.arrayVector<float>(bottomMLPBias1, REAL());

  std::vector<std::vector<float>> topMLPWeight2 =
      randomGenerator.genFloat2dVector(256, 128);
  auto topMLPWeight2Vector = maker.arrayVector<float>(bottomMLPWeight1, REAL());

  std::vector<std::vector<float>> topMLPBias2 =
      randomGenerator.genFloat2dVector(128, 1);
  auto topMLPBias2Vector = maker.arrayVector<float>(bottomMLPBias1, REAL());

  std::vector<std::vector<float>> topMLPWeight3 =
      randomGenerator.genFloat2dVector(128, 1);
  auto topMLPWeight3Vector = maker.arrayVector<float>(bottomMLPWeight1, REAL());

  std::vector<std::vector<float>> topMLPBias3 =
      randomGenerator.genFloat2dVector(1, 1);
  auto topMLPBias3Vector = maker.arrayVector<float>(bottomMLPBias1, REAL());

  optimization::registerVectorFunction(
      "mat_mul12_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              topMLPWeight1Vector->elements()->values()->asMutable<float>()),
          512,
          256),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_vector_add12_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              topMLPBias1Vector->elements()->values()->asMutable<float>()),
          256),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_mul12_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              topMLPWeight2Vector->elements()->values()->asMutable<float>()),
          256,
          128),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_vector_add12_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              topMLPBias2Vector->elements()->values()->asMutable<float>()),
          128),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_mul12_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(
              topMLPWeight3Vector->elements()->values()->asMutable<float>()),
          128,
          1),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "mat_vector_add12_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(
              topMLPBias3Vector->elements()->values()->asMutable<float>()),
          1),
      {},
      true,
      catalog);
};

void registerTPCxAIUC10ModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase10_ffnn_weight.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");

  optimization::registerVectorFunction(
      "mat_mul1_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 2, 32),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 32),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul1_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 32, 1),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 1),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "sigmoid",
      Sigmoid::signatures(),
      std::make_unique<Sigmoid>(),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(), {}, true, catalog);
};

void registerTPCxAIUC10MLModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase10_lr_model_weight.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");

  optimization::registerVectorFunction(
      "mat_mul1_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 2, 1),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 1),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "sigmoid",
      Sigmoid::signatures(),
      std::make_unique<Sigmoid>(),
      {},
      true,
      catalog);
};

void registerTPCxAIUC3ModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase3_ffnn_weight.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");
  std::vector<std::vector<float>> w3 = loadHDF5Array(ffnnModelPath, "w3");
  std::vector<std::vector<float>> b3 = loadHDF5Array(ffnnModelPath, "b3");

  optimization::registerVectorFunction(
      "mat_mul1_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 3, 256),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 256),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul1_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 256, 1024),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 1024),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul1_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w3)), 1024, 1),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b3)), 1),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(), {}, true, catalog);

  // init user encoder
  std::unordered_map<int, int> storeIdMapping;
  for (int i = 1; i < 12; i++) {
    storeIdMapping[i] = i - 1;
  }
  optimization::registerVectorFunction(
      "store_id_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(std::move(storeIdMapping)),
      {},
      true,
      catalog);
  std::vector<std::string> departmentList = {
      "AUTOMOTIVE",
      "BATH AND SHOWER",
      "BEAUTY",
      "BEDDING",
      "BOYS WEAR",
      "CANDY, TOBACCO, COOKIES",
      "CELEBRATION",
      "COMM BREAD",
      "COOK AND DINE",
      "DAIRY",
      "DSD GROCERY",
      "ELECTRONICS",
      "FABRICS AND CRAFTS",
      "FINANCIAL SERVICES",
      "FROZEN FOODS",
      "GIRLS WEAR, 4-6X  AND 7-14",
      "GROCERY DRY GOODS",
      "HARDWARE",
      "HOME DECOR",
      "HOME MANAGEMENT",
      "HORTICULTURE AND ACCESS",
      "HOUSEHOLD CHEMICALS/SUPP",
      "HOUSEHOLD PAPER GOODS",
      "IMPULSE MERCHANDISE",
      "INFANT APPAREL",
      "INFANT CONSUMABLE HARDLINES",
      "JEWELRY AND SUNGLASSES",
      "LADIESWEAR",
      "LAWN AND GARDEN",
      "LIQUOR,WINE,BEER",
      "MEAT - FRESH & FROZEN",
      "MEDIA AND GAMING",
      "MENS WEAR",
      "OFFICE SUPPLIES",
      "PAINT AND ACCESSORIES",
      "PERSONAL CARE",
      "PETS AND SUPPLIES",
      "PHARMACY OTC",
      "PHARMACY RX",
      "PLAYERS AND ELECTRONICS",
      "PRODUCE",
      "SERVICE DELI",
      "SHOES",
      "SPORTING GOODS",
      "TOYS",
      "WIRELESS"};
  std::unordered_map<std::string, int> departmentMapping;
  for (int i = 0; i < departmentList.size(); i++) {
    departmentMapping[departmentList[i]] = i;
  }
  optimization::registerVectorFunction(
      "department_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(departmentMapping)),
      {},
      true,
      catalog);
};

void registerTPCxAIUC8ModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string ffnnModelPath =
      "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase8_ffnn_weight.h5";
  std::vector<std::vector<float>> w1 = loadHDF5Array(ffnnModelPath, "w1");
  std::vector<std::vector<float>> b1 = loadHDF5Array(ffnnModelPath, "b1");
  std::vector<std::vector<float>> w2 = loadHDF5Array(ffnnModelPath, "w2");
  std::vector<std::vector<float>> b2 = loadHDF5Array(ffnnModelPath, "b2");
  std::vector<std::vector<float>> w3 = loadHDF5Array(ffnnModelPath, "w3");
  std::vector<std::vector<float>> b3 = loadHDF5Array(ffnnModelPath, "b3");
  std::vector<std::vector<float>> w4 = loadHDF5Array(ffnnModelPath, "w4");
  std::vector<std::vector<float>> b4 = loadHDF5Array(ffnnModelPath, "b4");

  optimization::registerVectorFunction(
      "mat_mul1_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w1)), 4, 256),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b1)), 256),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul1_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w2)), 256, 128),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_4",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b2)), 128),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul1_5",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w3)), 128, 64),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_6",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b3)), 64),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_mul1_7",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(w4)), 64, 1),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "mat_vector_add1_8",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(b4)), 1),
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
      "relu", Relu::signatures(), std::make_unique<Relu>(), {}, true, catalog);

  std::vector<std::string> departmentList = {
      "AUTOMOTIVE",
      "BATH AND SHOWER",
      "BEAUTY",
      "BEDDING",
      "BOYS WEAR",
      "CANDY, TOBACCO, COOKIES",
      "CELEBRATION",
      "COMM BREAD",
      "COOK AND DINE",
      "DAIRY",
      "DSD GROCERY",
      "ELECTRONICS",
      "FABRICS AND CRAFTS",
      "FINANCIAL SERVICES",
      "FROZEN FOODS",
      "GIRLS WEAR, 4-6X  AND 7-14",
      "GROCERY DRY GOODS",
      "HARDWARE",
      "HOME DECOR",
      "HOME MANAGEMENT",
      "HORTICULTURE AND ACCESS",
      "HOUSEHOLD CHEMICALS/SUPP",
      "HOUSEHOLD PAPER GOODS",
      "IMPULSE MERCHANDISE",
      "INFANT APPAREL",
      "INFANT CONSUMABLE HARDLINES",
      "JEWELRY AND SUNGLASSES",
      "LADIESWEAR",
      "LAWN AND GARDEN",
      "LIQUOR,WINE,BEER",
      "MEAT - FRESH & FROZEN",
      "MEDIA AND GAMING",
      "MENS WEAR",
      "OFFICE SUPPLIES",
      "PAINT AND ACCESSORIES",
      "PERSONAL CARE",
      "PETS AND SUPPLIES",
      "PHARMACY OTC",
      "PHARMACY RX",
      "PLAYERS AND ELECTRONICS",
      "PRODUCE",
      "SERVICE DELI",
      "SHOES",
      "SPORTING GOODS",
      "TOYS",
      "WIRELESS"};
  std::unordered_map<std::string, int> departmentMapping;
  for (int i = 0; i < departmentList.size(); i++) {
    departmentMapping[departmentList[i]] = i;
  }
  optimization::registerVectorFunction(
      "department_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(departmentMapping)),
      {},
      true,
      catalog);
};

void registerTPCxAIUC7MLModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string svdModelPath =
      "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase8_svd.h5";
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
      catalog);
}

void registerTPCxAIUC8MLModelFunctions(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::string xgboostModelPath =
      "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase8_ml_xgboost_model/0.txt";

  optimization::registerVectorFunction(
      "decision_tree_predict",
      TreePrediction::signatures(),
      std::make_unique<TreePrediction>(0, xgboostModelPath, 4, false),
      {},
      true,
      catalog);

  registerCustomType("tree_type", std::make_unique<TreeTypeFactories>());

  optimization::registerVectorFunction(
      "velox_decision_tree_predict",
      VeloxTreePrediction::signatures(),
      std::make_unique<VeloxTreePrediction>(4),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "velox_decision_tree_construct",
      VeloxTreeConstruction::signatures(),
      std::make_unique<VeloxTreeConstruction>(),
      {},
      true,
      catalog);

  optimization::registerVectorFunction(
      "decision_forest_predict",
      TreePrediction::signatures(),
      std::make_unique<ForestPrediction>(
          "/home/velox/resources/model/tpcxai_sf1/final/velox/usecase8_ml_xgboost_model",
          4,
          false),
      {},
      true,
      catalog);

  std::vector<std::string> departmentList = {
      "AUTOMOTIVE",
      "BATH AND SHOWER",
      "BEAUTY",
      "BEDDING",
      "BOYS WEAR",
      "CANDY, TOBACCO, COOKIES",
      "CELEBRATION",
      "COMM BREAD",
      "COOK AND DINE",
      "DAIRY",
      "DSD GROCERY",
      "ELECTRONICS",
      "FABRICS AND CRAFTS",
      "FINANCIAL SERVICES",
      "FROZEN FOODS",
      "GIRLS WEAR, 4-6X  AND 7-14",
      "GROCERY DRY GOODS",
      "HARDWARE",
      "HOME DECOR",
      "HOME MANAGEMENT",
      "HORTICULTURE AND ACCESS",
      "HOUSEHOLD CHEMICALS/SUPP",
      "HOUSEHOLD PAPER GOODS",
      "IMPULSE MERCHANDISE",
      "INFANT APPAREL",
      "INFANT CONSUMABLE HARDLINES",
      "JEWELRY AND SUNGLASSES",
      "LADIESWEAR",
      "LAWN AND GARDEN",
      "LIQUOR,WINE,BEER",
      "MEAT - FRESH & FROZEN",
      "MEDIA AND GAMING",
      "MENS WEAR",
      "OFFICE SUPPLIES",
      "PAINT AND ACCESSORIES",
      "PERSONAL CARE",
      "PETS AND SUPPLIES",
      "PHARMACY OTC",
      "PHARMACY RX",
      "PLAYERS AND ELECTRONICS",
      "PRODUCE",
      "SERVICE DELI",
      "SHOES",
      "SPORTING GOODS",
      "TOYS",
      "WIRELESS"};
  std::unordered_map<std::string, int> departmentMapping;
  for (int i = 0; i < departmentList.size(); i++) {
    departmentMapping[departmentList[i]] = i;
  }
  optimization::registerVectorFunction(
      "department_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(departmentMapping)),
      {},
      true,
      catalog);
};

void registerTPCxAIDepartmentEncoder(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  std::vector<std::string> departmentList = {
      "AUTOMOTIVE",
      "BATH AND SHOWER",
      "BEAUTY",
      "BEDDING",
      "BOYS WEAR",
      "CANDY, TOBACCO, COOKIES",
      "CELEBRATION",
      "COMM BREAD",
      "COOK AND DINE",
      "DAIRY",
      "DSD GROCERY",
      "ELECTRONICS",
      "FABRICS AND CRAFTS",
      "FINANCIAL SERVICES",
      "FROZEN FOODS",
      "GIRLS WEAR, 4-6X  AND 7-14",
      "GROCERY DRY GOODS",
      "HARDWARE",
      "HOME DECOR",
      "HOME MANAGEMENT",
      "HORTICULTURE AND ACCESS",
      "HOUSEHOLD CHEMICALS/SUPP",
      "HOUSEHOLD PAPER GOODS",
      "IMPULSE MERCHANDISE",
      "INFANT APPAREL",
      "INFANT CONSUMABLE HARDLINES",
      "JEWELRY AND SUNGLASSES",
      "LADIESWEAR",
      "LAWN AND GARDEN",
      "LIQUOR,WINE,BEER",
      "MEAT - FRESH & FROZEN",
      "MEDIA AND GAMING",
      "MENS WEAR",
      "OFFICE SUPPLIES",
      "PAINT AND ACCESSORIES",
      "PERSONAL CARE",
      "PETS AND SUPPLIES",
      "PHARMACY OTC",
      "PHARMACY RX",
      "PLAYERS AND ELECTRONICS",
      "PRODUCE",
      "SERVICE DELI",
      "SHOES",
      "SPORTING GOODS",
      "TOYS",
      "WIRELESS"};
  std::unordered_map<std::string, int> departmentMapping;
  for (int i = 0; i < departmentList.size(); i++) {
    departmentMapping[departmentList[i]] = i;
  }
  optimization::registerVectorFunction(
      "department_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(departmentMapping)),
      {},
      true,
      catalog);
}

void registerGenderEncoder(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  std::unordered_map<std::string, int> genderMapping;
  genderMapping["F"] = 0;
  genderMapping["M"] = 1;

  optimization::registerVectorFunction(
      "gender_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(std::move(genderMapping)),
      {},
      true,
      catalog);
}

void registerMovielensAgeMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_,
    const std::string& fileName) {
  VectorMaker maker{pool_.get()};

  optimization::registerVectorFunction(
      "user_age_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(fmt::format(
          "/home/velox/resources/model/movielens/final/velox/{}", fileName)),
      {},
      true,
      catalog);
}

void registerMovielensOccupationMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_,
    const std::string& fileName) {
  VectorMaker maker{pool_.get()};

  optimization::registerVectorFunction(
      "user_occupation_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(fmt::format(
          "/home/velox/resources/model/movielens/final/velox/{}", fileName)),
      {},
      true,
      catalog);
}

void registerMovielensGenerOneHotEncoder(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};

  std::unordered_map<std::string, int> genresMapping = {
      {"Animation", 1},
      {"Children's", 2},
      {"Comedy", 3},
      {"Adventure", 4},
      {"Fantasy", 5},
      {"Romance", 6},
      {"Drama", 7},
      {"Action", 8},
      {"Crime", 9},
      {"Thriller", 10},
      {"Horror", 11},
      {"Sci-Fi", 12},
      {"Documentary", 13},
      {"War", 14},
      {"Musical", 15},
      {"Mystery", 16},
      {"Film-Noir", 17},
      {"Western", 18}};

  optimization::registerVectorFunction(
      "genres_encoder",
      OneHotEncoder::signatures(),
      std::make_unique<OneHotEncoder>(std::move(genresMapping)),
      {},
      true,
      catalog);
}

void registerMovielensPopularityMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "movie_popularity_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(
          "/home/velox/resources/model/movielens/final/velox/q9_movie_popularity_minmax_scaler.txt"),
      /*extraParameters=*/{},
      /*isDeterministic=*/true,
      catalog);
}

void registerMovielensVoteAverageMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "movie_vote_average_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(
          "/home/velox/resources/model/movielens/final/velox/q9_movie_vote_avg_minmax_scaler.txt"),
      {},
      true,
      catalog);
}

void registerMovielensVoteCountMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "movie_vote_count_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(
          "/home/velox/resources/model/movielens/final/velox/q9_movie_vote_count_minmax_scaler.txt"),
      {},
      true,
      catalog);
}

void registerMovielensRatingMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "rating_minmax_scaler",
      MinMaxScaler::signatures(),
      std::make_unique<MinMaxScaler>(
          "/home/velox/resources/model/movielens/final/velox/q9_rating_rating_minmax_scaler.txt"),
      {},
      true,
      catalog);
}

void registerMovielensRatingMapToArray(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "map_to_array",
      RatingMapToArray::signatures(),
      std::make_unique<RatingMapToArray>(3706),
      {},
      true,
      catalog);
}

void registerTPCxAIHFTokenizer(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "hf_tokenizer",
      HuggingFaceTokenizer::signatures(),
      std::make_unique<HuggingFaceTokenizer>(
          "/home/velox/resources/model/tokenizer/roberta.json"),
      {},
      true,
      catalog);
}

void registerTPCxAITFFeatureExtractor(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool_) {
  VectorMaker maker{pool_.get()};
  optimization::registerVectorFunction(
      "extract_tf_features",
      TokenFreqVector::signatures(),
      std::make_unique<TokenFreqVector>(50265),
      {},
      true,
      catalog);
}

void registerOneHotInt(
    const std::string& functionName,
    const std::string& mappingFile,
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool) {
    
    VectorMaker maker{pool.get()};
    std::ifstream in(mappingFile);
    if (!in) {
        throw std::runtime_error("OneHot mapping file not found: " + mappingFile);
    }

    std::string line;
    std::getline(in, line); // skip header

    std::vector<std::pair<int64_t, int>> mapping;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string valStr, posStr;
        if (!std::getline(ss, valStr, ',')) continue;
        if (!std::getline(ss, posStr, ',')) continue;
        int64_t value = std::stoll(valStr);
        int position = std::stoi(posStr);
        mapping.emplace_back(value, position);
    }

    // std::cout << "Loaded one-hot mappings for " << functionName << ":\n";
    // for (auto& p : mapping) {
    //     std::cout << "  value=" << p.first << "  position=" << p.second << "\n";
    // }

    optimization::registerVectorFunction(
        functionName,
        OneHotEncoderInt::signatures(),
        std::make_unique<OneHotEncoderInt>(std::move(mapping)),
        /*extraTypeSignatures=*/{},
        /*deterministic=*/true,
        catalog);
}


void registerOneHotString(
    const std::string& functionName,
    const std::string& mappingFile,
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool) {
    
    VectorMaker maker{pool.get()};
    std::ifstream in(mappingFile);
    if (!in) {
        throw std::runtime_error("OneHot mapping file not found: " + mappingFile);
    }

    std::string line;
    std::getline(in, line); // skip header

    std::vector<std::pair<std::string, int>> mapping;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key, posStr;
        if (!std::getline(ss, key, ',')) continue;
        if (!std::getline(ss, posStr, ',')) continue;
        int position = std::stoi(posStr);
        mapping.emplace_back(key, position);
    }

    // std::cout << "Loaded string one-hot mappings for " << functionName << ":\n";
    // for (auto& p : mapping) {
    //     std::cout << "  value=\"" << p.first << "\"  position=" << p.second << "\n";
    // }

    optimization::registerVectorFunction(
        functionName,
        OneHotEncoderString::signatures(),
        std::make_unique<OneHotEncoderString>(std::move(mapping)),
        /*extraTypeSignatures=*/{},
        /*deterministic=*/true,
        catalog);
}


void registerTreePredictExpedia(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool) {
    
    // VectorMaker maker{pool.get()};

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
        catalog);
}

void registerFeatureMinMaxScaler(
    CataLog& catalog,
    std::shared_ptr<memory::MemoryPool> pool,
    const std::string& featureName,
    const std::string& fileName) {

  const std::string& statsPath = fileName;
  const std::string udfName = fmt::format("{}_minmax_scaler", featureName);

  optimization::registerVectorFunction(
      udfName,                          // function name
      MinMaxScaler::signatures(),       
      std::make_unique<MinMaxScaler>(statsPath),
      /*extra params=*/{},
      /*deterministic=*/true,
      catalog);
}

void registerForesttoRelationalFunctions(std::string modelPath, int numCols) {
    std::cout << "To register function for TreePrediction" << std::endl;

    exec::registerVectorFunction(
        "decision_tree_predict",
        TreePrediction::signatures(),
        std::make_unique<TreePrediction>(
            0, fmt::format("{}/0.txt", modelPath), numCols, true));

    std::cout << "To register type for Tree" << std::endl;

    registerCustomType("tree_type", std::make_unique<TreeTypeFactories>());

    std::cout << "To register function for VeloxTreePrediction" << std::endl;

    exec::registerVectorFunction(
        "velox_decision_tree_predict",
        VeloxTreePrediction::signatures(),
        std::make_unique<VeloxTreePrediction>(numCols));

    std::cout << "To register function for VeloxTreeConstruction" << std::endl;

    exec::registerVectorFunction(
        "velox_decision_tree_construct",
        VeloxTreeConstruction::signatures(),
        std::make_unique<VeloxTreeConstruction>());

    std::cout << "To register function for ForestPrediction" << std::endl;

    exec::registerVectorFunction(
        "decision_forest_predict",
        TreePrediction::signatures(),
        std::make_unique<ForestPrediction>(modelPath, numCols, true));
  }