#include <folly/init/Init.h>
#include <torch/torch.h>
#include <random>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/Dropout.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/SequencePooling.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

// Utility function to generate random float/int values

class RandomGenerator {
 public:
  RandomGenerator(float lb, float ub, int randomSeed = 0) {
    gen_ = std::mt19937(randomSeed);
    distR_ = std::uniform_real_distribution<float>(lb, ub);
    distI_ = std::uniform_int_distribution<int>((int)lb, (int)ub);
  }

  void setFloatRange(float lb, float ub) {
    distR_ = std::uniform_real_distribution<float>(lb, ub);
  }

  void setIntRange(int lb, int ub) {
    // NOTE: ub is included when sampling
    distI_ = std::uniform_int_distribution<int>(lb, ub);
  }

  float genRandomFloatValue() {
    return distR_(gen_);
  }

  int genRandomIntValue() {
    return distI_(gen_);
  }

  std::vector<std::vector<int>>
  genLookUpIndices(int numRow, int maxIndexNumber, int maxVariadic = 1) {
    // max variadic is the maximum number of sampled indices for each data
    // point, in two-tower model, the value is 6 for genre
    setIntRange(0, maxIndexNumber);
    std::vector<std::vector<int>> indicesVectors;
    for (int i = 0; i < numRow; i++) {
      std::vector<int> sampledIndices;
      int numSampledIndices = 1;
      if (maxVariadic > 1) {
        numSampledIndices = (i % maxVariadic == 0) ? 1 : i % maxVariadic;
      }
      for (int j = 0; j < numSampledIndices; j++) {
        sampledIndices.push_back(genRandomIntValue());
      }
      indicesVectors.push_back(sampledIndices);
    }
    return indicesVectors;
  }

  std::vector<std::vector<float>> genFloat2dVector(int numRow, int numCol) {
    // Initialize the input1 feature vector
    std::vector<std::vector<float>> float2dVector;

    for (int i = 0; i < numRow; i++) {
      std::vector<float> floatVector;
      for (int j = 0; j < numCol; j++) {
        floatVector.push_back(genRandomFloatValue());
      }
      float2dVector.push_back(floatVector);
    }
    return float2dVector;
  }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<float> distR_;
  std::uniform_int_distribution<int> distI_;
};

class TowTowerModelTest : public HiveConnectorTestBase {
 public:
  TowTowerModelTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    SetUp();
  }

  ~TowTowerModelTest() {}

  void run();
  void testTwoTowerModel();

  void TestBody() override {}

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() {
    HiveConnectorTestBase::TearDown();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::addDefaultLeafMemoryPool();
  VectorMaker maker{pool_.get()};
};

// Test Embedding Layer
void TowTowerModelTest::testTwoTowerModel() {
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  // embeddingDims is shared among all embedding layer
  int embeddingDims = 32;

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

  exec::registerVectorFunction(
      "user_id_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          userIdEmbeddingWeightsVector->elements()
              ->values()
              ->asMutable<float>(),
          userIdNumEmbedding,
          embeddingDims));

  exec::registerVectorFunction(
      "gender_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          genderEmbeddingWeightsVector->elements()
              ->values()
              ->asMutable<float>(),
          genderNumEmbedding,
          embeddingDims));

  exec::registerVectorFunction(
      "age_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          ageEmbeddingWeightsVector->elements()->values()->asMutable<float>(),
          ageNumEmbedding,
          embeddingDims));

  exec::registerVectorFunction(
      "occupation_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          occupationEmbeddingWeightsVector->elements()
              ->values()
              ->asMutable<float>(),
          occupationNumEmbedding,
          embeddingDims));

  int numSamples = 5;

  std::vector<std::vector<int>> userIndicesVector =
      randomGenerator.genLookUpIndices(numSamples, userIdNumEmbedding - 1);
  auto userIndicesArray = maker.arrayVector<int>(userIndicesVector, INTEGER());
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
      randomGenerator.genLookUpIndices(numSamples, occupationNumEmbedding - 1);
  auto occupationIndicesArray =
      maker.arrayVector<int>(occupationIndicesVector, INTEGER());
  auto occupationIndicesArrayRowVector =
      maker.rowVector({"occupation"}, {occupationIndicesArray});

  randomGenerator.setFloatRange(0, 1);
  std::vector<std::vector<float>> userMeanRatingVector =
      randomGenerator.genFloat2dVector(numSamples, 1);
  auto userMeanRatingArray =
      maker.arrayVector<float>(userMeanRatingVector, REAL());

  auto userPlan = exec::test::PlanBuilder(pool_.get())
                      .values({userIndicesArrayRowVector})
                      .project({"user_id_embedding(user_id)"})
                      .planNode();

  auto userEmbedding =
      exec::test::AssertQueryBuilder(userPlan).copyResults(pool_.get());

  auto genderPlan = exec::test::PlanBuilder(pool_.get())
                        .values({genderIndicesArrayRowVector})
                        .project({"gender_embedding(gender)"})
                        .planNode();

  auto genderEmbedding =
      exec::test::AssertQueryBuilder(genderPlan).copyResults(pool_.get());

  auto agePlan = exec::test::PlanBuilder(pool_.get())
                     .values({ageIndicesArrayRowVector})
                     .project({"age_embedding(age)"})
                     .planNode();

  auto ageEmbedding =
      exec::test::AssertQueryBuilder(agePlan).copyResults(pool_.get());

  auto occupationPlan = exec::test::PlanBuilder(pool_.get())
                            .values({occupationIndicesArrayRowVector})
                            .project({"occupation_embedding(occupation)"})
                            .planNode();

  auto occupationEmbedding =
      exec::test::AssertQueryBuilder(occupationPlan).copyResults(pool_.get());

  // std::cout << "[INFO] Results: \n" << results->toString() << std::endl;
  std::cout << "[INFO] Results: \n"
            << userEmbedding->toString(0, userEmbedding->size()) << std::endl;

  exec::registerVectorFunction(
      "concat1",
      Concat::signatures(),
      std::make_unique<Concat>(embeddingDims, embeddingDims));

  auto in1 = maker.rowVector(
      {"in1", "in2"}, {userEmbedding->childAt(0), genderEmbedding->childAt(0)});

  auto concatPlan1 = exec::test::PlanBuilder(pool_.get())
                         .values({in1})
                         .project({"concat1(in1, in2)"})
                         .planNode();

  auto out1 =
      exec::test::AssertQueryBuilder(concatPlan1).copyResults(pool_.get());

  exec::registerVectorFunction(
      "concat2",
      Concat::signatures(),
      std::make_unique<Concat>(2 * embeddingDims, embeddingDims));

  auto in2 = maker.rowVector(
      {"in1", "in2"}, {out1->childAt(0), ageEmbedding->childAt(0)});

  auto concatPlan2 = exec::test::PlanBuilder(pool_.get())
                         .values({in2})
                         .project({"concat2(in1, in2)"})
                         .planNode();
  auto out2 =
      exec::test::AssertQueryBuilder(concatPlan2).copyResults(pool_.get());

  exec::registerVectorFunction(
      "concat3",
      Concat::signatures(),
      std::make_unique<Concat>(3 * embeddingDims, embeddingDims));

  auto in3 = maker.rowVector(
      {"in1", "in2"}, {out2->childAt(0), occupationEmbedding->childAt(0)});

  auto concatPlan3 = exec::test::PlanBuilder(pool_.get())
                         .values({in3})
                         .project({"concat3(in1, in2)"})
                         .planNode();

  auto out3 =
      exec::test::AssertQueryBuilder(concatPlan3).copyResults(pool_.get());

  exec::registerVectorFunction(
      "concat4",
      Concat::signatures(),
      std::make_unique<Concat>(4 * embeddingDims, 1));

  auto in4 =
      maker.rowVector({"in1", "in2"}, {out3->childAt(0), userMeanRatingArray});

  auto concatPlan4 = exec::test::PlanBuilder(pool_.get())
                         .values({in4})
                         .project({"concat4(in1, in2) as user_nn_in"})
                         .planNode();

  auto out4 =
      exec::test::AssertQueryBuilder(concatPlan4).copyResults(pool_.get());

  std::cout << "[INFO] OUT4: \n"
            << out4->toString(0, out4->size()) << std::endl;

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

  exec::registerVectorFunction(
      "mat_mul1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          userNNweight1Vector->elements()->values()->asMutable<float>(),
          129,
          300));

  exec::registerVectorFunction(
      "mat_add1",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          userNNBias1Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          userNNweight2Vector->elements()->values()->asMutable<float>(),
          300,
          300));

  exec::registerVectorFunction(
      "mat_add2",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          userNNBias2Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          userNNweight3Vector->elements()->values()->asMutable<float>(),
          300,
          128));

  exec::registerVectorFunction(
      "mat_add3",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          userNNBias3Vector->elements()->values()->asMutable<float>(), 128));

  exec::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>());

  std::vector<std::vector<float>> batchNorm1Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm1WeightVector =
      maker.arrayVector<float>(batchNorm1Weight, REAL());
  std::vector<std::vector<float>> batchNorm1Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm1BiasVector = maker.arrayVector<float>(batchNorm1Bias, REAL());

  exec::registerVectorFunction(
      "batch_norm1",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          batchNorm1WeightVector->elements()->values()->asMutable<float>(),
          batchNorm1BiasVector->elements()->values()->asMutable<float>(),
          300));

  std::vector<std::vector<float>> batchNorm2Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2WeightVector =
      maker.arrayVector<float>(batchNorm2Weight, REAL());
  std::vector<std::vector<float>> batchNorm2Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2BiasVector = maker.arrayVector<float>(batchNorm2Bias, REAL());

  exec::registerVectorFunction(
      "batch_norm2",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          batchNorm2WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2BiasVector->elements()->values()->asMutable<float>(),
          300));

  std::vector<std::vector<float>> batchNorm3Weight =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm3WeightVector =
      maker.arrayVector<float>(batchNorm3Weight, REAL());
  std::vector<std::vector<float>> batchNorm3Bias =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm3BiasVector = maker.arrayVector<float>(batchNorm3Bias, REAL());

  exec::registerVectorFunction(
      "batch_norm3",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          batchNorm3WeightVector->elements()->values()->asMutable<float>(),
          batchNorm3BiasVector->elements()->values()->asMutable<float>(),
          128));

  auto userNNPlan =
      exec::test::PlanBuilder(pool_.get())
          .values({out4})
          .project(
              {"relu(batch_norm3(mat_add3(mat_mul3(relu(batch_norm2(mat_add2(mat_mul2(relu(batch_norm1(mat_add1(mat_mul1(user_nn_in)))))))))))) as user_nn_out"})
          .planNode();
  auto userNNOut =
      exec::test::AssertQueryBuilder(userNNPlan).copyResults(pool_.get());
  std::cout << "[INFO] user NN: \n"
            << userNNOut->toString(0, userNNOut->size()) << std::endl;

  // Item-Tower

  // movid_id
  int movieIdNumEmbedding = 3668;
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

  exec::registerVectorFunction(
      "movie_id_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          movieIdEmbeddingWeightsVector->elements()
              ->values()
              ->asMutable<float>(),
          movieIdNumEmbedding,
          embeddingDims));

  exec::registerVectorFunction(
      "genres_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          genderEmbeddingWeightsVector->elements()
              ->values()
              ->asMutable<float>(),
          genderNumEmbedding,
          embeddingDims));

  exec::registerVectorFunction(
      "sequence_pooling",
      SequencePooling::signatures(),
      std::make_unique<SequencePooling>(std::string("MEAN"), embeddingDims));

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

  auto itemPlan = exec::test::PlanBuilder(pool_.get())
                      .values({movieIndicesArrayRowVector})
                      .project({"movie_id_embedding(movie_id)"})
                      .planNode();

  auto itemEmbedding =
      exec::test::AssertQueryBuilder(itemPlan).copyResults(pool_.get());

  auto genresPlan = exec::test::PlanBuilder(pool_.get())
                        .values({genresIndicesArrayRowVector})
                        .project({"sequence_pooling(genres_embedding(genres))"})
                        .planNode();

  auto genresEmbedding =
      exec::test::AssertQueryBuilder(genresPlan).copyResults(pool_.get());

  exec::registerVectorFunction(
      "concat2_1",
      Concat::signatures(),
      std::make_unique<Concat>(embeddingDims, embeddingDims));

  auto in2_1 = maker.rowVector(
      {"in1", "in2"}, {itemEmbedding->childAt(0), genresEmbedding->childAt(0)});

  auto concatPlan2_1 = exec::test::PlanBuilder(pool_.get())
                           .values({in2_1})
                           .project({"concat2_1(in1, in2)"})
                           .planNode();

  auto out2_1 =
      exec::test::AssertQueryBuilder(concatPlan2_1).copyResults(pool_.get());

  exec::registerVectorFunction(
      "concat2_1",
      Concat::signatures(),
      std::make_unique<Concat>(2 * embeddingDims, 1));

  auto in2_2 = maker.rowVector(
      {"in1", "in2"}, {out2_1->childAt(0), itemMeanRatingArray});

  auto concatPlan2_2 = exec::test::PlanBuilder(pool_.get())
                           .values({in2_2})
                           .project({"concat2_1(in1, in2) as item_nn_in"})
                           .planNode();
  auto out2_2 =
      exec::test::AssertQueryBuilder(concatPlan2_2).copyResults(pool_.get());

  std::cout << "[INFO] item dnn input: \n"
            << out2_2->toString(0, out2_2->size()) << std::endl;

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

  exec::registerVectorFunction(
      "mat_mul2_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight1Vector->elements()->values()->asMutable<float>(),
          65,
          300));

  exec::registerVectorFunction(
      "mat_add2_1",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          itemNNBias1Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul2_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight2Vector->elements()->values()->asMutable<float>(),
          300,
          300));

  exec::registerVectorFunction(
      "mat_add2_2",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          itemNNBias2Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul2_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight3Vector->elements()->values()->asMutable<float>(),
          300,
          128));

  exec::registerVectorFunction(
      "mat_add2_3",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          itemNNBias3Vector->elements()->values()->asMutable<float>(), 128));

  std::vector<std::vector<float>> batchNorm2_1Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_1WeightVector =
      maker.arrayVector<float>(batchNorm2_1Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_1Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_1BiasVector =
      maker.arrayVector<float>(batchNorm2_1Bias, REAL());

  exec::registerVectorFunction(
      "batch_norm2_1",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          batchNorm2_1WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2_1BiasVector->elements()->values()->asMutable<float>(),
          300));

  std::vector<std::vector<float>> batchNorm2_2Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_2WeightVector =
      maker.arrayVector<float>(batchNorm2_2Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_2Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_2BiasVector =
      maker.arrayVector<float>(batchNorm2_2Bias, REAL());

  exec::registerVectorFunction(
      "batch_norm2_2",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          batchNorm2_2WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2_2BiasVector->elements()->values()->asMutable<float>(),
          300));

  std::vector<std::vector<float>> batchNorm2_3Weight =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm2_3WeightVector =
      maker.arrayVector<float>(batchNorm2_3Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_3Bias =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm2_3BiasVector =
      maker.arrayVector<float>(batchNorm2_3Bias, REAL());

  exec::registerVectorFunction(
      "batch_norm2_3",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          batchNorm2_3WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2_3BiasVector->elements()->values()->asMutable<float>(),
          128));

  auto itemNNPlan =
      exec::test::PlanBuilder(pool_.get())
          .values({out2_2})
          .project(
              {"relu(batch_norm2_3(mat_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_add2_1(mat_mul2_1(item_nn_in)))))))))))) as item_nn_out"})
          .planNode();
  auto itemNNOut =
      exec::test::AssertQueryBuilder(itemNNPlan).copyResults(pool_.get());
  std::cout << "[INFO] item NN: \n"
            << itemNNOut->toString(0, itemNNOut->size()) << std::endl;

  exec::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(128));

  auto finalInputRowVector = maker.rowVector(
      {"in1", "in2"}, {userNNOut->childAt(0), itemNNOut->childAt(0)});
  auto finalStagePlan = exec::test::PlanBuilder(pool_.get())
                            .values({finalInputRowVector})
                            .project({"cosine_similarity(in1, in2)"})
                            .planNode();
  auto scores =
      exec::test::AssertQueryBuilder(finalStagePlan).copyResults(pool_.get());
  std::cout << "[INFO] final score: \n"
            << scores->toString(0, scores->size()) << std::endl;
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  TowTowerModelTest demo;
  demo.testTwoTowerModel();
}