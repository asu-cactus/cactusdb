#include <folly/init/Init.h>
#include <torch/torch.h>
#include <random>
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/Dropout.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/Encoder.h"
#include "velox/ml_functions/SequencePooling.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

// Utility function to generate random float/int values

class TowTowerModelPipelineTest : public HiveConnectorTestBase {
 public:
  TowTowerModelPipelineTest() {
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
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~TowTowerModelPipelineTest() {}

  void run();
  void testEndtoEndPipelineMultiThreading(int numSamples, int numSplit);

  void TestBody() override {}

  void SetUp() {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
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


void TowTowerModelPipelineTest::testEndtoEndPipelineMultiThreading(
    int numSamples,
    int numSplit) {
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  int embeddingDims = 32;

  // init user encoder
  std::unordered_map<int, int> userIdMapping;
  for (int i = 1; i < 6041; i++) {
    userIdMapping[i] = i - 1;
  }

  exec::registerVectorFunction(
      "user_id_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(userIdMapping));

  // init movie encoder
  std::unordered_map<int, int> movieIdMapping;
  for (int i = 1; i < 3953; i++) {
    movieIdMapping[i] = i - 1;
  }

  exec::registerVectorFunction(
      "movie_id_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(movieIdMapping));

  // init age encoder
  std::unordered_map<int, int> ageMapping;
  ageMapping[1] = 0;
  ageMapping[18] = 1;
  ageMapping[25] = 2;
  ageMapping[35] = 3;
  ageMapping[45] = 4;
  ageMapping[50] = 5;
  ageMapping[56] = 6;

  exec::registerVectorFunction(
      "age_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(ageMapping));

  // init occupation  encoder
  std::unordered_map<int, int> occupationMapping;
  for (int i = 0; i < 21; i++) {
    occupationMapping[i] = i;
  }

  exec::registerVectorFunction(
      "occupation_encoder",
      IntEncoder::signatures(),
      std::make_unique<IntEncoder>(occupationMapping));

  std::unordered_map<std::string, int> genderMapping;
  genderMapping["F"] = 0;
  genderMapping["M"] = 1;

  exec::registerVectorFunction(
      "gender_encoder",
      StringEncoder::signatures(),
      std::make_unique<StringEncoder>(genderMapping));

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

  exec::registerVectorFunction(
      "genres_encoder",
      StringVariadicEncoder::signatures(),
      std::make_unique<StringVariadicEncoder>(genresMapping));

  exec::registerVectorFunction(
      "convert_int_array",
      ConvertToIntArray::signatures(),
      std::make_unique<ConvertToIntArray>());

  exec::registerVectorFunction(
      "convert_float_array",
      ConvertToFloatArray::signatures(),
      std::make_unique<ConvertToFloatArray>());

  exec::registerVectorFunction(
      "convert_double_to_float_array",
      ConvertDoubleToFloatArray::signatures(),
      std::make_unique<ConvertDoubleToFloatArray>());

  exec::registerVectorFunction(
      "change_rating",
      ChangeRating::signatures(),
      std::make_unique<ChangeRating>());

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
  exec::registerVectorFunction(
      "concat1",
      Concat::signatures(),
      std::make_unique<Concat>(embeddingDims, embeddingDims));

  exec::registerVectorFunction(
      "concat2",
      Concat::signatures(),
      std::make_unique<Concat>(2 * embeddingDims, embeddingDims));

  exec::registerVectorFunction(
      "concat3",
      Concat::signatures(),
      std::make_unique<Concat>(3 * embeddingDims, embeddingDims));
  exec::registerVectorFunction(
      "concat4",
      Concat::signatures(),
      std::make_unique<Concat>(4 * embeddingDims, 1));

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
      "mat_vector_add1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          userNNBias1Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          userNNweight2Vector->elements()->values()->asMutable<float>(),
          300,
          300));

  exec::registerVectorFunction(
      "mat_vector_add2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          userNNBias2Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          userNNweight3Vector->elements()->values()->asMutable<float>(),
          300,
          128));

  exec::registerVectorFunction(
      "mat_vector_add3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
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

  exec::registerVectorFunction(
      "concat2_1",
      Concat::signatures(),
      std::make_unique<Concat>(embeddingDims, embeddingDims));

  exec::registerVectorFunction(
      "concat2_2",
      Concat::signatures(),
      std::make_unique<Concat>(2 * embeddingDims, 1));

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
      "mat_vector_add2_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          itemNNBias1Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul2_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight2Vector->elements()->values()->asMutable<float>(),
          300,
          300));

  exec::registerVectorFunction(
      "mat_vector_add2_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          itemNNBias2Vector->elements()->values()->asMutable<float>(), 300));

  exec::registerVectorFunction(
      "mat_mul2_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          itemNNweight3Vector->elements()->values()->asMutable<float>(),
          300,
          128));

  exec::registerVectorFunction(
      "mat_vector_add2_3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
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

  exec::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(128));

  // read raw data

  auto inputRowType =
      ROW({"user_id",
           "movie_id",
           "rating",
           "timestamp",
           "gender",
           "age",
           "occupation",
           "zipcode",
           "title",
           "genres"},
          {INTEGER(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           VARCHAR(),
           INTEGER(),
           INTEGER(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR()});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;
  core::PlanNodeId readRawDataPlanNode;

  //   auto readRawDataPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
  //                              .tableScan(inputRowType, {}, "")
  //                              .capturePlanNodeId(readRawDataPlanNode)
  //                              //   .singleAggregation({}, {"count(*)"})
  //                              .planFragment();
  // .planNode();

  constexpr int64_t KB = 1024L;
  constexpr int64_t MB = 1024L * KB;
  constexpr int64_t GB = 1024L * MB;
  std::shared_ptr<memory::MemoryPool> rootPool{
      memory::defaultMemoryManager().addRootPool("root", 500 * MB)};
  queryCtx_->testingOverrideMemoryPool(rootPool);
  uint64_t kSizeKB = 1024UL;

  //   int numSplit = 2;
  auto hiveSplits = makeHiveConnectorSplits(
      {"/root/velox_latest/data/movielens.parquet"},
      numSplit,
      dwio::common::FileFormat::PARQUET);
  auto hiveSplits1 = makeHiveConnectorSplits(
      {"/root/velox_latest/data/movielens.parquet"},
      numSplit,
      dwio::common::FileFormat::PARQUET);
  //   int concurrency = 2;
  boost::interprocess::interprocess_semaphore semaphore(numSplit);

  std::vector<int> userIds = randomGenerator.gen1DInt(numSamples, 1, 6040);
  auto userIdFlatVector = maker.flatVector<int>(userIds, INTEGER());
  auto userRowVector = maker.rowVector({"u_user_id"}, {userIdFlatVector});
  std::vector<int> movieIds = randomGenerator.gen1DInt(numSamples, 1, 3706);
  auto movieIdFlatVector = maker.flatVector<int>(movieIds, INTEGER());
  auto movieRowVector = maker.rowVector({"m_movie_id"}, {movieIdFlatVector});

  auto changeRatingPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                              .tableScan(inputRowType, {}, "")
                              .capturePlanNodeId(readRawDataPlanNode)
                              .project(
                                  {"user_id",
                                   "movie_id",
                                   "gender",
                                   "age",
                                   "occupation",
                                   "title",
                                   "genres",
                                   "change_rating(rating) as rating"})
                              .planNode();
  core::PlanNodeId readRawDataPlanNode1;
  auto changeRatingPlan1 = PlanBuilder(planNodeIdGenerator, pool_.get())
                               .tableScan(inputRowType, {}, "")
                               .capturePlanNodeId(readRawDataPlanNode1)
                               .project(
                                   {"user_id",
                                    "movie_id",
                                    "gender",
                                    "age",
                                    "occupation",
                                    "title",
                                    "genres",
                                    "change_rating(rating) as rating"})
                               .planNode();
  //   // get average rating for the user data
  auto userDataPreprocessPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .values({userRowVector})
          .hashJoin(
              {"u_user_id"},
              {"user_id"},
              changeRatingPlan,
              "",
              {"u_user_id", "gender", "age", "occupation", "rating"})
          .singleAggregation(
              {"u_user_id", "gender", "age", "occupation"},
              {"avg(rating) as user_mean_rating"})
          .project(
              {"user_id_encoder(convert_int_array(u_user_id)) as user_id",
               "gender_encoder(gender) as gender",
               "age_encoder(convert_int_array(age)) as age",
               "occupation_encoder(convert_int_array(occupation)) as occupation",
               "convert_double_to_float_array(user_mean_rating) as user_mean_rating"})
          .planFragment();

  auto movieDataPreprocessPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .values({movieRowVector})
          .hashJoin(
              {"m_movie_id"},
              {"movie_id"},
              changeRatingPlan1,
              "",
              {"m_movie_id", "genres", "rating"})
          .singleAggregation(
              {"m_movie_id", "genres"}, {"avg(rating) as movie_mean_rating"})
          .project(
              {"movie_id_encoder(convert_int_array(m_movie_id)) as movie_id",
               "genres_encoder(split(genres, '|')) as genres",
               "convert_double_to_float_array(movie_mean_rating) as movie_mean_rating"})
          .planFragment();

  std::shared_ptr<std::vector<RowVectorPtr>> resultUserData =
      std::make_shared<std::vector<RowVectorPtr>>();

  std::shared_ptr<std::vector<RowVectorPtr>> resultMovieData =
      std::make_shared<std::vector<RowVectorPtr>>();
  auto taskUser = exec::Task::create(
      "0",
      userDataPreprocessPlan,
      0,
      queryCtx_,
      [resultUserData](RowVectorPtr vector, ContinueFuture* future) {
        if (vector) {
          resultUserData->push_back(vector);
        }
        return exec::BlockingReason::kNotBlocked;
      });

  auto taskMovie = exec::Task::create(
      "1",
      movieDataPreprocessPlan,
      1,
      queryCtx_,
      [resultMovieData](RowVectorPtr vector, ContinueFuture* future) {
        if (vector) {
          resultMovieData->push_back(vector);
        }
        return exec::BlockingReason::kNotBlocked;
      });

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  taskUser->start(taskUser, numSplit);
  taskMovie->start(taskMovie, numSplit);

  for (auto& split : hiveSplits) {
    taskUser->addSplit(readRawDataPlanNode, exec::Split(std::move(split)));
  }
  for (auto& split : hiveSplits1) {
    taskMovie->addSplit(readRawDataPlanNode1, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readRawDataPlanNode);
  taskMovie->noMoreSplits(readRawDataPlanNode1);
  waitForFinishedDrivers(taskUser);
  waitForFinishedDrivers(taskMovie);
 

  std::chrono::steady_clock::time_point preprocessEnd =
      std::chrono::steady_clock::now();
  auto finalInferencePlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .values((*resultUserData))
          .rowNumber({}, std::nullopt, true)
          .mergeJoin(
              {"row_number"},
              {"row_number"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .values((*resultMovieData))
                  .rowNumber({}, std::nullopt, true)
                  .planNode(),
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "movie_id",
               "genres",
               "movie_mean_rating"})
          .project(
              {"user_id_embedding(user_id) as user_id",
               "gender_embedding(gender) as gender",
               "age_embedding(age) as age",
               "occupation_embedding(occupation) as occupation",
               "user_mean_rating",
               "movie_id_embedding(movie_id) as movie_id",
               "sequence_pooling(genres_embedding(genres)) as genres",
               "movie_mean_rating"})
          .project(
              {"concat4(concat3(concat2(concat1(user_id, gender), age),occupation), user_mean_rating) as user_tower_features",
               "concat2_2(concat2_1(movie_id, genres), movie_mean_rating) as movie_tower_features"})
          .project(
              {"relu(batch_norm3(mat_vector_add3(mat_mul3(relu(batch_norm2(mat_vector_add2(mat_mul2(relu(batch_norm1(mat_vector_add1(mat_mul1(user_tower_features)))))))))))) as user_nn_out",
               "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
          .project({"cosine_similarity(user_nn_out, movie_nn_out)"})
          .planNode();
  // auto userTowe

  auto finalScore = exec::test::AssertQueryBuilder(finalInferencePlan)
                        .copyResults(pool_.get());
  //   std::cout << "[INFO] temp result: \n"
  //               << finalScore->toString(0, finalScore->size()) <<
  //               std::endl;
  //   std::cout << "flag 1" << std::endl;
  //   auto finalScore1 =
  //       exec::test::AssertQueryBuilder(movieTowerInferencePlan)
  //           .copyResults(pool_.get());
  //   std::cout << "flag 2" << std::endl;
  //   std::cout << "[INFO] temp result: \n"
  //                 << finalScore1->toString(0, finalScore1->size()) <<
  //                 std::endl;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  //   //   std::cout << "[INFO] temp result: \n"
  //   //             << finalScore->toString(0, finalScore->size()) <<
  std::cout << "Preprocess Time (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    preprocessEnd - begin)
                    .count()) /
          1e6
            << std::endl;
  std::cout << "Inference Time (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - preprocessEnd)
                    .count()) /
          1e6
            << std::endl;

  std::cout << "End-End Time (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1e6
            << std::endl;
}

int main(int argc, char** argv) {
  Eigen::setNbThreads(16);
  folly::init(&argc, &argv, false);
  TowTowerModelPipelineTest demo;

  int numSamples = 5000;
  int numSplit = 2;
  if (argc >= 2) {
    numSamples = std::stoi(argv[1]);
    numSplit = std::stoi(argv[2]);
  }
  std::cout << "[INFO] # Samples: " << numSamples << " # Split: " << numSplit
            << std::endl;
  demo.testEndtoEndPipelineMultiThreading(numSamples, numSplit);
}