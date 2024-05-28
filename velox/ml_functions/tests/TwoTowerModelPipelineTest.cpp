#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/program_options.hpp>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <torch/torch.h>
#include <random>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/ComplexLayer.h"
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
namespace po = boost::program_options;

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
    parquet::registerParquetWriterFactory();
    filesystems::registerLocalFileSystem();
    dwio::common::LocalFileSink::registerFactory();

    ioExecutor_ = std::make_unique<folly::IOThreadPoolExecutor>(2);

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>(), ioExecutor_.get());
    connector::registerConnector(hiveConnector);

    rootPool_ = memory::MemoryManager::getInstance()->addRootPool("TwoTowerTest");
    pool_ = rootPool_->addLeafChild("TwoTowerTest");

    // SetUp();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  // Function from ParquetTestBase.h
  std::unique_ptr<dwio::common::FileSink> createSink(
      const std::string& filePath) {
    auto sink = dwio::common::FileSink::create(
        fmt::format("file:{}", filePath), {.pool = rootPool_.get()});
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

  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;

  ~TowTowerModelPipelineTest() {
    HiveConnectorTestBase::TearDown();
  }

  void run();
  int64_t testEndtoEndPipelineMultiThreading(
      int numSamples,
      int numSplit,
      int batchSize,
      int numDriver,
      std::string dataPath);

  int64_t testEndtoEndPipelineFusedMultiThreading(
      int numSamples,
      int numSplit,
      int batchSize,
      int numDriver,
      std::string dataPath);

  int64_t testEndtoEndPipelineMultiThreadingmaterialize(
      int numSamples,
      int numSplit,
      std::string dataPath);

  void TestBody() override {}

  void SetUp() {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() {
    HiveConnectorTestBase::TearDown();
    connector::unregisterConnector(kHiveConnectorId);
    parquet::unregisterParquetReaderFactory();
  }
};

int64_t TowTowerModelPipelineTest::testEndtoEndPipelineMultiThreading(
    int numSamples,
    int numSplit,
    int batchSize,
    int numDriver,
    std::string dataPath) {
  VectorMaker maker{pool_.get()};
  std::cout
      << "[INFO]: TowTowerModelPipelineTest::testEndtoEndPipelineMultiThreading"
      << std::endl;
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

  auto userDataRowType = ROW(
      {
          "user_id",
          "gender",
          "age",
          "occupation",
          "zipcode",
      },
      {INTEGER(), VARCHAR(), INTEGER(), INTEGER(), VARCHAR()});

  auto movieDataRowType =
      ROW({"movie_id", "title", "genres"}, {INTEGER(), VARCHAR(), VARCHAR()});

  auto ratingDataRowType =
      ROW({"user_id", "movie_id", "rating", "timestamp"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()});

  auto queryDataRowType =
      ROW({"q_user_id", "q_movie_id"}, {INTEGER(), INTEGER()});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  constexpr int64_t KB = 1024L;
  constexpr int64_t MB = 1024L * KB;
  constexpr int64_t GB = 1024L * MB;
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kPreferredOutputBatchRows, std::to_string(batchSize)},
      //   {core::QueryConfig::kMaxOutputBatchRows, "600"}
  });
  uint64_t kSizeKB = 1024UL;

  auto userHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_user_s_8192.parquet", dataPath)},
      1,
      dwio::common::FileFormat::PARQUET);
  auto movieHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_movie_s_8192.parquet", dataPath)},
      1,
      dwio::common::FileFormat::PARQUET);

  auto ratingUserHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_rating_s_8192.parquet", dataPath)},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  auto ratingMovieHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_rating_s_8192.parquet", dataPath)},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  // Deprecated approach: use Python code to generate splitted query parquet
  // writer, ref: SinkTests.cpp
  //   std::string cmdToGenData = fmt::format(
  //       "python3 /root/velox_latest/data/gen_data.py -n {}", numSamples);
  //   int returnCode = system(cmdToGenData.c_str());

  std::vector<int> userIds = randomGenerator.gen1DInt(numSamples, 1, 6040);
  auto userIdFlatVector = maker.flatVector<int>(userIds, INTEGER());
  std::vector<int> movieIds = randomGenerator.gen1DInt(numSamples, 1, 3706);
  auto movieIdFlatVector = maker.flatVector<int>(movieIds, INTEGER());
  auto queryDataRowVector = maker.rowVector(
      {"q_user_id", "q_movie_id"}, {userIdFlatVector, movieIdFlatVector});

  auto tempPath = exec::test::TempDirectoryPath::create();
  auto filePath =
      fs::path(fmt::format("{}/query_data_test.parquet", tempPath->path));
  auto sink = createSink(filePath);
  auto sinkPtr = sink.get();
  uint64_t kRowsInRowGroup = 1000;
  uint64_t kBytesInRowGroup = 128 * 1024 * 1024;
  auto writer = createWriter(std::move(sink), [&]() {
    return std::make_unique<facebook::velox::parquet::LambdaFlushPolicy>(
        kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
  }, queryDataRowType);
  writer->write(queryDataRowVector);
  writer->flush();
  writer->close();

  auto queryDataHiveSplits = makeHiveConnectorSplits(
      //   {"/root/velox_latest/data/query_data.parquet"},
      {filePath},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  core::PlanNodeId readQueryDataPlanNodeId;
  core::PlanNodeId readUserDataPlanNodeId;
  core::PlanNodeId readRatingDataPlanNodeId1;
  core::PlanNodeId readRatingDataPlanNodeId2;
  core::PlanNodeId readMovieDataPlanNodeId;

  // plan node to join user table and rating table then run aggregation
  auto readUserAvgRatingDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(userDataRowType, {}, "")
          .capturePlanNodeId(readUserDataPlanNodeId)
          .hashJoin(
              {"user_id"},
              {"r_user_id"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(ratingDataRowType, {}, "")
                  .capturePlanNodeId(readRatingDataPlanNodeId1)
                  .project(
                      {"user_id as r_user_id",
                       "change_rating(rating) as rating"})
                  .partialAggregation(
                      {"r_user_id"}, {"avg(rating) as user_mean_rating"})
                  .localPartition({})
                  .finalAggregation()
                  .planNode(),
              "",
              {"user_id", "gender", "age", "occupation", "user_mean_rating"})
          .planNode();

  // plan node to join movie table and rating table then run aggregation
  auto readMovieAvgRatingDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(movieDataRowType, {}, "")
          .capturePlanNodeId(readMovieDataPlanNodeId)
          .hashJoin(
              {"movie_id"},
              {"r_movie_id"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(ratingDataRowType, {}, "")
                  .capturePlanNodeId(readRatingDataPlanNodeId2)
                  .project(
                      {"movie_id as r_movie_id",
                       "change_rating(rating) as rating"})
                  .partialAggregation(
                      {"r_movie_id"}, {"avg(rating) as movie_mean_rating"})
                  .localPartition({})
                  .finalAggregation()
                  .planNode(),
              "",
              {"movie_id", "genres", "movie_mean_rating"})
          .planNode();

  auto joinedUserAndMovieDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(queryDataRowType, {}, "")
          //   .tableScan(asRowType(queryDataRowVector->type()))
          .capturePlanNodeId(readQueryDataPlanNodeId)
          //   .values(queryDataRowVector)
          // if use parallelizable, there is a one thing needs to be resolved is
          // how to merge results together
          //   .values(queryDataRowVector, true /*parallelizable*/)
          //   .localPartition(std::vector<std::string>{})
          .hashJoin( // join with user-rating  table
              {"q_user_id"},
              {"user_id"},
              readUserAvgRatingDataPlan,
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "q_movie_id"})
          .hashJoin( // join with movie-rating table
              {"q_movie_id"},
              {"movie_id"},
              readMovieAvgRatingDataPlan,
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "movie_id",
               "genres",
               "movie_mean_rating"})
          .project( // pre processing, apply encoder
              {"user_id_encoder(convert_int_array(user_id)) as user_id",
               "gender_encoder(gender) as gender",
               "age_encoder(convert_int_array(age)) as age",
               "occupation_encoder(convert_int_array(occupation)) as occupation",
               "convert_double_to_float_array(user_mean_rating) as user_mean_rating",
               "movie_id_encoder(convert_int_array(movie_id)) as movie_id",
               "genres_encoder(split(genres, '|')) as genres",
               "convert_double_to_float_array(movie_mean_rating) as movie_mean_rating"})
          .project( // look-up embedding
              {"user_id_embedding(user_id) as user_id",
               "gender_embedding(gender) as gender",
               "age_embedding(age) as age",
               "occupation_embedding(occupation) as occupation",
               "user_mean_rating",
               "movie_id_embedding(movie_id) as movie_id",
               "sequence_pooling(genres_embedding(genres)) as genres",
               "movie_mean_rating"})
          .project( // concate embedding vectors
              {"concat4(concat3(concat2(concat1(user_id,gender),age),occupation), user_mean_rating) as user_tower_features",
               "concat2_2(concat2_1(movie_id, genres), movie_mean_rating) as movie_tower_features"})
          .project( // user/movie tower inference
              {"relu(batch_norm3(mat_vector_add3(mat_mul3(relu(batch_norm2(mat_vector_add2(mat_mul2(relu(batch_norm1(mat_vector_add1(mat_mul1(user_tower_features)))))))))))) as user_nn_out",
               "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
          .project({"cosine_similarity(user_nn_out, movie_nn_out)"});

  auto end2endPlanNodeFragment = joinedUserAndMovieDataPlan.planFragment();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  // TODO: Here we change how the plan is executed by using Velox's built-in
  // Cursor to execute it. We observed using task->start() will sometimes lead
  // to segmentation fault error for some reason. The previous implementation is
  // attached as a comment at the end of this function for future debugging
  // purposes.

  CursorParameters params;
  params.maxDrivers = numDriver;
  params.planNode = joinedUserAndMovieDataPlan.planNode();
  bool noMoreSplits = false;
  auto addSplits = [&](exec::Task* task) {
    if (!noMoreSplits) {
      for (auto& split : queryDataHiveSplits) {
        task->addSplit(readQueryDataPlanNodeId, exec::Split(std::move(split)));
      }
      task->noMoreSplits(readQueryDataPlanNodeId);

      for (auto& split : userHiveSplits) {
        task->addSplit(readUserDataPlanNodeId, exec::Split(std::move(split)));
      }
      task->noMoreSplits(readUserDataPlanNodeId);

      for (auto& split : movieHiveSplits) {
        task->addSplit(readMovieDataPlanNodeId, exec::Split(std::move(split)));
      }
      task->noMoreSplits(readMovieDataPlanNodeId);

      for (auto& split : ratingUserHiveSplits) {
        task->addSplit(
            readRatingDataPlanNodeId1, exec::Split(std::move(split)));
      }
      task->noMoreSplits(readRatingDataPlanNodeId1);

      for (auto& split : ratingMovieHiveSplits) {
        task->addSplit(
            readRatingDataPlanNodeId2, exec::Split(std::move(split)));
      }
      task->noMoreSplits(readRatingDataPlanNodeId2);
    }
    noMoreSplits = true;
  };

  auto [cursor, actualResults] = readCursor(params, addSplits);
  waitForTaskCompletion(cursor->task().get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  int totalNumOfRecord = 0;
  for (auto batchData : actualResults) {
    totalNumOfRecord += batchData->size();
  }

  std::cout << fmt::format(
                   "[DEBUG] # Batches: {}, # TotalRecords: {}",
                   actualResults.size(),
                   totalNumOfRecord)
            << std::endl;

  int64_t time =
      (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
           .count());

  return time;

  /* Previous implmentation for debugging purpose.
    std::vector<RowVectorPtr> resultUserData;
    boost::interprocess::interprocess_semaphore semaphore(2);
    std::shared_ptr<TaskQueue> taskQueue =
        std::make_shared<TaskQueue>(512 * 1024);
    auto taskUser = exec::Task::create(
        "0",
        std::move(end2endPlanNodeFragment),
        0,
        std::move(queryCtx_),
        [&resultUserData, &semaphore, &taskQueue, this](
            RowVectorPtr vector, ContinueFuture* future) {
          if (vector) {
            //   semaphore.post();
            //   for (auto& child : vector->children()) {
            //   child->loadedVector();
            // }
            for (auto& child : vector->children()) {
              child->loadedVector();
            }
            auto copy = BaseVector::create<RowVector>(
                vector->type(), vector->size(), pool_.get());
            copy->copy(vector.get(), 0, 0, vector->size());
            resultUserData.push_back(std::move(copy));
            //   taskQueue->enqueue(std::move(copy), future);
          }
          return exec::BlockingReason::kNotBlocked;
        });
    taskUser->start(numDriver);

    for (auto& split : queryDataHiveSplits) {
      // semaphore.wait();
      taskUser->addSplit(readQueryDataPlanNodeId,
      exec::Split(std::move(split)));
    }
    taskUser->noMoreSplits(readQueryDataPlanNodeId);

    for (auto& split : userHiveSplits) {
      // semaphore.wait();
      taskUser->addSplit(readUserDataPlanNodeId,
      exec::Split(std::move(split)));
    }
    taskUser->noMoreSplits(readUserDataPlanNodeId);

    for (auto& split : movieHiveSplits) {
      // semaphore.wait();
      taskUser->addSplit(readMovieDataPlanNodeId,
      exec::Split(std::move(split)));
    }
    taskUser->noMoreSplits(readMovieDataPlanNodeId);

    for (auto& split : ratingUserHiveSplits) {
      // semaphore.wait();
      taskUser->addSplit(
          readRatingDataPlanNodeId1, exec::Split(std::move(split)));
    }
    taskUser->noMoreSplits(readRatingDataPlanNodeId1);

    for (auto& split : ratingMovieHiveSplits) {
      // semaphore.wait();
      taskUser->addSplit(
          readRatingDataPlanNodeId2, exec::Split(std::move(split)));
    }
    taskUser->noMoreSplits(readRatingDataPlanNodeId2);

    waitForFinishedDrivers(taskUser);
    std::vector<RowVectorPtr> result;
    taskQueue->setNumProducers(1 * taskUser->numOutputDrivers());
    auto current_ = taskQueue->dequeue();
    int totalSize = 0;
    while (current_ != nullptr) {
      totalSize += current_->size();
      result.push_back(current_);
      current_ = taskQueue->dequeue();
    }

    waitForTaskCompletion(taskUser.get());

    auto movedData = std::move(resultUserData);
    int totalNumOfRecord = 0;
    for (auto batchData : movedData) {
      totalNumOfRecord += batchData->size();
    }

    std::cout << fmt::format(
                     "[DEBUG] # Batches: {}, # TotalRecords: {}",
                     movedData.size(),
                     totalNumOfRecord)
              << std::endl;
    */
}

int64_t TowTowerModelPipelineTest::testEndtoEndPipelineFusedMultiThreading(
    int numSamples,
    int numSplit,
    int batchSize,
    int numDriver,
    std::string dataPath) {
  VectorMaker maker{pool_.get()};
  std::cout
      << "[INFO]: TowTowerModelPipelineTest::testEndtoEndPipelineFusedMultiThreading"
      << std::endl;
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
      "relu", Relu::signatures(), std::make_unique<Relu>());

  std::vector<std::vector<float>> batchNorm1Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm1WeightVector =
      maker.arrayVector<float>(batchNorm1Weight, REAL());
  std::vector<std::vector<float>> batchNorm1Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm1BiasVector = maker.arrayVector<float>(batchNorm1Bias, REAL());

  std::vector<std::vector<float>> batchNorm2Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2WeightVector =
      maker.arrayVector<float>(batchNorm2Weight, REAL());
  std::vector<std::vector<float>> batchNorm2Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2BiasVector = maker.arrayVector<float>(batchNorm2Bias, REAL());

  std::vector<std::vector<float>> batchNorm3Weight =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm3WeightVector =
      maker.arrayVector<float>(batchNorm3Weight, REAL());
  std::vector<std::vector<float>> batchNorm3Bias =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm3BiasVector = maker.arrayVector<float>(batchNorm3Bias, REAL());

  exec::registerVectorFunction(
      "fully_layer_with_batch_norm1",
      FullyConnectWithBatchNormAndRelu::signatures(),
      std::make_unique<FullyConnectWithBatchNormAndRelu>(
          userNNweight1Vector->elements()->values()->asMutable<float>(),
          userNNBias1Vector->elements()->values()->asMutable<float>(),
          batchNorm1WeightVector->elements()->values()->asMutable<float>(),
          batchNorm1BiasVector->elements()->values()->asMutable<float>(),
          float(1e-5),
          129,
          300));

  exec::registerVectorFunction(
      "fully_layer_with_batch_norm2",
      FullyConnectWithBatchNormAndRelu::signatures(),
      std::make_unique<FullyConnectWithBatchNormAndRelu>(
          userNNweight2Vector->elements()->values()->asMutable<float>(),
          userNNBias2Vector->elements()->values()->asMutable<float>(),
          batchNorm2WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2BiasVector->elements()->values()->asMutable<float>(),
          float(1e-5),
          300,
          300));

  exec::registerVectorFunction(
      "fully_layer_with_batch_norm3",
      FullyConnectWithBatchNormAndRelu::signatures(),
      std::make_unique<FullyConnectWithBatchNormAndRelu>(
          userNNweight3Vector->elements()->values()->asMutable<float>(),
          userNNBias3Vector->elements()->values()->asMutable<float>(),
          batchNorm3WeightVector->elements()->values()->asMutable<float>(),
          batchNorm3BiasVector->elements()->values()->asMutable<float>(),
          float(1e-5),
          300,
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

  std::vector<std::vector<float>> batchNorm2_1Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_1WeightVector =
      maker.arrayVector<float>(batchNorm2_1Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_1Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_1BiasVector =
      maker.arrayVector<float>(batchNorm2_1Bias, REAL());

  std::vector<std::vector<float>> batchNorm2_2Weight =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_2WeightVector =
      maker.arrayVector<float>(batchNorm2_2Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_2Bias =
      randomGenerator.genFloat2dVector(1, 300);
  auto batchNorm2_2BiasVector =
      maker.arrayVector<float>(batchNorm2_2Bias, REAL());

  std::vector<std::vector<float>> batchNorm2_3Weight =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm2_3WeightVector =
      maker.arrayVector<float>(batchNorm2_3Weight, REAL());
  std::vector<std::vector<float>> batchNorm2_3Bias =
      randomGenerator.genFloat2dVector(1, 128);
  auto batchNorm2_3BiasVector =
      maker.arrayVector<float>(batchNorm2_3Bias, REAL());

  exec::registerVectorFunction(
      "fully_layer_with_batch_norm2_1",
      FullyConnectWithBatchNormAndRelu::signatures(),
      std::make_unique<FullyConnectWithBatchNormAndRelu>(
          itemNNweight1Vector->elements()->values()->asMutable<float>(),
          itemNNBias1Vector->elements()->values()->asMutable<float>(),
          batchNorm2_1WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2_1BiasVector->elements()->values()->asMutable<float>(),
          float(1e-5),
          65,
          300));

  exec::registerVectorFunction(
      "fully_layer_with_batch_norm2_2",
      FullyConnectWithBatchNormAndRelu::signatures(),
      std::make_unique<FullyConnectWithBatchNormAndRelu>(
          itemNNweight2Vector->elements()->values()->asMutable<float>(),
          itemNNBias2Vector->elements()->values()->asMutable<float>(),
          batchNorm2_2WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2_2BiasVector->elements()->values()->asMutable<float>(),
          float(1e-5),
          300,
          300));
  exec::registerVectorFunction(
      "fully_layer_with_batch_norm2_3",
      FullyConnectWithBatchNormAndRelu::signatures(),
      std::make_unique<FullyConnectWithBatchNormAndRelu>(
          itemNNweight3Vector->elements()->values()->asMutable<float>(),
          itemNNBias3Vector->elements()->values()->asMutable<float>(),
          batchNorm2_3WeightVector->elements()->values()->asMutable<float>(),
          batchNorm2_3BiasVector->elements()->values()->asMutable<float>(),
          float(1e-5),
          300,
          128));

  exec::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(128));

  auto userDataRowType = ROW(
      {
          "user_id",
          "gender",
          "age",
          "occupation",
          "zipcode",
      },
      {INTEGER(), VARCHAR(), INTEGER(), INTEGER(), VARCHAR()});

  auto movieDataRowType =
      ROW({"movie_id", "title", "genres"}, {INTEGER(), VARCHAR(), VARCHAR()});

  auto ratingDataRowType =
      ROW({"user_id", "movie_id", "rating", "timestamp"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()});

  auto queryDataRowType =
      ROW({"q_user_id", "q_movie_id"}, {INTEGER(), INTEGER()});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;

  constexpr int64_t KB = 1024L;
  constexpr int64_t MB = 1024L * KB;
  constexpr int64_t GB = 1024L * MB;
  queryCtx_->testingOverrideConfigUnsafe({
      {core::QueryConfig::kPreferredOutputBatchRows, std::to_string(batchSize)},
      //   {core::QueryConfig::kMaxOutputBatchRows, "600"}
  });
  uint64_t kSizeKB = 1024UL;
  // std::shared_ptr<memory::MemoryPool> rootPool{
  //     memory::defaultMemoryManager().addRootPool("root", 5000 * MB)};
  // queryCtx_->testingOverrideMemoryPool(rootPool);
  //   int numSplit = 2;

  auto userHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_user_s_8192.parquet", dataPath)},
      1,
      dwio::common::FileFormat::PARQUET);
  auto movieHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_movie_s_8192.parquet", dataPath)},
      1,
      dwio::common::FileFormat::PARQUET);

  auto ratingUserHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_rating_s_8192.parquet", dataPath)},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  auto ratingMovieHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_rating_s_8192.parquet", dataPath)},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  std::vector<int> userIds = randomGenerator.gen1DInt(numSamples, 1, 6040);
  auto userIdFlatVector = maker.flatVector<int>(userIds, INTEGER());
  std::vector<int> movieIds = randomGenerator.gen1DInt(numSamples, 1, 3706);
  auto movieIdFlatVector = maker.flatVector<int>(movieIds, INTEGER());
  auto queryDataRowVector = maker.rowVector(
      {"q_user_id", "q_movie_id"}, {userIdFlatVector, movieIdFlatVector});

  auto tempPath = exec::test::TempDirectoryPath::create();
  auto filePath =
      fs::path(fmt::format("{}/query_data_test.parquet", tempPath->path));
  auto sink = createSink(filePath);
  auto sinkPtr = sink.get();
  uint64_t kRowsInRowGroup = 1000;
  uint64_t kBytesInRowGroup = 128 * 1024 * 1024;
  auto writer = createWriter(std::move(sink), [&]() {
    return std::make_unique<facebook::velox::parquet::LambdaFlushPolicy>(
        kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
  }, queryDataRowType);
  writer->write(queryDataRowVector);
  writer->flush();
  writer->close();

  auto queryDataHiveSplits = makeHiveConnectorSplits(
      //   {"/root/velox_latest/data/query_data.parquet"},
      {filePath},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  core::PlanNodeId readQueryDataPlanNodeId;
  core::PlanNodeId readUserDataPlanNodeId;
  core::PlanNodeId readRatingDataPlanNodeId1;
  core::PlanNodeId readRatingDataPlanNodeId2;
  core::PlanNodeId readMovieDataPlanNodeId;

  // plan node to join user table and rating table then run aggregation
  auto readUserAvgRatingDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(userDataRowType, {}, "")
          .capturePlanNodeId(readUserDataPlanNodeId)
          .hashJoin(
              {"user_id"},
              {"r_user_id"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(ratingDataRowType, {}, "")
                  .capturePlanNodeId(readRatingDataPlanNodeId1)
                  .project(
                      {"user_id as r_user_id",
                       "change_rating(rating) as rating"})
                  .partialAggregation(
                      {"r_user_id"}, {"avg(rating) as user_mean_rating"})
                  .localPartition({})
                  .finalAggregation()
                  .planNode(),
              "",
              {"user_id", "gender", "age", "occupation", "user_mean_rating"})
          .planNode();

  // plan node to join movie table and rating table then run aggregation
  auto readMovieAvgRatingDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(movieDataRowType, {}, "")
          .capturePlanNodeId(readMovieDataPlanNodeId)
          .hashJoin(
              {"movie_id"},
              {"r_movie_id"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(ratingDataRowType, {}, "")
                  .capturePlanNodeId(readRatingDataPlanNodeId2)
                  .project(
                      {"movie_id as r_movie_id",
                       "change_rating(rating) as rating"})
                  .partialAggregation(
                      {"r_movie_id"}, {"avg(rating) as movie_mean_rating"})
                  .localPartition({})
                  .finalAggregation()
                  .planNode(),
              "",
              {"movie_id", "genres", "movie_mean_rating"})
          .planNode();

  auto joinedUserAndMovieDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(queryDataRowType, {}, "")
          //   .tableScan(asRowType(queryDataRowVector->type()))
          .capturePlanNodeId(readQueryDataPlanNodeId)
          //   .values(queryDataRowVector)
          // if use parallelizable, there is a one thing needs to be resolved is
          // how to merge results together
          //   .values(queryDataRowVector, true /*parallelizable*/)
          //   .localPartition(std::vector<std::string>{})
          .hashJoin( // join with user-rating  table
              {"q_user_id"},
              {"user_id"},
              readUserAvgRatingDataPlan,
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "q_movie_id"})
          .hashJoin( // join with movie-rating table
              {"q_movie_id"},
              {"movie_id"},
              readMovieAvgRatingDataPlan,
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "movie_id",
               "genres",
               "movie_mean_rating"})
          .project( // pre processing, apply encoder
              {"user_id_encoder(convert_int_array(user_id)) as user_id",
               "gender_encoder(gender) as gender",
               "age_encoder(convert_int_array(age)) as age",
               "occupation_encoder(convert_int_array(occupation)) as occupation",
               "convert_double_to_float_array(user_mean_rating) as user_mean_rating",
               "movie_id_encoder(convert_int_array(movie_id)) as movie_id",
               "genres_encoder(split(genres, '|')) as genres",
               "convert_double_to_float_array(movie_mean_rating) as movie_mean_rating"})
          .project( // look-up embedding
              {"user_id_embedding(user_id) as user_id",
               "gender_embedding(gender) as gender",
               "age_embedding(age) as age",
               "occupation_embedding(occupation) as occupation",
               "user_mean_rating",
               "movie_id_embedding(movie_id) as movie_id",
               "sequence_pooling(genres_embedding(genres)) as genres",
               "movie_mean_rating"})
          .project( // concate embedding vectors
              {"concat4(concat3(concat2(concat1(user_id, gender),age),occupation), user_mean_rating) as user_tower_features",
               "concat2_2(concat2_1(movie_id, genres), movie_mean_rating) as movie_tower_features"})
          .project( // user/movie tower inference
              {"fully_layer_with_batch_norm3(fully_layer_with_batch_norm2(fully_layer_with_batch_norm1(user_tower_features))) as user_nn_out",
               "fully_layer_with_batch_norm2_3(fully_layer_with_batch_norm2_2(fully_layer_with_batch_norm2_1(movie_tower_features))) as movie_nn_out"})
          .project({"cosine_similarity(user_nn_out, movie_nn_out)"})
          .planFragment();

  std::vector<RowVectorPtr> resultUserData;
  boost::interprocess::interprocess_semaphore semaphore(2);
  auto taskUser = exec::Task::create(
      "0",
      joinedUserAndMovieDataPlan,
      0,
      queryCtx_,
      [&resultUserData, &semaphore](
          RowVectorPtr vector, ContinueFuture* future) {
        if (vector) {
          semaphore.post();
          //   for (auto& child : vector->children()) {
          //   child->loadedVector();
          // }
          resultUserData.push_back(vector);
        }
        return exec::BlockingReason::kNotBlocked;
      });

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  taskUser->start(numDriver);
  //   taskReadRating->start(taskReadRating, numSplit);

  for (auto& split : queryDataHiveSplits) {
    // semaphore.wait();
    taskUser->addSplit(readQueryDataPlanNodeId, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readQueryDataPlanNodeId);

  for (auto& split : userHiveSplits) {
    // semaphore.wait();
    taskUser->addSplit(readUserDataPlanNodeId, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readUserDataPlanNodeId);

  for (auto& split : movieHiveSplits) {
    // semaphore.wait();
    taskUser->addSplit(readMovieDataPlanNodeId, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readMovieDataPlanNodeId);

  for (auto& split : ratingUserHiveSplits) {
    // semaphore.wait();
    taskUser->addSplit(
        readRatingDataPlanNodeId1, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readRatingDataPlanNodeId1);

  for (auto& split : ratingMovieHiveSplits) {
    // semaphore.wait();
    taskUser->addSplit(
        readRatingDataPlanNodeId2, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readRatingDataPlanNodeId2);

  waitForFinishedDrivers(taskUser);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  auto movedData = std::move(resultUserData);
  int totalNumOfRecord = 0;
  for (auto batchData : movedData) {
    totalNumOfRecord += batchData->size();
  }

  //   for (int i = 0; i < movedData.size(); i++) {
  //     RowVectorPtr printData = movedData[i];
  //     std::cout << "[DEBUG], batch : " << i << "\n"
  //                 << printData->toString(0, printData->size())
  //                 << std::endl;
  //   }

  //    std::move(resultUserData);
  std::cout << fmt::format(
                   "[DEBUG] # Batches: {}, # TotalRecords: {}",
                   movedData.size(),
                   totalNumOfRecord)
            << std::endl;

  int64_t time =
      (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
           .count());

  //   std::cout << "End-End Time (sec) = "
  //             << (std::chrono::duration_cast<std::chrono::microseconds>(
  //                     end - begin)
  //                     .count()) /
  //           1e6
  //             << std::endl;
  return time;
}

int64_t
TowTowerModelPipelineTest::testEndtoEndPipelineMultiThreadingmaterialize(
    int numSamples,
    int numSplit,
    std::string dataPath) {
  VectorMaker maker{pool_.get()};
  std::cout
      << "[INFO]: TowTowerModelPipelineTest::testEndtoEndPipelineMultiThreadingmaterialize"
      << std::endl;

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

  auto userRatingDataRowType =
      ROW({"user_id", "gender", "age", "occupation", "user_mean_rating"},
          {INTEGER(), VARCHAR(), INTEGER(), INTEGER(), DOUBLE()});

  auto movieRatingDataRowType =
      ROW({"movie_id", "genres", "movie_mean_rating"},
          {INTEGER(), VARCHAR(), DOUBLE()});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  CursorParameters params;

  constexpr int64_t KB = 1024L;
  constexpr int64_t MB = 1024L * KB;
  constexpr int64_t GB = 1024L * MB;
  //   std::shared_ptr<memory::MemoryPool> rootPool{
  //       memory::defaultMemoryManager().addRootPool("root", 5000 * MB)};
  //   queryCtx_->testingOverrideMemoryPool(rootPool);
  uint64_t kSizeKB = 1024UL;

  //   int numSplit = 2;
  auto userRatingHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_user_rating.parquet", dataPath)},
      numSplit,
      dwio::common::FileFormat::PARQUET);
  auto movieRatingHiveSplits = makeHiveConnectorSplits(
      {fmt::format("file:{}/movielens_movie_rating.parquet", dataPath)},
      numSplit,
      dwio::common::FileFormat::PARQUET);

  std::vector<int> userIds = randomGenerator.gen1DInt(numSamples, 1, 6040);
  auto userIdFlatVector = maker.flatVector<int>(userIds, INTEGER());
  std::vector<int> movieIds = randomGenerator.gen1DInt(numSamples, 1, 3706);
  auto movieIdFlatVector = maker.flatVector<int>(movieIds, INTEGER());
  auto queryDataRowVector = maker.rowVector(
      {"q_user_id", "q_movie_id"}, {userIdFlatVector, movieIdFlatVector});

  core::PlanNodeId readUserRatingDataPlanNodeId;
  core::PlanNodeId readMovieRatingDataPlanNodeId;

  auto readUserRatingDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(userRatingDataRowType, {}, "")
          .capturePlanNodeId(readUserRatingDataPlanNodeId)
          .planNode();

  auto readMovieRatingDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .tableScan(movieRatingDataRowType, {}, "")
          .capturePlanNodeId(readMovieRatingDataPlanNodeId)
          .planNode();

  auto joinedUserAndMovieDataPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .values({queryDataRowVector})
          .hashJoin( // join with user-rating table
              {"q_user_id"},
              {"user_id"},
              readUserRatingDataPlan,
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "q_movie_id"})
          .hashJoin( // join with movie-rating table
              {"q_movie_id"},
              {"movie_id"},
              readMovieRatingDataPlan,
              "",
              {"user_id",
               "gender",
               "age",
               "occupation",
               "user_mean_rating",
               "movie_id",
               "genres",
               "movie_mean_rating"})
          .project( // pre-processing, run encoder
              {"user_id_encoder(convert_int_array(user_id)) as user_id",
               "gender_encoder(gender) as gender",
               "age_encoder(convert_int_array(age)) as age",
               "occupation_encoder(convert_int_array(occupation)) as occupation",
               "convert_double_to_float_array(user_mean_rating) as user_mean_rating",
               "movie_id_encoder(convert_int_array(movie_id)) as movie_id",
               "genres_encoder(split(genres, '|')) as genres",
               "convert_double_to_float_array(movie_mean_rating) as movie_mean_rating"})
          .project( // look-up embedding for user/movie features
              {"user_id_embedding(user_id) as user_id",
               "gender_embedding(gender) as gender",
               "age_embedding(age) as age",
               "occupation_embedding(occupation) as occupation",
               "user_mean_rating",
               "movie_id_embedding(movie_id) as movie_id",
               "sequence_pooling(genres_embedding(genres)) as genres",
               "movie_mean_rating"})
          .project( // concat embedding vectors
              {"concat4(concat3(concat2(concat1(user_id, gender),age),occupation), user_mean_rating) as user_tower_features",
               "concat2_2(concat2_1(movie_id, genres), movie_mean_rating) as movie_tower_features"})
          .project( // inference for user/movie twoer
              {"relu(batch_norm3(mat_vector_add3(mat_mul3(relu(batch_norm2(mat_vector_add2(mat_mul2(relu(batch_norm1(mat_vector_add1(mat_mul1(user_tower_features)))))))))))) as user_nn_out",
               "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
          .project( // compute recommendation score
              {"cosine_similarity(user_nn_out, movie_nn_out)"})
          .planFragment();

  std::vector<RowVectorPtr> resultUserData;
  boost::interprocess::interprocess_semaphore semaphore(2);
  auto taskUser = exec::Task::create(
      "0",
      joinedUserAndMovieDataPlan,
      0,
      queryCtx_,
      [&resultUserData, &semaphore](
          RowVectorPtr vector, ContinueFuture* future) {
        if (vector) {
          resultUserData.push_back(vector);
        }
        return exec::BlockingReason::kNotBlocked;
      });

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  taskUser->start(numSplit);

  for (auto& split : userRatingHiveSplits) {
    taskUser->addSplit(
        readUserRatingDataPlanNodeId, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readUserRatingDataPlanNodeId);

  for (auto& split : movieRatingHiveSplits) {
    taskUser->addSplit(
        readMovieRatingDataPlanNodeId, exec::Split(std::move(split)));
  }
  taskUser->noMoreSplits(readMovieRatingDataPlanNodeId);

  waitForFinishedDrivers(taskUser);

  std::chrono::steady_clock::time_point preprocessEnd =
      std::chrono::steady_clock::now();

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  auto movedData = std::move(resultUserData);
  RowVectorPtr printData = movedData[0];
  std::cout << "[DEBUG] # Batches: " << movedData.size() << std::endl;
  //   std::cout << "[DEBUG]: \n"
  //             << printData->toString(0, printData->size()) << std::endl;

  int64_t time =
      (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
           .count());

  std::cout << "End-End Time (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1e6
            << std::endl;
  return time;
}

DEFINE_string(data_path, "../../../../data", "Path to data dir");
DEFINE_int32(num_sample, 50000, "Number of samples");
DEFINE_int32(num_split, 10, "Number of splits");
DEFINE_int32(
    benchmark_mode,
    0,
    "Benchmark Mode, 0-non-materialize, 1-materialize, 2-both");
DEFINE_int32(batch_size, 5000, "Batch size");
DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(num_driver, 8, "Number of driver");

int main(int argc, char** argv) {
  Eigen::setNbThreads(16);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  int numSamples = FLAGS_num_sample;
  int numSplit = FLAGS_num_split; // default 2
  int benchmarkMode = FLAGS_benchmark_mode; // default 0 0: non-materialize 1:
                                            // materialize 2: both
  int batchSize = FLAGS_batch_size; // default 500
  int numRepeat = FLAGS_num_repeat; // default 1
  int numDriver = FLAGS_num_driver;
  std::string dataPath = FLAGS_data_path;

  std::cout
      << fmt::format(
             "[INFO] # Samples: {}, # Batch Size: {}, # Split: {}, #Driver: {}, numRepeat: {}",
             numSamples,
             batchSize,
             numSplit,
             numDriver,
             numRepeat)
      << std::endl;

  TowTowerModelPipelineTest demo;

  int64_t nonMaterializeLatency = 0;
  int64_t materializeLatency = 0;

  if (benchmarkMode == 0 or benchmarkMode == 2) {
    for (int i = 0; i < numRepeat; i++) {
      nonMaterializeLatency += demo.testEndtoEndPipelineMultiThreading(
          numSamples, numSplit, batchSize, numDriver, dataPath);
    }
    nonMaterializeLatency = nonMaterializeLatency / numRepeat;
  }

  if (benchmarkMode == 1 or benchmarkMode == 2) {
    for (int i = 0; i < numRepeat; i++) {
      materializeLatency += demo.testEndtoEndPipelineMultiThreadingmaterialize(
          numSamples, numSplit, dataPath);
    }
    materializeLatency = materializeLatency / numRepeat;
  }

  std::cout << "==========================================" << std::endl;
  std::cout << "Benchmark Result EndtoEndPipeline (sec) = "
            << nonMaterializeLatency / 1e6 << std::endl;
  std::cout << "Benchmark Result EndtoEndPipeline Materialize (sec) = "
            << materializeLatency / 1e6 << std::endl;
}