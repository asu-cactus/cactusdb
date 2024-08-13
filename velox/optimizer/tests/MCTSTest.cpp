#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <unistd.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>

// Velox headers
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/ComplexLayer.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/Dropout.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/Encoder.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/ml_functions/SequencePooling.h"
#include "velox/ml_functions/ChatGPT.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

// Custom headers
#include <json/json.h>
#include "velox/cost_model/CostEstimator.h"
#include "velox/optimizer/Helper.h"
#include "velox/optimizer/Mul2JoinAggRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/TwoLayerUDF2TorchNNRewriteAction.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

#define BUFFER_SIZE 1024

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

class IntegratedMCTSTest : public HiveConnectorTestBase {
 public:
  IntegratedMCTSTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    Type::registerSerDe();
    common::Filter::registerSerDe();
    connector::hive::HiveTableHandle::registerSerDe();
    connector::hive::LocationHandle::registerSerDe();
    connector::hive::HiveColumnHandle::registerSerDe();
    connector::hive::HiveInsertTableHandle::registerSerDe();
    registerPartitionFunctionSerDe();
    core::PlanNode::registerSerDe();
    core::ITypedExpr::registerSerDe();
    parquet::registerParquetReaderFactory();
    parquet::registerParquetWriterFactory();
    filesystems::registerLocalFileSystem();
    // Register hiveconnector for file splits.
    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    tempDirPath_ = exec::test::TempDirectoryPath::create();
  }

  ~IntegratedMCTSTest() {
    TearDown();
  }

  void SetUp() override {}

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}
  // Wait for all drivers to finish work.
  void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  int cacheQueryPlan(PlanBuilder& planBuilder) {
    int queryPlanCacheId = queryPlanCacheId_++;
    auto serializedPlan = planBuilder.planNode()->serialize();
    // queryPlanCaches_[queryPlanCacheId] = serializedPlan;

    return queryPlanCacheId;
  }

  void resetQueryPlanFromCache(
      PlanBuilder& planBuilder,
      int queryPlanCacheId) {
    auto it = queryPlanCaches_.find(queryPlanCacheId);
    if (it != queryPlanCaches_.end()) {
      auto serializedPlan = it->second;
      auto deserlizedUpdatedPlanNode =
                    ISerializable::deserialize<core::PlanNode>(
                        serializedPlan, pool_.get());
      planBuilder.setRoot(deserlizedUpdatedPlanNode);
    } else {
      throw std::runtime_error(
          fmt::format("[ERROR]queryPlanCacheId: {} was not found.", queryPlanCacheId));
    }
  }

  // Function from ParquetTestBase.h
  std::unique_ptr<dwio::common::FileSink> createSink(
      const std::string& filePath) {
    auto sink = dwio::common::FileSink::create(
        fmt::format("file:{}", filePath), {.pool = pool_.get()});
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

  std::vector<std::string> generateTwoTowerQueryData(
      int numSample,
      int maxUserId,
      int maxMovieId,
      int numSplit = 8) {
    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    auto queryDataRowType =
        ROW({"q_user_id", "q_movie_id"}, {INTEGER(), INTEGER()});
    std::vector<std::string> inputPaths;
    size_t partitionSize = ceil(numSample / float(numSplit));

    for (size_t i = 0; i < numSplit; i++) {
      size_t numSamplesInPartition = (i + 1) * partitionSize <= numSample
          ? partitionSize
          : numSample - i * partitionSize;
      if (numSamplesInPartition == 0) {
        continue;
      }
      std::vector<int> userIds =
          randomGenerator.gen1DInt(numSamplesInPartition, 1, maxUserId);
      auto userIdFlatVector = maker.flatVector<int>(userIds, INTEGER());
      std::vector<int> movieIds =
          randomGenerator.gen1DInt(numSamplesInPartition, 1, maxMovieId);
      auto movieIdFlatVector = maker.flatVector<int>(movieIds, INTEGER());
      auto queryDataRowVector = maker.rowVector(
          {"q_user_id", "q_movie_id"}, {userIdFlatVector, movieIdFlatVector});
      auto filePath = fs::path(
          fmt::format("{}/query_data_part_{}.parquet", tempDirPath_->path, i));

      auto sink = createSink(filePath);
      auto sinkPtr = sink.get();
      uint64_t kRowsInRowGroup = 100000;
      uint64_t kBytesInRowGroup = 1280 * 1024 * 1024;
      auto writer = createWriter(
          std::move(sink),
          [&]() {
            return std::make_unique<
                facebook::velox::parquet::LambdaFlushPolicy>(
                kRowsInRowGroup, kBytesInRowGroup, [&]() { return false; });
          },
          queryDataRowType);
      writer->write(queryDataRowVector);
      writer->flush();
      writer->close();

      inputPaths.push_back(std::move(filePath));
    }
    return inputPaths;
  }

  void registerTwoTowerFunc(
      CataLog& catalog,
      std::shared_ptr<memory::MemoryPool> pool_,
      bool isVerticalPartition) {
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
    optimization::registerVectorFunction(
        "concat1",
        Concat::signatures(),
        std::make_unique<Concat>(embeddingDims, embeddingDims),
        {},
        true,
        catalog,
        isVerticalPartition);

    optimization::registerVectorFunction(
        "concat2",
        Concat::signatures(),
        std::make_unique<Concat>(2 * embeddingDims, embeddingDims),
        {},
        true,
        catalog,
        isVerticalPartition);

    optimization::registerVectorFunction(
        "concat3",
        Concat::signatures(),
        std::make_unique<Concat>(3 * embeddingDims, embeddingDims),
        {},
        true,
        catalog,
        isVerticalPartition);
    optimization::registerVectorFunction(
        "concat4",
        Concat::signatures(),
        std::make_unique<Concat>(4 * embeddingDims, 1),
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
    auto batchNorm1BiasVector =
        maker.arrayVector<float>(batchNorm1Bias, REAL());

    optimization::registerVectorFunction(
        "batch_norm1_1",
        BatchNorm1D::signatures(),
        std::make_unique<BatchNorm1D>(
            std::move(batchNorm1WeightVector->elements()
                          ->values()
                          ->asMutable<float>()),
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
    auto batchNorm2BiasVector =
        maker.arrayVector<float>(batchNorm2Bias, REAL());

    optimization::registerVectorFunction(
        "batch_norm1_2",
        BatchNorm1D::signatures(),
        std::make_unique<BatchNorm1D>(
            std::move(batchNorm2WeightVector->elements()
                          ->values()
                          ->asMutable<float>()),
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
    auto batchNorm3BiasVector =
        maker.arrayVector<float>(batchNorm3Bias, REAL());

    optimization::registerVectorFunction(
        "batch_norm1_3",
        BatchNorm1D::signatures(),
        std::make_unique<BatchNorm1D>(
            std::move(batchNorm3WeightVector->elements()
                          ->values()
                          ->asMutable<float>()),
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

    std::vector<int> userTowerdims = {
        129, 300, 300, 300, 300, 300, 300, 300, 300, 128, 128, 128, 128};

    exec::registerVectorFunction(
        "user_torchNN",
        TorchDNNV2::signatures(),
        std::make_unique<TorchDNNV2>(
            userTowerKernelTypes, userTowerWeights, userTowerdims));

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
            genderNumEmbedding,
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

    optimization::registerVectorFunction(
        "concat2_1",
        Concat::signatures(),
        std::make_unique<Concat>(embeddingDims, embeddingDims),
        {},
        true,
        catalog,
        isVerticalPartition);

    optimization::registerVectorFunction(
        "concat2_2",
        Concat::signatures(),
        std::make_unique<Concat>(2 * embeddingDims, 1),
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
            std::move(batchNorm2_1BiasVector->elements()
                          ->values()
                          ->asMutable<float>()),
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
            std::move(batchNorm2_2BiasVector->elements()
                          ->values()
                          ->asMutable<float>()),
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
            std::move(batchNorm2_3BiasVector->elements()
                          ->values()
                          ->asMutable<float>()),
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
    std::vector<int> movieTowerdims = {
        65, 300, 300, 300, 300, 300, 300, 300, 300, 128, 128, 128, 128};

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

  /**
   * @brief A function to run logical plan.
   *
   * @param numThreads The number of Velox executor threads.
   * @param numSplits The number of file splits.
   * @param myPlan The pointer to the planBuilder which builds the logical plan.
   * @param cataLog A class storing metadata and information related to UDFs and
   * data sources.
   */
  float runPlanWithCataLog(
      int numThreads,
      int numSplits,
      PlanBuilder& myPlan,
      CataLog& cataLog,
      int repeatRun = 1,
      int verbose = 1) {
    float totalElapsedTime = 0;
    std::vector<RowVectorPtr> finalResult;
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
            auto hiveSplits = makeHiveConnectorSplits(fileAddr, fileFormat);

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
        finalResult = actualResults;
        dataIdx = 0;
        totalDataNum = 0;
        for (auto batchedData : finalResult) {
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
                      << batchedData->toString(0, batchedData->size())
                      << std::endl;
          }
          dataIdx += 1;
          totalDataNum += batchSize;
        }
        finalResult = std::move(finalResult);
      }
    }
    if (verbose >= 1) {
      std::cout << fmt::format(
                       "[INFO] Total # of Batch: {}, Total # of Data: {}",
                       dataIdx,
                       totalDataNum)
                << std::endl;
    }
    // finalResult = std::move(finalResult);

    return totalElapsedTime / repeatRun;
  }

  struct DataFrame {
    std::vector<std::vector<float>> features;
    std::vector<float*> weights;
    std::vector<float*> bias;
    float* featuresFloat;
    std::vector<std::shared_ptr<TempFilePath>> feature_paths;
  };

  /**
   * @brief A function generates random data source.
   *
   * @param features The number of features (column count) in the data source.
   * @param samples The number of samples (row count) in the data source.
   * @param first_layer The output size of the first layer in the network.
   * @param second_layer The output size of the second layer in the network.
   *
   * @return DataFrame The structure used to denote the generated data.
   */
  DataFrame data_generate(
      int features,
      int samples,
      int first_layer,
      int second_layer,
      int num_split = 8) {
    // Example:
    // ( 1000 * 597540 x 597540 * 1024 + 1000*1024) first layer, data x weights
    // + bias. ( 1000 * 1024 x 1024 * 14588 + 1000*14588) second layer, data x
    // weights + bias.
    int input_features_size = features;
    int num_samples = samples;

    int first_layer_output_size = first_layer;
    int second_layer_output_size = second_layer;

    long input_total_size = input_features_size * num_samples;

    int weight_layer1_size = input_features_size * first_layer_output_size;
    int weight_layer2_size = first_layer_output_size * second_layer_output_size;

    int bias_layer1_size = first_layer_output_size;
    int bias_layer2_size = second_layer_output_size;
    // Seed the random number generator
    std::random_device rd;
    // Initialize the Mersenne Twister engine
    std::mt19937 gen(rd());
    // Define the range
    std::uniform_real_distribution<float> distribution(0.0009, 0.0011);
    // Generate input
    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);

    std::vector<std::shared_ptr<TempFilePath>> feature_paths;
    size_t partition_size = ceil(num_samples / float(num_split));
    for (size_t i = 0; i < num_split; i++) {
      size_t num_samples_in_partition = (i + 1) * partition_size <= num_samples
          ? partition_size
          : num_samples - i * partition_size;
      if (num_samples_in_partition == 0) {
        continue;
      }
      std::vector<int> indexes = randomGenerator.genIntRange(
          i * partition_size, i * partition_size + num_samples_in_partition);
      auto indexFlaVector = maker.flatVector<int>(indexes);
      std::vector<std::vector<float>> partialFeature =
          randomGenerator.genFloat2dVector(
              num_samples_in_partition, input_features_size);
      auto featureArrayVector =
          maker.arrayVector<float>(std::move(partialFeature), REAL());
      auto inputRowVector = maker.rowVector(
          {"idx", "v"},
          {std::move(indexFlaVector), std::move(featureArrayVector)});
      auto file = TempFilePath::create();
      auto config = std::make_shared<facebook::velox::dwrf::Config>();
      writeToFile(file->path, {std::move(inputRowVector)}, config);
      feature_paths.push_back(file);
      inputRowVector.reset();
      indexFlaVector.reset();
      indexes.clear();
      featureArrayVector.reset();
      partialFeature.clear();
      partialFeature.shrink_to_fit();
    }

    // Generate weight
    float* weight_layer1 = new float[weight_layer1_size];

    for (int i = 0; i < weight_layer1_size; ++i) {
      // weight_layer1[i] = i;
      weight_layer1[i] = 0.000001;
    }
    float* weight_layer2 = new float[weight_layer2_size];

    for (int i = 0; i < weight_layer2_size; ++i) {
      weight_layer2[i] = 0.000001;
    }

    std::vector<float*> weights;
    weights.push_back(std::move(weight_layer1));
    weights.push_back(std::move(weight_layer2));

    // Generate bias
    float* bias_layer1 = new float[bias_layer1_size];

    for (int i = 0; i < bias_layer1_size; ++i) {
      bias_layer1[i] = 0.00001;
    }
    float* bias_layer2 = new float[bias_layer2_size];

    for (int i = 0; i < bias_layer2_size; ++i) {
      bias_layer2[i] = 0.00001;
    }
    std::vector<float*> bias;
    bias.push_back(std::move(bias_layer1));
    bias.push_back(std::move(bias_layer2));
    // Create DataFrame
    DataFrame data;
    // data.features = featureVectors;
    data.weights = std::move(weights);
    data.bias = std::move(bias);
    data.feature_paths = std::move(feature_paths);
    return data;
  }

  /**
   * @brief Registers a series of vector functions in the optimization
   * namespace.
   *
   * @param units1 Number of units in the first layer.
   * @param units2 Number of units in the second layer.
   * @param input_size1 Size of the input for the first layer.
   * @param input_size2 Size of the input for the second layer.
   * @param weightsFile_1 Pointer to the weights for the first layer.
   * @param weightsFile_2 Pointer to the weights for the second layer.
   * @param biasFile_1 Pointer to the bias for the first layer.
   * @param biasFile_2 Pointer to the bias for the second layer.
   * @param catalog Reference to a CataLog object to store metadata and
   * information.
   *
   * @return A string representing the composed vector function expression.
   */
  std::string registerFunctions(
      int units1,
      int units2,
      int input_size1,
      int input_size2,
      float* weightsFile_1,
      float* weightsFile_2,
      float* biasFile_1,
      float* biasFile_2,
      CataLog& catalog,
      bool isVerticalPartition) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "mat_mul0_0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(weightsFile_1), input_size1, units1),
        {},
        true,
        catalog,
        isVerticalPartition);
    // Register matrix addition function for the first layer
    optimization::registerVectorFunction(
        "mat_add0_0",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(std::move(biasFile_1), units1),
        {},
        true,
        catalog);
    // Register ReLU activation function for the first layer
    optimization::registerVectorFunction(
        "relu",
        Relu::signatures(),
        std::make_unique<Relu>(),
        {},
        true,
        catalog);
    // Register matrix multiplication function for the second layer
    optimization::registerVectorFunction(
        "mat_mul0_1",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(weightsFile_2), input_size2, units2),
        {},
        true,
        catalog,
        isVerticalPartition);
    // Register matrix addition function for the second layer
    optimization::registerVectorFunction(
        "mat_add0_1",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(std::move(biasFile_2), units2),
        {},
        true,
        catalog);
    // Register softmax activation function for the second layer
    optimization::registerVectorFunction(
        "softmax",
        Softmax::signatures(),
        std::make_unique<Softmax>(),
        {},
        true,
        catalog);
    // Compose and return the vector function expression
    // return "mat_mul0({})";
    return "softmax(mat_add0_1(mat_mul0_1(relu(mat_add0_0(mat_mul0_0({})))))) as v";
  }

  std::vector<std::vector<float>>
  loadFeaturesFromCSV(std::string filePath, int numSamples, int numFeature) {
    int size = numSamples * numFeature;

    std::cout << "Loading tensor of size " << size << " from " << filePath
              << std::endl;

    std::ifstream file(filePath.c_str());

    std::vector<std::vector<float>> inputArrayVector;

    int index = 0;

    std::string line;

    while (numSamples--) { // Read a line from the file

      std::vector<float> curRow(numFeature);

      std::getline(file, line);

      std::istringstream iss(
          line); // Create an input string stream from the line

      std::string numberStr;

      int colIndex = 0;

      while (std::getline(
          iss, numberStr, ',')) { // Read each number separated by comma
        //
        float number = std::stof(numberStr); // Convert the string to float

        if (colIndex < numFeature)

          curRow[colIndex] = number;

        colIndex++;
      }

      inputArrayVector.push_back(std::move(curRow));
    }

    file.close();

    return inputArrayVector;
  }

  void registerDecisionForestFunctions() {
    std::cout << "To register function for TreePrediction" << std::endl;

    exec::registerVectorFunction(
        "decision_tree_predict",
        TreePrediction::signatures(),
        std::make_unique<TreePrediction>(
            0,
            "/home/velox/resources/model/fraud_xgboost_10_8/0.txt",
            28,
            false));

    std::cout << "To register type for Tree" << std::endl;

    registerCustomType("tree_type", std::make_unique<TreeTypeFactories>());

    std::cout << "To register function for VeloxTreePrediction" << std::endl;

    exec::registerVectorFunction(
        "velox_decision_tree_predict",
        VeloxTreePrediction::signatures(),
        std::make_unique<VeloxTreePrediction>(28));

    std::cout << "To register function for VeloxTreeConstruction" << std::endl;

    exec::registerVectorFunction(
        "velox_decision_tree_construct",
        VeloxTreeConstruction::signatures(),
        std::make_unique<VeloxTreeConstruction>());

    std::cout << "To register function for ForestPrediction" << std::endl;

    exec::registerVectorFunction(
        "decision_forest_predict",
        TreePrediction::signatures(),
        std::make_unique<ForestPrediction>(
            "/home/velox/resources/model/fraud_xgboost_10_8", 28, true));
  }

  std::vector<std::shared_ptr<TempFilePath>> splitDataToFiles(
      std::vector<std::vector<float>> data,
      int numSplit = 4,
      bool createIndex = false) {
    std::vector<std::shared_ptr<TempFilePath>> paths;

    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    size_t numSamples = data.size();
    size_t partitionSize = ceil(numSamples / numSplit);
    for (size_t i = 0; i < numSplit; i++) {
      auto startIdx = data.begin() + i * partitionSize;
      auto endIdx = data.begin() + (i + 1) * partitionSize;
      endIdx = (endIdx < data.end()) ? endIdx : data.end();
      size_t numSampleInPartition = (i + 1) * partitionSize <= numSamples
          ? partitionSize
          : numSamples - i * partitionSize;

      std::vector<std::vector<float>> partialData(startIdx, endIdx);
      auto featureArrayVector = maker.arrayVector<float>(partialData, REAL());
      RowVectorPtr inputRowVector;
      if (createIndex) {
        std::vector<int> indexes = randomGenerator.genIntRange(
            i * partitionSize, i * partitionSize + numSampleInPartition);
        auto indexFlatVector = maker.flatVector<int>(indexes);
        inputRowVector = maker.rowVector(
            {"idx", "v"},
            {std::move(indexFlatVector), std::move(featureArrayVector)});
      } else {
        inputRowVector = maker.rowVector({"v"}, {featureArrayVector});
      }
      auto file = TempFilePath::create();
      auto config = std::make_shared<facebook::velox::dwrf::Config>();
      writeToFile(file->path, {inputRowVector}, config);
      paths.push_back(file);
    }

    return paths;
  }

  std::string process_mem_usage() {
    using std::ifstream;
    using std::ios_base;
    using std::string;

    double vm_usage = 0.0;
    double resident_set = 0.0;

    // Read data from /proc/self/stat
    ifstream stat_stream("/proc/self/stat", ios_base::in);
    if (!stat_stream) {
      std::cerr << "Error opening /proc/self/stat" << std::endl;
      return "";
    }

    // Extract relevant fields
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
        tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
        stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
        starttime >> vsize >> rss;

    stat_stream.close();

    // Get page size in KB
    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024 / 1024 / 1024;

    // Calculate memory usage
    vm_usage = vsize / 1024.0 / 1024.0 / 1024.0;
    resident_set = rss * page_size_kb;
    std::cout << fmt::format(
                     " vm_usage: {:.2f} , resident_set: {:.2f}",
                     vm_usage,
                     resident_set)
              << std::endl;
    return "";
  }

  /**
   * @brief A test function to test the rewrite rule of
   * Mul2JoinAggRewriteAction.
   *
   * @param rewrite A boolean value indicating whether to perform a rewrite.
   */
  void testSingleRewrite(
      bool rewrite,
      int repeatRun,
      int featureSize,
      int numSamples,
      int numDriver,
      std::string benchmarkMode,
      int blockSize,
      int verbose,
      bool getCost) {
    // Set data source config.
    int input_features_size = featureSize; // 597540
    int num_samples = numSamples;
    int first_layer_output_size = 1024;
    int second_layer_output_size = 14588;
    // Set splits number
    // Initialize CataLog
    CataLog cataLog;
    cataLog.setDefaultBlocksSize(blockSize);
    cataLog.setBlockingThreshold(1);
    int numSplit = 8;
    if (input_features_size == 597540) {
      if (num_samples == 50000) {
        numSplit = 16;
      } else if (num_samples == 100000) {
        numSplit = 32;
      }
    }
    // Generate data source
    auto data = data_generate(
        input_features_size,
        num_samples,
        first_layer_output_size,
        second_layer_output_size,
        numSplit);
    // Split inputs into many splits and return paths as a std::vector
    std::vector<std::shared_ptr<TempFilePath>> files = data.feature_paths;
    RowTypePtr inputRowType = ROW({"idx", "v"}, {INTEGER(), ARRAY(REAL())});
    //  Check the input size against the blocking threshold in cataLog.
    //  If yes, preblock the input vector, store it, and add information in
    //  cataLog. If not, set dataSource in cataLog.
    if (input_features_size > cataLog.getBlockingThreshold()) {
      // FIXME: temporary disable the following blocking for vertical partition
      // since we have deallocate the data.features in data_generate function to
      // save unnecessary memory usage.
      /* // If input size is larger than blocking threshold, preblock and store
      in
      // cataLog
      std::vector<std::vector<float>> valuesBlock =
          optimization::create_input_block(
              input_features_size * num_samples,
              data.features,
              cataLog.getDefaultBlocksNum());
      optimization::FileStructure values = optimization::block_to_files(
          valuesBlock, cataLog.getDefaultBlocksNum(), 0);
      // Set data source blocks in cataLog
      cataLog.setDataSourceBlocks(values.schema, values.paths); */
    } else {
      // If input size is not larger than blocking threshold, set dataSource in
      // cataLog
      cataLog.setDataSource(inputRowType, files);
    }
    // Set data source statistics in cataLog
    cataLog.setDataSourceStat({num_samples, input_features_size});
    cataLog.setUDFSchema("value", inputRowType);
    // Build two dense layers UDFs using registerFunction in optimization
    // namespace
    bool isVerticalPartition = true;
    if (benchmarkMode.find("mul2joinAggHorizontal") != std::string::npos) {
      isVerticalPartition = false;
    }
    std::string computationStr = registerFunctions(
        first_layer_output_size,
        second_layer_output_size,
        input_features_size,
        first_layer_output_size,
        data.weights[0],
        data.weights[1],
        data.bias[0],
        data.bias[1],
        cataLog,
        isVerticalPartition);
    // Initialize planNodeID
    core::PlanNodeId p0;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    // Create a plan for FFNN using two dense layers UDFs
    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(inputRowType)
                      .capturePlanNodeId(p0)
                      .project({fmt::format(computationStr, "v"), "v as v1"})
                      .project({"v", "v1"});

    std::cout << "computationStr: " << fmt::format(computationStr, "v")
              << std::endl;
    std::cout << "query plan:" << myPlan.planNode()->toString(true, true)
              << std::endl;
    // auto myPlan0 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
    //                   .tableScan(inputRowType)
    //                   .capturePlanNodeId(p0)
    //                   .project({fmt::format(computationStr, "v")});
    // auto myPlan = myPlan0.project({"v", "v"});

    // Set original plan nodeId and file address of data source
    cataLog.setIdAddressMap(p0, files);
    // Set vector name and nodeId of data source
    cataLog.setVectorIdMap(p0, "v");
    // Add source to catalog
    std::shared_ptr<OutputStat> stat =
        std::make_shared<OutputStat>(OutputStat(numSamples, featureSize));
    Source src = Source(p0, Source::Type::FILE, std::move(stat));
    cataLog.addSource(std::make_shared<Source>(src));

    // Get the logical plan
    auto planNode = myPlan.planNode();
    // auto planNode = myPlan.planNodeModifiable();
    // Create ruleManager
    RuleManager ruleManager;
    ruleManager.rules.emplace(
        "TwoLayerUDF2TorchNNRewriteAction",
        std::make_shared<optimization::TwoLayerUDF2TorchNNRewriteAction>());
    // std::cout<<"rule size" << ruleManager.rules.size() << std::endl;
    // auto it = ruleManager.rules.find("TwoLayerUDF2TorchNNRewriteAction");
    // ruleManager.rules.erase(it);
    // std::cout<<"rule size" << ruleManager.rules.size() << std::endl;
    // Create planState
    PlanState planState(ruleManager);

    // Run rewriten rule
    if (rewrite) {
      // Get possible actions for this plan
      planState.getPossibleActions(planNode, cataLog);

      if (verbose != 0) {
        std::cout << "[INFO] All possible actions:" << std::endl;
        for (auto entry : planState.actionsPair) {
          std::cout << entry.first << ": " << entry.second << std::endl;
        }
      }

      std::pair<std::string, std::string> testAction;
      if (benchmarkMode == "mul2joinAgg") {
        testAction = std::make_pair("mat_mul0", "Mul2JoinAggRewriteAction");
      } else if (benchmarkMode == "udf2torchNN") {
        testAction = std::make_pair(
            "softmax0(mat_add1(mat_mul1(relu0(mat_add0(mat_mul0(ROW[\"v\"]))))))",
            "MultiLayerUDF2TorchNNRewriteAction");
      } else if (benchmarkMode == "mul2joinAggHorizontal") {
        testAction =
            std::make_pair("mat_mul0", "Mul2JoinAggHorizontalRewriteAction");
      } else if (benchmarkMode == "mul2joinAggHorizontal1") {
        testAction =
            std::make_pair("mat_mul1", "Mul2JoinAggHorizontalRewriteAction");
      } else {
        throw std::runtime_error(
            fmt::format("Non-supported benchmark mode: {}", benchmarkMode));
      }
      if (verbose != 0) {
        std::cout << "Taken action: " << testAction << std::endl;
      }
      // Take one rewritten action
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);
      // Update the planState (getPossibleAction after apply one action)
      planState.update(myPlan, cataLog);
    }

    // Run the rewritten plan
    if (verbose != 0) {
      std::cout << "Executed Query Plan: \n"
                << myPlan.planNode()->toString(true, true) << std::endl;
    }

    if (getCost) {
      // std::shared_ptr<Catalog> catalog =
      //       std::make_shared<Catalog>(Catalog("db-catalog"));

      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      CostModel* cm = new SimpleCostModel(cataLog);
      CostEstimator* ce =
          new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

      planNode = myPlan.planNode();
      CostEstimate cost = ce->estimateCost(planNode);

      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      auto costComputationTime =
          (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
               .count()) /
          1000000.0;
      std::cout << "[INFO] Current query plan cost: " << cost.cost
                << ", Computation Time: " << costComputationTime << std::endl;
      return;
    }

    float averageExectuionTime = runPlanWithCataLog(
        numDriver, numDriver, myPlan, cataLog, repeatRun, verbose);
    std::cout << averageExectuionTime << std::endl;
  }

  PlanBuilder setupQueryPlan(
      std::string model,
      std::string computationStr,
      std::vector<std::string> inputFilePaths,
      std::vector<std::shared_ptr<TempFilePath>> inputTempFiles,
      int numSamples,
      int numFeatures,
      CataLog& cataLog,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
    PlanBuilder myPlan;
    if (model != "two-tower") {
      auto inputRowType = ROW({"idx", "v"}, {INTEGER(), ARRAY(REAL())});
      core::PlanNodeId p0;
      myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                   .tableScan(inputRowType)
                   .capturePlanNodeId(p0)
                   .project({fmt::format(computationStr, "v")});
      cataLog.setIdAddressMap(p0, inputTempFiles);
      std::shared_ptr<OutputStat> stat =
          std::make_shared<OutputStat>(OutputStat(numSamples, numFeatures));
      Source src = Source(p0, Source::Type::FILE, std::move(stat));
      cataLog.addSource(std::make_shared<Source>(src));
      cataLog.setFileSchema(p0, inputRowType);
    } else {
      core::PlanNodeId readQueryDataPlanNodeId;
      core::PlanNodeId readUserDataPlanNodeId;
      core::PlanNodeId readRatingDataPlanNodeId1;
      core::PlanNodeId readRatingDataPlanNodeId2;
      core::PlanNodeId readMovieDataPlanNodeId;

      auto userDataRowType = ROW(
          {
              "user_id",
              "gender",
              "age",
              "occupation",
              "zipcode",
          },
          {INTEGER(), VARCHAR(), INTEGER(), INTEGER(), VARCHAR()});

      auto movieDataRowType = ROW(
          {"movie_id", "title", "genres"}, {INTEGER(), VARCHAR(), VARCHAR()});

      auto ratingDataRowType =
          ROW({"user_id", "movie_id", "rating", "timestamp"},
              {INTEGER(), INTEGER(), INTEGER(), INTEGER()});

      auto queryDataRowType =
          ROW({"q_user_id", "q_movie_id"}, {INTEGER(), INTEGER()});

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
                  {"user_id",
                   "gender",
                   "age",
                   "occupation",
                   "user_mean_rating"})
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
              .capturePlanNodeId(readQueryDataPlanNodeId)
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
                  {"user_id",
                   "movie_id",
                   "user_id_encoder(convert_int_array(user_id)) as user_id_embed",
                   "gender_encoder(gender) as gender",
                   "age_encoder(convert_int_array(age)) as age",
                   "occupation_encoder(convert_int_array(occupation)) as occupation",
                   "convert_double_to_float_array(user_mean_rating) as user_mean_rating",
                   "movie_id_encoder(convert_int_array(movie_id)) as movie_id_embed",
                   "genres_encoder(split(genres, '|')) as genres",
                   "convert_double_to_float_array(movie_mean_rating) as movie_mean_rating"})
              .project( // look-up embedding
                  {"user_id",
                   "movie_id",
                   "user_id_embedding(user_id_embed) as user_id_embed",
                   "gender_embedding(gender) as gender",
                   "age_embedding(age) as age",
                   "occupation_embedding(occupation) as occupation",
                   "user_mean_rating",
                   "movie_id_embedding(movie_id_embed) as movie_id_embed",
                   "sequence_pooling(genres_embedding(genres)) as genres",
                   "movie_mean_rating"})
              .project( // concate embedding vectors
                  {"user_id",
                   "movie_id",
                   "concat4(concat3(concat2(concat1(user_id_embed,gender),age),occupation), user_mean_rating) as user_tower_features",
                   "concat2_2(concat2_1(movie_id_embed, genres), movie_mean_rating) as movie_tower_features"})
              // .project( // user/movie tower inference
              // {"user_torchNN(user_tower_features) as user_nn_out",
              //  "movie_torchNN(movie_tower_features) as movie_nn_out"})
              .project( // user/movie tower inferenc e
                  {"user_id",
                   "movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"user_id",
                   "movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});

      myPlan = joinedUserAndMovieDataPlan;

      std::vector<std::string> userDataPaths =
          getFilePathsFromDir("/home/velox/data/movielens/user");
      std::vector<std::string> movieDataPaths =
          getFilePathsFromDir("/home/velox/data/movielens/movie");
      std::vector<std::string> ratingDataPaths =
          getFilePathsFromDir("/home/velox/data/movielens/rating");

      cataLog.setIdAddressMap(
          readQueryDataPlanNodeId,
          inputFilePaths,
          dwio::common::FileFormat::PARQUET);
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

      std::shared_ptr<OutputStat> stat =
          std::make_shared<OutputStat>(OutputStat(numSamples, 2));
      Source src =
          Source(readQueryDataPlanNodeId, Source::Type::FILE, std::move(stat));
      cataLog.addSource(std::make_shared<Source>(src));

      std::shared_ptr<OutputStat> userStat =
          std::make_shared<OutputStat>(OutputStat(6040, 5));
      Source userSrc = Source(
          readUserDataPlanNodeId, Source::Type::FILE, std::move(userStat));
      cataLog.addSource(std::make_shared<Source>(userSrc));

      std::shared_ptr<OutputStat> movieStat =
          std::make_shared<OutputStat>(OutputStat(3706, 3));
      Source movieSrc = Source(
          readMovieDataPlanNodeId, Source::Type::FILE, std::move(movieStat));
      cataLog.addSource(std::make_shared<Source>(movieSrc));

      std::shared_ptr<OutputStat> ratingStat =
          std::make_shared<OutputStat>(OutputStat(3706, 3));
      Source ratingSrc1 = Source(
          readRatingDataPlanNodeId1, Source::Type::FILE, std::move(ratingStat));
      Source ratingSrc2 = Source(
          readRatingDataPlanNodeId2, Source::Type::FILE, std::move(ratingStat));
      cataLog.addSource(std::make_shared<Source>(ratingSrc1));
      cataLog.addSource(std::make_shared<Source>(ratingSrc2));
      cataLog.setFileSchema(readUserDataPlanNodeId, userDataRowType);
      cataLog.setFileSchema(readMovieDataPlanNodeId, movieDataRowType);
      cataLog.setFileSchema(readRatingDataPlanNodeId1, ratingDataRowType);
      cataLog.setFileSchema(readRatingDataPlanNodeId1, ratingDataRowType);
      cataLog.setFileSchema(readQueryDataPlanNodeId, queryDataRowType);
    }

    return myPlan;
  }

  void testIntegratedMCTS(
      std::string model,
      int featureSize,
      int numSamples,
      int repeatRun,
      int blockSize,
      int verbose) {
    PlanBuilder myPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    std::vector<std::string> inputFilePaths;
    std::vector<std::shared_ptr<TempFilePath>> inputTempFiles;
    std::string computationStr;

    if (model == "ffnn") {
      // Set data source config.
      int input_features_size = featureSize; // 597540
      int num_samples = numSamples;
      int first_layer_output_size = 1024;
      int second_layer_output_size = 14588;
      cataLog.setDefaultBlocksSize(blockSize);
      cataLog.setBlockingThreshold(1);
      // Set splits number
      // Generate data source
      auto data = data_generate(
          input_features_size,
          num_samples,
          first_layer_output_size,
          second_layer_output_size);
      // Split inputs into many splits and return paths as a std::vector
      inputTempFiles = data.feature_paths;

      bool isVerticalPartition = false;
      computationStr = registerFunctions(
          first_layer_output_size,
          second_layer_output_size,
          input_features_size,
          first_layer_output_size,
          data.weights[0],
          data.weights[1],
          data.bias[0],
          data.bias[1],
          cataLog,
          isVerticalPartition);

    } else if (model == "df") {
      // TODO: current the data is load froma file not generated
      numSamples = 56962;
      featureSize = 28;
      registerDecisionForestFunctions();

      std::string dataFilePath =
          "/home/velox/resources/data/creditcard_test.csv";

      std::vector<std::vector<float>> inputFeatureVectors =
          loadFeaturesFromCSV(dataFilePath, numSamples, featureSize);
      inputTempFiles = splitDataToFiles(
          inputFeatureVectors, 4 /*numSplit*/, true /*createIndex*/);
      computationStr = "decision_forest_predict({}) as v";
    } else if (model == "two-tower") {
      registerTwoTowerFunc(cataLog, pool_, false /*isVerticalPartition*/);
      inputFilePaths = generateTwoTowerQueryData(numSamples, 6040, 3706, 1);
      featureSize = 2;
      std::cout << "inputDataPaths : " << inputFilePaths << std::endl;
    } else {
      throw std::runtime_error(fmt::format("Non-supported model: {}", model));
    }

    myPlan = setupQueryPlan(
        model,
        computationStr,
        inputFilePaths,
        inputTempFiles,
        numSamples,
        featureSize,
        cataLog,
        planNodeIdGenerator);

    // Get the logical plan
    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    planState.getPossibleActions(planNode, cataLog);

    // Set up socket
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
      std::cerr << "Error creating socket\n";
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    serverAddr.sin_port = htons(12345);

    if (connect(
            clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) ==
        -1) {
      std::cerr << "Error connecting to server\n";
      close(clientSocket);
    }

    std::cout << "Start optimization." << std::endl;

    // send start message to start MCTS optimization
    // start flag and initial query plan
    Json::Value startJsonMessage;
    startJsonMessage["mctsAction"] = "start";
    startJsonMessage["queryPlan"] = planNode->toString(true, true);
    std::cout << "json message: " << startJsonMessage << std::endl;
    sendJsonBySocket(startJsonMessage, clientSocket);
    bool optimizationIsFinished = false;

    while (!optimizationIsFinished) {
      planNode = myPlan.planNode();
      // received json message from MCTS
      Json::Value receivedJsonMessage = receiveJsonFromSocket(clientSocket);
      std::string mctsAction = receivedJsonMessage["mctsAction"].asString();
      LOG(INFO) << "===================================" << std::endl;
      LOG(INFO) << "Received message with mcts action: " << mctsAction
                << std::endl;
      if (mctsAction == "") {
        LOG(INFO) << "Un-captured error happened" << std::endl;
        return;
      }
      LOG(INFO) << "JSON Message: " << receivedJsonMessage << std::endl;
      if (mctsAction == "resetPlan") {
        // if it is root node, it needs to start with original plan
        // the p0 will be increased after capturePlanNodeId is called
        // so it is required to clean the old IdAddressMap and VectorIdMap
        // before reset the myPlan
        cataLog.clearIdAddressMap();
        cataLog.clearVectorIdMap();
        cataLog.clearSourceMap();
        myPlan = setupQueryPlan(
            model,
            computationStr,
            inputFilePaths,
            inputTempFiles,
            numSamples,
            featureSize,
            cataLog,
            planNodeIdGenerator);
        planNode = myPlan.planNode();
        planState.getPossibleActions(planNode, cataLog);
        // std::cout << "[INFO] All possible actions:" << std::endl;
        // for (auto entry : planState.actionsPair) {
        //   std::cout << entry.first << ": " << entry.second << std::endl;
        // }
        // send acknowledgement for synchronization
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getQueryPlan") {
        Json::Value jsonMessage;
        jsonMessage["communicateFlag"] = true;
        jsonMessage["mctsAction"] = "recQueryPlan";
        jsonMessage["queryPlan"] =
            "\"" + myPlan.planNode()->toString(true, true) + "\"";
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "getActionSpace") {
        planState.getPossibleActions(planNode, cataLog);
        Json::Value jsonMessage;
        jsonMessage["actionSpace"] = Json::arrayValue;
        for (const auto& entry : planState.actionsPair) {
          // LOG(INFO) << "[ACTION SBACE] " << entry.first << s't'd
          // entry.second << std::endl;
          Json::Value jsonEntry;
          jsonEntry["expression"] = entry.first;
          jsonEntry["action"] = Json::arrayValue;
          for (auto action : entry.second) {
            jsonEntry["action"].append(Json::Value(action));
          }
          jsonMessage["actionSpace"].append(jsonEntry);
        }
        // cache current state
        int queryPlanCacheId = cacheQueryPlan(myPlan);
        jsonMessage["queryPlanCacheId"] = queryPlanCacheId;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "takeAction") {
        std::pair<std::string, std::string> targetAction;
        targetAction.first = receivedJsonMessage["targetString"].asString();
        targetAction.second = receivedJsonMessage["targetAction"].asString();

        LOG(INFO) << "[INFO] take action: " << targetAction << std::endl;
        if (targetAction.first != "None") {
          // None action is selected
          planState.takeAction(
              planNode,
              nullptr,
              maker,
              myPlan,
              pool_,
              planNodeIdGenerator,
              {targetAction},
              cataLog);
          planState.update(myPlan, cataLog);
        }
        LOG(INFO) << "[INFO] current my query plan"
                  << myPlan.planNode()->toString(true, true) << std::endl;
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "cacheState") {
        int queryPlanCacheId = cacheQueryPlan(myPlan);
        Json::Value jsonMessage;
        jsonMessage["queryPlanCacheId"] = queryPlanCacheId;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "resetState") {
        int queryPlanCacheId = receivedJsonMessage["queryPlanCacheId"].asInt();
        resetQueryPlanFromCache(myPlan, queryPlanCacheId);
        sendAcknowledgment(clientSocket);
      } else if (mctsAction == "getCost") {
        Json::Value jsonMessage;
        if (receivedJsonMessage["costMode"] == "offline") {
          float executeTime =
              runPlanWithCataLog(8, 8, myPlan, cataLog, 4, verbose);
          jsonMessage["reward"] = executeTime;
          LOG(INFO) << "[INFO] get Cost(offline): " << " time: " << executeTime
                    << std::endl;
        } else if (receivedJsonMessage["costMode"] == "online") {
          CostModel* cm = new SimpleCostModel(cataLog);
          CostEstimator* ce =
              new SimpleCostEstimator(std::unique_ptr<CostModel>(cm));

          planNode = myPlan.planNode();
          CostEstimate cost = ce->estimateCost(planNode);
          jsonMessage["reward"] = cost.cost;
          LOG(INFO) << "[INFO] get Cost(online): " << cost.cost << std::endl;
        }
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);

      } else if (mctsAction == "runPlan") {
        auto latency = runPlanWithCataLog(8, 8, myPlan, cataLog, 4, verbose);
        Json::Value jsonMessage;
        jsonMessage["latency"] = latency;
        sendAcknowledgment(clientSocket);
        sendJsonBySocket(jsonMessage, clientSocket);
      } else if (mctsAction == "finished") {
        // finished
        // nothing to do
      }

      optimizationIsFinished =
          receivedJsonMessage["optimizationIsFinished"].asBool();
      LOG(INFO) << "[INFO] reached end of the loop, current opt flag: "
                << optimizationIsFinished << std::endl;
    };

    // Run the rewritten plan
    // LOG(INFO) << "[INFO] MCTS finished, run the optimized query plan"
    //           << std::endl;
    // LOG(INFO) << "[INFO] Optimized query plan"
    //           << myPlan.planNode()->toString(true, true) << std::endl;
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::MemoryManager::getInstance()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};
  std::shared_ptr<TempDirectoryPath> tempDirPath_;

  VectorMaker maker{pool_.get()};
  static inline int queryPlanCacheId_ = 0;
  std::map<int, folly::dynamic> queryPlanCaches_;
};

DEFINE_string(mode, "mcts", "Mode: mcts or benchmark");
DEFINE_string(model, "ffnn", "Model: ffnn, df, two-tower");
DEFINE_bool(rewrite, true, "Whether  rewrite");
DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(feature_size, 1000, "FFNN Feature size");
DEFINE_int32(num_sample, 1000, "Number of samples");
DEFINE_int32(num_driver, 8, "Number of drivers");
DEFINE_int32(verbose, 2, "Verbose");
DEFINE_int32(block_size, 256, "Block Size");
DEFINE_bool(cost, false, "Whether get cost");

int main(int argc, char** argv) {
  memory::MemoryManager::initialize({});
  folly::init(&argc, &argv, false);
  // gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::string mode = FLAGS_mode;
  std::string model = FLAGS_model;

  bool rewrite = FLAGS_rewrite;
  int repeatRun = FLAGS_num_repeat;
  int featureSize = FLAGS_feature_size;
  int numSample = FLAGS_num_sample;
  int numDriver = FLAGS_num_driver;
  int verbose = FLAGS_verbose;
  int blockSize = FLAGS_block_size;
  bool getCost = FLAGS_cost;
  IntegratedMCTSTest demo;

  // available single benchmark mode: mul2joinAgg, udf2torchNN,
  // mul2joinAggHorizontal
  if (mode == "mcts") {
    demo.testIntegratedMCTS(
        model, featureSize, numSample, repeatRun, blockSize, verbose);
  } else {
    // Benchmark a single rewrite action
    demo.testSingleRewrite(
        rewrite,
        repeatRun,
        featureSize,
        numSample,
        numDriver,
        mode,
        blockSize,
        verbose,
        getCost);
  }
}
