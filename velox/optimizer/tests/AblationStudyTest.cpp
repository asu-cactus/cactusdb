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
#include <folly/init/Init.h>
#include <torch/torch.h>
#include <iostream>
#include <random>
#include <string>
#include "velox/optimizer/Helper.h"

// Velox headers
#include <H5Cpp.h>
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
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/FraudDetectionFunctions.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/functions.h"
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
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include "velox/optimizer/tests/ModelRegister.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

class AblationStudyTest : public HiveConnectorTestBase {
 public:
  AblationStudyTest() {
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

  ~AblationStudyTest() {
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

  void resetQueryPlanFromCache(PlanBuilder& planBuilder, int queryPlanCacheId) {
    auto it = queryPlanCaches_.find(queryPlanCacheId);
    if (it != queryPlanCaches_.end()) {
      auto serializedPlan = it->second;
      auto deserlizedUpdatedPlanNode =
          ISerializable::deserialize<core::PlanNode>(
              serializedPlan, pool_.get());
      planBuilder.setRoot(deserlizedUpdatedPlanNode);
    } else {
      throw std::runtime_error(
          fmt::format(
              "[ERROR]queryPlanCacheId: {} was not found.", queryPlanCacheId));
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

  RowVectorPtr getOrderData(std::string filePath) {
    std::ifstream file(filePath.c_str());

    if (file.fail()) {
      std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
      exit(1);
    }

    std::vector<int> oOrderId;
    std::vector<int> oCustomerSk;
    std::vector<std::string> oWeekday;
    std::vector<std::string> oDate;

    std::string line;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
      // std::cout << "Ignoring header: " << line << std::endl;
    }

    while (std::getline(file, line)) { // Read a line from the file

      // std::vector<float> curRow(numCols);

      // std::getline(file, line);

      std::istringstream iss(
          line); // Create an input string stream from the line

      std::string numberStr;

      int colIndex = 0;

      while (std::getline(
          iss, numberStr, ',')) { // Read each number separated by comma
        /*if (index < 5) {
            std::cout << colIndex << ": " << numberStr << std::endl;
        }*/
        // Trim leading and trailing whitespace from the input string (if any)
        if (numberStr.size() >= 2 && numberStr.front() == '"' &&
            numberStr.back() == '"') {
          numberStr = numberStr.substr(1, numberStr.size() - 2);
        }
        if (colIndex == 0) {
          oOrderId.push_back(std::stoi(numberStr));
        } else if (colIndex == 1) {
          oCustomerSk.push_back(std::stoi(numberStr));
        } else if (colIndex == 2) {
          oWeekday.push_back(numberStr);
        } else if (colIndex == 3) {
          oDate.push_back(numberStr);
        }

        colIndex++;
      }
    }

    file.close();

    // Prepare Customer table
    auto oOrderIdVector = maker.flatVector<int>(oOrderId);
    auto oCustomerSkVector = maker.flatVector<int>(oCustomerSk);
    auto oWeekdayVector = maker.flatVector<std::string>(oWeekday);
    auto oDateVector = maker.flatVector<std::string>(oDate);
    auto orderRowVector = maker.rowVector(
        {"o_order_id", "o_customer_sk", "o_weekday", "o_date"},
        {oOrderIdVector, oCustomerSkVector, oWeekdayVector, oDateVector});

    return orderRowVector;
  }

  RowVectorPtr getTransactionData(std::string filePath) {
    std::ifstream file(filePath.c_str());

    if (file.fail()) {
      std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
      exit(1);
    }

    std::vector<float> tAmount;
    std::vector<int> tSender;
    std::vector<std::string> tReceiver;
    std::vector<int64_t> transactionId;
    std::vector<std::string> tTime;

    std::string line;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
      // std::cout << "Ignoring header: " << line << std::endl;
    }

    while (std::getline(file, line)) { // Read a line from the file

      // std::vector<float> curRow(numCols);

      // std::getline(file, line);

      std::istringstream iss(
          line); // Create an input string stream from the line

      std::string numberStr;

      int colIndex = 0;

      while (std::getline(
          iss, numberStr, ',')) { // Read each number separated by comma
        /*if (index < 5) {
            std::cout << colIndex << ": " << numberStr << std::endl;
        }*/
        // Trim leading and trailing whitespace from the input string (if any)
        if (numberStr.size() >= 2 && numberStr.front() == '"' &&
            numberStr.back() == '"') {
          numberStr = numberStr.substr(1, numberStr.size() - 2);
        }
        if (colIndex == 0) {
          tAmount.push_back(std::stof(numberStr));
        } else if (colIndex == 1) {
          tSender.push_back(std::stoi(numberStr));
        } else if (colIndex == 2) {
          tReceiver.push_back(numberStr);
        } else if (colIndex == 3) {
          // Convert to double first using std::stod
          double numberDouble = std::stod(numberStr);
          transactionId.push_back(static_cast<long long>(numberDouble));
        } else if (colIndex == 4) {
          tTime.push_back(numberStr);
        }

        colIndex++;
      }
    }

    file.close();

    // Prepare Customer table
    auto tAmountVector = maker.flatVector<float>(tAmount);
    auto tSenderVector = maker.flatVector<int>(tSender);
    auto tReceiverVector = maker.flatVector<std::string>(tReceiver);
    auto transactionIdVector = maker.flatVector<int64_t>(transactionId);
    auto tTimeVector = maker.flatVector<std::string>(tTime);
    auto transactionRowVector = maker.rowVector(
        {"t_amount", "t_sender", "t_receiver", "transaction_id", "t_time"},
        {tAmountVector,
         tSenderVector,
         tReceiverVector,
         transactionIdVector,
         tTimeVector});

    return transactionRowVector;
  }

  RowVectorPtr getCustomerData(std::string filePath) {
    std::ifstream file(filePath.c_str());

    if (file.fail()) {
      std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
      exit(1);
    }

    std::vector<int> cCustomerSk;
    std::vector<int> cAddrerssNum;
    std::vector<int> cCustFlag;
    std::vector<int> cBirthYear;
    std::vector<int> cBirthCountry;

    std::unordered_map<std::string, int> countryMap = getCountryMap();
    int countryIndex = countryMap.size();

    std::string line;

    // Ignore the first line (header)
    if (std::getline(file, line)) {
      // std::cout << "Ignoring header: " << line << std::endl;
    }

    while (std::getline(file, line)) { // Read a line from the file

      // std::vector<float> curRow(numCols);

      // std::getline(file, line);

      std::istringstream iss(
          line); // Create an input string stream from the line

      std::string numberStr;

      int colIndex = 0;

      while (std::getline(
          iss, numberStr, ',')) { // Read each number separated by comma
        /*if (index < 5) {
            std::cout << colIndex << ": " << numberStr << std::endl;
        }*/
        // Trim leading and trailing whitespace from the input string (if any)
        if (numberStr.size() >= 2 && numberStr.front() == '"' &&
            numberStr.back() == '"') {
          numberStr = numberStr.substr(1, numberStr.size() - 2);
        }

        size_t first = numberStr.find_first_not_of(' ');
        if (first == std::string::npos)
          numberStr = "0";
        else {
          size_t last = numberStr.find_last_not_of(' ');
          numberStr = numberStr.substr(first, (last - first + 1));
        }

        // std::cout << "Column Index: " << colIndex << ", Value: " << numberStr
        // << std::endl;
        if (colIndex == 0) {
          cCustomerSk.push_back(std::stoi(numberStr));
        } else if (colIndex == 2) {
          cAddrerssNum.push_back(std::stoi(numberStr));
        } else if (colIndex == 5) {
          if (numberStr == "N")
            cCustFlag.push_back(0);
          else
            cCustFlag.push_back(1);
        } else if (colIndex == 8) {
          cBirthYear.push_back(std::stoi(numberStr));
        } else if (colIndex == 9) {
          if (countryMap.find(numberStr) == countryMap.end()) {
            // Key does not exist, insert it
            countryMap[numberStr] = countryIndex;
            cBirthCountry.push_back(countryIndex);
            countryIndex++;
          } else {
            // Key exists, retrieve its value
            cBirthCountry.push_back(countryMap[numberStr]);
          }
        }

        colIndex++;
      }
    }

    file.close();

    // Prepare Customer table
    auto cCustomerSkVector = maker.flatVector<int>(cCustomerSk);
    auto cAddrerssNumVector = maker.flatVector<int>(cAddrerssNum);
    auto cCustFlagVector = maker.flatVector<int>(cCustFlag);
    auto cBirthYearVector = maker.flatVector<int>(cBirthYear);
    auto cBirthCountryVector = maker.flatVector<int>(cBirthCountry);
    auto customerRowVector = maker.rowVector(
        {"c_customer_sk",
         "c_address_num",
         "c_cust_flag",
         "c_birth_year",
         "c_birth_country"},
        {cCustomerSkVector,
         cAddrerssNumVector,
         cCustFlagVector,
         cBirthYearVector,
         cBirthCountryVector});

    return customerRowVector;
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
   * @param input_size Size of the input for the first layer.
   * @param weights1 Pointer to the weights for the first layer.
   * @param weights2 Pointer to the weights for the second layer.
   * @param bias1 Pointer to the bias for the first layer.
   * @param bias2 Pointer to the bias for the second layer.
   * @param catalog Reference to a CataLog object to store metadata and
   * information.
   *
   * @return A string representing the composed vector function expression.
   */
  std::string registerFFNNFunctions(
      int units1,
      int units2,
      int input_size,
      float* weights1,
      float* weights2,
      float* bias1,
      float* bias2,
      CataLog& catalog,
      bool isVerticalPartition) {
    // Register matrix multiplication function for the first layer
    optimization::registerVectorFunction(
        "mat_mul0_0",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(weights1), input_size, units1),
        {},
        true,
        catalog,
        isVerticalPartition);
    // Register matrix addition function for the first layer
    optimization::registerVectorFunction(
        "mat_add0_0",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(std::move(bias1), units1),
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
        std::make_unique<MatrixMultiply>(std::move(weights2), units1, units2),
        {},
        true,
        catalog,
        isVerticalPartition);
    // Register matrix addition function for the second layer
    optimization::registerVectorFunction(
        "mat_add0_1",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(std::move(bias2), units2),
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

  void registerLLMFunctions(
      int units1,
      int units2,
      int input_size,
      CataLog& catalog,
      std::shared_ptr<memory::MemoryPool> pool_) {
    VectorMaker maker{pool_.get()};
    std::cout << "[INFO]: Register LLM Function functions" << std::endl;

    std::vector<std::vector<float>> w1 = loadHDF5Array(
        "/home/velox/resources/model/llm_mr/velox/llm_ffnn.h5", "w1");
    std::vector<std::vector<float>> b1 = loadHDF5Array(
        "/home/velox/resources/model/llm_mr/velox/llm_ffnn.h5", "b1");
    std::vector<std::vector<float>> w2 = loadHDF5Array(
        "/home/velox/resources/model/llm_mr/velox/llm_ffnn.h5", "w2");
    std::vector<std::vector<float>> b2 = loadHDF5Array(
        "/home/velox/resources/model/llm_mr/velox/llm_ffnn.h5", "b2");

    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    std::vector<std::vector<float>> ffnnWeight1 =
        randomGenerator.genFloat2dVector(input_size, units1);
    auto ffnnWeight1Vector = maker.arrayVector<float>(w1, REAL());

    std::vector<std::vector<float>> ffnnBias1 =
        randomGenerator.genFloat2dVector(units1, 1);
    auto ffnnBias1Vector = maker.arrayVector<float>(b1, REAL());

    std::vector<std::vector<float>> ffnnWeight2 =
        randomGenerator.genFloat2dVector(units1, units2);
    auto ffnnWeight2Vector = maker.arrayVector<float>(w2, REAL());

    std::vector<std::vector<float>> ffnnBias2 =
        randomGenerator.genFloat2dVector(units2, 1);
    auto ffnnBias2Vector = maker.arrayVector<float>(b2, REAL());

    optimization::registerVectorFunction(
        "mat_mul3_1",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(
                ffnnWeight1Vector->elements()->values()->asMutable<float>()),
            input_size,
            units1),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_vector_add3_2",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            std::move(
                ffnnBias1Vector->elements()->values()->asMutable<float>()),
            units1),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_mul3_3",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(
                ffnnWeight2Vector->elements()->values()->asMutable<float>()),
            units1,
            units2),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_vector_add3_4",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            std::move(
                ffnnBias2Vector->elements()->values()->asMutable<float>()),
            units2),
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

    // Register softmax activation function for the second layer
    optimization::registerVectorFunction(
        "softmax",
        Softmax::signatures(),
        std::make_unique<Softmax>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "chatgpt_server",
        ChatGPT::signatures(),
        std::make_unique<ChatGPT>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "chatgpt_recommender",
        ChatGPTRecommender::signatures(),
        std::make_unique<ChatGPTRecommender>(),
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
        "convert_double_array_to_float_array",
        ConvertDoubleArrayToFloatArray::signatures(),
        std::make_unique<ConvertDoubleArrayToFloatArray>(),
        {},
        true,
        catalog);
    optimization::registerVectorFunction(
        "llm_ffnn_minmax_scaler",
        MinMaxScaler::signatures(),
        std::make_unique<MinMaxScaler>(
            "/home/velox/resources/model/llm_mr/velox/llm_mr_minmax_scaler.txt"),
        {},
        true,
        catalog);
  }

  void registerLLM2Functions(
      CataLog& catalog,
      std::shared_ptr<memory::MemoryPool> pool_) {
    VectorMaker maker{pool_.get()};
    std::cout << "[INFO]: Register LLM2 Function functions" << std::endl;

    std::vector<std::string> document = readTextFile(
        "/home/velox/resources/model/llm_mr/velox/rag_document.txt");

    std::vector<std::vector<float>> documentEmbedding = loadHDF5Array(
        "/home/velox/resources/model/llm_mr/velox/rag_document.h5",
        "embedding");

    optimization::registerVectorFunction(
        "rag",
        RAG::signatures(),
        std::make_unique<RAG>(document, documentEmbedding, 384),
        {},
        true,
        catalog);

    std::string textEmbeddingExtractionMiniLMAPI =
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2";

    optimization::registerVectorFunction(
        "hf_minilm_embedding_extractor",
        HuggingFaceServerless::signatures(),
        std::make_unique<HuggingFaceServerless>(
            textEmbeddingExtractionMiniLMAPI,
            HuggingFaceTaskType::TEXT_FEATURE_EXTRACTION),
        {},
        true,
        catalog);
  }

  void registerFraudDetectionFunctions(
      int numCols,
      CataLog& catalog,
      std::shared_ptr<memory::MemoryPool> pool_) {
    // Register Pre-processing functions
    optimization::registerVectorFunction(
        "is_weekday",
        IsWeekday::signatures(),
        std::make_unique<IsWeekday>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "date_to_timestamp_1",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%d"),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "date_to_timestamp_2",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%dT%H:%M"),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "time_diff_in_days",
        TimeDiffInDays::signatures(),
        std::make_unique<TimeDiffInDays>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "get_transaction_features",
        GetTransactionFeatures::signatures(),
        std::make_unique<GetTransactionFeatures>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "get_customer_features",
        GetCustomerFeatures::signatures(),
        std::make_unique<GetCustomerFeatures>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "get_age",
        GetAge::signatures(),
        std::make_unique<GetAge>(),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "get_binary_class",
        GetBinaryClass::signatures(),
        std::make_unique<GetBinaryClass>(),
        {},
        true,
        catalog);

    // Register Random Forest model
    std::string xgboost_fraud_model_path =
        "/home/velox/resources/model/fraud_xgboost_9_1600";
    optimization::registerVectorFunction(
        "xgboost_fraud_predict",
        ForestPrediction::signatures(),
        std::make_unique<ForestPrediction>(xgboost_fraud_model_path, 9, true),
        {},
        true,
        catalog);

    std::string xgboost_fraud_transaction_path =
        "/home/velox/resources/model/fraud_xgboost_5_16";
    optimization::registerVectorFunction(
        "xgboost_fraud_transaction",
        ForestPrediction::signatures(),
        std::make_unique<ForestPrediction>(
            xgboost_fraud_transaction_path, 5, true),
        {},
        true,
        catalog);

    // Register FFNN model
    std::vector<std::vector<float>> w1 = loadHDF5Array(
        "/home/velox/resources/model/fraud_dnn_weights.h5", "fc1.weight");
    std::vector<std::vector<float>> b1 = loadHDF5Array(
        "/home/velox/resources/model/fraud_dnn_weights.h5", "fc1.bias");
    std::vector<std::vector<float>> w2 = loadHDF5Array(
        "/home/velox/resources/model/fraud_dnn_weights.h5", "fc2.weight");
    std::vector<std::vector<float>> b2 = loadHDF5Array(
        "/home/velox/resources/model/fraud_dnn_weights.h5", "fc2.bias");
    std::vector<std::vector<float>> w3 = loadHDF5Array(
        "/home/velox/resources/model/fraud_dnn_weights.h5", "fc3.weight");
    std::vector<std::vector<float>> b3 = loadHDF5Array(
        "/home/velox/resources/model/fraud_dnn_weights.h5", "fc3.bias");

    auto itemNNweight1Vector = maker.arrayVector<float>(w1, REAL());
    auto itemNNweight2Vector = maker.arrayVector<float>(w2, REAL());
    auto itemNNweight3Vector = maker.arrayVector<float>(w3, REAL());
    auto itemNNBias1Vector = maker.arrayVector<float>(b1, REAL());
    auto itemNNBias2Vector = maker.arrayVector<float>(b2, REAL());
    auto itemNNBias3Vector = maker.arrayVector<float>(b3, REAL());

    optimization::registerVectorFunction(
        "mat_mul1_1",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(
                itemNNweight1Vector->elements()->values()->asMutable<float>()),
            numCols,
            32),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_vector_add1_2",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            std::move(
                itemNNBias1Vector->elements()->values()->asMutable<float>()),
            32),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_mul1_3",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(
                itemNNweight2Vector->elements()->values()->asMutable<float>()),
            32,
            16),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_vector_add1_4",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            std::move(
                itemNNBias2Vector->elements()->values()->asMutable<float>()),
            16),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_mul1_5",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(
                itemNNweight3Vector->elements()->values()->asMutable<float>()),
            16,
            2),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "mat_vector_add1_6",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            std::move(
                itemNNBias3Vector->elements()->values()->asMutable<float>()),
            2),
        {},
        true,
        catalog);

    optimization::registerVectorFunction(
        "relu",
        Relu::signatures(),
        std::make_unique<Relu>(),
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
      std::string model,
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
    PlanBuilder myPlan;
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
          second_layer_output_size,
          numSplit);
      // Split inputs into many splits and return paths as a std::vector
      inputTempFiles = data.feature_paths;

      bool isVerticalPartition = false;
      computationStr = registerFFNNFunctions(
          first_layer_output_size,
          second_layer_output_size,
          input_features_size,
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
    } else if (model == "llm") {
      registerLLMFunctions(64, 2, 3, cataLog, pool_);
    } else if (model == "ml-q1") {
      registerTwoTowerFunc(cataLog, pool_, false /*isVerticalPartition*/);
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

    RuleManager ruleManager;
    PlanState planState(ruleManager);
    auto planNode = myPlan.planNode();

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
            "softmax(mat_add0_1(mat_mul0_1(relu(mat_add0_0(mat_mul0_0(ROW[\"v\"]))))))",
            "MultiLayerUDF2TorchNNRewriteAction");
      } else if (benchmarkMode == "mul2joinAggHorizontal") {
        testAction =
            std::make_pair("mat_mul0_0", "Mul2JoinAggHorizontalRewriteAction");
      } else if (benchmarkMode == "mul2joinAggHorizontal1") {
        testAction =
            std::make_pair("mat_mul0_1", "Mul2JoinAggHorizontalRewriteAction");
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
        pool_, numDriver, myPlan, cataLog, repeatRun, verbose);
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
    if (model == "ffnn" || model == "df") {
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
    } else if (model == "llm" || model == "llm-op") {
      std::vector<std::string> userDataPaths =
          getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/user");
      std::vector<std::string> movieDataPaths = getFilePathsFromDir(
          "/home/velox/resources/data/parquet/llm_mr/movie");
      auto userDataRowType =
          ROW({"user_id", "description"}, {INTEGER(), VARCHAR()});
      auto movieDataRowType =
          ROW({"id",
               "description",
               "popularity",
               "vote_average",
               "vote_count",
               "spoken_languages"},
              {INTEGER(), VARCHAR(), REAL(), REAL(), INTEGER(), VARCHAR()});

      std::string llmDataStatsFilePath =
          "/home/velox/resources/data/parquet/llm_mr/llm_mr_statistics.txt";
      std::ifstream llmStatistics(llmDataStatsFilePath);
      if (!llmStatistics) {
        throw std::runtime_error(
            "Unable to open file: " + llmDataStatsFilePath);
      }
      // TODO: need more smart way to do this
      std::string line1, line2;
      int numUser, numMovies;
      std::getline(llmStatistics, line1);
      std::getline(llmStatistics, line2);
      numUser = std::stoi(line1);
      numMovies = std::stoi(line2);

      core::PlanNodeId readUserDataPlanNodeId;
      core::PlanNodeId readMoviewDataPlanNodeId;

      if (model == "llm") {
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(userDataRowType, {}, "")
                .capturePlanNodeId(readUserDataPlanNodeId)
                .project(
                    {"CAST(user_id AS VARCHAR) as user_id",
                     "description AS user_description"})
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator, pool_.get())
                        .tableScan(movieDataRowType, {}, "")
                        .capturePlanNodeId(readMoviewDataPlanNodeId)
                        .project({
                            "CAST(id AS VARCHAR) AS movie_id",
                            "description AS movie_description",
                            "llm_ffnn_minmax_scaler(convert_double_array_to_float_array(array_constructor(popularity, vote_average, vote_count))) AS movie_description_array",
                            "spoken_languages",
                        })
                        .planNode(),
                    {"user_id",
                     "movie_id",
                     "user_description",
                     "movie_description",
                     "spoken_languages",
                     "movie_description_array"})
                .project(
                    {"user_id",
                     "movie_id",
                     "spoken_languages",
                     "movie_description_array",
                     "CONCAT(user_id, user_description) AS user_description_processed",
                     "CONCAT(movie_id, movie_description) AS movie_description_processed"})
                .project(
                    {"user_id",
                     "movie_id",
                     "spoken_languages",
                     "movie_description_array",
                     "chatgpt_server(user_description_processed, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description_summerized",
                     "chatgpt_server(movie_description_processed, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description_summerized"})
                .project(
                    {"user_id",
                     "movie_id",
                     "spoken_languages",
                     "movie_description_array",
                     "chatgpt_recommender(user_description_summerized, movie_description_summerized, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.') AS result"})
                .project({
                    "user_id",
                    "movie_id",
                    "spoken_languages",
                    "result",
                    "argmax(softmax(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array))))))) AS trending_prediction",
                })
                .filter("spoken_languages LIKE '\%English\%'")
                .filter("trending_prediction = 1");
      } else if (model == "llm-op") {
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(userDataRowType, {}, "")
                .capturePlanNodeId(readUserDataPlanNodeId)
                .project(
                    {"CAST(user_id AS VARCHAR) as user_id",
                     "description AS user_description"})
                .project(
                    {"user_id",
                     "CONCAT(user_id, user_description) AS user_description_processed"})
                .project({
                    "user_id",
                    "chatgpt_server(user_description_processed, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description_summerized",
                })
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator, pool_.get())
                        .tableScan(movieDataRowType, {}, "")
                        .capturePlanNodeId(readMoviewDataPlanNodeId)
                        .filter("spoken_languages LIKE '\%English\%'")
                        .project({
                            "CAST(id AS VARCHAR) AS movie_id",
                            "description AS movie_description",
                            "llm_ffnn_minmax_scaler(convert_double_array_to_float_array(array_constructor(popularity, vote_average, vote_count))) AS movie_description_array",
                            "spoken_languages",
                        })
                        .project(
                            {"movie_id",
                             "movie_description",
                             "argmax(softmax(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array))))))) AS trending_prediction",
                             "spoken_languages"})
                        .filter("trending_prediction = 1")
                        .project({
                            "movie_id",
                            "CONCAT(movie_id, movie_description) AS movie_description_processed",
                        })
                        .project(
                            {"movie_id",
                             "chatgpt_server(movie_description_processed, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description_summerized"})
                        .planNode(),
                    {"user_id",
                     "movie_id",
                     "user_description_summerized",
                     "movie_description_summerized"})
                .project(
                    {"user_id",
                     "movie_id",
                     "chatgpt_recommender(user_description_summerized, movie_description_summerized, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.') AS result"});
      }

      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readMoviewDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      std::shared_ptr<OutputStat> userStat =
          std::make_shared<OutputStat>(OutputStat(numUser, 2));
      std::shared_ptr<OutputStat> movieStat =
          std::make_shared<OutputStat>(OutputStat(numMovies, 2));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStat);
      Source movieSrc =
          Source(readMoviewDataPlanNodeId, Source::Type::FILE, movieStat);
      cataLog.addSource(std::make_shared<Source>(userSrc));
      cataLog.addSource(std::make_shared<Source>(movieSrc));
    } else if (model == "llm2" || model == "llm2-op") {
      std::vector<std::string> userDataPaths =
          getFilePathsFromDir("/home/velox/resources/data/parquet/llm_mr/user");
      std::vector<std::string> movieDataPaths = getFilePathsFromDir(
          "/home/velox/resources/data/parquet/llm_mr/movie");
      auto userDataRowType =
          ROW({"user_id", "description"}, {INTEGER(), VARCHAR()});
      auto movieDataRowType =
          ROW({"id",
               "title",
               "description",
               "popularity",
               "vote_average",
               "vote_count",
               "spoken_languages"},
              {INTEGER(),
               VARCHAR(),
               VARCHAR(),
               REAL(),
               REAL(),
               INTEGER(),
               VARCHAR()});

      std::string llmDataStatsFilePath =
          "/home/velox/resources/data/parquet/llm_mr/llm_mr_statistics.txt";
      std::ifstream llmStatistics(llmDataStatsFilePath);
      if (!llmStatistics) {
        throw std::runtime_error(
            "Unable to open file: " + llmDataStatsFilePath);
      }
      // TODO: need more smart way to do this
      std::string line1, line2;
      int numUser, numMovies;
      std::getline(llmStatistics, line1);
      std::getline(llmStatistics, line2);
      numUser = std::stoi(line1);
      numMovies = std::stoi(line2);

      core::PlanNodeId readUserDataPlanNodeId;
      core::PlanNodeId readMoviewDataPlanNodeId;

      if (model == "llm2") {
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(userDataRowType, {}, "")
                .capturePlanNodeId(readUserDataPlanNodeId)
                .project(
                    {"CAST(user_id AS VARCHAR) as user_id",
                     "description AS user_description"})
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator, pool_.get())
                        .tableScan(movieDataRowType, {}, "")
                        .capturePlanNodeId(readMoviewDataPlanNodeId)
                        .project({
                            "CAST(id AS VARCHAR) AS movie_id",
                            "description AS movie_description",
                            "title",
                            "llm_ffnn_minmax_scaler(convert_double_array_to_float_array(array_constructor(popularity, vote_average, vote_count))) AS movie_description_array",
                            "spoken_languages",
                        })
                        .planNode(),
                    {"user_id",
                     "movie_id",
                     "title",
                     "user_description",
                     "movie_description",
                     "spoken_languages",
                     "movie_description_array"})
                .project({
                    "user_id",
                    "movie_id",
                    "spoken_languages",
                    "movie_description_array",
                    "CONCAT(user_id, user_description) AS user_description_processed",
                    "rag(hf_minilm_embedding_extractor(title)) AS movie_description_processed"
                    //  "CONCAT(movie_id, movie_description) AS
                    //  movie_description_processed"
                })
                .project(
                    {"user_id",
                     "movie_id",
                     "spoken_languages",
                     "movie_description_array",
                     "chatgpt_server(user_description_processed, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description_summerized",
                     "chatgpt_server(movie_description_processed, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description_summerized"})
                .project(
                    {"user_id",
                     "movie_id",
                     "spoken_languages",
                     "movie_description_array",
                     "chatgpt_recommender(user_description_summerized, movie_description_summerized, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.') AS result"})
                .project({
                    "user_id",
                    "movie_id",
                    "spoken_languages",
                    "result",
                    "argmax(softmax(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array))))))) AS trending_prediction",
                })
                .filter("spoken_languages LIKE '\%English\%'")
                .filter("trending_prediction = 1");
      } else if (model == "llm2-op") {
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(userDataRowType, {}, "")
                .capturePlanNodeId(readUserDataPlanNodeId)
                .project(
                    {"CAST(user_id AS VARCHAR) as user_id",
                     "description AS user_description"})
                .project(
                    {"user_id",
                     "CONCAT(user_id, user_description) AS user_description_processed"})
                .project({
                    "user_id",
                    "chatgpt_server(user_description_processed, 'Please summarize the users description. The following are the average ratings given by users to movies in each genre.') AS user_description_summerized",
                })
                .nestedLoopJoin(
                    PlanBuilder(planNodeIdGenerator, pool_.get())
                        .tableScan(movieDataRowType, {}, "")
                        .capturePlanNodeId(readMoviewDataPlanNodeId)
                        .filter("spoken_languages LIKE '\%English\%'")
                        .project({
                            "CAST(id AS VARCHAR) AS movie_id",
                            "hf_minilm_embedding_extractor(title) as title_embed",
                            "description AS movie_description",
                            "llm_ffnn_minmax_scaler(convert_double_array_to_float_array(array_constructor(popularity, vote_average, vote_count))) AS movie_description_array",
                            "spoken_languages",
                        })
                        .project(
                            {"movie_id",
                             "title_embed",
                             "movie_description",
                             "argmax(softmax(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array))))))) AS trending_prediction",
                             "spoken_languages"})
                        .filter("trending_prediction = 1")
                        .project({
                            "movie_id",
                            "rag(title_embed) AS movie_description_processed",
                        })
                        .project(
                            {"movie_id",
                             "chatgpt_server(movie_description_processed, 'Please summarize the movies description. The following are the detailed information of the movie.') AS movie_description_summerized"})
                        .planNode(),
                    {"user_id",
                     "movie_id",
                     "user_description_summerized",
                     "movie_description_summerized"})
                .project(
                    {"user_id",
                     "movie_id",
                     "chatgpt_recommender(user_description_summerized, movie_description_summerized, 'Given the user description and movie description, please return a recommendation score from 0-5 and explain the reason? Your response should be formatted as recommendation score and reason.') AS result"});
      }

      cataLog.setIdAddressMap(
          readUserDataPlanNodeId,
          userDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readMoviewDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      std::shared_ptr<OutputStat> userStat =
          std::make_shared<OutputStat>(OutputStat(numUser, 2));
      std::shared_ptr<OutputStat> movieStat =
          std::make_shared<OutputStat>(OutputStat(numMovies, 2));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStat);
      Source movieSrc =
          Source(readMoviewDataPlanNodeId, Source::Type::FILE, movieStat);
      cataLog.addSource(std::make_shared<Source>(userSrc));
      cataLog.addSource(std::make_shared<Source>(movieSrc));
    } else if (model == "llm3-op" || model == "llm3") {
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
      std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

      if (dataDirPrefix == "") {
        // use default value:
        dataDirPrefix = "/home/velox/resources/data/parquet/movielens/final/";
      }
      std::vector<std::string> movieDataPaths =
          getFilePathsFromDir(dataDirPrefix + "movie");
      int movieNumRows, movieNumCols;
      readDataStats(
          dataDirPrefix + "movie_stats.txt", movieNumRows, movieNumCols);

      core::PlanNodeId readMoviewDataPlanNodeId;
      if (model == "llm3") {
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(movieDataRowType, {}, "")
                .capturePlanNodeId(readMoviewDataPlanNodeId)
                .project(
                    {"m_movie_id",
                     "m_genres",
                     "chatgpt_server(m_title, 'Please return the country where this movie was produced:') AS country_of_produce",
                     "chatgpt_server(m_title, 'Please return the year this movie was released:') AS year_of_release"})
                .filter("m_genres LIKE '\%Action\%'");
      } else if (model == "llm3-op") {
        myPlan =
            PlanBuilder(planNodeIdGenerator, pool_.get())
                .tableScan(movieDataRowType, {}, "")
                .capturePlanNodeId(readMoviewDataPlanNodeId)
                .filter("m_genres LIKE '\%Action\%'")
                .project(
                    {"m_movie_id",
                     "chatgpt_server(m_title, 'Please return the country where this movie was produced:') AS country_of_produce",
                     "chatgpt_server(m_title, 'Please return the year this movie was released:') AS year_of_release"});
      }
      cataLog.setIdAddressMap(
          readMoviewDataPlanNodeId,
          movieDataPaths,
          dwio::common::FileFormat::PARQUET);
      std::shared_ptr<OutputStat> movieStat =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMoviewDataPlanNodeId, Source::Type::FILE, movieStat);
      cataLog.addSource(std::make_shared<Source>(movieSrc));
    } else if (model == "two-tower") {
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
                   //  "concat4(concat3(concat2(concat1(user_id_embed,gender),age),occupation),
                   //  user_mean_rating) as user_tower_features",
                   "concat(user_id_embed, gender, age, occupation, user_mean_rating) as user_tower_features",
                   //  "concat2_2(concat2_1(movie_id_embed, genres),
                   //  movie_mean_rating) as movie_tower_features"
                   "concat(movie_id_embed, genres, movie_mean_rating) as movie_tower_features"})
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
    } else if (model == "fraud") {
      auto orderDataRowType =
          ROW({"o_order_id", "o_customer_sk", "o_weekday", "o_date"},
              {INTEGER(), INTEGER(), VARCHAR(), VARCHAR()});

      auto transactionDataRowType = ROW(
          {"t_amount", "t_sender", "t_receiver", "transaction_id", "t_time"},
          {REAL(), INTEGER(), VARCHAR(), BIGINT(), VARCHAR()});

      auto customerDataRowType =
          ROW({"c_customer_sk",
               "c_address_num",
               "c_cust_flag",
               "c_birth_year",
               "c_birth_country"},
              {INTEGER(), INTEGER(), INTEGER(), INTEGER(), INTEGER()});
      std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");
      if (dataDirPrefix == "") {
        // use default value:
        dataDirPrefix = "/home/velox/resources/data/parquet/fraud/50_mb/";
      }

      std::vector<std::string> orderDataPaths =
          getFilePathsFromDir(dataDirPrefix + "order");
      std::vector<std::string> transactionDataPaths =
          getFilePathsFromDir(dataDirPrefix + "financial_transactions");
      std::vector<std::string> customerDataPaths =
          getFilePathsFromDir(dataDirPrefix + "customer");

      int orderNumRows, orderNumCols, transactionNumRows, transactionNumCols,
          customerNumRows, customerNumCols;

      readDataStats(
          dataDirPrefix + "order_stats.txt", orderNumRows, orderNumCols);
      readDataStats(
          dataDirPrefix + "financial_transactions_stats.txt",
          transactionNumRows,
          transactionNumCols);
      readDataStats(
          dataDirPrefix + "customer_stats.txt",
          customerNumRows,
          customerNumCols);

      std::cout << "[INFO] orderNumRows: " << orderNumRows
                << ", orderNumCols: " << orderNumCols << std::endl;
      std::cout << "[INFO] transactionNumRows: " << transactionNumRows
                << ", transactionNumCols: " << transactionNumCols << std::endl;
      std::cout << "[INFO] customerNumRows: " << customerNumRows
                << ", customerNumCols: " << customerNumCols << std::endl;

      PlanNodeId readOrderDataPlanNodeId;
      PlanNodeId readTransactionDataPlanNodeId;
      PlanNodeId readCustomerDataPlanNodeId;

      myPlan =
          exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
              // .values({orderRowVector})
              .tableScan(orderDataRowType)
              .capturePlanNodeId(readOrderDataPlanNodeId)
              .project(
                  {"o_customer_sk",
                   "o_order_id",
                   "date_to_timestamp_1(o_date) AS o_timestamp"})
              .filter("o_timestamp IS NOT NULL")
              .filter("is_weekday(o_timestamp) = 1")
              .partialAggregation(
                  {"o_customer_sk"},
                  {"count(o_order_id) as total_order",
                   "max(o_timestamp) as o_last_order_time"})
              .finalAggregation()
              .hashJoin(
                  {"o_customer_sk"},
                  {"t_sender"},
                  exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      // .values({transactionRowVector})
                      .tableScan(transactionDataRowType)
                      .capturePlanNodeId(readTransactionDataPlanNodeId)
                      .project(
                          {"t_amount",
                           "t_sender",
                           "t_receiver",
                           "transaction_id",
                           "date_to_timestamp_2(t_time) as t_timestamp"})
                      .filter("t_timestamp IS NOT NULL")
                      .planNode(),
                  "",
                  {"o_customer_sk",
                   "total_order",
                   "o_last_order_time",
                   "transaction_id",
                   "t_amount",
                   "t_timestamp"},
                  core::JoinType::kInner)
              .project(
                  {"o_customer_sk",
                   "total_order",
                   "transaction_id",
                   "t_amount",
                   "t_timestamp",
                   "time_diff_in_days(o_last_order_time, t_timestamp) as time_diff"})
              .filter("time_diff <= 500")
              .project(
                  {"o_customer_sk",
                   "transaction_id",
                   "get_transaction_features(total_order, t_amount, time_diff, t_timestamp) as transaction_features"})
              .filter("xgboost_fraud_transaction(transaction_features) >= 0.5")
              .hashJoin(
                  {"o_customer_sk"},
                  {"c_customer_sk"},
                  exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      // .values({customerRowVector})
                      .tableScan(customerDataRowType)
                      .capturePlanNodeId(readCustomerDataPlanNodeId)
                      .project(
                          {"c_customer_sk",
                           "c_address_num",
                           "c_cust_flag",
                           "c_birth_country",
                           "get_age(c_birth_year) as c_age"})
                      .project(
                          {"c_customer_sk",
                           "get_customer_features(c_address_num, c_cust_flag, c_birth_country, c_age) as customer_features"})
                      .planNode(),
                  "",
                  {"transaction_id",
                   "transaction_features",
                   "customer_features"})
              .project(
                  {"transaction_id",
                   "concat(customer_features, transaction_features) AS all_features"})
              .project(
                  {"transaction_id",
                   "all_features",
                   "softmax(mat_vector_add1_6(mat_mul1_5(relu(mat_vector_add1_4(mat_mul1_3(relu(mat_vector_add1_2(mat_mul1_1(all_features))))))))) AS fraudulent_probs"})
              .filter("get_binary_class(fraudulent_probs) = 1")
              .filter("xgboost_fraud_predict(all_features) >= 0.5")
              .project({"transaction_id"});
      cataLog.setIdAddressMap(
          readOrderDataPlanNodeId,
          orderDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readTransactionDataPlanNodeId,
          transactionDataPaths,
          dwio::common::FileFormat::PARQUET);
      cataLog.setIdAddressMap(
          readCustomerDataPlanNodeId,
          customerDataPaths,
          dwio::common::FileFormat::PARQUET);

      cataLog.setFileSchema(readOrderDataPlanNodeId, orderDataRowType);
      cataLog.setFileSchema(
          readTransactionDataPlanNodeId, transactionDataRowType);
      cataLog.setFileSchema(readCustomerDataPlanNodeId, customerDataRowType);

      std::shared_ptr<OutputStat> orderStat =
          std::make_shared<OutputStat>(OutputStat(orderNumRows, orderNumCols));
      Source orderSrc =
          Source(readOrderDataPlanNodeId, Source::Type::FILE, orderStat);
      cataLog.addSource(std::make_shared<Source>(orderSrc));
      std::shared_ptr<OutputStat> transactionStat =
          std::make_shared<OutputStat>(
              OutputStat(transactionNumRows, transactionNumCols));
      Source transactionSrc = Source(
          readTransactionDataPlanNodeId, Source::Type::FILE, transactionStat);
      cataLog.addSource(std::make_shared<Source>(transactionSrc));
      std::shared_ptr<OutputStat> customerStat = std::make_shared<OutputStat>(
          OutputStat(customerNumRows, customerNumCols));
      Source customerSrc =
          Source(readCustomerDataPlanNodeId, Source::Type::FILE, customerStat);
      cataLog.addSource(std::make_shared<Source>(customerSrc));

    } else if (model == "ml-q1" || model == "ml-q2" || model == "ml-q3") {
      myPlan =
          setupMovielensDBQuery(model, cataLog, pool_, planNodeIdGenerator);
    } else {
      throw std::runtime_error(fmt::format("Non-supported model: {}", model));
    }

    return myPlan;
  }
  void print_action(
      std::pair<std::string, std::string>& testAction,
      PlanState& planState,
      PlanBuilder& myPlan) {
    std::cout << "################## print info #################" << std::endl;
    std::cout << "[INFO] Action taken:" << testAction.first << " "
              << testAction.second << std::endl;
    std::cout << "[INFO] All possible actions:" << std::endl;
    for (auto entry : planState.actionsPair) {
      std::cout << entry.first << ": " << entry.second << std::endl;
    }
    std::cout << "[DEBUG] execution plan(After action taken): \n"
              << myPlan.planNode()->toString(true, true) << std::endl;
    std::cout << "################## print info end #################"
              << std::endl;
  }
  std::pair<std::string, std::string> getMostSparsityAction(
      PlanState& planState,
      CataLog& cataLog) {
    // Find the action with the most sparsity
    std::pair<std::string, std::string> mostSparsityAction;
    double maxSparsity = 1.1;
    for (const auto& action : planState.actionsPair) {
      if (action.second.size() < 1) {
        continue; // Skip actions with no parameters
      }
      if (action.second[0] != "Dense2SparseRewriteAction") {
        continue; // Skip other action
      }
      auto inputName = getInputExprName(action.first);
      double sparsity = cataLog.getSparsity(inputName);
      std::cout << "inputName: " << inputName << ", sparsity: " << sparsity
                << std::endl;
      if (sparsity < maxSparsity) {
        maxSparsity = sparsity;
        mostSparsityAction.first = action.first;
        mostSparsityAction.second = action.second[0];
      }
    }
    std::cout << "The most Sparsity Action:\n"
              << mostSparsityAction.first << ": " << mostSparsityAction.second
              << "Sparsity ratio:" << maxSparsity << std::endl;
    return mostSparsityAction;
  }

  void testAblationStudy(
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
      computationStr = registerFFNNFunctions(
          first_layer_output_size,
          second_layer_output_size,
          input_features_size,
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
    } else if (
        model == "llm" || model == "llm-op" || model == "llm3" ||
        model == "llm3-op") {
      registerLLMFunctions(64, 2, 3, cataLog, pool_);
    } else if (model == "llm2" || model == "llm2-op") {
      registerLLMFunctions(64, 2, 3, cataLog, pool_);
      registerLLM2Functions(cataLog, pool_);
    } else if (model == "fraud") {
      registerFraudDetectionFunctions(9, cataLog, pool_);
    } else if (model == "ml-q1") {
      registerTwoTowerFunc(cataLog, pool_, false /*isVerticalPartition*/);
      registerMLTrendingModelFunctions(cataLog, pool_);
    } else if (model == "ml-q2") {
      registerMLTrendingModelFunctions(cataLog, pool_);
      registerMLInterestMovieModelFunctions(cataLog, pool_);
      registerMLMovieTagEncoderModelFunctions(cataLog, pool_);
      registerMLDLRMModelFunctions(cataLog, pool_);
    } else if (model == "ml-q3") {
      registerMLQ3UserMovieInterestModelFunctions(cataLog, pool_);
      registerMLQ3UserMovieRatingModelFunctions(cataLog, pool_);
      registerMLMovieTagEncoderModelFunctions(cataLog, pool_);
      registerMLMovieTagEncoderModelFunctions1(cataLog, pool_);
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

    std::cout << "[INFO] All possible actions:" << std::endl;
    for (auto entry : planState.actionsPair) {
      std::cout << entry.first << ": " << entry.second << std::endl;
    }
    std::pair<std::string, std::string> mostSparsityAction =
        getMostSparsityAction(planState, cataLog);

    std::string queryOptType = getEnvVar("CD_VELOX_QUERY_OPT_TYPE");
    std::cout << "QueryOptType:" << queryOptType << std::endl;
    std::pair<std::string, std::string> testAction;

    if (queryOptType == "mlq1-fusion" || queryOptType == "mlq1-optimized") {
      testAction = std::make_pair(
          "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(ROW[\"movie_description_array\"]))))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);

      planNode = myPlan.planNode();

      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(ROW[\"user_tower_features\"]))))))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);

      planNode = myPlan.planNode();

      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(ROW[\"movie_tower_features\"]))))))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);

      planNode = myPlan.planNode();

      planState.getPossibleActions(planNode, cataLog);

    } else if (
        queryOptType == "mlq1-fusion-gpu" ||
        queryOptType == "mlq1-optimized-gpu") {
      testAction = std::make_pair(
          "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(ROW[\"movie_description_array\"]))))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);

      planNode = myPlan.planNode();

      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(ROW[\"user_tower_features\"]))))))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);

      planNode = myPlan.planNode();

      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(ROW[\"movie_tower_features\"]))))))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);

      planNode = myPlan.planNode();

      planState.getPossibleActions(planNode, cataLog);
    }
    if (queryOptType == "mlq2-mul2join" || queryOptType == "mlq2-optimized") {
      testAction =
          std::make_pair("mat_mul10_1", "Mul2JoinAggHorizontalRewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);
    }
    if (queryOptType == "mlq2-fusion" || queryOptType == "mlq2-optimized") {
      testAction = std::make_pair(
          "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(ROW[\"top_mlp_input\"])))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(ROW[\"u_final_interest_features\"])))))))",
          "MultiLayerUDF2TorchNNRewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(ROW[\"m_trending_features\"]))))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "relu(mat_vector_add11_2(mat_mul11_1(ROW[\"mt_relevance_score\"])))",
          "MultiLayerUDF2TorchNNRewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);
    }

    if (queryOptType == "mlq2-fusion-gpu" ||
        queryOptType == "mlq2-optimized-gpu") {
      testAction = std::make_pair(
          "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(ROW[\"top_mlp_input\"])))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(ROW[\"u_final_interest_features\"])))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(ROW[\"m_trending_features\"]))))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      testAction = std::make_pair(
          "relu(mat_vector_add11_2(mat_mul11_1(ROW[\"mt_relevance_score\"])))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);
    }

    std::cout << "[DEBUG] Initial execution plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;

    if (queryOptType == "mlq3-mul2join" || queryOptType == "mlq3-optimized1" ||
        queryOptType == "mlq3-dense2sparse") {
      if (queryOptType == "mlq3-mul2join") {
        testAction =
            std::make_pair("mat_mul20_1", "Mul2JoinAggHorizontalRewriteAction");
      } else if (
          queryOptType == "mlq3-optimized1" ||
          queryOptType == "mlq3-optimized-gpu") {
        testAction =
            std::make_pair("mat_mul10_1", "Mul2JoinAggHorizontalRewriteAction");
      } else if (queryOptType == "mlq3-dense2sparse") {
        testAction = std::make_pair(
            "mat_mul20_1(ROW[\"mt_relevance_score\"])",
            "Dense2SparseRewriteAction");
      }

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);
      print_action(testAction, planState, myPlan);
    }
    if (queryOptType == "mlq3-dense2sparse1" ||
        queryOptType == "mlq3-dense2sparse") {
      testAction = std::make_pair(
          "mat_mul10_1(ROW[\"mt_relevance_score\"])",
          "Dense2SparseRewriteAction");

      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);
      print_action(testAction, planState, myPlan);
    }

    if (queryOptType == "mlq3-fusion" || queryOptType == "mlq3-optimized" ||
        queryOptType == "mlq3-optimized-d2s") {
      testAction = std::make_pair(
          "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(ROW[\"model_features\"])))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      print_action(testAction, planState, myPlan);

      testAction = std::make_pair(
          "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(ROW[\"model_features\"])))))))))",
          "MultiLayerUDF2TorchNNRewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      print_action(testAction, planState, myPlan);

      if (queryOptType == "mlq3-optimized-d2s") {
        testAction = std::make_pair(
            "mat_mul10_1(ROW[\"mt_relevance_score\"])",
            "Dense2SparseRewriteAction");

        planState.takeAction(
            planNode,
            nullptr,
            maker,
            myPlan,
            pool_,
            planNodeIdGenerator,
            {testAction},
            cataLog);

        planState.update(myPlan, cataLog);
        planNode = myPlan.planNode();
        planState.getPossibleActions(planNode, cataLog);
        print_action(testAction, planState, myPlan);
      }
    }

    if (queryOptType == "mlq3-fusion-gpu" ||
        queryOptType == "mlq3-optimized-gpu") {
      testAction = std::make_pair(
          "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(ROW[\"model_features\"])))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      print_action(testAction, planState, myPlan);

      testAction = std::make_pair(
          "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(ROW[\"model_features\"])))))))))",
          "MultiLayerUDF2TorchNNCUDARewriteAction");
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          myPlan,
          pool_,
          planNodeIdGenerator,
          {testAction},
          cataLog);

      planState.update(myPlan, cataLog);
      planNode = myPlan.planNode();
      planState.getPossibleActions(planNode, cataLog);

      print_action(testAction, planState, myPlan);
    }

    std::cout << "[INFO] All possible actions:" << std::endl;
    for (auto entry : planState.actionsPair) {
      std::cout << entry.first << ": " << entry.second << std::endl;
    }

    std::cout << "[DEBUG] final executed plan: \n"
              << myPlan.planNode()->toString(true, true) << std::endl;

    float executeTime =
        runPlanWithCataLog(pool_, 8, myPlan, cataLog, repeatRun, verbose);

    std::cout << "[INFO] Execution time: " << executeTime << std::endl;
    // auto serializedPlan = myPlan.planNode()->serialize();

    // std::cout << "[DEBUG] serialized plan: \n" << serializedPlan <<
    // std::endl;

    return;

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
        planState.clearTransformedExpr();
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
              runPlanWithCataLog(pool_, 8, myPlan, cataLog, repeatRun, verbose);
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
        auto latency =
            runPlanWithCataLog(pool_, 8, myPlan, cataLog, 4, verbose);
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

DEFINE_string(mode, "ablation", "Mode: ablation or benchmark");
DEFINE_string(model, "ffnn", "Model: ffnn, df, two-tower, llm");
DEFINE_bool(rewrite, true, "Whether  rewrite");
DEFINE_int32(num_repeat, 1, "Number of repeat run");
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
  AblationStudyTest demo;

  // available single benchmark mode: mul2joinAgg, udf2torchNN,
  // mul2joinAggHorizontal
  if (mode == "ablation") {
    demo.testAblationStudy(
        model, featureSize, numSample, repeatRun, blockSize, verbose);
  } else {
    // Benchmark a single rewrite action
    demo.testSingleRewrite(
        model,
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
