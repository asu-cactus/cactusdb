/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <folly/init/Init.h>
#include <pstl/pstl_config.h>
#include <torch/torch.h>
#include <cwchar>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/XGBoost.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include <ctime>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <locale>
#include "velox/functions/Udf.h"
#include <unordered_map>
#include <H5Cpp.h>

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;



class IsWeekday : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<int> results;

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto inputTimes = decodedArray->base()->as<FlatVector<int64_t>>();

    const int secondsInADay = 86400;
    for (int i = 0; i < rows.size(); i++) {
        int64_t timestamp = inputTimes->valueAt(i);

        std::time_t time = static_cast<std::time_t>(timestamp);
        std::tm* time_info = std::localtime(&time);
        int dayOfWeek = time_info->tm_wday;

        /*int64_t daysSinceEpoch = timestamp / secondsInADay;
        // Unix epoch (Jan 1, 1970) was a Thursday, so dayOfWeek for epoch is 4 (0=Sunday, 6=Saturday)
        int dayOfWeekEpoch = 4;  // Thursday
        // Calculate the current day of the week (0=Sunday, ..., 6=Saturday)
        int dayOfWeek = (daysSinceEpoch + dayOfWeekEpoch) % 7;*/

        // Return true if the day is Saturday (6) or Sunday (0)
        if (dayOfWeek == 0 || dayOfWeek == 6) {
            results.push_back(0);
        }
        else {
            results.push_back(1);
        }
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "is_weekday";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};


class GetAge : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<int> results;

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto birthYears = decodedArray->base()->as<FlatVector<int>>();

    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* localTime = std::localtime(&currentTime);
    int currentYear = 1900 + localTime->tm_year;

    for (int i = 0; i < rows.size(); i++) {
        int birthYear = birthYears->valueAt(i);
        results.push_back(currentYear - birthYear);
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "get_age";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};


class GetTransactionFeatures : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int secondsInADay = 86400;
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();
    BaseVector* base2 = args[2].get();
    BaseVector* base3 = args[3].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    auto totalOrders = decodedArray0->base()->as<FlatVector<int64_t>>();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();
    auto tAmounts = decodedArray1->base()->as<FlatVector<float>>();

    exec::LocalDecodedVector thirdHolder(context, *base2, rows);
    auto decodedArray2 = thirdHolder.get();
    auto timeDiffs = decodedArray2->base()->as<FlatVector<int64_t>>();

    exec::LocalDecodedVector fourthHolder(context, *base3, rows);
    auto decodedArray3 = fourthHolder.get();
    auto tTimestamps = decodedArray3->base()->as<FlatVector<int64_t>>();

    for (int i = 0; i < rows.size(); i++) {
        float totalOrder = (static_cast<float>(totalOrders->valueAt(i)))/79.0;
        float tAmount = (tAmounts->valueAt(i))/16048.0;
        float timeDiff = (static_cast<float>(timeDiffs->valueAt(i)))/729.0;
        int64_t tTimestamp = tTimestamps->valueAt(i);

        // Calculate day of week
        std::time_t time = static_cast<std::time_t>(tTimestamp);
        std::tm* time_info = std::localtime(&time);
        float dayOfWeek = (static_cast<float>(time_info->tm_wday))/6.0;

        // Calculate the number of days since Unix epoch
        float daysSinceEpoch = (static_cast<float>(tTimestamp / secondsInADay))/15338.0;

        std::vector<float> vec;
        vec.push_back(totalOrder);
        vec.push_back(tAmount);
        vec.push_back(timeDiff);
        vec.push_back(dayOfWeek);
        vec.push_back(daysSinceEpoch);

        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .argumentType("REAL")
                .argumentType("BIGINT")
                .argumentType("BIGINT")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_transaction_features";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class GetCustomerFeatures : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int secondsInADay = 86400;
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();
    BaseVector* base2 = args[2].get();
    BaseVector* base3 = args[3].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    auto cAddressNums = decodedArray0->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();
    auto cCustFlags = decodedArray1->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector thirdHolder(context, *base2, rows);
    auto decodedArray2 = thirdHolder.get();
    auto cBirthCountries = decodedArray2->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector fourthHolder(context, *base3, rows);
    auto decodedArray3 = fourthHolder.get();
    auto cAges = decodedArray3->base()->as<FlatVector<int>>();

    for (int i = 0; i < rows.size(); i++) {
        float cAddressNum = (static_cast<float>(cAddressNums->valueAt(i)))/35352.0;
        float cCustFlag = static_cast<float>(cCustFlags->valueAt(i));
        float cBirthCountry = (static_cast<float>(cBirthCountries->valueAt(i)))/211.0;
        float cAge = (static_cast<float>(cAges->valueAt(i)))/94.0;

        std::vector<float> vec;
        vec.push_back(cAddressNum);
        vec.push_back(cCustFlag);
        vec.push_back(cBirthCountry);
        vec.push_back(cAge);

        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_customer_features";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class TimeDiffInDays : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto inputTimes1 = decodedLeftArray->base()->as<FlatVector<int64_t>>();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto inputTimes2 = decodedRightArray->base()->as<FlatVector<int64_t>>();

    std::vector<int64_t> results;
    int secondsInADay = 86400;



    for (int i = 0; i < rows.size(); i++) {

        int64_t timestamp1 = inputTimes1->valueAt(i);
        int64_t timestamp2 = inputTimes2->valueAt(i);

        int64_t differenceInSeconds = std::abs(timestamp1 - timestamp2);
        int64_t differenceInDays = differenceInSeconds / secondsInADay;

        results.push_back(differenceInDays);
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int64_t>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .argumentType("BIGINT")
                .returnType("BIGINT")
                .build()};
  }

  static std::string getName() {
    return "time_diff_in_days";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class DateToTimestamp : public MLFunction {
 public:
 DateToTimestamp (const char* dateFormat_) {
     dateFormat = dateFormat_;
 }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    //auto inputStrings = args[0]->as<FlatVector<StringView>>();
    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();

    std::vector<int64_t> results;
    struct std::tm t = {};

    for (int i = 0; i < rows.size(); i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::string inputStr = std::string(val);

      std::istringstream ss(inputStr);
      ss >> std::get_time(&t, dateFormat);

      // Check if parsing was successful
      if (ss.fail()) {
          std::cerr << "Failed to parse date string " << inputStr << std::endl;
          results.push_back(0);
          continue;
      }

      // Convert tm struct to time_t (timestamp)
      time_t tt = mktime(&t);
      // Cast time_t to int64_t
      int64_t timestamp = static_cast<int64_t>(tt);
      results.push_back(timestamp);

    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int64_t>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("BIGINT")
                .build()};
  }

  static std::string getName() {
    return "date_to_timestamp";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

  private:
    const char* dateFormat;

};



class GetBinaryClass : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<int> results;

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto inputProbs = decodedArray->base()->as<ArrayVector>();
    auto inputProbsValues = inputProbs->elements()->asFlatVector<float>();

    for (int i = 0; i < rows.size(); i++) {
        int32_t offset = inputProbs->offsetAt(i);
        float prob_0 = inputProbsValues->valueAt(offset);
        float prob_1 = inputProbsValues->valueAt(offset + 1);
        if (std::isnan(prob_0) || std::isnan(prob_1)) {
            results.push_back(0);
        }
        else {
            int predicted_class = (prob_0 > prob_1) ? 0 : 1;
            results.push_back(predicted_class);
        }
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("ARRAY(REAL)")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "get_binary_class";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class FraudDetectionTest : public HiveConnectorTestBase {
 public:
  FraudDetectionTest() {
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
            ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~FraudDetectionTest() {}

  void registerFunctions(std::string modelFilePath, int numCols);
  void registerNNFunctions(int numCols);
  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath);

  RowVectorPtr getOrderData(std::string filePath);
  RowVectorPtr getTransactionData(std::string filePath);
  RowVectorPtr getCustomerData(std::string filePath);
  std::vector<std::vector<float>> loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint);
  std::unordered_map<std::string, int> getCountryMap();
  void testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath);

  ArrayVectorPtr parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols);

  RowVectorPtr writeDataToFile(std::string csvFilePath, int numRows, int numCols, int numDataSplits, 
                               std::string outPath, int dataBatchSize);


  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {
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

  std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};
  VectorMaker maker{pool_.get()};
};

void FraudDetectionTest::registerFunctions(std::string modelFilePath, int numCols) {
  
  exec::registerVectorFunction(
      "is_weekday",
      IsWeekday::signatures(),
      std::make_unique<IsWeekday>());
  std::cout << "Completed registering function for is_weekday" << std::endl;

  exec::registerVectorFunction(
        "date_to_timestamp_1",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%d"));
  std::cout << "Completed registering function for date_to_timestamp_1" << std::endl;

  exec::registerVectorFunction(
        "date_to_timestamp_2",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%dT%H:%M"));
  std::cout << "Completed registering function for date_to_timestamp_2" << std::endl;

  exec::registerVectorFunction(
        "time_diff_in_days",
        TimeDiffInDays::signatures(),
        std::make_unique<TimeDiffInDays>());
  std::cout << "Completed registering function for time_diff_in_days" << std::endl;

  exec::registerVectorFunction(
          "get_transaction_features",
          GetTransactionFeatures::signatures(),
          std::make_unique<GetTransactionFeatures>());
  std::cout << "Completed registering function for get_transaction_features" << std::endl;

  exec::registerVectorFunction(
            "get_customer_features",
            GetCustomerFeatures::signatures(),
            std::make_unique<GetCustomerFeatures>());
  std::cout << "Completed registering function for get_customer_features" << std::endl;

  exec::registerVectorFunction(
          "get_age",
          GetAge::signatures(),
          std::make_unique<GetAge>());
  std::cout << "Completed registering function for get_age" << std::endl;

  exec::registerVectorFunction(
        "concat_vectors2",
        Concat::signatures(),
        std::make_unique<Concat>(4, 5));

  exec::registerVectorFunction(
            "get_binary_class",
            GetBinaryClass::signatures(),
            std::make_unique<GetBinaryClass>());
  std::cout << "Completed registering function for is_anomalous" << std::endl;

  std::string xgboost_fraud_model_path = "resources/model/fraud_xgboost_9_1600";
  exec::registerVectorFunction(
        "xgboost_fraud_predict",
        TreePrediction::signatures(),
        std::make_unique<ForestPrediction>(xgboost_fraud_model_path, 9, true));
  std::cout << "Completed registering function for xgboost_fraud_predict" << std::endl;

  std::string xgboost_fraud_transaction_path = "resources/model/fraud_xgboost_5_16";
    exec::registerVectorFunction(
          "xgboost_fraud_transaction",
          TreePrediction::signatures(),
          std::make_unique<ForestPrediction>(xgboost_fraud_transaction_path, 5, true));
    std::cout << "Completed registering function for xgboost_fraud_transaction" << std::endl;

}


void FraudDetectionTest::registerNNFunctions(int numCols) {

  std::vector<std::vector<float>> w1 = loadHDF5Array("resources/model/fraud_dnn_weights.h5", "fc1.weight", 0);
  std::vector<std::vector<float>> b1 = loadHDF5Array("resources/model/fraud_dnn_weights.h5", "fc1.bias", 0);
  std::vector<std::vector<float>> w2 = loadHDF5Array("resources/model/fraud_dnn_weights.h5", "fc2.weight", 0);
  std::vector<std::vector<float>> b2 = loadHDF5Array("resources/model/fraud_dnn_weights.h5", "fc2.bias", 0);
  std::vector<std::vector<float>> w3 = loadHDF5Array("resources/model/fraud_dnn_weights.h5", "fc3.weight", 0);
  std::vector<std::vector<float>> b3 = loadHDF5Array("resources/model/fraud_dnn_weights.h5", "fc3.bias", 0);

  auto itemNNweight1Vector = maker.arrayVector<float>(w1, REAL());
  auto itemNNweight2Vector = maker.arrayVector<float>(w2, REAL());
  auto itemNNweight3Vector = maker.arrayVector<float>(w3, REAL());
  auto itemNNBias1Vector = maker.arrayVector<float>(b1, REAL());
  auto itemNNBias2Vector = maker.arrayVector<float>(b2, REAL());
  auto itemNNBias3Vector = maker.arrayVector<float>(b3, REAL());

  exec::registerVectorFunction(
      "mat_mul_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight1Vector->elements()->values()->asMutable<float>()),
          numCols,
          32));

  exec::registerVectorFunction(
      "mat_vector_add_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias1Vector->elements()->values()->asMutable<float>()), 32));

  exec::registerVectorFunction(
      "mat_mul_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight2Vector->elements()->values()->asMutable<float>()),
          32,
          16));

  exec::registerVectorFunction(
      "mat_vector_add_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias2Vector->elements()->values()->asMutable<float>()), 16));

  exec::registerVectorFunction(
      "mat_mul_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight3Vector->elements()->values()->asMutable<float>()),
          16,
          2));

  exec::registerVectorFunction(
      "mat_vector_add_3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias3Vector->elements()->values()->asMutable<float>()), 2));

  exec::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(),
          {},
          true);

  exec::registerVectorFunction(
      "softmax", Softmax::signatures(), std::make_unique<Softmax>());

}



ArrayVectorPtr FraudDetectionTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

    int size = numRows * numCols;

    std::cout << "Loading tensor of size " << size << " from " << filePath << std::endl;

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<std::vector<float>> inputArrayVector;

    
    int index = 0;
    
    std::string line;
    
    while (numRows--) { // Read a line from the file

        std::vector<float> curRow(numCols);
	
        std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
						    //
            float number = std::stof(numberStr);    // Convert the string to float

	        if (colIndex < numCols)					    

                curRow[colIndex] = number;

	        colIndex ++;

        }

	    inputArrayVector.push_back(curRow);
    }

    file.close();

    ArrayVectorPtr tensor = maker.arrayVector<float>(inputArrayVector);
    
    return tensor;

}

RowVectorPtr FraudDetectionTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
                                                 int numDataSplits, std::string outPath, int dataBatchSize) {

    ArrayVectorPtr inputArrayVector = parseCSVFile(maker, csvFilePath, numRows, numCols);
  
    std::vector<int32_t> indexVector;
  
    for (int i = 0; i < numRows; i++) {
  
       indexVector.push_back(i);
  
    }
  
    auto inputIndexVector = maker.flatVector<int32_t>(indexVector);
  
    auto inputRowVector = maker.rowVector({"row_id", "x"}, {inputIndexVector, inputArrayVector});

    auto dataConfig = std::make_shared<facebook::velox::dwrf::Config>();
  
    // affects the number of splits
    // number of bites in each stripe (collection of rows)
    // strip size should be <= split size (total_size / total splits)
    // to have the desired number of splits
    uint64_t kDataSizeKB = 512UL;

    uint32_t numDataRows = dataBatchSize;
  
    dataConfig->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 799 * kDataSizeKB);
  
    dataConfig->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, numDataRows);
  
    // auto dataFile = TempFilePath::create();

    // outPath = dataFile->path;

    writeToFile(outPath, {inputRowVector}, dataConfig);

    return inputRowVector;

}


std::unordered_map<std::string, int> FraudDetectionTest::getCountryMap() {
    std::unordered_map<std::string, int> countryMap;

    // Open the txt file
    std::string filePath = "resources/data/country_mapping.txt";
    std::ifstream file(filePath.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file!" << std::endl;
        exit(1);
    }

    std::string line;
    // Read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key;
        std::string value_str;

        // Get the key before the comma
        std::getline(ss, key, ',');

        // Get the value after the comma
        std::getline(ss, value_str);

        // Convert the string value to an integer
        int value = std::stoi(value_str);

        // Insert into the unordered_map
        countryMap[key] = value;
    }

    // Close the file
    file.close();

    return countryMap;
}


std::vector<std::vector<float>> FraudDetectionTest::loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint) {
    /*if (!std::filesystem::exists(filename)) {
          throw std::runtime_error("File not found: " + filename);
    }*/

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();

    // Get the number of dimensions
    int rank = dataspace.getSimpleExtentNdims();
    // std::cout << "Rank: " << rank << std::endl;

    // Allocate space for the dimensions
    std::vector<hsize_t> dims(rank);

    // Get the dataset dimensions
    dataspace.getSimpleExtentDims(dims.data(), nullptr);

    size_t rows;
    size_t cols;

    if (rank == 1) {
      rows = dims[0];
      cols = 1;
    } else if (rank == 2) {
      rows = dims[0];
      cols = dims[1];
    } else {
      throw std::runtime_error("Unsupported rank: " + std::to_string(rank));
    }

    //std::cout << "Num Rows: " << rows << ", Num Columns " << cols << std::endl;

    // Read data into a 1D vector
    std::vector<float> flatData(rows * cols);
    dataset.read(flatData.data(), H5::PredType::NATIVE_FLOAT);

    // Convert to 2D vector
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = flatData[i * cols + j];
            if (doPrint == 1)
                std::cout << result[i][j] << ", ";
        }
        if (doPrint == 1)
            std::cout << std::endl;
    }

    // Close the dataset and file
    dataset.close();
    file.close();

    return result;
}


RowVectorPtr FraudDetectionTest::getOrderData(std::string filePath) {

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
        std::cout << "Ignoring header: " << line << std::endl;
    }
    
    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);
	
        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == 0) {
                oOrderId.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 1) {
                oCustomerSk.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 2) {
                oWeekday.push_back(numberStr);
            }
            else if (colIndex == 3) {
                oDate.push_back(numberStr);
            }

	        colIndex ++;

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
         {oOrderIdVector, oCustomerSkVector, oWeekdayVector, oDateVector}
     );

     return orderRowVector;
}


RowVectorPtr FraudDetectionTest::getTransactionData(std::string filePath) {

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
        std::cout << "Ignoring header: " << line << std::endl;
    }

    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);

        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == 0) {
                tAmount.push_back(std::stof(numberStr));
            }
            else if (colIndex == 1) {
                tSender.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 2) {
                tReceiver.push_back(numberStr);
            }
            else if (colIndex == 3) {
                transactionId.push_back(std::stoll(numberStr));
            }
            else if (colIndex == 4) {
                tTime.push_back(numberStr);
           }

	        colIndex ++;

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
         {tAmountVector, tSenderVector, tReceiverVector, transactionIdVector, tTimeVector}
     );

     return transactionRowVector;
}


RowVectorPtr FraudDetectionTest::getCustomerData(std::string filePath) {

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
        std::cout << "Ignoring header: " << line << std::endl;
    }

    while (std::getline(file, line)) { // Read a line from the file

        //std::vector<float> curRow(numCols);

        //std::getline(file, line);

        std::istringstream iss(line); // Create an input string stream from the line

        std::string numberStr;

	    int colIndex = 0;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }

            size_t first = numberStr.find_first_not_of(' ');
            if (first == std::string::npos)
                numberStr = "0";
            else {
                size_t last = numberStr.find_last_not_of(' ');
                numberStr = numberStr.substr(first, (last - first + 1));
            }

            //std::cout << "Column Index: " << colIndex << ", Value: " << numberStr << std::endl;
            if (colIndex == 0) {
                cCustomerSk.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 2) {
                cAddrerssNum.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 5) {
                if (numberStr == "N")
                    cCustFlag.push_back(0);
                else
                    cCustFlag.push_back(1);
            }
            else if (colIndex == 8) {
                cBirthYear.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 9) {
                if (countryMap.find(numberStr) == countryMap.end()) {
                     // Key does not exist, insert it
                     countryMap[numberStr] = countryIndex;
                     cBirthCountry.push_back(countryIndex);
                     countryIndex ++;
                } else {
                    // Key exists, retrieve its value
                    cBirthCountry.push_back(countryMap[numberStr]);
                }
           }

	        colIndex ++;

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
         {"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_year", "c_birth_country"},
         {cCustomerSkVector, cAddrerssNumVector, cCustFlagVector, cBirthYearVector, cBirthCountryVector}
     );

     return customerRowVector;
}



void FraudDetectionTest::testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string orderFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();
                      
     std::string path = dataFile->path;

     RowVectorPtr orderRowVector = getOrderData(orderFilePath);
     std::cout << "orderRowVector data generated" << std::endl;
     RowVectorPtr transactionRowVector = getTransactionData("resources/data/financial_transactions.csv");
     std::cout << "transactionRowVector data generated" << std::endl;
     RowVectorPtr customerRowVector = getCustomerData("resources/data/customer.csv");
     std::cout << "customerRowVector data generated" << std::endl;

     registerNNFunctions(9);

     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

     auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({orderRowVector})
                         .project({"o_customer_sk", "o_order_id", "date_to_timestamp_1(o_date) AS o_timestamp"})
                         .filter("o_timestamp IS NOT NULL")
                         .filter("is_weekday(o_timestamp) = 1")
                         .singleAggregation({"o_customer_sk"}, {"count(o_order_id) as total_order", "max(o_timestamp) as o_last_order_time"})
                         .hashJoin({"o_customer_sk"},
                             {"t_sender"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({transactionRowVector})
                             .project({"t_amount", "t_sender", "t_receiver", "transaction_id", "date_to_timestamp_2(t_time) as t_timestamp"})
                             .filter("t_timestamp IS NOT NULL")
                             .planNode(),
                             "",
                             {"o_customer_sk", "total_order", "o_last_order_time", "transaction_id", "t_amount", "t_timestamp"},
                             core::JoinType::kInner
                         )
                         .project({"o_customer_sk", "total_order", "transaction_id", "t_amount", "t_timestamp", "time_diff_in_days(o_last_order_time, t_timestamp) as time_diff"})
                         .filter("time_diff <= 500")
                         .project({"o_customer_sk", "transaction_id", "get_transaction_features(total_order, t_amount, time_diff, t_timestamp) as transaction_features"})
                         .filter("xgboost_fraud_transaction(transaction_features) >= 0.5")
                         .hashJoin({"o_customer_sk"},
                             {"c_customer_sk"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({customerRowVector})
                             .project({"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_country", "get_age(c_birth_year) as c_age"})
                             .project({"c_customer_sk", "get_customer_features(c_address_num, c_cust_flag, c_birth_country, c_age) as customer_features"})
                             .planNode(),
                             "",
                             {"transaction_id", "transaction_features", "customer_features"}
                         )
                         .project({"transaction_id", "concat_vectors2(customer_features, transaction_features) AS all_features"})
                         .project({"transaction_id", "all_features", "softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(mat_vector_add_1(mat_mul_1(all_features))))))))) AS fraudulent_probs"})
                         .filter("get_binary_class(fraudulent_probs) = 1")
                         .filter("xgboost_fraud_predict(all_features) >= 0.5")
                         .project({"transaction_id"})
                         .planNode();
   
 
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << "Results Size: " << results->size() << std::endl;
    std::cout << results->toString(0, 5) << std::endl;
   
    std::cout << "Time for Executing with Real Data (sec): " << std::endl;

    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath) {

  std::cout << "Option is " << option << std::endl;
  registerFunctions(modelFilePath, 9);

  if (option == 0) {
      testingWithRealData(numDataSplits, dataBatchSize, numRows, numCols, orderDataFilePath, modelFilePath);
  }

  else
      std::cout << "Seleceted test is not supported" << std::endl;
}



DEFINE_int32(rewriteOrNot, 0, "0 for all tests");
DEFINE_int32(numDataSplits, 16, "number of data splits");
DEFINE_int32(numTreeSplits, 16, "number of tree splits");
DEFINE_int32(numTreeRows, 100, "batch size for processing trees");
DEFINE_int32(dataBatchSize, 100, "batch size for processing input samples");
DEFINE_int32(numRows, 10, "number of tuples in the dataset to be predicted");
DEFINE_int32(numCols, 9, "number of columns in the dataset to be predicted");
DEFINE_string(dataFilePath, "resources/data/creditcard_test.csv", "path to input dataset to be predicted");
DEFINE_string(modelFilePath, "resources/model/fraud_xgboost_1600_8", "path to the model used for prediction");
DEFINE_string(orderDataFilePath, "resources/data/order.csv", "path to input order dataset");

int main(int argc, char** argv) {

  setlocale(LC_TIME, "C");

  folly::init(&argc, &argv, false);

  memory::MemoryManager::initialize({});

  int option = FLAGS_rewriteOrNot;
  int numDataSplits = FLAGS_numDataSplits;
  int numTreeSplits = FLAGS_numTreeSplits;
  int numTreeRows = FLAGS_numTreeRows;
  int dataBatchSize = FLAGS_dataBatchSize;
  int numRows = FLAGS_numRows;
  int numCols = FLAGS_numCols;
  std::string dataFilePath = FLAGS_dataFilePath;
  std::string modelFilePath = FLAGS_modelFilePath;
  std::string orderDataFilePath = FLAGS_orderDataFilePath;

  FraudDetectionTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}, orderDataFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath, orderDataFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath, orderDataFilePath);

}
