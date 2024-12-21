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
#include <Eigen/Dense>
#include <sys/time.h>
#include <sys/resource.h>

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;




class CPUUtilizationTracker {
public:
    CPUUtilizationTracker() {
        startCPUTime = getCPUTime();
        startWallTime = std::chrono::steady_clock::now();
    }

    ~CPUUtilizationTracker() {
        auto endWallTime = std::chrono::steady_clock::now();
        double endCPUTime = getCPUTime();

        // Calculate total CPU time (user + system)
        double cpuTimeUsed = endCPUTime - startCPUTime;

        // Calculate real elapsed time (wall time)
        auto elapsedWallTime = std::chrono::duration_cast<std::chrono::milliseconds>(endWallTime - startWallTime).count() / 1000.0;

        // CPU utilization as a percentage
        double cpuUtilization = (cpuTimeUsed / elapsedWallTime) * 100.0;

        std::cout << "CPU Utilization for the method: " << cpuUtilization << "%\n";
    }

private:
    double startCPUTime;
    std::chrono::steady_clock::time_point startWallTime;

    // Method to get current CPU time (user + system)
    double getCPUTime() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        double userCPUTime = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
        double sysCPUTime = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
        return userCPUTime + sysCPUTime;
    }
};




class VectorAddition : public MLFunction {
 public:
  VectorAddition(int inputDims) {
    inputDims_ = inputDims;
  }

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
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = rightHolder->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input1Matrix(input1Values, numInput, inputDims_);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input2Matrix(input2Values, numInput, inputDims_);

    std::vector<std::vector<float>> results;

    for (int i = 0; i < numInput; i++) {
      //Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> vSum = input1Matrix.row(i) + input2Matrix.row(i);
      Eigen::VectorXf vSum = input1Matrix.row(i) + input2Matrix.row(i);
      std::vector<float> curVec(vSum.data(), vSum.data() + vSum.size());
      //std::vector<float> std_vector(vSum.data(), vSum.data() + vSum.size());
      results.push_back(curVec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "vector_addition";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int inputDims_;
};



class IsWorkingDay : public MLFunction {
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
    //auto inputTimes = decodedArray->base()->as<FlatVector<int64_t>>();

    //const int secondsInADay = 86400;
    for (int i = 0; i < rows.size(); i++) {
        int64_t timestamp = decodedArray->valueAt<int64_t>(i);

        std::time_t time = static_cast<std::time_t>(timestamp);
        std::tm* time_info = std::localtime(&time);
        int dayOfWeek = static_cast<int>(time_info->tm_wday);

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
    //output = maker.flatVector<int>(results);
    auto localResult = maker.flatVector<int>(results);
    context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "is_working_day";
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


class AgeDuringTransaction : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<int> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();

    exec::LocalDecodedVector vecHolder0(context, *base0, rows);
    auto decodedArray0 = vecHolder0.get();

    exec::LocalDecodedVector vecHolder1(context, *base1, rows);
    auto decodedArray1 = vecHolder1.get();
    //auto birthYears = decodedArray->base()->as<FlatVector<int>>();

    for (int i = 0; i < rows.size(); i++) {
        int64_t tTimestamp = decodedArray0->valueAt<int64_t>(i);
        int birthYear = decodedArray1->valueAt<int>(i);

        // Calculate year
        std::time_t time = static_cast<std::time_t>(tTimestamp);
        std::tm* time_info = std::localtime(&time);
        int year = 1900 + time_info->tm_year;

        results.push_back(year - birthYear);
    }

    VectorMaker maker{context.pool()};
    //output = maker.flatVector<int>(results);
    auto localResult = maker.flatVector<int>(results);
    context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .argumentType("INTEGER")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "age_during_transaction";
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

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    //auto totalOrders = decodedArray0->base()->as<FlatVector<int64_t>>();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();
    //auto tAmounts = decodedArray1->base()->as<FlatVector<float>>();

    for (int i = 0; i < rows.size(); i++) {
        float tAmount = (decodedArray0->valueAt<float>(i))/16048.0;
        int64_t tTimestamp = decodedArray1->valueAt<int64_t>(i);

        // Calculate day of week
        std::time_t time = static_cast<std::time_t>(tTimestamp);
        std::tm* time_info = std::localtime(&time);
        float day = (static_cast<float>(time_info->tm_mday))/31.0;
        float month = (static_cast<float>(time_info->tm_mon))/12.0;
        float year = (static_cast<float>(1900 + (time_info->tm_year)))/2011.0;
        float dayOfWeek = (static_cast<float>(time_info->tm_wday))/6.0;

        std::vector<float> vec;
        vec.push_back(tAmount);
        vec.push_back(day);
        vec.push_back(month);
        vec.push_back(year);
        vec.push_back(dayOfWeek);

        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("REAL")
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

    //int secondsInADay = 86400;
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();
    BaseVector* base2 = args[2].get();
    BaseVector* base3 = args[3].get();
    BaseVector* base4 = args[4].get();
    BaseVector* base5 = args[5].get();
    BaseVector* base6 = args[6].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    //auto cAddressNums = decodedArray0->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();
    //auto cCustFlags = decodedArray1->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector thirdHolder(context, *base2, rows);
    auto decodedArray2 = thirdHolder.get();
    //auto cAges = decodedArray3->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector fourthHolder(context, *base3, rows);
    auto decodedArray3 = fourthHolder.get();

    exec::LocalDecodedVector fifthHolder(context, *base4, rows);
    auto decodedArray4 = fifthHolder.get();

    exec::LocalDecodedVector sixthHolder(context, *base5, rows);
    auto decodedArray5 = sixthHolder.get();
    //auto cBirthCountries = decodedArray2->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector seventhHolder(context, *base6, rows);
    auto decodedArray6 = seventhHolder.get();


    for (int i = 0; i < rows.size(); i++) {
        float cAddressNum = (static_cast<float>(decodedArray0->valueAt<int>(i)))/35352.0;
        float cCustFlag = static_cast<float>(decodedArray1->valueAt<int>(i));
        float cBirthDay = (static_cast<float>(decodedArray2->valueAt<int>(i)))/31.0;
        float cBirthMonth = (static_cast<float>(decodedArray3->valueAt<int>(i)))/12.0;
        float cBirthYear = (static_cast<float>(decodedArray4->valueAt<int>(i)))/2002.0;
        float cBirthCountry = (static_cast<float>(decodedArray5->valueAt<int>(i)))/211.0;
        float cTransactionLimit = (static_cast<float>(decodedArray6->valueAt<float>(i)))/16116.0;

        std::vector<float> vec;
        vec.push_back(cAddressNum);
        vec.push_back(cCustFlag);
        vec.push_back(cBirthDay);
        vec.push_back(cBirthMonth);
        vec.push_back(cBirthYear);
        vec.push_back(cBirthCountry);
        vec.push_back(cTransactionLimit);

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
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .argumentType("REAL")
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

    for (int i = 0; i < rows.size(); i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::string inputStr = std::string(val);

      struct std::tm t = {};
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
      int64_t timestamp = static_cast<int64_t>(tt) + 25200;
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

    auto* flatResult = output->asFlatVector<int32_t>();

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto inputProbs = decodedArray->base()->as<ArrayVector>();
    auto inputProbsValues = inputProbs->elements()->asFlatVector<float>();

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->setNull(row, true);  // Handle nulls
        return;
      }

      // Get the array index, size, and offset for this row
      auto arrayIndex = decodedArray->index(row);
      auto offset = inputProbs->offsetAt(arrayIndex);

      float prob_0 = inputProbsValues->valueAt(offset);
      float prob_1 = inputProbsValues->valueAt(offset + 1);
      int predicted_class = (prob_0 > prob_1) ? 0 : 1;

      flatResult->set(row, predicted_class);  // Set result in output vector
    });

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

  RowVectorPtr getAccountData(std::string filePath);
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
      "is_working_day",
      IsWorkingDay::signatures(),
      std::make_unique<IsWorkingDay>());
  std::cout << "Completed registering function for is_working_day" << std::endl;

  exec::registerVectorFunction(
        "date_to_timestamp",
        DateToTimestamp::signatures(),
        std::make_unique<DateToTimestamp>("%Y-%m-%dT%H:%M"));
  std::cout << "Completed registering function for date_to_timestamp_2" << std::endl;


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
          "age_during_transaction",
          AgeDuringTransaction::signatures(),
          std::make_unique<AgeDuringTransaction>());
  std::cout << "Completed registering function for get_age" << std::endl;

  exec::registerVectorFunction(
            "get_binary_class",
            GetBinaryClass::signatures(),
            std::make_unique<GetBinaryClass>());
  std::cout << "Completed registering function for is_anomalous" << std::endl;

  std::string xgboost_fraud_transaction_path = "resources/model/fraud_xgboost_trans_5_1600";
    exec::registerVectorFunction(
          "xgboost_fraud_transaction",
          TreePrediction::signatures(),
          std::make_unique<ForestPrediction>(xgboost_fraud_transaction_path, 5, true));
    std::cout << "Completed registering function for xgboost_fraud_transaction" << std::endl;
 
  std::cout << "To register type for Tree" << std::endl;
    
  registerCustomType("tree_type", std::make_unique<TreeTypeFactories>());


  std::cout << "To register function for VeloxTreePrediction" << std::endl;

  exec::registerVectorFunction(
      "velox_decision_tree_predict",
      VeloxTreePrediction::signatures(),
      std::make_unique<VeloxTreePrediction>(5));

  std::cout << "To register function for VeloxTreeConstruction" << std::endl;

  exec::registerVectorFunction(
       "velox_decision_tree_construct",
       VeloxTreeConstruction::signatures(),
       std::make_unique<VeloxTreeConstruction>());

}


void FraudDetectionTest::registerNNFunctions(int numCols) {

  std::vector<std::vector<float>> w1 = loadHDF5Array("resources/model/fraud_detection.h5", "fc1.weight", 0);
  std::vector<std::vector<float>> b1 = loadHDF5Array("resources/model/fraud_detection.h5", "fc1.bias", 0);
  std::vector<std::vector<float>> w2 = loadHDF5Array("resources/model/fraud_detection.h5", "fc2.weight", 0);
  std::vector<std::vector<float>> b2 = loadHDF5Array("resources/model/fraud_detection.h5", "fc2.bias", 0);
  std::vector<std::vector<float>> w3 = loadHDF5Array("resources/model/fraud_detection.h5", "fc3.weight", 0);
  std::vector<std::vector<float>> b3 = loadHDF5Array("resources/model/fraud_detection.h5", "fc3.bias", 0);
  std::vector<std::vector<float>> w11 = loadHDF5Array("resources/model/fraud_detection.h5", "w11", 0);
  std::vector<std::vector<float>> w12 = loadHDF5Array("resources/model/fraud_detection.h5", "w12", 0);

  /*std::vector<std::vector<float>> b11;
  int n_row = b1.size();
  int n_col = b1[0].size();
  for (int i = 0; i < n_row; i++) {
      std::vector<float> temp;
      for (int j = 0; j < n_col; j++) {
          temp.push_back(b1[i][j]/2.0);
      }
      b11.push_back(temp);
  }*/

  auto itemNNweight1Vector = maker.arrayVector<float>(w1, REAL());
  auto itemNNweight2Vector = maker.arrayVector<float>(w2, REAL());
  auto itemNNweight3Vector = maker.arrayVector<float>(w3, REAL());
  auto itemNNBias1Vector = maker.arrayVector<float>(b1, REAL());
  auto itemNNBias2Vector = maker.arrayVector<float>(b2, REAL());
  auto itemNNBias3Vector = maker.arrayVector<float>(b3, REAL());
  auto itemNNweight11Vector = maker.arrayVector<float>(w11, REAL());
  auto itemNNweight12Vector = maker.arrayVector<float>(w12, REAL());
  //auto itemNNBias11Vector = maker.arrayVector<float>(b11, REAL());

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
          12));

  exec::registerVectorFunction(
      "mat_vector_add_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias2Vector->elements()->values()->asMutable<float>()), 12));

  exec::registerVectorFunction(
      "mat_mul_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight3Vector->elements()->values()->asMutable<float>()),
          12,
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

  exec::registerVectorFunction(
      "mat_mul_11",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight11Vector->elements()->values()->asMutable<float>()),
          7,
          32));

  exec::registerVectorFunction(
      "mat_mul_12",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight12Vector->elements()->values()->asMutable<float>()),
          5,
          32));

  /*exec::registerVectorFunction(
      "mat_vector_add_11",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias11Vector->elements()->values()->asMutable<float>()), 32));*/

  exec::registerVectorFunction(
          "vector_addition",
          VectorAddition::signatures(),
          std::make_unique<VectorAddition>(32));


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


RowVectorPtr FraudDetectionTest::getAccountData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> faCustomerSk;
    std::vector<float> faTransactionLimit;
    
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
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == 0) {
                faCustomerSk.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 1) {
                faTransactionLimit.push_back(std::stof(numberStr));
            }

	        colIndex ++;

        }

    }

    file.close();

     // Prepare Customer table
     auto faCustomerSkVector = maker.flatVector<int>(faCustomerSk);
     auto faTransactionLimitVector = maker.flatVector<float>(faTransactionLimit);
     auto accountRowVector = maker.rowVector(
         {"fa_customer_sk", "fa_transaction_limit"},
         {faCustomerSkVector, faTransactionLimitVector}
     );

     return accountRowVector;
}


RowVectorPtr FraudDetectionTest::getTransactionData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<float> tAmount;
    std::vector<int> tSender;
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
            else if (colIndex == 3) {
                double numberDouble = std::stod(numberStr);
                transactionId.push_back(static_cast<long long>(numberDouble));
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
     auto transactionIdVector = maker.flatVector<int64_t>(transactionId);
     auto tTimeVector = maker.flatVector<std::string>(tTime);
     auto transactionRowVector = maker.rowVector(
         {"transaction_id", "t_sender", "t_amount", "t_time"},
         {transactionIdVector, tSenderVector, tAmountVector, tTimeVector}
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
    std::vector<int> cBirthDay;
    std::vector<int> cBirthMonth;
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
            else if (colIndex == 6) {
                cBirthDay.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 7) {
                cBirthMonth.push_back(std::stoi(numberStr));
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
     auto cBirthDayVector = maker.flatVector<int>(cBirthDay);
     auto cBirthMonthVector = maker.flatVector<int>(cBirthMonth);
     auto cBirthYearVector = maker.flatVector<int>(cBirthYear);
     auto cBirthCountryVector = maker.flatVector<int>(cBirthCountry);
     auto customerRowVector = maker.rowVector(
         {"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year", "c_birth_country"},
         {cCustomerSkVector, cAddrerssNumVector, cCustFlagVector, cBirthDayVector, cBirthMonthVector, cBirthYearVector, cBirthCountryVector}
     );

     return customerRowVector;
}



void FraudDetectionTest::testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string orderFilePath, std::string modelFilePath) {

     std::vector<std::string> pathVectors;

  string forestFolderPath = "resources/model/fraud_xgboost_trans_5_1600";

  Forest::vectorizeForestFolder(forestFolderPath, pathVectors);

  int numTrees = pathVectors.size();

  auto treeModels = makeFlatVector<StringView> (pathVectors.size());

  for (int i = 0; i < numTrees; i++) {

      treeModels->set(i, StringView(pathVectors[i].c_str()));

  }


  auto treeIndexVector = maker.flatVector<int16_t>(numTrees);

   for (int i = 0; i < numTrees; i++) {

     treeIndexVector->set(i, i);

  }
  auto treeRowVector = maker.rowVector({"tree_id", "tree_path"}, {treeIndexVector, treeModels});

     auto dataFile = TempFilePath::create();
                      
     std::string path = dataFile->path;

     RowVectorPtr accountRowVector = getAccountData("resources/data/financial_account.csv");
     std::cout << "accountRowVector data generated" << std::endl;
     RowVectorPtr transactionRowVector = getTransactionData("resources/data/financial_transactions.csv");
     std::cout << "transactionRowVector data generated" << std::endl;
     RowVectorPtr customerRowVector = getCustomerData("resources/data/customer.csv");
     std::cout << "customerRowVector data generated" << std::endl;

     int totalRowsAccount = accountRowVector->size();
     int totalRowsTransaction = transactionRowVector->size();
     int totalRowsCustomer = customerRowVector->size();

     std::cout << "account data size: " << totalRowsAccount << ",  transaction data size: " << totalRowsTransaction << ",  customer data size: " << totalRowsCustomer << std::endl;

     int batch_counts = 16;
     int batchSizeAccount = totalRowsAccount / batch_counts;
     int batchSizeTransaction = totalRowsTransaction / batch_counts;
     int batchSizeCustomer = totalRowsCustomer / batch_counts;

     std::vector<RowVectorPtr> batchesAccount;
     std::vector<RowVectorPtr> batchesTransaction;
     std::vector<RowVectorPtr> batchesCustomer;

     for (int i = 0; i < batch_counts; ++i) {
         int start = i * batchSizeAccount;
         int end = (i == (batch_counts - 1)) ? totalRowsAccount : (i + 1) * batchSizeAccount;  // Handle remainder for last batch
         batchesAccount.push_back(std::dynamic_pointer_cast<RowVector>(accountRowVector->slice(start, end - start)));

         start = i * batchSizeTransaction;
         end = (i == (batch_counts - 1)) ? totalRowsTransaction : (i + 1) * batchSizeTransaction;  // Handle remainder for last batch
         batchesTransaction.push_back(std::dynamic_pointer_cast<RowVector>(transactionRowVector->slice(start, end - start)));

         start = i * batchSizeCustomer;
         end = (i == (batch_counts - 1)) ? totalRowsCustomer : (i + 1) * batchSizeCustomer;  // Handle remainder for last batch
         batchesCustomer.push_back(std::dynamic_pointer_cast<RowVector>(customerRowVector->slice(start, end - start)));
     }

     registerNNFunctions(12);
     CPUUtilizationTracker tracker;

     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

     auto myPlan1 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values(batchesTransaction)
                         //.localPartition({"t_sender"})
                         .project({"transaction_id", "t_sender", "t_amount", "date_to_timestamp(t_time) as t_timestamp"})
                         .filter("is_working_day(t_timestamp) = 1")
                         .project({"transaction_id", "t_sender", "t_timestamp", "get_transaction_features(t_amount, t_timestamp) as transaction_feature"})
                         .filter("xgboost_fraud_transaction(transaction_feature) >= 0.5")
                         .project({"transaction_id", "t_sender", "t_timestamp", "mat_mul_12(transaction_feature) AS dnn_part12"})
                         .hashJoin(
                             {"t_sender"},
                             {"c_customer_sk"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values(batchesCustomer)
                             //.localPartition({"c_customer_sk"})
                             .project({"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year", "c_birth_country"})
                             .hashJoin(
                                 {"c_customer_sk"},
                                 {"fa_customer_sk"},
                                 exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                                 .values(batchesAccount)
                                 //.localPartition({"fa_customer_sk"})
                                 .project({"fa_customer_sk", "fa_transaction_limit"})
                                 .planNode(),
                                 "",
                                 {"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year", "c_birth_country", "fa_transaction_limit"}
                             )
                             .project({"c_customer_sk", "c_birth_year", "mat_vector_add_1(mat_mul_11(get_customer_features(c_address_num, c_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, fa_transaction_limit))) as dnn_part11"})
                             .planNode(),
                             "",
                             {"transaction_id", "t_timestamp", "dnn_part12", "c_birth_year", "dnn_part11"}
                         )
                         .filter("age_during_transaction(t_timestamp, c_birth_year) >= 18")
                         //.project({"transaction_id", "get_binary_class(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(mat_vector_add_1(mat_mul_1(concat(customer_feature, transaction_feature))))))))))) AS fraud_type"})
                         .project({"transaction_id", "get_binary_class(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(vector_addition(dnn_part11, dnn_part12))))))))) AS fraud_type"})
                         //.filter("get_binary_class(fraudulent_probs) = 1")
                         //.project({"transaction_id", "get_binary_class(fraudulent_probs)"})
                         .planNode();
      

      // The following plan converts decision tree to join and aggregation

     /*auto myPlan1 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                        .values({customerRowVector})
                        .localPartition({"c_customer_sk"})
                        .project({"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year", "c_birth_country"})
                        .hashJoin(
                             {"c_customer_sk"},
                             {"fa_customer_sk"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({accountRowVector})
                             .localPartition({"fa_customer_sk"})
                             .project({"fa_customer_sk", "fa_transaction_limit"})
                             .planNode(),
                             "",
                             {"c_customer_sk", "c_address_num", "c_cust_flag", "c_birth_day", "c_birth_month", "c_birth_year", "c_birth_country", "fa_transaction_limit"}
                        )
                        .project({"c_customer_sk", "c_birth_year", "mat_vector_add_1(mat_mul_11(get_customer_features(c_address_num, c_cust_flag, c_birth_day, c_birth_month, c_birth_year, c_birth_country, fa_transaction_limit))) as dnn_part11"})
                         .hashJoin(
                             {"c_customer_sk"},
                             {"t_sender"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({transactionRowVector})
                             .localPartition({"t_sender"})
                             .project({"transaction_id", "t_sender", "t_amount", "date_to_timestamp(t_time) as t_timestamp"})
                             .filter("is_working_day(t_timestamp) = 1")
                             .project({"transaction_id", "t_sender", "t_timestamp", "get_transaction_features(t_amount, t_timestamp) as transaction_feature"})
                             .nestedLoopJoin(
		                         exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                                 .values({treeRowVector})
			                     .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
                                 .planNode(),
                                 {"transaction_id", "t_sender", "t_timestamp", "transaction_feature", "tree_id", "tree"})
                             .project({"transaction_id", "t_sender", "t_timestamp", "transaction_feature", "tree_id", "velox_decision_tree_predict(transaction_feature, tree) as single_prediction"})
                             .aggregation({"transaction_id", "t_sender", "t_timestamp", "transaction_feature"},
				             {"sum(single_prediction) as xgboost_prediction"},
				             {},
				             core::AggregationNode::Step::kPartial,
				             false)
                             .filter("xgboost_prediction >= 0.5")
                             .project({"transaction_id", "t_sender", "t_timestamp", "mat_mul_12(transaction_feature) AS dnn_part12"})
                             .planNode(),
                             "",
                             {"transaction_id", "t_timestamp", "dnn_part12", "c_birth_year", "dnn_part11"}
                         )
                         //.filter("is_working_day(t_timestamp) = 1")
                         //.filter("age_during_transaction(t_timestamp, c_birth_year) >= 18")
                         //.project({"transaction_id", "t_timestamp", "c_birth_year", "get_binary_class(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(mat_vector_add_1(mat_mul_1(concat(customer_feature, transaction_feature))))))))))) AS fraud_type"})
                         .project({"transaction_id", "t_timestamp", "c_birth_year", "get_binary_class(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(vector_addition(dnn_part11, dnn_part12))))))))) AS fraud_type"})
                         .filter("age_during_transaction(t_timestamp, c_birth_year) >= 18")
                         //.filter("xgboost_fraud_transaction(transaction_feature) >= 0.5")
                         //.filter("is_working_day(t_timestamp) = 1")
                         //.filter("get_binary_class(fraudulent_probs) = 1")
                         //.project({"transaction_id", "get_binary_class(fraudulent_probs)"})
                         .planNode();*/



    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto results = exec::test::AssertQueryBuilder(myPlan1).maxDrivers(8).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << "Single Batch with DNN first Results Size: " << results->size() << std::endl;
    std::cout << results->toString(0, 5) << std::endl;
    std::cout << "Time for Executing with Single Batch (sec): " << std::endl;
    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
 
}


void FraudDetectionTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath) {

  std::cout << "Option is " << option << std::endl;
  registerFunctions(modelFilePath, 12);

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
