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
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/ml_functions/functions.h"
//#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/Embedding.h"
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
    int current_year = 2024;

    BaseVector* base0 = args[0].get();

    exec::LocalDecodedVector vecHolder0(context, *base0, rows);
    auto decodedArray0 = vecHolder0.get();

    for (int i = 0; i < rows.size(); i++) {
        int birthYear = decodedArray0->valueAt<int>(i);
        results.push_back(current_year - birthYear);
    }

    VectorMaker maker{context.pool()};
    auto localResult = maker.flatVector<int>(results);
    context.moveOrCopyResult(localResult, rows, output);
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


class GetCustomerExtraFeature : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

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
        int custFlag = (decodedArray0->valueAt<int>(i));
        float cCustFlag = (static_cast<float>(custFlag));
        float avg_rating = decodedArray1->valueAt<float>(i);
        avg_rating = avg_rating/5.0;

        std::vector<float> vec;
        vec.push_back(cCustFlag);
        vec.push_back(avg_rating);

        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .argumentType("REAL")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_customer_extra_feature";
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


class GetProductRating : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    //auto totalOrders = decodedArray0->base()->as<FlatVector<int64_t>>();

    for (int i = 0; i < rows.size(); i++) {
        float avg_rating = decodedArray0->valueAt<float>(i);
        avg_rating = avg_rating/5.0;

        std::vector<float> vec;
        vec.push_back(avg_rating);

        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("REAL")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_product_rating";
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


class FraudTwoTowerTest : public HiveConnectorTestBase {
 public:
  FraudTwoTowerTest() {
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

  ~FraudTwoTowerTest() {}

  void registerFunctions();
  void registerNNFunctionsCustomer(int numCols);
  void registerNNFunctionsProduct(int numCols);
  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath);

  std::unordered_map<std::string, int> getCountryMap();
  std::unordered_map<std::string, int> getDepartmentMap();
  RowVectorPtr getCustomerData(std::string filePath);
  RowVectorPtr getProductData(std::string filePath);
  RowVectorPtr getRatingData(std::string filePath);
  std::vector<std::vector<float>> loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint);
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

void FraudTwoTowerTest::registerFunctions() {

  exec::registerVectorFunction(
          "get_age",
          GetAge::signatures(),
          std::make_unique<GetAge>());
  std::cout << "Completed registering function for get_age" << std::endl;

  exec::registerVectorFunction(
          "get_customer_extra_feature",
          GetCustomerExtraFeature::signatures(),
          std::make_unique<GetCustomerExtraFeature>());
  std::cout << "Completed registering function for get_customer_extra_feature" << std::endl;

  exec::registerVectorFunction(
          "get_product_rating",
          GetProductRating::signatures(),
          std::make_unique<GetProductRating>());
  std::cout << "Completed registering function for get_product_rating" << std::endl;

  exec::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(16));

  exec::registerVectorFunction(
          "vector_addition",
          VectorAddition::signatures(),
          std::make_unique<VectorAddition>(16));

}


void FraudTwoTowerTest::registerNNFunctionsCustomer(int numCols) {
  std::vector<std::vector<float>> emb_customer = loadHDF5Array("resources/model/customer_encoder.h5", "emb_customer.weight", 0);
  std::vector<std::vector<float>> emb_addr = loadHDF5Array("resources/model/customer_encoder.h5", "emb_addr.weight", 0);
  std::vector<std::vector<float>> emb_age = loadHDF5Array("resources/model/customer_encoder.h5", "emb_age.weight", 0);
  std::vector<std::vector<float>> emb_country = loadHDF5Array("resources/model/customer_encoder.h5", "emb_country.weight", 0);
  std::vector<std::vector<float>> w1 = loadHDF5Array("resources/model/customer_encoder.h5", "fc1.weight", 0);
  std::vector<std::vector<float>> b1 = loadHDF5Array("resources/model/customer_encoder.h5", "fc1.bias", 0);
  std::vector<std::vector<float>> w2 = loadHDF5Array("resources/model/customer_encoder.h5", "fc2.weight", 0);
  std::vector<std::vector<float>> b2 = loadHDF5Array("resources/model/customer_encoder.h5", "fc2.bias", 0);
  std::vector<std::vector<float>> w3 = loadHDF5Array("resources/model/customer_encoder.h5", "fc3.weight", 0);
  std::vector<std::vector<float>> b3 = loadHDF5Array("resources/model/customer_encoder.h5", "fc3.bias", 0);
  std::vector<std::vector<float>> bn1_w = loadHDF5Array("resources/model/customer_encoder.h5", "batch_norm1.weight", 0);
  std::vector<std::vector<float>> bn1_b = loadHDF5Array("resources/model/customer_encoder.h5", "batch_norm1.bias", 0);
  std::vector<std::vector<float>> bn2_w = loadHDF5Array("resources/model/customer_encoder.h5", "batch_norm2.weight", 0);
  std::vector<std::vector<float>> bn2_b = loadHDF5Array("resources/model/customer_encoder.h5", "batch_norm2.bias", 0);
  std::vector<std::vector<float>> bn3_w = loadHDF5Array("resources/model/customer_encoder.h5", "batch_norm3.weight", 0);
  std::vector<std::vector<float>> bn3_b = loadHDF5Array("resources/model/customer_encoder.h5", "batch_norm3.bias", 0);

  auto itemEmbCustomerVector = maker.arrayVector<float>(emb_customer, REAL());
  auto itemEmbAddrVector = maker.arrayVector<float>(emb_addr, REAL());
  auto itemEmbAgeVector = maker.arrayVector<float>(emb_age, REAL());
  auto itemEmbCountryVector = maker.arrayVector<float>(emb_country, REAL());
  auto itemNNweight1Vector = maker.arrayVector<float>(w1, REAL());
  auto itemNNweight2Vector = maker.arrayVector<float>(w2, REAL());
  auto itemNNweight3Vector = maker.arrayVector<float>(w3, REAL());
  auto itemNNBias1Vector = maker.arrayVector<float>(b1, REAL());
  auto itemNNBias2Vector = maker.arrayVector<float>(b2, REAL());
  auto itemNNBias3Vector = maker.arrayVector<float>(b3, REAL());
  auto itemBNweight1Vector = maker.arrayVector<float>(bn1_w, REAL());
  auto itemBNweight2Vector = maker.arrayVector<float>(bn2_w, REAL());
  auto itemBNweight3Vector = maker.arrayVector<float>(bn3_w, REAL());
  auto itemBNBias1Vector = maker.arrayVector<float>(bn1_b, REAL());
  auto itemBNBias2Vector = maker.arrayVector<float>(bn2_b, REAL());
  auto itemBNBias3Vector = maker.arrayVector<float>(bn3_b, REAL());


  exec::registerVectorFunction(
      "convert_int_array",
      ConvertToIntArray::signatures(),
      std::make_unique<ConvertToIntArray>());

  exec::registerVectorFunction(
      "convert_double_to_float_array",
      ConvertDoubleToFloatArray::signatures(),
      std::make_unique<ConvertDoubleToFloatArray>());

  exec::registerVectorFunction(
      "embedding_customer",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmbCustomerVector->elements()
              ->values()
              ->asMutable<float>(),
          70710,
          16));

  exec::registerVectorFunction(
      "embedding_addr",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmbAddrVector->elements()
              ->values()
              ->asMutable<float>(),
          35355,
          16));

  exec::registerVectorFunction(
      "embedding_age",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmbAgeVector->elements()
              ->values()
              ->asMutable<float>(),
          95,
          8));

  exec::registerVectorFunction(
      "embedding_country",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmbCountryVector->elements()
              ->values()
              ->asMutable<float>(),
          213,
          8));

  exec::registerVectorFunction(
      "mat_mul_1_customer",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight1Vector->elements()->values()->asMutable<float>()),
          numCols,
          128));

  exec::registerVectorFunction(
      "mat_vector_add_1_customer",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias1Vector->elements()->values()->asMutable<float>()), 128));

  exec::registerVectorFunction(
      "mat_mul_2_customer",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight2Vector->elements()->values()->asMutable<float>()),
          128,
          64));

  exec::registerVectorFunction(
      "mat_vector_add_2_customer",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias2Vector->elements()->values()->asMutable<float>()), 64));

  exec::registerVectorFunction(
      "mat_mul_3_customer",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight3Vector->elements()->values()->asMutable<float>()),
          64,
          16));

  exec::registerVectorFunction(
      "mat_vector_add_3_customer",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias3Vector->elements()->values()->asMutable<float>()), 16));

  exec::registerVectorFunction(
      "batch_norm1_customer",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          itemBNweight1Vector->elements()->values()->asMutable<float>(),
          itemBNBias1Vector->elements()->values()->asMutable<float>(),
          128));

  exec::registerVectorFunction(
      "batch_norm2_customer",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          itemBNweight2Vector->elements()->values()->asMutable<float>(),
          itemBNBias2Vector->elements()->values()->asMutable<float>(),
          64));

  exec::registerVectorFunction(
      "batch_norm3_customer",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          itemBNweight3Vector->elements()->values()->asMutable<float>(),
          itemBNBias3Vector->elements()->values()->asMutable<float>(),
          16));

  exec::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(),
          {},
          true);


}



void FraudTwoTowerTest::registerNNFunctionsProduct(int numCols) {
  std::vector<std::vector<float>> emb_product = loadHDF5Array("resources/model/product_encoder.h5", "emb_product.weight", 0);
  std::vector<std::vector<float>> emb_dept = loadHDF5Array("resources/model/product_encoder.h5", "emb_dept.weight", 0);
  std::vector<std::vector<float>> w1 = loadHDF5Array("resources/model/product_encoder.h5", "fc1.weight", 0);
  std::vector<std::vector<float>> b1 = loadHDF5Array("resources/model/product_encoder.h5", "fc1.bias", 0);
  std::vector<std::vector<float>> w2 = loadHDF5Array("resources/model/product_encoder.h5", "fc2.weight", 0);
  std::vector<std::vector<float>> b2 = loadHDF5Array("resources/model/product_encoder.h5", "fc2.bias", 0);
  std::vector<std::vector<float>> w3 = loadHDF5Array("resources/model/product_encoder.h5", "fc3.weight", 0);
  std::vector<std::vector<float>> b3 = loadHDF5Array("resources/model/product_encoder.h5", "fc3.bias", 0);
  std::vector<std::vector<float>> bn1_w = loadHDF5Array("resources/model/product_encoder.h5", "batch_norm1.weight", 0);
  std::vector<std::vector<float>> bn1_b = loadHDF5Array("resources/model/product_encoder.h5", "batch_norm1.bias", 0);
  std::vector<std::vector<float>> bn2_w = loadHDF5Array("resources/model/product_encoder.h5", "batch_norm2.weight", 0);
  std::vector<std::vector<float>> bn2_b = loadHDF5Array("resources/model/product_encoder.h5", "batch_norm2.bias", 0);
  std::vector<std::vector<float>> bn3_w = loadHDF5Array("resources/model/product_encoder.h5", "batch_norm3.weight", 0);
  std::vector<std::vector<float>> bn3_b = loadHDF5Array("resources/model/product_encoder.h5", "batch_norm3.bias", 0);

  auto itemEmbProductVector = maker.arrayVector<float>(emb_product, REAL());
  auto itemEmbDeptVector = maker.arrayVector<float>(emb_dept, REAL());
  auto itemNNweight1Vector = maker.arrayVector<float>(w1, REAL());
  auto itemNNweight2Vector = maker.arrayVector<float>(w2, REAL());
  auto itemNNweight3Vector = maker.arrayVector<float>(w3, REAL());
  auto itemNNBias1Vector = maker.arrayVector<float>(b1, REAL());
  auto itemNNBias2Vector = maker.arrayVector<float>(b2, REAL());
  auto itemNNBias3Vector = maker.arrayVector<float>(b3, REAL());
  auto itemBNweight1Vector = maker.arrayVector<float>(bn1_w, REAL());
  auto itemBNweight2Vector = maker.arrayVector<float>(bn2_w, REAL());
  auto itemBNweight3Vector = maker.arrayVector<float>(bn3_w, REAL());
  auto itemBNBias1Vector = maker.arrayVector<float>(bn1_b, REAL());
  auto itemBNBias2Vector = maker.arrayVector<float>(bn2_b, REAL());
  auto itemBNBias3Vector = maker.arrayVector<float>(bn3_b, REAL());

  exec::registerVectorFunction(
      "embedding_product",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmbProductVector->elements()
              ->values()
              ->asMutable<float>(),
          708,
          16));

  exec::registerVectorFunction(
      "embedding_dept",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmbDeptVector->elements()
              ->values()
              ->asMutable<float>(),
          46,
          8));

  exec::registerVectorFunction(
      "mat_mul_1_product",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight1Vector->elements()->values()->asMutable<float>()),
          numCols,
          48));

  exec::registerVectorFunction(
      "mat_vector_add_1_product",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias1Vector->elements()->values()->asMutable<float>()), 48));

  exec::registerVectorFunction(
      "mat_mul_2_product",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight2Vector->elements()->values()->asMutable<float>()),
          48,
          32));

  exec::registerVectorFunction(
      "mat_vector_add_2_product",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias2Vector->elements()->values()->asMutable<float>()), 32));

  exec::registerVectorFunction(
      "mat_mul_3_product",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight3Vector->elements()->values()->asMutable<float>()),
          32,
          16));

  exec::registerVectorFunction(
      "mat_vector_add_3_product",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias3Vector->elements()->values()->asMutable<float>()), 16));

  exec::registerVectorFunction(
      "batch_norm1_product",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          itemBNweight1Vector->elements()->values()->asMutable<float>(),
          itemBNBias1Vector->elements()->values()->asMutable<float>(),
          48));

  exec::registerVectorFunction(
      "batch_norm2_product",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          itemBNweight2Vector->elements()->values()->asMutable<float>(),
          itemBNBias2Vector->elements()->values()->asMutable<float>(),
          32));

  exec::registerVectorFunction(
      "batch_norm3_product",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          itemBNweight3Vector->elements()->values()->asMutable<float>(),
          itemBNBias3Vector->elements()->values()->asMutable<float>(),
          16));

}



ArrayVectorPtr FraudTwoTowerTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

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

RowVectorPtr FraudTwoTowerTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
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


std::unordered_map<std::string, int> FraudTwoTowerTest::getCountryMap() {
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


std::unordered_map<std::string, int> FraudTwoTowerTest::getDepartmentMap() {
    std::unordered_map<std::string, int> deptMap;

    // Open the txt file
    std::string filePath = "resources/data/department_mapping.txt";
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
        std::getline(ss, key, '=');

        // Get the value after the comma
        std::getline(ss, value_str);

        // Convert the string value to an integer
        int value = std::stoi(value_str);

        // Insert into the unordered_map
        deptMap[key] = value;
    }

    // Close the file
    file.close();

    return deptMap;
}


std::vector<std::vector<float>> FraudTwoTowerTest::loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint) {
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


RowVectorPtr FraudTwoTowerTest::getCustomerData(std::string filePath) {

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
                     countryMap[numberStr] = 212;
                     cBirthCountry.push_back(212);
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


RowVectorPtr FraudTwoTowerTest::getProductData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> pProductId;
    std::vector<int> pDept;
    
    std::string line;

    std::unordered_map<std::string, int> deptMap = getDepartmentMap();
    int deptIndex = deptMap.size();

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
                pProductId.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 2) {
                if (deptMap.find(numberStr) == deptMap.end()) {
                     // Key does not exist, insert it
                     pDept.push_back(0);
                } else {
                    // Key exists, retrieve its value
                    pDept.push_back(deptMap[numberStr]);
                }
           }

	        colIndex ++;

        }

    }

    file.close();

     // Prepare Customer table
     auto pProductIdVector = maker.flatVector<int>(pProductId);
     auto pDeptVector = maker.flatVector<int>(pDept);
     //auto oTripTypeVector = maker.flatVector<int>(oTripType);
     auto productRowVector = maker.rowVector(
         {"p_product_id", "p_dept"},
         {pProductIdVector, pDeptVector}
     );

     return productRowVector;
}


RowVectorPtr FraudTwoTowerTest::getRatingData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> rUserId;
    std::vector<int> rProductId;
    std::vector<int> rRating;


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
                rUserId.push_back(std::stoi(numberStr));
            }
            else if (colIndex == 1) {
                rProductId.push_back(std::stoi(numberStr));
            }
            else {
                rRating.push_back(std::stoi(numberStr));
            }

	        colIndex ++;

        }

    }

    file.close();

     // Prepare Customer table
     auto rUserIdVector = maker.flatVector<int>(rUserId);
     auto rProductIdVector = maker.flatVector<int>(rProductId);
     auto rRatingVector = maker.flatVector<int>(rRating);
     auto ratingRowVector = maker.rowVector(
         {"r_user_id", "r_product_id", "r_rating"},
         {rUserIdVector, rProductIdVector, rRatingVector}
     );

     return ratingRowVector;
}



void FraudTwoTowerTest::testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string orderFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();
                      
     std::string path = dataFile->path;

     RowVectorPtr customerRowVector = getCustomerData("resources/data/customer.csv");
     std::cout << "customerRowVector data generated" << std::endl;
     RowVectorPtr productRowVector = getProductData("resources/data/product.csv");
     std::cout << "productRowVector data generated" << std::endl;
     RowVectorPtr ratingRowVector = getRatingData("resources/data/ProductRating.csv");
     std::cout << "ratingRowVector data generated" << std::endl;

     int totalRowsCustomer = customerRowVector->size();
     int totalRowsProduct = productRowVector->size();
     int totalRowsRating = ratingRowVector->size();

     std::cout << "customer data size: " << totalRowsCustomer << ",  product data size: " << totalRowsProduct << ",  rating data size: " << totalRowsRating << std::endl;

     int batch_counts = 4;
     int batchSizeCustomer = totalRowsCustomer / batch_counts;
     int batchSizeProduct = totalRowsProduct / batch_counts;
     int batchSizeRating = totalRowsRating / batch_counts;

     std::vector<RowVectorPtr> batchesCustomer;
     std::vector<RowVectorPtr> batchesProduct;
     std::vector<RowVectorPtr> batchesRating;

     for (int i = 0; i < batch_counts; ++i) {
         int start = i * batchSizeCustomer;
         int end = (i == (batch_counts - 1)) ? totalRowsCustomer : (i + 1) * batchSizeCustomer;  // Handle remainder for last batch
         batchesCustomer.push_back(std::dynamic_pointer_cast<RowVector>(customerRowVector->slice(start, end - start)));

         start = i * batchSizeProduct;
         end = (i == (batch_counts - 1)) ? totalRowsProduct : (i + 1) * batchSizeProduct;  // Handle remainder for last batch
         batchesProduct.push_back(std::dynamic_pointer_cast<RowVector>(productRowVector->slice(start, end - start)));

         start = i * batchSizeRating;
         end = (i == (batch_counts - 1)) ? totalRowsRating : (i + 1) * batchSizeRating;  // Handle remainder for last batch
         batchesRating.push_back(std::dynamic_pointer_cast<RowVector>(ratingRowVector->slice(start, end - start)));
     }

     registerNNFunctionsCustomer(50);
     registerNNFunctionsProduct(25);
     CPUUtilizationTracker tracker;

     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();


    auto myPlanProduct = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({productRowVector})
                         .localPartition({"p_product_id"})
                         .project({"p_product_id", "p_dept"})
                         .hashJoin({"p_product_id"},
                             {"r_product_id"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({ratingRowVector})
                             .localPartition({"r_product_id"})
                             .project({"r_product_id", "r_rating"})
                             .singleAggregation({"r_product_id"}, {"avg(r_rating) as avg_product_rating"})
                             .planNode(),
                             "",
                             {"p_product_id", "p_dept", "avg_product_rating"}
                         )
                         .project({"p_product_id", "concat(embedding_product(convert_int_array(p_product_id)), embedding_dept(convert_int_array(p_dept)), get_product_rating(CAST(avg_product_rating AS REAL))) as product_feature"})
                         .project({"p_product_id", "relu(batch_norm3_product(mat_vector_add_3_product(mat_mul_3_product(relu(batch_norm2_product(mat_vector_add_2_product(mat_mul_2_product(relu(batch_norm1_product(mat_vector_add_1_product(mat_mul_1_product(product_feature)))))))))))) AS product_encoding"})
                         .planNode();

    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      .values({customerRowVector})
                      .localPartition({"c_customer_sk"})
                      .project({"c_customer_sk", "c_address_num", "get_age(c_birth_year) as age", "c_birth_country", "c_cust_flag"})
                      .hashJoin({"c_customer_sk"},
                             {"r_user_id"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({ratingRowVector})
                             .localPartition({"r_user_id"})
                             .project({"r_user_id", "r_rating"})
                             .singleAggregation({"r_user_id"}, {"avg(r_rating) as avg_customer_rating"})
                             .filter("avg_customer_rating >= 4.0")
                             .planNode(),
                             "",
                             {"c_customer_sk", "c_address_num", "age", "c_birth_country", "c_cust_flag", "avg_customer_rating"}
                      )
                      .project({"c_customer_sk", "avg_customer_rating", "concat(embedding_customer(convert_int_array(c_customer_sk)), embedding_addr(convert_int_array(c_address_num)), embedding_age(convert_int_array(age)), embedding_country(convert_int_array(c_birth_country)), get_customer_extra_feature(c_cust_flag, CAST(avg_customer_rating AS REAL))) as customer_feature"})
                      .project({"c_customer_sk", "avg_customer_rating", "relu(batch_norm3_customer(mat_vector_add_3_customer(mat_mul_3_customer(relu(batch_norm2_customer(mat_vector_add_2_customer(mat_mul_2_customer(relu(batch_norm1_customer(mat_vector_add_1_customer(mat_mul_1_customer(customer_feature)))))))))))) AS customer_encoding"})
                      .nestedLoopJoin(
                             myPlanProduct,
                             {"c_customer_sk", "avg_customer_rating", "customer_encoding", "p_product_id", "product_encoding"}
                      )
                      .project({"c_customer_sk", "p_product_id", "avg_customer_rating", "vector_addition(customer_encoding, product_encoding) as final_encoding"})
                      .planNode();


    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    auto results2 = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << "Two Tower Model Results Size: " << results2->size() << std::endl;
    std::cout << results2->toString(0, 5) << std::endl;
    std::cout << "Time for Executing Two Tower Model with Single Batch (sec): " << std::endl;
    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count()) /1000000.0 << std::endl;

 
}


void FraudTwoTowerTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath) {

  std::cout << "Option is " << option << std::endl;
  registerFunctions();

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

  FraudTwoTowerTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}, orderDataFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath, orderDataFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath, orderDataFilePath);

}
