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



class GetOrderFeatures : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    const char* dateFormat = "%Y-%m-%d";
    int secondsInADay = 86400;
    std::string days_of_week[7] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();
    //auto cAddressNums = decodedArray0->base()->as<FlatVector<int>>();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();
    //auto cCustFlags = decodedArray1->base()->as<FlatVector<int>>();

    for (int i = 0; i < rows.size(); i++) {
        std::vector<float> vec;

        StringView valTimestamp = decodedArray0->valueAt<StringView>(i);
        std::string sTimestamp = std::string(valTimestamp);
        StringView valWeekday = decodedArray1->valueAt<StringView>(i);
        std::string sWeekday = std::string(valWeekday);

        struct std::tm t = {};
        std::istringstream ss(sTimestamp);
        ss >> std::get_time(&t, dateFormat);

      // Check if parsing was successful
      if (ss.fail()) {
          std::cerr << "Failed to parse date string " << sTimestamp << std::endl;
          vec.push_back(0.0);
          continue;
      }
      else {
          // Convert tm struct to time_t (timestamp)
          time_t tt = mktime(&t);
          // Cast time_t to int64_t
          int64_t longTimestamp = static_cast<int64_t>(tt);
          vec.push_back((static_cast<float>(longTimestamp/secondsInADay))/15340.0);
      }

      for (int i = 0; i < 7; i++) {
          if (sWeekday == days_of_week[i]) {
              vec.push_back(1.0);
          }
          else {
              vec.push_back(0.0);
          }
      }
      results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    //auto localResult = maker.flatVector<int64_t>(results);
    //context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .argumentType("VARCHAR")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_order_features";
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



class IsPopularStore : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto* flatResult = output->asFlatVector<int32_t>();
    BaseVector* parameter = args[0].get();

    exec::LocalDecodedVector vecHolder(context, *parameter, rows);
    auto decodedArray = vecHolder.get();
    auto arrayVector = decodedArray->base()->as<ArrayVector>();
    auto flatElements = arrayVector->elements()->asFlatVector<float>();

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->set(row, 0);  // Handle nulls
        return;
      }

      auto arrayIndex = decodedArray->index(row);
      int size = arrayVector->sizeAt(arrayIndex);  // Get the size of the array
      int offset = arrayVector->offsetAt(arrayIndex);  // Get the offset

      // Use std::accumulate directly over the flat elements vector
      float sum = std::accumulate(
          flatElements->rawValues() + offset,
          flatElements->rawValues() + offset + size,
          0.0f);
      float firstElem = flatElements->valueAt(offset);
      float mean = (sum - firstElem) / (size - 1);
      flatResult->set(row, mean >= 0.4f ? 1 : 0);  // Directly set result
    });

  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "is_popular_store";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class GetMaxIndex : public MLFunction {
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
    auto arrayVector = decodedArray->base()->as<ArrayVector>();
    auto flatElements = arrayVector->elements()->asFlatVector<float>();

    rows.applyToSelected([&](vector_size_t row) {
      if (decodedArray->isNullAt(row)) {
        flatResult->setNull(row, true);  // Handle nulls
        return;
      }

      // Get the array index, size, and offset for this row
      auto arrayIndex = decodedArray->index(row);
      auto size = arrayVector->sizeAt(arrayIndex);
      auto offset = arrayVector->offsetAt(arrayIndex);

      // Find the index of the maximum element in the array
      auto maxIter = std::max_element(
          flatElements->rawValues() + offset,
          flatElements->rawValues() + offset + size);

      int index_of_max = std::distance(flatElements->rawValues() + offset, maxIter);
      flatResult->set(row, index_of_max);  // Set result in output vector
    });

  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("ARRAY(REAL)")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "get_max_index";
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



class TripTypeDetectionTest : public HiveConnectorTestBase {
 public:
  TripTypeDetectionTest() {
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

  ~TripTypeDetectionTest() {}

  void registerFunctions(std::string modelFilePath, int numCols);
  void registerNNFunctions(int numCols);
  void run( int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath);

  RowVectorPtr getOrderData(std::string filePath);
  RowVectorPtr getStoreData(std::string filePath);
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

void TripTypeDetectionTest::registerFunctions(std::string modelFilePath, int numCols) {

  exec::registerVectorFunction(
          "get_order_features",
          GetOrderFeatures::signatures(),
          std::make_unique<GetOrderFeatures>());
  std::cout << "Completed registering function for get_order_features" << std::endl;

  exec::registerVectorFunction(
          "is_popular_store",
          IsPopularStore::signatures(),
          std::make_unique<IsPopularStore>());
  std::cout << "Completed registering function for is_popular_store" << std::endl;

  exec::registerVectorFunction(
            "get_max_index",
            GetMaxIndex::signatures(),
            std::make_unique<GetMaxIndex>());
  std::cout << "Completed registering function for get_binary_class" << std::endl;

}


void TripTypeDetectionTest::registerNNFunctions(int numCols) {
  std::vector<std::vector<float>> emb1 = loadHDF5Array("resources/model/trip_type_classify.h5", "embedding.weight", 0);
  std::vector<std::vector<float>> w1 = loadHDF5Array("resources/model/trip_type_classify.h5", "fc1.weight", 0);
  std::vector<std::vector<float>> b1 = loadHDF5Array("resources/model/trip_type_classify.h5", "fc1.bias", 0);
  std::vector<std::vector<float>> w2 = loadHDF5Array("resources/model/trip_type_classify.h5", "fc2.weight", 0);
  std::vector<std::vector<float>> b2 = loadHDF5Array("resources/model/trip_type_classify.h5", "fc2.bias", 0);
  std::vector<std::vector<float>> w3 = loadHDF5Array("resources/model/trip_type_classify.h5", "fc3.weight", 0);
  std::vector<std::vector<float>> b3 = loadHDF5Array("resources/model/trip_type_classify.h5", "fc3.bias", 0);
  std::vector<std::vector<float>> w11 = loadHDF5Array("resources/model/trip_type_classify.h5", "w11", 0);
  std::vector<std::vector<float>> w12 = loadHDF5Array("resources/model/trip_type_classify.h5", "w12", 0);

  auto itemEmb1Vector = maker.arrayVector<float>(emb1, REAL());
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
      "convert_int_array",
      ConvertToIntArray::signatures(),
      std::make_unique<ConvertToIntArray>());

  exec::registerVectorFunction(
      "customer_id_embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          itemEmb1Vector->elements()
              ->values()
              ->asMutable<float>(),
          70710,
          16));

  exec::registerVectorFunction(
      "mat_mul_1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight1Vector->elements()->values()->asMutable<float>()),
          numCols,
          48));

  exec::registerVectorFunction(
      "mat_vector_add_1",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias1Vector->elements()->values()->asMutable<float>()), 48));

  exec::registerVectorFunction(
      "mat_mul_2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight2Vector->elements()->values()->asMutable<float>()),
          48,
          24));

  exec::registerVectorFunction(
      "mat_vector_add_2",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias2Vector->elements()->values()->asMutable<float>()), 24));

  exec::registerVectorFunction(
      "mat_mul_3",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight3Vector->elements()->values()->asMutable<float>()),
          24,
          1000));

  exec::registerVectorFunction(
      "mat_vector_add_3",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(itemNNBias3Vector->elements()->values()->asMutable<float>()), 1000));

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
          24,
          48));

  exec::registerVectorFunction(
      "mat_mul_12",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(itemNNweight12Vector->elements()->values()->asMutable<float>()),
          69,
          48));

  exec::registerVectorFunction(
          "vector_addition",
          VectorAddition::signatures(),
          std::make_unique<VectorAddition>(48));


}



ArrayVectorPtr TripTypeDetectionTest::parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {

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

RowVectorPtr TripTypeDetectionTest::writeDataToFile(std::string csvFilePath, int numRows, int numCols, 
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


std::vector<std::vector<float>> TripTypeDetectionTest::loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint) {
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


RowVectorPtr TripTypeDetectionTest::getOrderData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> oOrderId;
    std::vector<int> oCustomerSk;
    std::vector<std::string> oWeekday;
    std::vector<std::string> oDate;
    std::vector<int> oStore;
    //std::vector<int> oTripType;
    
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
            else if (colIndex == 4) {
                oStore.push_back(std::stoi(numberStr));
            }
            //else if (colIndex == 5) {
            //    oTripType.push_back(std::stoi(numberStr));
            //}

	        colIndex ++;

        }

    }

    file.close();

     // Prepare Customer table
     auto oOrderIdVector = maker.flatVector<int>(oOrderId);
     auto oCustomerSkVector = maker.flatVector<int>(oCustomerSk);
     auto oWeekdayVector = maker.flatVector<std::string>(oWeekday);
     auto oDateVector = maker.flatVector<std::string>(oDate);
     auto oStoreVector = maker.flatVector<int>(oStore);
     //auto oTripTypeVector = maker.flatVector<int>(oTripType);
     auto orderRowVector = maker.rowVector(
         {"o_order_id", "o_customer_sk", "o_weekday", "o_date", "o_store"},
         {oOrderIdVector, oCustomerSkVector, oWeekdayVector, oDateVector, oStoreVector}
     );

     return orderRowVector;
}


RowVectorPtr TripTypeDetectionTest::getStoreData(std::string filePath) {

    std::ifstream file(filePath.c_str());

    if (file.fail()) {

        std::cerr << "Data File:" << filePath << " => Read Error" << std::endl;
        exit(1);

    }

    std::vector<int> sStore;
    std::vector<std::vector<float>> sFeatures;


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
	    std::vector<float> features;

        while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
            /*if (index < 5) {
                std::cout << colIndex << ": " << numberStr << std::endl;
            }*/
            // Trim leading and trailing whitespace from the input string (if any)
            if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                numberStr = numberStr.substr(1, numberStr.size() - 2);
            }
            if (colIndex == 0) {
                sStore.push_back(std::stoi(numberStr));
                features.push_back(std::stof(numberStr));
            }
            else {
                features.push_back(std::stof(numberStr));
            }

	        colIndex ++;

        }
        sFeatures.push_back(features);

    }

    file.close();

     // Prepare Customer table
     auto sStoreVector = maker.flatVector<int>(sStore);
     auto sFeaturesVector = maker.arrayVector<float>(sFeatures, REAL());
     auto storeRowVector = maker.rowVector(
         {"s_store", "s_features"},
         {sStoreVector, sFeaturesVector}
     );

     return storeRowVector;
}



void TripTypeDetectionTest::testingWithRealData(int numDataSplits, int dataBatchSize, int numRows, int numCols, std::string orderFilePath, std::string modelFilePath) {

     auto dataFile = TempFilePath::create();
                      
     std::string path = dataFile->path;

     RowVectorPtr orderRowVector = getOrderData("resources/data/order.csv");
     std::cout << "orderRowVector data generated" << std::endl;
     RowVectorPtr storeRowVector = getStoreData("resources/data/store_dept.csv");
     std::cout << "storeRowVector data generated" << std::endl;

     int totalRowsOrder = orderRowVector->size();
     int totalRowsStore = storeRowVector->size();

     std::cout << "order data size: " << totalRowsOrder << ",  store data size: " << totalRowsStore << std::endl;

     int batch_counts = 4;
     int batchSizeOrder = totalRowsOrder / batch_counts;
     int batchSizeStore = totalRowsStore / batch_counts;

     std::vector<RowVectorPtr> batchesOrder;
     std::vector<RowVectorPtr> batchesStore;

     for (int i = 0; i < batch_counts; ++i) {
         int start = i * batchSizeOrder;
         int end = (i == (batch_counts - 1)) ? totalRowsOrder : (i + 1) * batchSizeOrder;  // Handle remainder for last batch
         batchesOrder.push_back(std::dynamic_pointer_cast<RowVector>(orderRowVector->slice(start, end - start)));

         start = i * batchSizeStore;
         end = (i == (batch_counts - 1)) ? totalRowsStore : (i + 1) * batchSizeStore;  // Handle remainder for last batch
         batchesStore.push_back(std::dynamic_pointer_cast<RowVector>(storeRowVector->slice(start, end - start)));
     }

     registerNNFunctions(93);
     CPUUtilizationTracker tracker;

     auto dataHiveSplits =  makeHiveConnectorSplits(path, numDataSplits, dwio::common::FileFormat::DWRF);

     auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();


    auto myPlan2 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                         .values({storeRowVector})
                         .localPartition({"s_store"})
                         .project({"s_store", "s_features as store_feature"})
                         .filter("is_popular_store(store_feature) = 1")
                         .project({"s_store", "mat_vector_add_1(mat_mul_12(store_feature)) as dnn_part2"})
                         .hashJoin({"s_store"},
                             {"o_store"},
                             exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                             .values({orderRowVector})
                             .localPartition({"o_store"})
                             .filter("o_weekday != 'Sunday'")
                             .project({"o_order_id", "o_customer_sk", "o_store", "o_date", "o_weekday"})
                             //.project({"o_order_id", "o_store", "customer_id_embedding(convert_int_array(o_customer_sk)) as customer_id_feature", "get_order_features(o_date, o_weekday) AS order_feature"})
                             .project({"o_order_id", "o_store", "mat_mul_11(concat(customer_id_embedding(convert_int_array(o_customer_sk)), get_order_features(o_date, o_weekday))) as dnn_part1"})
                             .planNode(),
                             "",
                             {"o_order_id", "dnn_part1", "dnn_part2"}
                         )
                         .project({"o_order_id", "get_max_index(softmax(mat_vector_add_3(mat_mul_3(relu(mat_vector_add_2(mat_mul_2(relu(vector_addition(dnn_part1, dnn_part2))))))))) AS predicted_trip_type"})
                         .planNode();


    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    auto results2 = exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();

    //std::cout << "Results:" << results->toString() << std::endl;
    std::cout << "Single Batch with DNN first Results Size: " << results2->size() << std::endl;
    std::cout << results2->toString(0, 5) << std::endl;
    std::cout << "Time for Executing with Single Batch (sec): " << std::endl;
    std::cout << (std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count()) /1000000.0 << std::endl;

 
}


void TripTypeDetectionTest::run(int option, int numDataSplits, int numTreeSplits, int numTreeRows, int dataBatchSize, int numRows, int numCols, std::string dataFilePath, std::string modelFilePath, std::string orderDataFilePath) {

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

  TripTypeDetectionTest demo;

  std::cout << fmt::format("Option: {}, numDataSplits: {}, numTreeSplits: {}, numTreeRows: {}, dataBatchSize: {}, numRows: {}, numCols: {}, dataFilePath: {}, modelFilePath: {}, orderDataFilePath: {}", 
                           option, numDataSplits, numTreeSplits, numTreeRows, numRows, numCols, dataBatchSize, dataFilePath, modelFilePath, orderDataFilePath) 
      << std::endl;

  demo.run(option, numDataSplits, numTreeSplits, numTreeRows, dataBatchSize, numRows, numCols, dataFilePath, modelFilePath, orderDataFilePath);

}
