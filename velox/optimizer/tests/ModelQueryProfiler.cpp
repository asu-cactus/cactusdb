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
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <torch/torch.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
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
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/FraudDetectionFunctions.h"
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
#include "velox/cost_model/Stat.h"
#include "velox/optimizer/Mul2JoinAggRewriteAction.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RewriteAction.h"
#include "velox/optimizer/RuleManager.h"
#include "velox/optimizer/TwoLayerUDF2TorchNNRewriteAction.h"
#include "velox/optimizer/tests/BenchmarkQueryTemplates.h"
#include "velox/optimizer/tests/BenchmarkUtils.h"
#include "velox/optimizer/tests/ModelRegister.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

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
    rootPool_ = memory::MemoryManager::getInstance()->addRootPool(
        "ProfileQueryGenerator");
    pool_ = rootPool_->addLeafChild("ProfileQueryGenerator");
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

  int cacheQueryPlanAndCateLog(PlanBuilder& planBuilder, CataLog& cataLog) {
    int queryPlanCacheId = queryPlanCacheId_++;
    auto serializedPlan = planBuilder.planNode()->serialize();
    queryPlanCaches_[queryPlanCacheId] = serializedPlan;
    cataLogIdAddressMapCaches_[queryPlanCacheId] = cataLog.getIdAddressMap();

    return queryPlanCacheId;
  }

  void resetQueryPlanAndQueryPlanFromCache(
      PlanBuilder& planBuilder,
      CataLog& cataLog,
      int queryPlanCacheId) {
    auto it1 = queryPlanCaches_.find(queryPlanCacheId);
    if (it1 != queryPlanCaches_.end()) {
      auto serializedPlan = it1->second;
      auto deserlizedUpdatedPlanNode =
          ISerializable::deserialize<core::PlanNode>(
              serializedPlan, pool_.get());
      planBuilder.setRoot(deserlizedUpdatedPlanNode);
    } else {
      throw std::runtime_error(
          fmt::format(
              "[ERROR]queryPlanCacheId: {} was not found queryPlanCaches.",
              queryPlanCacheId));
    }

    auto it2 = cataLogIdAddressMapCaches_.find(queryPlanCacheId);
    if (it2 != cataLogIdAddressMapCaches_.end()) {
      cataLog.setIdAddressMap(it2->second);
    } else {
      throw std::runtime_error(
          fmt::format(
              "[ERROR]queryPlanCacheId: {} was not found in cataLogIdAddressMapCaches.",
              queryPlanCacheId));
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

  std::string registerNNModelFromParams(
      std::vector<int> kernelSizes,
      std::vector<std::string> kernelNames,
      int featureSize,
      CataLog& cataLog,
      int& modelGroupId_) {
    checkOrAbort(
        kernelSizes.size(),
        kernelNames.size(),
        "[ERROR] kernelSizes and kernelNames should have the same size.");
    // use input size as random seed
    RandomGenerator randomGenerator = RandomGenerator(-1, 1);
    int modelGroupId = modelGroupId_++;
    int functionId = 0;
    int numberOfLayers = kernelNames.size();

    optimization::registerVectorFunction(
        "relu",
        Relu::signatures(),
        std::make_unique<Relu>(),
        {},
        true,
        cataLog);
    optimization::registerVectorFunction(
        "softmax",
        Softmax::signatures(),
        std::make_unique<Softmax>(),
        {},
        true,
        cataLog);
    optimization::registerVectorFunction(
        "argmax",
        Argmax::signatures(),
        std::make_unique<Argmax>(),
        {},
        true,
        cataLog);

    std::string modelComputationStr = "{}";
    int lastSize = featureSize;

    for (int i = 0; i < kernelSizes.size(); i++) {
      int layerSize = kernelSizes[i];
      std::string kernelName = kernelNames[i];

      if (kernelName == "MatMul") {
        std::vector<std::vector<float>> weights =
            randomGenerator.genFloat2dVector(lastSize, layerSize);
        std::string matMulName =
            fmt::format("mat_mul{}_{}", modelGroupId_, functionId++);
        optimization::registerVectorFunction(
            matMulName,
            MatrixMultiply::signatures(),
            std::make_unique<MatrixMultiply>(
                std::move(flattenVectorToPointer(weights)),
                lastSize,
                layerSize),
            {},
            true,
            cataLog);
        modelComputationStr = matMulName + "(" + modelComputationStr + ")";
      } else if (kernelName == "MatAdd") {
        std::vector<std::vector<float>> bias =
            randomGenerator.genFloat2dVector(1, layerSize);

        std::string matVectorAddName =
            fmt::format("mat_vector_add{}_{}", modelGroupId_, functionId++);
        optimization::registerVectorFunction(
            matVectorAddName,
            MatrixVectorAddition::signatures(),
            std::make_unique<MatrixVectorAddition>(
                std::move(flattenVectorToPointer(bias)), layerSize),
            {},
            true,
            cataLog);
        modelComputationStr =
            matVectorAddName + "(" + modelComputationStr + ")";
      } else if (kernelName == "ReLu") {
        modelComputationStr = "relu(" + modelComputationStr + ")";
      } else if (kernelName == "Softmax") {
        modelComputationStr = "softmax(" + modelComputationStr + ")";
      } else if (kernelName == "Argmax") {
        modelComputationStr = "argmax(" + modelComputationStr + ")";
      } else if (kernelName == "BatchNorm") {
        std::vector<std::vector<float>> batchNormWeights =
            randomGenerator.genFloat2dVector(1, layerSize);
        std::vector<std::vector<float>> batchNormBias =
            randomGenerator.genFloat2dVector(1, layerSize);
        std::string batchNormName =
            fmt::format("batch_norm_1d{}_{}", modelGroupId_, functionId++);
        optimization::registerVectorFunction(
            batchNormName,
            BatchNorm1D::signatures(),
            std::make_unique<BatchNorm1D>(
                std::move(flattenVectorToPointer(batchNormWeights)),
                std::move(flattenVectorToPointer(batchNormBias)),
                layerSize),
            {},
            true,
            cataLog);
        modelComputationStr = batchNormName + "(" + modelComputationStr + ")";
      } else {
        throw std::runtime_error(
            fmt::format("[ERROR] Unsupported kernel name: {}", kernelName));
      }
      lastSize = layerSize;
    }
    return modelComputationStr;
  }

  void benchmarkModel(
      std::string modelType,
      int numThreads,
      int repeatRun,
      int verbose,
      bool rewrite,
      int numData,
      int featureSize,
      int dataBatchSize = 256) {
    VectorMaker maker{pool_.get()};
    PlanBuilder queryPlan;
    CataLog cataLog;
    // Initialize planNodeIdGenerator
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    std::vector<std::vector<int>> sampledModelKernelSizes =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/_sampledModel/model_layer_size.txt");

    std::vector<std::vector<std::string>> sampledModelKernelNames =
        readModelKernelStrFromFile(
            "/home/velox/velox/optimizer/tests/_sampledModel/model_kernel_name.txt");

    std::cout << "[INFO] sampledModelKernelSizes: " << sampledModelKernelSizes
              << std::endl;
    std::cout << "[INFO] sampledModelKernelNames: " << sampledModelKernelNames
              << std::endl;

    // sample data
    RandomGenerator randomGenerator = RandomGenerator(-1, 1);
    std::vector<std::vector<float>> inputData =
        randomGenerator.genFloat2dVector(numData, featureSize);

    auto inputDataVector = maker.arrayVector<float>(inputData, REAL());
    auto inputRowVector = maker.rowVector({"x"}, {inputDataVector});

    // register model functions
    std::string modelComputationStr = registerNNModelFromParams(
        sampledModelKernelSizes[0],
        sampledModelKernelNames[0],
        featureSize,
        cataLog,
        modelGroupId_);

    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({inputRowVector})
                    .project({fmt::format(modelComputationStr, "x")});

    float executeTime = runPlanWithCataLog(
        pool_, numThreads, queryPlan, cataLog, repeatRun, verbose);
    std::cout << "[INFO] Query Plan: "
              << queryPlan.planNode()->toString(true, true) << std::endl;
    std::cout << "[INFO] Execution time: " << executeTime << std::endl;

    // std::string latencyOutputPath =
    //     "/home/velox/velox/optimizer/tests/executionLatency.txt";
    // writeStringToFile(std::to_string(executeTime), latencyOutputPath);

    // auto serializedPlan = queryPlan.planNode()->serialize();
    // std::string queryOutPutPath =
    //     "/home/velox/velox/optimizer/tests/serializedQueryPlan.json";
    // augmentSerializedPlan(serializedPlan, cataLog);
    // writeStringToFile(folly::toJson(serializedPlan), queryOutPutPath);

    // auto queryPlanStr = queryPlan.planNode()->toString(true, true);
    // std::string queryPlanStrOutputPath =
    //     "/home/velox/velox/optimizer/tests/queryPlanStr.txt";
    // writeStringToFile(queryPlanStr, queryPlanStrOutputPath);

    // std::cout << "[INFO] Execution time: " << executeTime << std::endl;
  }

 private:
  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::MemoryManager::getInstance()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};
  std::shared_ptr<TempDirectoryPath> tempDirPath_;
  std::map<int, std::map<core::PlanNodeId, std::vector<std::string>>>
      cataLogIdAddressMapCaches_;

  VectorMaker maker{pool_.get()};
  static inline int queryPlanCacheId_ = 0;
  std::map<int, folly::dynamic> queryPlanCaches_;
  static inline int modelGroupId_ = 0;
};

DEFINE_string(modelType, "ffnn", "Model: ffnn, df, two-tower, llm");
DEFINE_int32(num_repeat, 1, "Number of repeat run");
DEFINE_int32(num_data, 1000, "Number of data to generate for the benchmark");
DEFINE_int32(feature_size, 256, "Feature size for the model");
DEFINE_int32(num_driver, 8, "Number of drivers");
DEFINE_int32(verbose, 2, "Verbose");
DEFINE_int32(data_batch_size, 256, "Data batch size");
DEFINE_string(data_path, "", "Data path to store the generated data");

int main(int argc, char** argv) {
  memory::MemoryManager::initialize({});
  folly::init(&argc, &argv, false);
  std::string modelType = FLAGS_modelType;

  // bool rewrite = FLAGS_rewrite;
  int repeatRun = FLAGS_num_repeat;
  int numData = FLAGS_num_data;
  int featureSize = FLAGS_feature_size;
  int numDriver = FLAGS_num_driver;
  int verbose = FLAGS_verbose;
  int dataBatchSize = FLAGS_data_batch_size;
  std::string dataPath = FLAGS_data_path;
  IntegratedMCTSTest demo;

  demo.benchmarkModel(
      modelType,
      numDriver,
      repeatRun,
      verbose,
      numData,
      featureSize,
      dataBatchSize);
}
