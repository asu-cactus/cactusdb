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

class DLKernelBenchmarkTest : public HiveConnectorTestBase {
 public:
  DLKernelBenchmarkTest() {
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
            ->newConnector(
                kHiveConnectorId,
                std::make_shared<core::MemConfig>(),
                ioExecutor_.get());
    connector::registerConnector(hiveConnector);

    rootPool_ =
        memory::MemoryManager::getInstance()->addRootPool("DLKernelBenchmark");
    pool_ = rootPool_->addLeafChild("DLKernelBenchmark");

    randomGenerator = RandomGenerator(-1, 1, 0);
    // SetUp();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::unique_ptr<folly::IOThreadPoolExecutor> ioExecutor_;

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;
  RandomGenerator randomGenerator;

  ~DLKernelBenchmarkTest() {
    HiveConnectorTestBase::TearDown();
  }

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

  float runPlan(PlanBuilder& plan, int repeatRun = 1, int verbose = 1) {
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
          {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000000000"},
           {core::QueryConfig::kMaxOutputBatchRows, "1000000"},
           {core::QueryConfig::kPreferredOutputBatchRows, "1000000"}});

      // Add hivesplits to the target plan node (data source node).
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      CursorParameters params;
      params.maxDrivers = 1;
      params.planNode = plan.planNode();
      params.queryCtx = queryCtx_;
      bool noMoreSplits = false;
      auto addSplits = [&noMoreSplits](exec::Task* task) {
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
        dataIdx = 0;
        totalDataNum = 0;
        for (auto batchedData : actualResults) {
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
      }
      finalResult = std::move(finalResult);
    }
    if (verbose >= 1) {
      std::cout << fmt::format(
                       "[INFO] Total # of Batch: {}, Total # of Data: {}",
                       dataIdx,
                       totalDataNum)
                << std::endl;
    }

    return totalElapsedTime / repeatRun;
  }

  PlanBuilder getMatMulPlan(int batchSize, int featureSize, int dim2) {
    VectorMaker maker{pool_.get()};
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(batchSize, featureSize);
    std::vector<std::vector<float>> rightMatrix =
        randomGenerator.genFloat2dVector(featureSize, dim2);
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());
    auto rightMatrixVector = maker.arrayVector<float>(rightMatrix, REAL());
    // std::cout << "[DEBUG] right size: " << rightMatrix.size() << " " << rightMatrix[0].size() << std::endl;
    // std::cout << rightData[0] << "," << rightData[1] << std::endl;
    exec::registerVectorFunction(
        "mat_mul",
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            rightMatrixVector->elements()->values()->asMutable<float>(),
            featureSize,
            dim2));
    auto inputRowVector = maker.rowVector({"x"}, {leftMatrixVector});

    auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"mat_mul(x)"});

    return myPlan;
  }

  PlanBuilder getMatAddPlan(int batchSize, int featureSize) {
    VectorMaker maker{pool_.get()};
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(batchSize, featureSize);
    std::vector<std::vector<float>> rightMatrix =
        randomGenerator.genFloat2dVector(featureSize, 1);
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());
    auto rightMatrixVector = maker.arrayVector<float>(rightMatrix, REAL());

    exec::registerVectorFunction(
        "mat_vector_add",
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            rightMatrixVector->elements()->values()->asMutable<float>(), featureSize));

    auto inputRowVector = maker.rowVector({"x"}, {leftMatrixVector});

    auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"mat_vector_add(x)"});
    return myPlan;
  }

  PlanBuilder getReluPlan(int batchSize, int featureSize) {
    VectorMaker maker{pool_.get()};
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(batchSize, featureSize);
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());

    exec::registerVectorFunction(
        "relu", Relu::signatures(), std::make_unique<Relu>());

    auto inputRowVector = maker.rowVector({"x"}, {leftMatrixVector});

    auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"relu(x)"});
    return myPlan;
  }

  PlanBuilder getSoftmaxPlan(int batchSize, int featureSize) {
    VectorMaker maker{pool_.get()};
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(batchSize, featureSize);
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());

    exec::registerVectorFunction(
        "softmax", Softmax::signatures(), std::make_unique<Softmax>());

    auto inputRowVector = maker.rowVector({"x"}, {leftMatrixVector});

    auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"softmax(x)"});
    return myPlan;
  }

  void benchmarkKernel(
      std::string kernel,
      int batchSize,
      int featureSize,
      int dim2,
      int verbose,
      int numRepeat) {
    PlanBuilder plan;
    if (kernel == "MatMul") {
      plan = getMatMulPlan(batchSize, featureSize, dim2);
    } else if (kernel == "MatAdd") {
      plan = getMatAddPlan(batchSize, featureSize);
    } else if (kernel == "Relu") {
      plan = getReluPlan(batchSize, featureSize);
    } else if (kernel == "Softmax") {
      plan = getSoftmaxPlan(batchSize, featureSize);
    }

    if (verbose == 1) {
      std::cout << fmt::format(
                       "[INFO] Query Plan: \n {}",
                       plan.planNode()->toString(true, true))
                << std::endl;
    }

    float latency = runPlan(plan, numRepeat, verbose);
    std::cout << latency << std::endl;
  }
};

// DEFINE_string(data_path, "../../../../data", "Path to data dir");
DEFINE_int32(batch_size, 500, "Batch Size");
DEFINE_string(kernel, "MatMul", "Benchmark kernel");
DEFINE_int32(feature_size, 100, "Feature Size");
DEFINE_int32(dim2, 100, "Second dimension of right table");
DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(verbose, 0, "Verbose");
// DEFINE_int32(num_driver, 8, "Number of driver");

int main(int argc, char** argv) {
  Eigen::setNbThreads(16);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  int batchSize = FLAGS_batch_size;
  int numRepeat = FLAGS_num_repeat;
  std::string kernel = FLAGS_kernel;
  int featureSize = FLAGS_feature_size;
  int dim2 = FLAGS_dim2;
  int verbose = FLAGS_verbose;

  if (verbose >= 1) {
    std::cout
        << fmt::format(
              "[INFO] Benchmark Kernel: {} \n \t # Batch Size: {}, numRepeat: {}, featureSize: {}, dim2: {}",
              kernel,
              batchSize,
              numRepeat,
              featureSize,
              dim2)
        << std::endl;
  }

  DLKernelBenchmarkTest dlKernelBenchmark;

  dlKernelBenchmark.benchmarkKernel(
      kernel, batchSize, featureSize, dim2, verbose, numRepeat);
}