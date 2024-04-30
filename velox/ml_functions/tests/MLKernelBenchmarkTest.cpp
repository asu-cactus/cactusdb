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
            rightMatrixVector->elements()->values()->asMutable<float>(),
            featureSize));

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

  PlanBuilder getTorchDNNPlan(int batchSize, int featureSize, int layer1Size, int layer2Size) {
    VectorMaker maker{pool_.get()};
    std::vector<std::vector<float>> inputValue =
        randomGenerator.genFloat2dVector(batchSize, featureSize);
    auto inputValueVector = maker.arrayVector<float>(inputValue, REAL());

    float* w1Weight = randomGenerator.genFloat1dArray(featureSize*layer1Size);
    float* w1Bias = randomGenerator.genFloat1dArray(layer1Size);
    float* w2Weight = randomGenerator.genFloat1dArray(layer1Size*layer2Size);
    float* w2Bias = randomGenerator.genFloat1dArray(layer2Size);
    std::vector<float*> weights = {w1Weight, w2Weight};
    std::vector<float*> bias = {w1Bias, w2Bias};
    std::vector<int> dims = {featureSize, layer1Size, layer2Size};


    registerVectorFunction(
                          "torchDNN",
                          TorchDNN::signatures(),
                          std::make_unique<TorchDNN>(
                              weights, bias, dims));

    auto inputRowVector = maker.rowVector({"x"}, {inputValueVector});

    auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"torchDNN(x)"});
    return myPlan;
  }

  PlanBuilder getTorchDNNKernelPlan(std::string kernel, int batchSize, int featureSize, int outputSize) {
    VectorMaker maker{pool_.get()};
    std::vector<std::vector<float>> inputValue =
        randomGenerator.genFloat2dVector(batchSize, featureSize);
    auto inputValueVector = maker.arrayVector<float>(inputValue, REAL());

    float* w1Weight;
    float* w1Bias;
    std::vector<int> dims;
    
    if (kernel == "Dense") {
      w1Weight = randomGenerator.genFloat1dArray(featureSize*outputSize);
      w1Bias = randomGenerator.genFloat1dArray(outputSize);
      dims = {featureSize, outputSize};
    } else if (kernel == "Relu" || kernel == "Softmax") {
      dims = {featureSize};
    } else {
      throw std::runtime_error(
          fmt::format("Non-supported TorchNN model: {}", kernel));
    }
    // std::vector<float*> weights = {w1Weight, w2Weight};
    // std::vector<float*> bias = {w1Bias, w2Bias};
     

    registerVectorFunction(
                          "torchDNNKernel",
                          TorchDNNKernel::signatures(),
                          std::make_unique<TorchDNNKernel>(
                              kernel, w1Weight, w1Bias, dims));

    auto inputRowVector = maker.rowVector({"x"}, {inputValueVector});

    auto myPlan = exec::test::PlanBuilder(pool_.get())
                      .values({inputRowVector})
                      .project({"torchDNNKernel(x)"});
    return myPlan;
  }

  PlanBuilder getHashJoinPlan(
      int lIndexMax,
      int rIndexMax,
      int dim1,
      int dim2,
      bool reverseOrder) {
    VectorMaker maker{pool_.get()};
    std::vector<int> leftIds = randomGenerator.genIntRange(0, lIndexMax);
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(lIndexMax, dim1);

    std::vector<int> rightIds = randomGenerator.genIntRange(0, rIndexMax);
    std::vector<std::vector<float>> rightMatrix =
        randomGenerator.genFloat2dVector(rIndexMax, dim2);
    auto leftIdVector = maker.flatVector<int>(leftIds, INTEGER());
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());
    auto rightIdVector = maker.flatVector<int>(rightIds, INTEGER());
    auto rightMatrixVector = maker.arrayVector<float>(rightMatrix, REAL());

    RowVectorPtr leftRowVector;
    RowVectorPtr rightRowVector;
    if (!reverseOrder) {
      leftRowVector =
          maker.rowVector({"l_id", "m1"}, {leftIdVector, leftMatrixVector});
      rightRowVector =
          maker.rowVector({"r_id", "m2"}, {rightIdVector, rightMatrixVector});
    } else {
      leftRowVector =
          maker.rowVector({"l_id", "m1"}, {rightIdVector, rightMatrixVector});
      rightRowVector =
          maker.rowVector({"r_id", "m2"}, {leftIdVector, leftMatrixVector});
    }

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    core::PlanNodeId leftTableScanNodeId;
    core::PlanNodeId rightTableScanNodeId;

    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                      .values({leftRowVector})
                      .capturePlanNodeId(leftTableScanNodeId)
                      .hashJoin(
                          {"l_id"},
                          {"r_id"},
                          PlanBuilder(planNodeIdGenerator)
                              .values({rightRowVector})
                              .capturePlanNodeId(rightTableScanNodeId)
                              .planNode(),
                          "",
                          {"l_id", "r_id", "m1", "m2"});

    return myPlan;
  }

  PlanBuilder getNestedLoopJoinPlan(
      int lIndexMax,
      int rIndexMax,
      int dim1,
      int dim2,
      bool reverseOrder) {
    VectorMaker maker{pool_.get()};
    std::vector<int> leftIds = randomGenerator.genIntRange(0, lIndexMax);
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(lIndexMax, dim1);

    std::vector<int> rightIds = randomGenerator.genIntRange(0, rIndexMax);
    std::vector<std::vector<float>> rightMatrix =
        randomGenerator.genFloat2dVector(rIndexMax, dim2);
    auto leftIdVector = maker.flatVector<int>(leftIds, INTEGER());
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());
    auto rightIdVector = maker.flatVector<int>(rightIds, INTEGER());
    auto rightMatrixVector = maker.arrayVector<float>(rightMatrix, REAL());

    RowVectorPtr leftRowVector;
    RowVectorPtr rightRowVector;
    if (!reverseOrder) {
      leftRowVector =
          maker.rowVector({"l_id", "m1"}, {leftIdVector, leftMatrixVector});
      rightRowVector =
          maker.rowVector({"r_id", "m2"}, {rightIdVector, rightMatrixVector});
    } else {
      leftRowVector =
          maker.rowVector({"l_id", "m1"}, {rightIdVector, rightMatrixVector});
      rightRowVector =
          maker.rowVector({"r_id", "m2"}, {leftIdVector, leftMatrixVector});
    }

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    core::PlanNodeId leftTableScanNodeId;
    core::PlanNodeId rightTableScanNodeId;

    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                      .values({leftRowVector})
                      .capturePlanNodeId(leftTableScanNodeId)
                      .nestedLoopJoin(
                          PlanBuilder(planNodeIdGenerator)
                              .values({rightRowVector})
                              .capturePlanNodeId(rightTableScanNodeId)
                              .planNode(),
                          "",
                          {"l_id", "r_id", "m1", "m2"});

    return myPlan;
  }

  PlanBuilder getRowNumber(
      int lIndexMax,
      int dim1) {
    VectorMaker maker{pool_.get()};
    std::vector<int> leftIds = randomGenerator.genIntRange(0, lIndexMax);
    std::vector<std::vector<float>> leftMatrix =
        randomGenerator.genFloat2dVector(lIndexMax, dim1);

    auto leftIdVector = maker.flatVector<int>(leftIds, INTEGER());
    auto leftMatrixVector = maker.arrayVector<float>(leftMatrix, REAL());

    RowVectorPtr leftRowVector =
          maker.rowVector({"l_id", "m1"}, {leftIdVector, leftMatrixVector});


    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    core::PlanNodeId leftTableScanNodeId;

    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator)
                      .values({leftRowVector})
                      .capturePlanNodeId(leftTableScanNodeId)
                      .rowNumber({});

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
    } else {
      throw std::runtime_error(
          fmt::format("Non-supported benchmark kernel: {}", kernel));
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

  void benchmarkTorchNN(
      std::string model,
      int batchSize,
      int featureSize,
      int layer1Size,
      int layer2Size,
      int verbose,
      int numRepeat) {
    PlanBuilder plan;
    if (model == "FFNN") {
      plan = getTorchDNNPlan(batchSize, featureSize, layer1Size, layer2Size);
    } else if (model == "Dense" || model == "Relu" || model == "Softmax") {
      plan = getTorchDNNKernelPlan(model, batchSize, featureSize, layer1Size);
    } else {
      throw std::runtime_error(
          fmt::format("Non-supported benchmark model: {}", model));
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

  void benchmarkOperator(
      std::string op,
      int lIndexMax,
      int rIndexMax,
      int dim1,
      int dim2,
      bool reverseOrder,
      int verbose,
      int numRepeat) {
    PlanBuilder plan;
    if (op == "HashJoin") {
      plan = getHashJoinPlan(lIndexMax, rIndexMax, dim1, dim2, reverseOrder);
    } else if (op == "NestedLoopJoin") {
      plan =
          getNestedLoopJoinPlan(lIndexMax, rIndexMax, dim1, dim2, reverseOrder);
    } else {
      throw std::runtime_error(
          fmt::format("Non-supported benchmark Op: {}", op));
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
DEFINE_string(mode, "DL", "Benchmark mode: DL or DB");
DEFINE_string(model, "", "FFNN");
DEFINE_int32(batch_size, 500, "Batch Size");
DEFINE_string(kernel, "", "Benchmark DL kernel");
DEFINE_string(op, "", "Benchmark DB Op");
DEFINE_int32(feature_size, 100, "Feature Size");
DEFINE_int32(l_max_index, 100, "");
DEFINE_int32(r_max_index, 100, "");
DEFINE_int32(dim1, 100, "Second dimension of left table");
DEFINE_int32(dim2, 100, "Second dimension of right table");
DEFINE_int32(l1size, 100, "Size of hidden layer 1");
DEFINE_int32(l2size, 100, "Size of hidden layer 2");
// DEFINE_int32(dim3, 100, "Second dimension of right table");
// DEFINE_int32(dim4, 100, "Second dimension of right table");
DEFINE_int32(num_repeat, 5, "Number of repeat run");
DEFINE_int32(verbose, 0, "Verbose");
DEFINE_bool(
    reverse_order,
    false,
    "Whether reverse order(if applicable, like JOIN)");
// DEFINE_int32(num_driver, 8, "Number of driver");

int main(int argc, char** argv) {
  Eigen::setNbThreads(16);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  std::string mode = FLAGS_mode;
  std::string model = FLAGS_model;
  int batchSize = FLAGS_batch_size;
  int numRepeat = FLAGS_num_repeat;
  std::string kernel = FLAGS_kernel;
  std::string op = FLAGS_op;
  int featureSize = FLAGS_feature_size;
  int dim2 = FLAGS_dim2;
  int verbose = FLAGS_verbose;

  DLKernelBenchmarkTest dlKernelBenchmark;

  // Check if flag is valid
  if (mode == "DL" && kernel != "") {
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
    dlKernelBenchmark.benchmarkKernel(
        kernel, batchSize, featureSize, dim2, verbose, numRepeat);
  } else if (mode == "TorchNN" && model != "") { 
    int layer1Size = FLAGS_l1size;
    int layer2Size = FLAGS_l2size;

    if (verbose >= 1) {
      std::cout
          << fmt::format(
                 "[INFO] Benchmark TorchNN Model: {} \n \t # Batch Size: {}, numRepeat: {}, featureSize: {}, l1Size: {}, l2Size: {}",
                 model,
                 batchSize,
                 numRepeat,
                 featureSize,
                 layer1Size,
                 layer2Size)
          << std::endl;
    }
    dlKernelBenchmark.benchmarkTorchNN(model, batchSize, featureSize, layer1Size, layer2Size, verbose, numRepeat);
  
  } else if (mode == "DB" && op != "") {
    bool reverseOrder = FLAGS_reverse_order;
    int lIndexMax = FLAGS_l_max_index;
    int rIndexMax = FLAGS_r_max_index;
    int dim1 = FLAGS_dim1;
    // int dim3 = FLAGS_dim3;
    // int dim4 = FLAGS_dim4;

    if (verbose >= 1) {
      std::cout
          << fmt::format(
                 "[INFO] Benchmark Op: {} \n \t # lIndexMax: {}, rIndexMax: {}, numRepeat: {}, dim1: {}, dim2: {}, reverseOrder: {}",
                 op,
                 lIndexMax,
                 rIndexMax,
                 numRepeat,
                 dim1,
                 dim2,
                 reverseOrder)
          << std::endl;
    }
    dlKernelBenchmark.benchmarkOperator(
        op, lIndexMax, rIndexMax, dim1, dim2, reverseOrder, verbose, numRepeat);
  } else {
    throw std::runtime_error(fmt::format(
        "Non-valid Configs! kernel needs to work with mode: DL. Op needs to work with mode: DB. \n \t\t Current Setting: mode: {}, kernel: {}, op: {} ",
        mode,
        kernel,
        op));
  }
}