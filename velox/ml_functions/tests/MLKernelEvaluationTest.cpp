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
#define EIGEN_USE_BLAS
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <cblas.h>
#include <folly/init/Init.h>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include "velox/common/base/Fs.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/exec/Task.h"

#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::memory;
using namespace facebook::velox::core;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

// TODO: Refactor
class MLKernelEvaluationTest : public HiveConnectorTestBase {
 public:
  MLKernelEvaluationTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    SetUp();
  }

  ~MLKernelEvaluationTest() {}

  /// Run the demo.
  void run(
      int numDriver,
      int memoryPoolSizeMB,
      int spillMemThresholdMB,
      bool enableSpill,
      int repeatRun);
  void testMatMul();
  void testMatAdd();

  std::unique_ptr<MemoryManager> memoryManager_;

  uint64_t kMemoryCapacity = 512 * MB;
  uint64_t kInitMemoryPoolCapacity = 16 * MB;
  uint64_t kMinMemoryPoolCapacityTransferSize = 8 * MB;

  std::shared_ptr<core::QueryCtx> newQueryCtx(int64_t memoryCapacity) {
    std::unordered_map<std::string, std::shared_ptr<Config>> configs;
    std::shared_ptr<MemoryPool> pool =
        memory::MemoryManager::getInstance()->addRootPool(
            "", memoryCapacity, memory::MemoryReclaimer::create());
    std::unordered_map<std::string, std::string> queryConfigValues = {};
    std::unordered_map<std::string, std::string> myMapWithValues = {
        {core::QueryConfig::kSpillEnabled, "true"},
        {core::QueryConfig::kJoinSpillEnabled, "true"},
        {core::QueryConfig::kJoinSpillMemoryThreshold, "10485760"},
        //  {core::QueryConfig::kSpillableReservationGrowthPct, "1"},
        /*
        kSpillPartitionBits is removed after PR 5890,
        kJoinSpillPartitionBits and kAggregationSpillPartitionBits are
        introduced Please consider how to replace it by check the following
        link: https://github.com/facebookincubator/velox/pull/5890
        */
        //  {core::QueryConfig::kSpillPartitionBits, "1"}
    };
    auto queryCtx = std::make_shared<core::QueryCtx>(
        executor_.get(),
        queryConfigValues,
        configs,
        cache::AsyncDataCache::getInstance(),
        std::move(pool));
    return queryCtx;
  }
  FlatVectorPtr<float> get_tensor(std::ifstream& file, int size, int lines);
  FlatVectorPtr<float>
  get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines);

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() {
    HiveConnectorTestBase::TearDown();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  void TestBody() override {}

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::MemoryManager::getInstance()->addLeafPool();
  VectorMaker maker{pool_.get()};
};

void MLKernelEvaluationTest::testMatMul() {
  // Test case of Matrix Multiplication
  // input feature: 8 x 4
  // weights: 4 x 2
  // output: 8 x 2
  std::vector<int> filterColumn = {0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<std::vector<float>> inputFeatures = {
      {1, 1, 1, 1},
      {2, 2, 2, 2},
      {3, 3, 3, 3},
      {4, 4, 4, 4},
      {5, 5, 5, 5},
      {6, 6, 6, 6},
      {7, 7, 7, 7},
      {8, 8, 8, 8}};

  std::vector<std::vector<float>> weights = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
  std::vector<std::vector<float>> bias = {{1, 2}};

  exec::registerVectorFunction(
      "mat_mul",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(weights)), 4, 2));
  exec::registerVectorFunction(
      "mat_vector_add",
      MatrixVectorAddition::signatures(),
      std::make_unique<MatrixVectorAddition>(
          std::move(flattenVectorToPointer(bias)), 2));

  std::vector<std::vector<float>> weights2 = {{1, 1}, {2, 2}};
  exec::registerVectorFunction(
      "mat_mul2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          std::move(flattenVectorToPointer(weights2)), 2, 2));

  auto inputRowVector = maker.rowVector(
      {"attr", "x"},
      {maker.flatVector<int>(filterColumn),
       maker.arrayVector<float>(inputFeatures, REAL())});

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .filter("attr = 1")
                    // .project({"attr", "mat_mul(x)"})
                    // .project({"attr", "mat_mul2(mat_vector_add(mat_mul(x)))"})
                    .project({"attr", "mat_vector_add(mat_mul(x))"})
                    .planNode();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).maxDrivers(4).copyResults(
          pool_.get());

  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void MLKernelEvaluationTest::testMatAdd() {
  std::cout << "==================" << std::endl;
  std::cout << "Test case of Matrix Addtion" << std::endl;
  // Eigen::setNbThreads(48);
  int num_rows = 5;
  int num_cols = 10;
  int size = num_rows * num_cols;

  auto weights = maker.flatVector<float>(size);
  for (int i = 0; i < size; i++) {
    weights->set(i, i * 10);
  }

  std::vector<std::vector<float>> inputVectors;
  for (int i = 0; i < num_rows; i++) {
    std::vector<float> inputVector;
    for (int j = 0; j < num_cols; j++) {
      inputVector.push_back(i * j);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});

  // step1: Register
  exec::registerVectorFunction(
      "mat_add",
      MatrixAddition::signatures(),
      std::make_unique<MatrixAddition>(
          weights->values()->asMutable<float>(), num_cols));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"mat_add(x)"})
                    .planNode();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Matrix Addition (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  // std::cout << "Results:" << results->toString() << std::endl;
  // std::cout << results->toString(0, results->size()) << std::endl;
}

void MLKernelEvaluationTest::run(
    int numDriver,
    int memoryPoolSizeMB,
    int spillMemThresholdMB,
    bool enableSpill,
    int repeatRun) {
  testMatMul();
  // testMatAdd();
}

DEFINE_bool(spill, false, "Whether enable spilling");
DEFINE_int32(spill_threshold, 30, "Set spill memory threshold");
DEFINE_int32(memory_pool, 100, "Set memory pool size");
DEFINE_int32(repeat, 5, "Number of repeat run");
DEFINE_int32(num_driver, 1, "Number of driver");
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  bool enableSpill = FLAGS_spill;
  int memoryPoolSizeMB = FLAGS_memory_pool;
  int spillMemoryThresholdMB = FLAGS_spill_threshold;
  int repeatRun = FLAGS_repeat;
  int numDriver = FLAGS_num_driver;
  MLKernelEvaluationTest demo;
  demo.run(
      numDriver,
      memoryPoolSizeMB,
      spillMemoryThresholdMB,
      enableSpill,
      repeatRun);
}
