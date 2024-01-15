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
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/expression/VectorFunction.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/exec/FilterProject.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"

// Custom headers
#include "RewriteAction.h"
#include "TwoLayerUDF2TorchNNRewriteAction.h"
#include "RuleManager.h"
#include "PlanState.h"
#include "DecisionForestUDF2RelationRewriteAction.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/core/Expressions.h"
#include "velox/core/ITypedExpr.h"
#include "velox/core/PlanNode.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"

#include "RewriteAction.h"
#include "RuleManager.h"
#include "PlanState.h"
#include "DecisionForestUDF2RelationRewriteAction.h"

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

class DecisionForestUDF2RelationRewriteActionTest : public HiveConnectorTestBase {
 public:
  DecisionForestUDF2RelationRewriteActionTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, nullptr);
    connector::registerConnector(hiveConnector);
  }

  ~DecisionForestUDF2RelationRewriteActionTest() {
    TearDown();
  }

  void SetUp() override {}

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  void registerFunctions() {
    std::cout << "To register function for TreePrediction" << std::endl;

    exec::registerVectorFunction(
        "decision_tree_predict",
        TreePrediction::signatures(),
        std::make_unique<TreePrediction>(
            0, "resources/model/fraud_xgboost_10_8/0.txt", 28, false));

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
            "resources/model/fraud_xgboost_10_8", 28, true));
  }

  ArrayVectorPtr parseCSVFile(
      VectorMaker& maker,
      std::string filePath,
      int numRows,
      int numCols) {
    int size = numRows * numCols;

    std::cout << "Loading tensor of size " << size << " from " << filePath
              << std::endl;

    std::ifstream file(filePath.c_str());

    std::vector<std::vector<float>> inputArrayVector;

    int index = 0;

    std::string line;

    while (numRows--) { // Read a line from the file

      std::vector<float> curRow(numCols);

      std::getline(file, line);

      std::istringstream iss(
          line); // Create an input string stream from the line

      std::string numberStr;

      int colIndex = 0;

      while (std::getline(
          iss, numberStr, ',')) { // Read each number separated by comma
        //
        float number = std::stof(numberStr); // Convert the string to float

        if (colIndex < numCols)

          curRow[colIndex] = number;

        colIndex++;
      }

      inputArrayVector.push_back(curRow);
    }

    file.close();

    ArrayVectorPtr tensor = maker.arrayVector<float>(inputArrayVector);

    return tensor;
  }

  RowVectorPtr loadData(std::string& path, int numRows, int numCols) {
    ArrayVectorPtr inputArrayVector =
        parseCSVFile(maker, path, numRows, numCols);

    std::vector<int32_t> indexVector;

    for (int i = 0; i < numRows; i++) {
      indexVector.push_back(i);
    }

    auto inputIndexVector = maker.flatVector<int32_t>(indexVector);

    return maker.rowVector(
        {"row_id", "x"}, {inputIndexVector, inputArrayVector});
  }

  void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  void writeToVeloxFile(
      RowVectorPtr rowVectors,
      int numRows,
      int numSplits,
      std::string filePath) {
    auto config = std::make_shared<facebook::velox::dwrf::Config>();

    uint64_t kSizeKB = 1024UL;

    uint32_t rows = numRows / numSplits + 1;

    config->set(facebook::velox::dwrf::Config::STRIPE_SIZE, 779 * kSizeKB);

    config->set(facebook::velox::dwrf::Config::ROW_INDEX_STRIDE, rows);

    writeToFile(filePath, {rowVectors}, config);
  }

  void runDecisionForestPlan(
      std::string filePath,
      int numRows,
      int numSplits,
      PlanBuilder& myPlan,
      core::PlanNodeId p0) {
    auto hiveSplits = makeHiveConnectorSplits(
        filePath, numSplits, dwio::common::FileFormat::DWRF);

    std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};

    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

    queryCtx_->testingOverrideConfigUnsafe(
        {{core::QueryConfig::kPreferredOutputBatchBytes, "1000000"},
         {core::QueryConfig::kMaxOutputBatchRows, "10000"}});

    auto task = exec::Task::create(
        "0",
        myPlan.planFragment(),
        0,
        queryCtx_,
        [](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if (result)
            std::cout << result->toString() << std::endl;
          return exec::BlockingReason::kNotBlocked;
        });

    std::cout << "Hive splits:" << std::endl;

    for (auto& split : hiveSplits) {
      // std::cout << split->toString() << std::endl;
      task->addSplit(p0, exec::Split(std::move(split)));
    }
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    int veloxThreads = 8;

    task->start(veloxThreads);

    task->noMoreSplits(p0);

    // Start task with 2 as maximum drivers and wait for execution to finish

    waitForFinishedDrivers(task);

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::stringstream ss;

    ss << numRows << "," << numSplits << "," << veloxThreads << ",";

    std::cout << "Time for Decision Forest Prediction with Input Data (sec): "
              << std::endl;

    std::cout << ss.str()
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << " secs" << std::endl;

    unregisterCustomType("tree_type");
  }

  void testRewriteDecisionForestUDFPlan(bool rewrite) {
    // register functions and types that are needed for this test
    registerFunctions();

    // prepare features that are needed for this test
    int numRows = 56962;

    int numCols = 28;

    std::string dataFilePath = "resources/data/creditcard_test.csv";

    auto inputRowVector = loadData(dataFilePath, numRows, numCols);

    // write the features to a file

    int numSplits = 8;

    auto file = TempFilePath::create();

    writeToVeloxFile(inputRowVector, numRows, numSplits, file->path);

    // create a plan for decision forest using UDF-centric style
    core::PlanNodeId p0;

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                      //.values({inputRowVector})
                      .tableScan(asRowType(inputRowVector->type()))
                      .capturePlanNodeId(p0)
                      .project({"decision_forest_predict(x)"});

    auto planNode = myPlan.planNode();
    // Create ruleManager
    RuleManager ruleManager;
    // Create planState
    PlanState planState(ruleManager);

    if (rewrite) {
      // Get possible actions for this plan
      planState.getPossibleActions(planNode);
      // Print possible actions
      for (const auto& entry : planState.actionsPair) {
        std::cout << entry.first << ": " << entry.second << std::endl;
      }
      // Choose one action from possible actions (Now we only pick the first one, later it would be choosen by MCTS)
      auto it = planState.actionsPair.begin();
      std::string testAction  = it->first;
      // Take the action
      planState.takeAction(planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, {testAction});
      // Update the planState (getPossibleAction after apply one action)
      planState.update(myPlan);
    }

    // Run the rewritten plan
    runDecisionForestPlan(file->path, numRows, numSplits, myPlan, p0);
  }

 private:
  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::addDefaultLeafMemoryPool();

  VectorMaker maker{pool_.get()};
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);

  DecisionForestUDF2RelationRewriteActionTest demo;

  bool rewrite = true;

  if (argc > 1) {
    if (strcmp(argv[1], "N") == 0) {
      rewrite = false;
    }
  }

  if (rewrite) {
    std::cout
        << "================= Run UDF-Centric Decision Forest w/ Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testRewriteDecisionForestUDFPlan(true);

  } else {
    std::cout
        << "================= Run UDF-Centric Decision Forest w/o Rewriting ==================="
        << std::endl
        << std::endl;

    demo.testRewriteDecisionForestUDFPlan(false);
  }

  std::cout
      << "--" << std::endl
      << "[Usage] " << std::endl
      << "./_build/release/velox/optimizer/rewrite_test Y  //run decision forest model with rewriting"
      << std::endl
      << "./_build/release/velox/optimizer/rewrite_test N  //run decision forest model with rewriting"
      << std::endl
      << "By default: Y is used" << std::endl;
}
