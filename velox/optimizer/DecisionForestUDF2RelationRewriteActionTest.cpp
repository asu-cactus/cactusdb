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
#include "CataLog.h"

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
        // Register hiveconnector for file splits.
        auto hiveConnector =
            connector::getConnectorFactory(
                                           connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
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

    void registerFunctions(std::string modelPath, int numCols) {
        std::cout << "To register function for TreePrediction" << std::endl;

        exec::registerVectorFunction(
                                     "decision_tree_predict",
                                     TreePrediction::signatures(),
                                     std::make_unique<TreePrediction>(
                                                                      0, fmt::format("{}/0.txt", modelPath), numCols, false));

        std::cout << "To register type for Tree" << std::endl;

        registerCustomType("tree_type", std::make_unique<TreeTypeFactories>());

        std::cout << "To register function for VeloxTreePrediction" << std::endl;

        exec::registerVectorFunction(
                                     "velox_decision_tree_predict",
                                     VeloxTreePrediction::signatures(),
                                     std::make_unique<VeloxTreePrediction>(numCols));

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
                                                                        modelPath, numCols, true));
    }

     std::vector<std::vector<float>> parseCSVFile(std::string filePath,
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

        return inputArrayVector;
    }

    std::vector<std::shared_ptr<TempFilePath>> loadData(std::string& path, int numRows, int numCols, int numSplits = 8) {
        std::vector<std::vector<float>> inputData =
            parseCSVFile(path, numRows, numCols);

        RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
        std::vector<std::shared_ptr<TempFilePath>> paths;
        
        // split input data into multiple partitions
        size_t partitionSize = ceil(numRows / float(numSplits));
        size_t countProcessedRows = 0;
        for (size_t i = 0; i < numSplits; i++) {
            auto startIdx = inputData.begin() + i * partitionSize;
            auto endIdx = inputData.begin() + (i + 1) * partitionSize;
            endIdx = (endIdx < inputData.end()) ? endIdx : inputData.end();
            size_t numSampleInPartition = (i + 1) * partitionSize <= numRows
                ? partitionSize
                : numRows - i * partitionSize;

            std::vector<std::vector<float>> partialData(startIdx, endIdx);
            auto featureArrayVector = maker.arrayVector<float>(partialData, REAL());
            std::vector<int> indexes = randomGenerator.genIntRange(
                i * partitionSize, i * partitionSize + numSampleInPartition);
            auto indexFlatVector = maker.flatVector<int>(indexes);
            RowVectorPtr inputRowVector = maker.rowVector(
                {"idx", "v"},
                {std::move(indexFlatVector), std::move(featureArrayVector)});
            auto file = TempFilePath::create();
            auto config = std::make_shared<facebook::velox::dwrf::Config>();
            writeToFile(file->path, {inputRowVector}, config);
            paths.push_back(file);
            countProcessedRows += numSampleInPartition;
        }
        assert(countProcessedRows == numRows);
        return paths;
    }

    void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
        while (!task->isFinished()) {
            usleep(1000); // 0.01 second.
        }
    }

    /**
     * @brief A function to run logical plan.
     * 
     * @param filePath The file path for the data source file to be split.
     * @param numRows The number of rows for data source.
     * @param numSplits The number of file splits.
     * @param myPlan The pointer to the planBuilder which builds the logical plan.
     * @param p0 The planNodeID for the plan node that needs to add file splits.
     */
    void runDecisionForestPlan(
                               int numThreads,
                               PlanBuilder& myPlan,
                               CataLog &cataLog,
                               int batchRows) {
        // Create hivesplits for file.
        // Initializes executor.
        std::shared_ptr<folly::Executor> executor_{
            std::make_shared<folly::CPUThreadPoolExecutor>(
                                                           std::thread::hardware_concurrency())};
        // Initializes queryCtx.
        std::shared_ptr<core::QueryCtx> queryCtx_{
            std::make_shared<core::QueryCtx>(executor_.get())};
        // Set queryCtx config.
        queryCtx_->testingOverrideConfigUnsafe(
                                               {{core::QueryConfig::kPreferredOutputBatchBytes, "100000"},
                                               {core::QueryConfig::kMaxOutputBatchRows, std::to_string(batchRows)}});

        std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();

        CursorParameters params;
        params.maxDrivers = numThreads;
        params.planNode = myPlan.planNode();
        params.queryCtx = queryCtx_;
        bool noMoreSplits = false;

        auto addSplits = [&noMoreSplits, &cataLog](exec::Task* task) {
            auto idFileAddrMap = cataLog.getIdAddressMap();
            std::vector<core::PlanNodeId> ids;
            if (!noMoreSplits) {
                for (const auto& entry : idFileAddrMap) {
                    core::PlanNodeId key = entry.first;
                    const std::vector<std::shared_ptr<TempFilePath>> fileAddr = entry.second;
                    auto hiveSplits = makeHiveConnectorSplits(fileAddr);
                    for (auto& split : hiveSplits) {
                        task->addSplit(key, exec::Split(std::move(split)));
                    }
                    ids.push_back(key);
                }

                for (auto id: ids){
                    task->noMoreSplits(id);
                }
            }
            noMoreSplits = true;
        };

        auto [cursor, actualResults] = readCursor(params, addSplits);
        waitForTaskCompletion(cursor->task().get());

        int totalNumData = 0;
        int totalNumBatch = 0;
        for (auto batchedData : actualResults) {
            batchedData = std::move(batchedData);
            int batchSize = batchedData->size();
            totalNumBatch += 1;
            totalNumData += batchSize;

        }

        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();

        std::stringstream ss;

        ss << "numThreads:" << numThreads << ", numBatch:" << totalNumBatch << ", numData:" << totalNumData << std::endl;

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

    void testRewriteDecisionForestUDFPlan(std::string modelPath, std::string dataPath, bool rewrite, int numSplits, int numTreeSplit, int batchRows) {
        // register functions and types that are needed for this test
        int numRows, numCols;
        countRowsAndColumnsFromCSV(dataPath, numRows, numCols);
        std::cout << fmt::format("[INFO] Stats. from CSV file: numRows: {}, numCols: {}", numRows, numCols) << std::endl;
        registerFunctions(modelPath, numCols);

        // prepare features that are needed for this test
        // inputs are split and paths are returned
        auto inputPaths = loadData(dataPath, numRows, numCols, numSplits);
        RowTypePtr inputRowType = ROW({"idx", "v"}, {INTEGER(), ARRAY(REAL())});

        // create a plan for decision forest using UDF-centric style
        core::PlanNodeId p0;

        auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
        CataLog cataLog;
        auto myPlan = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(inputRowType)
            .capturePlanNodeId(p0)
            .project({"decision_forest_predict(v)"});
        cataLog.setIdAddressMap(p0, inputPaths);
        cataLog.setVectorIdMap(p0, "v");
        // Get the logical plan                  
        auto planNode = myPlan.planNode();
        // Create ruleManager
        RuleManager1 ruleManager;
        // Create planState
        PlanState1 planState(ruleManager);

        if (rewrite) {
            // Get possible actions for this plan
            planState.getPossibleActions(planNode, cataLog);
            // Print possible actions
            for (const auto& entry : planState.actionsPair) {
                std::cout << entry.first << ": " << entry.second << std::endl;
            }
            // Choose one action from possible actions (Now we only pick the first one, later it would be choosen by MCTS)
            auto it = planState.actionsPair.begin();
            std::pair<std::string, std::string> testAction("decision_forest_predict", "DecisionForestUDF2RelationRewriteAction");
            // Take one rewritten action
            planState.takeAction(planNode, nullptr, maker, myPlan, pool_, planNodeIdGenerator, {testAction}, cataLog, numTreeSplit);
            // Update the planState (getPossibleAction after apply one action)
            planState.update(myPlan, cataLog);
        }
        std::cout << "Query Plan: \n" << myPlan.planNode()->toString(true, true) << std::endl;
        // Run the rewritten plan
        runDecisionForestPlan(8 /*numThreads*/, myPlan, cataLog, batchRows);
    }

private:
    std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

    VectorMaker maker{pool_.get()};
};

DEFINE_string(model_path, "/home/velox/resources/model/fraud_xgboost_10_8", "Path to model");
DEFINE_string(data_path, "/home/velox/resources/data/creditcard_test.csv", "Path to csv file");
DEFINE_bool(rewrite, true, "Rewrite or not");
DEFINE_int32(num_split, 1, "Number of splits for input");
DEFINE_int32(numTreeSplit, 8, "Number of splits for tree");
DEFINE_int32(batchRows, 10000, "Number of rows in a batch");

int main(int argc, char** argv) {
    folly::init(&argc, &argv, false);
    memory::MemoryManager::initialize({});

    std::string modelPath = FLAGS_model_path;
    std::string dataPath = FLAGS_data_path;
    bool rewrite = FLAGS_rewrite;
    int numSplits = FLAGS_num_split;
    int numTreeSplit = FLAGS_numTreeSplit;
    int batchRows = FLAGS_batchRows;

    DecisionForestUDF2RelationRewriteActionTest demo;

    std::cout << fmt::format("Model: {}, Data: {}, Rewrite: {}", modelPath, dataPath, rewrite) << std::endl;

    demo.testRewriteDecisionForestUDFPlan(modelPath, dataPath, rewrite, numSplits, numTreeSplit, batchRows);

}
