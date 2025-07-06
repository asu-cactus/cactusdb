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
#pragma once
#include <H5Cpp.h>
#include <json/json.h>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/optimizer/CataLog.h"
#include "velox/optimizer/Helper.h"
#include "velox/optimizer/PlanState.h"
#include "velox/optimizer/Register.h"
#include "velox/optimizer/RuleManager.h"

#define BUFFER_SIZE 1024
using namespace optimization;
using namespace facebook::velox;
using namespace facebook::velox::core;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;

facebook::velox::RowVectorPtr copyRowVector(
    const facebook::velox::RowVectorPtr& rowVector,
    std::shared_ptr<memory::MemoryPool>& pool) {
  // Vector to hold the deep copies of child vectors.
  std::vector<facebook::velox::VectorPtr> childCopies;

  // Iterate through the children of the RowVector.
  for (const auto& child : rowVector->children()) {
    if (child) {
      // Allocate a new vector and copy the data into it.
      auto childCopy = facebook::velox::BaseVector::create(
          child->type(), child->size(), pool.get());

      // Copy the data from the source vector to the new vector.
      childCopy->copy(child.get(), 0, 0, child->size());

      childCopies.push_back(childCopy);
    } else {
      // If the child is null, push a null vector.
      childCopies.push_back(nullptr);
    }
  }

  // Create a new RowVector with the copied children.
  return std::make_shared<facebook::velox::RowVector>(
      pool.get(), // Memory pool for allocation.
      rowVector->type(), // The RowType of the original RowVector.
      nullptr, // Nulls buffer (nullptr for no nulls).
      rowVector->size(), // Number of rows in the RowVector.
      std::move(childCopies)); // Null count (if available).
};

/**
 * @brief A function to run logical plan.
 *
 * @param pool The memory pool for the Velox executor.
 * @param numThreads The number of Velox executor threads.
 * @param numSplits The number of file splits.
 * @param myPlan The pointer to the planBuilder which builds the logical plan.
 * @param cataLog A class storing metadata and information related to UDFs and
 * data sources.
 */
float runPlanWithCataLog(
    std::shared_ptr<memory::MemoryPool>& pool,
    int numThreads,
    PlanBuilder& myPlan,
    CataLog& cataLog,
    std::vector<RowVectorPtr>& finalResult,
    int repeatRun = 1,
    int verbose = 1,
    bool copyResult = false) {
  float totalElapsedTime = 0;
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
        {{core::QueryConfig::kPreferredOutputBatchBytes, "10000000"},
         {core::QueryConfig::kMaxOutputBatchRows, "1000000"},
         {core::QueryConfig::kPreferredOutputBatchRows, "1000"}});

    // Add hivesplits to the target plan node (data source node).
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
          const std::vector<std::string> fileAddr = entry.second;
          // check file exists
          for (const auto& addr : fileAddr) {
            if (!fs::exists(addr)) {
              LOG(ERROR) << "[ERROR] File not exists: " << addr << std::endl;
              return;
            }
          }
          auto fileFormat = cataLog.getIdFileFormat(key);
          auto hiveSplits = HiveConnectorTestBase::makeHiveConnectorSplits(
              fileAddr, fileFormat);

          for (auto& split : hiveSplits) {
            task->addSplit(key, exec::Split(std::move(split)));
          }

          ids.push_back(key);
        }

        for (auto id : ids) {
          task->noMoreSplits(id);
        }
      }
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
        int batchSize = batchedData->size();
        if (verbose == 3) {
          std::cout << fmt::format(
                           "[INFO] Batched Data: {}, Batch Size:{} \n",
                           dataIdx,
                           batchSize)
                    << batchedData->toString() << std::endl;
        } else if (verbose == 4) {
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
        // finalResult.push_back(batchedData);
        if (copyResult) {
          finalResult.push_back(copyRowVector(batchedData, pool));
        }
      }
    }
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

float runPlanWithCataLog(
    std::shared_ptr<memory::MemoryPool>& pool,
    int numThreads,
    PlanBuilder& myPlan,
    CataLog& cataLog,
    int repeatRun = 1,
    int verbose = 1) {
  std::vector<RowVectorPtr> finalResult;
  return runPlanWithCataLog(
      pool,
      numThreads,
      myPlan,
      cataLog,
      finalResult,
      repeatRun,
      verbose,
      false /* copyResult */);
}

Json::Value receiveJsonFromSocket(int clientSocket) {
  char messageBuffer[BUFFER_SIZE];
  memset(messageBuffer, 0, BUFFER_SIZE);
  recv(clientSocket, messageBuffer, BUFFER_SIZE, 0);
  Json::CharReaderBuilder jsonReader;
  Json::Value receivedJsonMessage;
  std::istringstream jsonStream(messageBuffer);
  Json::parseFromStream(jsonReader, jsonStream, &receivedJsonMessage, nullptr);
  return receivedJsonMessage;
}

void sendJsonBySocket(Json::Value jsonMessage, int clientSocket) {
  std::string jsonMessageStr = jsonMessage.toStyledString();
  send(clientSocket, jsonMessageStr.c_str(), jsonMessageStr.length(), 0);
}

void sendAcknowledgment(int clientSocket) {
  const char* ack_message = "ACK";
  send(clientSocket, ack_message, strlen(ack_message), 0);
}

std::vector<std::vector<float>> loadHDF5Array(
    const std::string& filename,
    const std::string& datasetName) {
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File not found: " + filename);
  }
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

  // Read data into a 1D vector
  std::vector<float> flatData(rows * cols);
  dataset.read(flatData.data(), H5::PredType::NATIVE_FLOAT);

  // Convert to 2D vector
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result[i][j] = flatData[i * cols + j];
    }
  }

  // Close the dataset and file
  dataset.close();
  file.close();

  return result;
};

std::vector<std::string> readTextFile(const std::string& filename) {
  std::vector<std::string> lines;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  return lines;
}

std::vector<std::string> loadHDF5StringArray(
    const std::string& filename,
    const std::string& datasetName) {
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File not found: " + filename);
  }
  H5::H5File file(filename, H5F_ACC_RDONLY);
  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  // Get the number of dimensions
  int rank = dataspace.getSimpleExtentNdims();

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

  // Read data into a 1D vector of strings
  std::vector<std::string> flatData(rows * cols);

  // Assuming the data is stored as fixed-size strings or strings encoded in a
  // way that can be read as a single block
  dataset.read(flatData.data(), H5::PredType::C_S1); // C_S1 for string types

  // Convert to 2D vector if the dataset is 2-dimensional
  std::vector<std::string> result;
  if (rank == 1) {
    result = std::move(flatData);
  } else if (rank == 2) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result.push_back(flatData[i * cols + j]);
      }
    }
  }

  // Close the dataset and file
  dataset.close();
  file.close();

  return result;
}

PlanBuilder setupTPCxAIQuery(
    std::string queryType,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
  std::string queryOptType =
      getEnvVar("CD_VELOX_QUERY_OPT_TYPE"); // env used for ablation study of
  PlanBuilder queryPlan;

  auto finicialAccountDataRowType =
      ROW({"fa_customer_sk", "transaction_limit"}, {BIGINT(), DOUBLE()});
  auto finicialTransactionsDataRowType = ROW(
      {"amount", "iban", "sender_id", "receiver_id", "transaction_id", "time"},
      {DOUBLE(), VARCHAR(), BIGINT(), VARCHAR(), BIGINT(), VARCHAR()});
  auto orderDataRowType =
      ROW({"o_order_id", "o_customer_sk", "weekday", "date", "store"},
          {BIGINT(), BIGINT(), VARCHAR(), VARCHAR(), BIGINT()});
  auto lineitemDataRowType =
      ROW({"li_order_id", "li_product_id", "quantity", "price"},
          {BIGINT(), BIGINT(), BIGINT(), DOUBLE()});
  auto productDataRowType = ROW(
      {"p_product_id", "name", "department"}, {BIGINT(), VARCHAR(), VARCHAR()});
  auto storeDeptDataRowType = ROW(
      {"store", "department", "num_of_week"}, {BIGINT(), VARCHAR(), BIGINT()});
  auto productRatingRowType =
      ROW({"user_id", "product_id"}, {BIGINT(), BIGINT()});
  auto customerDataRowType =
      ROW({"c_customer_sk",
           "c_customer_id",
           "c_current_addr_sk",
           "c_first_name",
           "c_last_name",
           "c_preferred_cust_flag",
           "c_birth_day",
           "c_birth_month",
           "c_birth_year",
           "c_birth_country",
           "c_login",
           "c_email_address"},
          {INTEGER(),
           VARCHAR(),
           INTEGER(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           INTEGER(),
           INTEGER(),
           INTEGER(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR()});
  auto orderReturnDataRowType =
      ROW({"or_order_id", "or_product_id", "or_return_quantity"},
          {INTEGER(), INTEGER(), INTEGER()});
  auto reviewDataRowType = ROW({"id", "text"}, {INTEGER(), VARCHAR()});

  std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

  if (dataDirPrefix == "") {
    // use default value:
    dataDirPrefix =
        "/home/velox/resources/data/parquet/tpcxai_sf1/final/serving/";
  }

  std::vector<std::string> finicialAccountDataPaths =
      getFilePathsFromDir(dataDirPrefix + "financial_account");
  std::vector<std::string> finicialTransactionsDataPaths =
      getFilePathsFromDir(dataDirPrefix + "financial_transactions");
  std::vector<std::string> orderDataPaths =
      getFilePathsFromDir(dataDirPrefix + "order");
  std::vector<std::string> lineitemDataPaths =
      getFilePathsFromDir(dataDirPrefix + "lineitem");
  std::vector<std::string> productDataPaths =
      getFilePathsFromDir(dataDirPrefix + "product");
  std::vector<std::string> storeDeptDataPaths =
      getFilePathsFromDir(dataDirPrefix + "store_dept");
  std::vector<std::string> productRatingDataPaths =
      getFilePathsFromDir(dataDirPrefix + "product_rating");
  std::vector<std::string> customerDataPaths =
      getFilePathsFromDir(dataDirPrefix + "customer");
  std::vector<std::string> orderReturnDataPaths =
      getFilePathsFromDir(dataDirPrefix + "order_returns");
  std::vector<std::string> reviewDataPaths =
      getFilePathsFromDir(dataDirPrefix + "review");

  int finicialAccountNumRows, finicialAccountNumCols,
      finicialTransactionsNumRows, finicialTransactionsNumCols, orderNumRows,
      orderNumCols, lineitemNumRows, lineitemNumCols, productNumRows,
      productNumCols, storeDeptNumRows, storeDeptNumCols, productRatingNumRows,
      productRatingNumCols, customerNumRows, customerNumCols,
      orderReturnNumRows, orderReturnNumCols, reviewNumRows, reviewNumCols;

  readDataStats(
      dataDirPrefix + "financial_account_stats.txt",
      finicialAccountNumRows,
      finicialAccountNumCols);
  readDataStats(
      dataDirPrefix + "financial_transactions_stats.txt",
      finicialTransactionsNumRows,
      finicialTransactionsNumCols);
  readDataStats(dataDirPrefix + "order_stats.txt", orderNumRows, orderNumCols);
  readDataStats(
      dataDirPrefix + "lineitem_stats.txt", lineitemNumRows, lineitemNumCols);
  readDataStats(
      dataDirPrefix + "product_stats.txt", productNumRows, productNumCols);
  readDataStats(
      dataDirPrefix + "store_dept_stats.txt",
      storeDeptNumRows,
      storeDeptNumCols);
  readDataStats(
      dataDirPrefix + "product_rating_stats.txt",
      productRatingNumRows,
      productRatingNumCols);
  readDataStats(
      dataDirPrefix + "customer_stats.txt", customerNumRows, customerNumCols);
  readDataStats(
      dataDirPrefix + "order_returns_stats.txt",
      orderReturnNumRows,
      orderReturnNumCols);
  readDataStats(
      dataDirPrefix + "review_stats.txt", reviewNumRows, reviewNumCols);
  if (queryType.find("uc3") != std::string::npos) {
    PlanNodeId readStoreDeptDataPlanNodeId;
    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(storeDeptDataRowType, {}, "")
            .capturePlanNodeId(readStoreDeptDataPlanNodeId)
            .project({
                "store",
                "department",
                "num_of_week",
                "store_id_encoder(array_constructor(CAST(store as INTEGER))) as store_id_encoded",
                "department_encoder(department) as department_encoded",
                "CAST(num_of_week / 156 AS REAL) as num_of_week_norm",
            })
            .project(
                {"store",
                 "department",
                 "num_of_week",
                 "transform(concat(store_id_encoded, department_encoded), x-> CAST(x as REAL))  as features1",
                 "array_constructor(num_of_week_norm) as features2"})
            .project(
                {"store",
                 "department",
                 "num_of_week",
                 "concat(features1, features2) as features"})
            .project(
                {"store",
                 "department",
                 "num_of_week",
                 "mat_vector_add1_6(mat_mul1_5(relu(mat_vector_add1_4(mat_mul1_3(relu(mat_vector_add1_2(mat_mul1_1(features)))))))) as prediction"});
    cataLog.setIdAddressMap(
        readStoreDeptDataPlanNodeId,
        storeDeptDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(readStoreDeptDataPlanNodeId, "store_dept");
    Source storeDeptSrc = Source(
        readStoreDeptDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(storeDeptNumRows, storeDeptNumCols)));
    cataLog.addSource(std::make_shared<Source>(storeDeptSrc));

  } else if (queryType.find("uc7-ml") != std::string::npos) {
    PlanNodeId readProductRatingDataPlanNodeId;

    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(productRatingRowType, {}, "")
            .capturePlanNodeId(readProductRatingDataPlanNodeId)
            .project(
                {"CAST (user_id AS INTEGER) AS user_id",
                 "CAST (product_id as INTEGER) AS product_id"})
            .project(
                {"user_id", "product_id", "svd(user_id, product_id) as pred"});
    cataLog.setIdAddressMap(
        readProductRatingDataPlanNodeId,
        productRatingDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(
        readProductRatingDataPlanNodeId, "product_rating");
    Source productRatingSrc = Source(
        readProductRatingDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(productRatingNumRows, productRatingNumCols)));
    cataLog.addSource(std::make_shared<Source>(productRatingSrc));
  } else if (queryType.find("uc8") != std::string::npos) {
    PlanNodeId readOrderDataPlanNodeId;
    PlanNodeId readLineitemDataPlanNodeId;
    PlanNodeId readProductDataPlanNodeId;

    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(orderDataRowType, {}, "")
            .capturePlanNodeId(readOrderDataPlanNodeId)
            .project({
                "o_order_id",
                "store",
                "CAST (date AS TIMESTAMP) AS date",
            })
            .project({
                "o_order_id",
                "store",
                "date",
                "day_of_week(date) as weekday",
            })
            .hashJoin(
                {"o_order_id"},
                {"li_order_id"},
                PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(lineitemDataRowType, {}, "")
                    .capturePlanNodeId(readLineitemDataPlanNodeId)
                    .planNode(),
                "",
                {"li_order_id",
                 "li_product_id",
                 "o_order_id",
                 "quantity",
                 "date",
                 "weekday"})
            .hashJoin(
                {"li_product_id"},
                {"p_product_id"},
                PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(productDataRowType, {}, "")
                    .capturePlanNodeId(readProductDataPlanNodeId)
                    .planNode(),
                "",
                {"li_order_id",
                 "o_order_id",
                 "quantity",
                 "date",
                 "department",
                 "weekday"})
            .partialAggregation(
                {"o_order_id", "date", "department", "quantity"},
                {"sum(quantity) as scan_count", "min(weekday) as weekday"})
            .finalAggregation()
            .project(
                {"o_order_id",
                 "date",
                 "array_constructor(quantity, scan_count, weekday) as features",
                 "department_encoder(department) as department_encoded"});
    if (queryType.find("ml") == std::string::npos) {
      queryPlan
          .project(
              {"o_order_id",
               "date",
               "transform(concat(features, department_encoded), x-> CAST(x as REAL)) as features"})
          .project(
              {"o_order_id",
               "date",
               "softmax(mat_vector_add1_8(mat_mul1_7(relu(mat_vector_add1_6(mat_mul1_5(relu(mat_vector_add1_4(mat_mul1_3(relu(mat_vector_add1_2(mat_mul1_1(features)))))))))))) as prediction"});
    } else {
      queryPlan =
          queryPlan
              .project(
                  {"o_order_id",
                   "date",
                   "transform(concat(department_encoded, features), x-> CAST(x as REAL)) as features"})
              .project(
                  {"o_order_id", "date", "decision_forest_predict(features)"});
    }

    cataLog.setIdAddressMap(
        readOrderDataPlanNodeId,
        orderDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readLineitemDataPlanNodeId,
        lineitemDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readProductDataPlanNodeId,
        productDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(readOrderDataPlanNodeId, "order");
    cataLog.addNodeIdRelationName(readLineitemDataPlanNodeId, "lineitem");
    cataLog.addNodeIdRelationName(readProductDataPlanNodeId, "product");
    Source orderSrc = Source(
        readOrderDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(OutputStat(orderNumRows, orderNumCols)));
    Source lineitemSrc = Source(
        readLineitemDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(lineitemNumRows, lineitemNumCols)));
    Source productSrc = Source(
        readProductDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(productNumRows, productNumCols)));
    cataLog.addSource(std::make_shared<Source>(orderSrc));
    cataLog.addSource(std::make_shared<Source>(lineitemSrc));
    cataLog.addSource(std::make_shared<Source>(productSrc));
  } else if (queryType.find("uc10") != std::string::npos) {
    PlanNodeId readFinancialAccountDataPlanNodeId;
    PlanNodeId readFinancialTransactionsDataPlanNodeId;
    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(finicialAccountDataRowType, {}, "")
            .capturePlanNodeId(readFinancialAccountDataPlanNodeId)
            .hashJoin(
                {"fa_customer_sk"},
                {"sender_id"},
                PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(finicialTransactionsDataRowType, {}, "")
                    .capturePlanNodeId(readFinancialTransactionsDataPlanNodeId)
                    .project(
                        {"transaction_id",
                         "sender_id",
                         "CAST(hour(CAST(time AS TIMESTAMP)) as DOUBLE) as business_hour_norm",
                         "amount"})
                    .planNode(),
                "",
                {"transaction_id",
                 "sender_id",
                 "business_hour_norm",
                 "amount",
                 "transaction_limit"})
            .project(
                {"transaction_id",
                 "amount / transaction_limit as amount_norm",
                 "business_hour_norm"})
            .project(
                {"transaction_id",
                 "transform(array_constructor(amount_norm, business_hour_norm), x-> CAST(X as REAL)) as features"});
    if (queryType.find("ml") == std::string::npos) {
      queryPlan = queryPlan.project(
          {"transaction_id",
           "sigmoid(mat_vector_add1_4(mat_mul1_3(relu(mat_vector_add1_2(mat_mul1_1(features)))))) as prediction"});
    } else {
      queryPlan = queryPlan.project(
          {"transaction_id",
           "sigmoid(mat_vector_add1_2(mat_mul1_1(features))) as prediction"});
    }

    cataLog.setIdAddressMap(
        readFinancialAccountDataPlanNodeId,
        finicialAccountDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readFinancialTransactionsDataPlanNodeId,
        finicialTransactionsDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(
        readFinancialAccountDataPlanNodeId, "financial_account");
    cataLog.addNodeIdRelationName(
        readFinancialTransactionsDataPlanNodeId, "financial_transactions");
    Source financialAccountSrc = Source(
        readFinancialAccountDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(finicialAccountNumRows, finicialAccountNumCols)));
    Source financialTransactionsSrc = Source(
        readFinancialTransactionsDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(OutputStat(
            finicialTransactionsNumRows, finicialTransactionsNumCols)));
    cataLog.addSource(std::make_shared<Source>(financialAccountSrc));
    cataLog.addSource(std::make_shared<Source>(financialTransactionsSrc));
  } else if (queryType == "readCustomer") {
    PlanNodeId readCustomerDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(customerDataRowType, {}, "")
                    .capturePlanNodeId(readCustomerDataPlanNodeId)
                    .project(
                        {"c_customer_sk",
                         "c_customer_id",
                         "c_current_addr_sk",
                         "c_first_name",
                         "c_last_name",
                         "c_preferred_cust_flag",
                         "c_birth_day",
                         "c_birth_month",
                         "c_birth_year",
                         "c_birth_country",
                         "c_login",
                         "c_email_address"});
    cataLog.setIdAddressMap(
        readCustomerDataPlanNodeId,
        customerDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(readCustomerDataPlanNodeId, "customer");
    Source customerSrc = Source(
        readCustomerDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(customerNumRows, customerNumCols)));
    cataLog.addSource(std::make_shared<Source>(customerSrc));
  } else if (queryType == "readReview") {
    PlanNodeId readReviewDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(reviewDataRowType, {}, "")
                    .capturePlanNodeId(readReviewDataPlanNodeId)
                    .project({"id", "text"});
    cataLog.setIdAddressMap(
        readReviewDataPlanNodeId,
        reviewDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(readReviewDataPlanNodeId, "review");
    Source reviewSrc = Source(
        readReviewDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(OutputStat(reviewNumRows, reviewNumCols)));
    cataLog.addSource(std::make_shared<Source>(reviewSrc));
  } else if (queryType == "readOrderReturn") {
    PlanNodeId readOrderReturnDataPlanNodeId;
    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(orderReturnDataRowType, {}, "")
            .capturePlanNodeId(readOrderReturnDataPlanNodeId)
            .project({"or_order_id", "or_product_id", "or_return_quantity"});
    cataLog.setIdAddressMap(
        readOrderReturnDataPlanNodeId,
        orderReturnDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.addNodeIdRelationName(
        readOrderReturnDataPlanNodeId, "order_returns");
    Source orderReturnSrc = Source(
        readOrderReturnDataPlanNodeId,
        Source::Type::FILE,
        std::make_shared<OutputStat>(
            OutputStat(orderReturnNumRows, orderReturnNumCols)));
    cataLog.addSource(std::make_shared<Source>(orderReturnSrc));

  } else {
    throw std::runtime_error(
        "[setupTPCxAIQuery] Unsupported query type: " + queryType);
  }
  return queryPlan;
}

PlanBuilder setupMovielensDBQuery(
    std::string queryType,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
  std::string queryOptType =
      getEnvVar("CD_VELOX_QUERY_OPT_TYPE"); // env used for ablation study of
                                            // rewrite rules
  PlanBuilder queryPlan;

  auto movieTagDataRowType =
      ROW({"mt_movie_id", "mt_relevance_score"}, {INTEGER(), ARRAY(REAL())});
  auto movieDataRowType =
      ROW({"m_movie_id",
           "m_title",
           "m_genres",
           "m_spoken_languages",
           "m_popularity",
           "m_vote_average",
           "m_vote_count",
           "m_overview"},
          {INTEGER(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           REAL(),
           REAL(),
           INTEGER(),
           VARCHAR()});
  auto userDataRowType =
      ROW({"u_user_id", "u_gender", "u_age", "u_occupation", "u_zipcode"},
          {INTEGER(), VARCHAR(), INTEGER(), INTEGER(), VARCHAR()});
  auto ratingDataRowType =
      ROW({"r_user_id", "r_movie_id", "r_rating", "r_timestamp"},
          {INTEGER(), INTEGER(), INTEGER(), INTEGER()});

  std::string dataDirPrefix = getEnvVar("CD_DATA_DIR_PREFIX");

  if (dataDirPrefix == "") {
    // use default value:
    dataDirPrefix = "/home/velox/resources/data/parquet/movielens/final/";
  }

  std::vector<std::string> movieTagDataPaths =
      getFilePathsFromDir(dataDirPrefix + "movie_tag_relevance");
  std::vector<std::string> movieDataPaths =
      getFilePathsFromDir(dataDirPrefix + "movie");
  std::vector<std::string> userDataPaths =
      getFilePathsFromDir(dataDirPrefix + "user");
  std::vector<std::string> ratingDataPaths =
      getFilePathsFromDir(dataDirPrefix + "rating");

  int movieTagNumRows, movieTagNumCols, movieNumRows, movieNumCols, userNumRows,
      userNumCols, ratingNumRows, ratingNumCols;

  readDataStats(
      dataDirPrefix + "movie_tag_relevance_stats.txt",
      movieTagNumRows,
      movieTagNumCols);
  readDataStats(dataDirPrefix + "movie_stats.txt", movieNumRows, movieNumCols);
  readDataStats(dataDirPrefix + "user_stats.txt", userNumRows, userNumCols);
  readDataStats(
      dataDirPrefix + "rating_stats.txt", ratingNumRows, ratingNumCols);
  if (queryType.find("user_only") != std::string::npos) {
    PlanNodeId readUserDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(userDataRowType, {}, "")
                    .capturePlanNodeId(readUserDataPlanNodeId)
                    .project(
                        {"u_user_id",
                         "u_gender",
                         "u_age",
                         "u_occupation",
                         "u_zipcode"});
    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("movie_only") != std::string::npos) {
    PlanNodeId readMovieDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(movieDataRowType, {}, "")
                    .capturePlanNodeId(readMovieDataPlanNodeId)
                    .project(
                        {"m_movie_id",
                         "m_title",
                         "m_genres",
                         "m_spoken_languages",
                         "m_popularity",
                         "m_vote_average",
                         "m_vote_count",
                         "m_overview"});
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("movie_rating_only") != std::string::npos) {
    PlanNodeId readRatingDataPlanNodeId;
    queryPlan =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .tableScan(ratingDataRowType, {}, "")
            .capturePlanNodeId(readRatingDataPlanNodeId)
            .project({"r_user_id", "r_movie_id", "r_rating", "r_timestamp"});
    cataLog.setIdAddressMap(
        readRatingDataPlanNodeId,
        ratingDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("movie_tag_only") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    queryPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .tableScan(movieTagDataRowType, {}, "")
                    .capturePlanNodeId(readMovieTagDataPlanNodeId)
                    .project({"mt_movie_id", "mt_relevance_score"});
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
  } else if (queryType.find("q1") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    PlanNodeId readRatingDataPlanNodeId1;
    PlanNodeId readRatingDataPlanNodeId2;
    if (queryOptType.empty() || queryOptType == "" ||
        queryOptType.find("fusion") != std::string::npos) {
      // default query
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "movie_description_array"});

      queryPlan =
          readUserAvgRatingPlan
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.planNode(),
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "movie_description_array"})
              .filter("m_genres LIKE '\%Action\%'")
              .project(
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres_embed",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
    } else if (queryOptType.find("ffnn_pushdown") != std::string::npos) {
      // ffnn pushdown
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .project(
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .filter("m_genres LIKE '\%Action\%'")
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating"});

      queryPlan =
          readUserAvgRatingPlan
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.planNode(),
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres_embed",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});

    } else if (
        queryOptType.find("ffnn_pushdown_n_reorder") != std::string::npos) {
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .filter("m_genres LIKE '\%Action\%'")
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .project(
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating"});

      queryPlan =
          readUserAvgRatingPlan
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.planNode(),
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating",
                   "m_movie_id",
                   "m_genres",
                   "m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating",
                   "m_movie_id",
                   "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                   "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres_embed",
                   "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features",
                   "m_movie_id",
                   "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out",
                   "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
    } else if (
        queryOptType.find("decomposition_pushdown") != std::string::npos) {
      // optimized query
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating",
                   "movie_description_array"});

      queryPlan =
          readUserAvgRatingPlan
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   //  "user_id_embedding(user_id_encoder(array_constructor(u_user_id)))
                   //  as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features"})
              .project(
                  {"u_user_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out"})
              // .orderBy({"u_user_id"}, false)
              .nestedLoopJoin(
                  readMovieAvgRatingPlan.orderBy({"m_movie_id"}, false)
                      .project(
                          {"m_movie_id",
                           "movie_description_array",
                           "m_genres AS m_genres1",
                           "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                           "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres_embed",
                           "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
                      .project({
                          "m_movie_id",
                          "movie_description_array",
                          "m_genres1",
                          "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features",
                      })
                      .project(
                          {"m_movie_id",
                           "movie_description_array",
                           "m_genres1",
                           "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
                      .planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "user_nn_out",
                   "movie_nn_out",
                   "m_genres1",
                   "movie_description_array"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "user_nn_out",
                   "movie_nn_out",
                   "m_genres1",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction"})
              .filter("trending_prediction = 1")
              .filter("m_genres1 LIKE '\%Action\%'")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)",
                   "trending_prediction",
                   "m_genres1"});

    } else if (queryOptType.find("optimized") != std::string::npos) {
      // optimized query
      auto readUserAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(userDataRowType, {}, "")
              .capturePlanNodeId(readUserDataPlanNodeId)
              .hashJoin(
                  {"u_user_id"},
                  {"r_user_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId1)
                      .project(
                          {"r_user_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_user_id"},
                          {"avg(r_rating) as u_user_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"u_user_id",
                   "u_gender",
                   "u_age",
                   "u_occupation",
                   "u_user_mean_rating"});

      auto readMovieAvgRatingPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieDataRowType, {}, "")
              .capturePlanNodeId(readMovieDataPlanNodeId)
              .filter("m_genres LIKE '\%Action\%'")
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS movie_description_array",
              })
              .project({
                  "m_movie_id",
                  "m_genres",
                  "m_spoken_languages",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(movie_description_array)))))))))) AS trending_prediction",
              })
              .filter("trending_prediction = 1")
              .hashJoin(
                  {"m_movie_id"},
                  {"r_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(ratingDataRowType, {}, "")
                      .capturePlanNodeId(readRatingDataPlanNodeId2)
                      .project(
                          {"r_movie_id", "if (r_rating > 3, 1, 0) as r_rating"})
                      .partialAggregation(
                          {"r_movie_id"},
                          {"avg(r_rating) as m_movie_mean_rating"})
                      .finalAggregation()
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "m_genres",
                   "m_spoken_languages",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "m_movie_mean_rating"});

      queryPlan =
          readUserAvgRatingPlan
              .project(
                  {"u_user_id",
                   "user_id_embedding(user_id_encoder(convert_int_array(u_user_id))) as u_user_id_embed",
                   "gender_embedding(gender_encoder(u_gender)) as u_gender",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation",
                   "transform(array_constructor(u_user_mean_rating), x -> CAST(x as REAL)) as u_user_mean_rating"})
              .project(
                  {"u_user_id",
                   "concat(u_user_id_embed, u_gender, u_age, u_occupation,u_user_mean_rating) as user_tower_features"})
              .project(
                  {"u_user_id",
                   "relu(batch_norm1_3(mat_vector_add1_3(mat_mul1_3(relu(batch_norm1_2(mat_vector_add1_2(mat_mul1_2(relu(batch_norm1_1(mat_vector_add1_1(mat_mul1_1(user_tower_features)))))))))))) as user_nn_out"})
              .nestedLoopJoin(
                  readMovieAvgRatingPlan
                      // .orderBy({"m_movie_id"}, false)
                      .project(
                          {"m_movie_id",
                           "movie_id_embedding(movie_id_encoder(convert_int_array(m_movie_id))) as m_movie_id_embed",
                           "sequence_pooling(genres_embedding(genres_encoder(split(m_genres, '|')))) as m_genres_embed",
                           "transform(array_constructor(m_movie_mean_rating), x -> CAST(x as REAL)) as m_movie_mean_rating"})
                      .project({
                          "m_movie_id",
                          "concat(m_movie_id_embed, m_genres_embed, m_movie_mean_rating) as movie_tower_features",
                      })
                      .project(
                          {"m_movie_id",
                           "relu(batch_norm2_3(mat_vector_add2_3(mat_mul2_3(relu(batch_norm2_2(mat_vector_add2_2(mat_mul2_2(relu(batch_norm2_1(mat_vector_add2_1(mat_mul2_1(movie_tower_features)))))))))))) as movie_nn_out"})
                      .planNode(),
                  {"u_user_id", "m_movie_id", "user_nn_out", "movie_nn_out"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity(user_nn_out, movie_nn_out)"});
    } else {
      // NON-SUPPORTED
      throw std::invalid_argument(
          "Unsupported query optimization type: " + queryOptType);
    }

    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readRatingDataPlanNodeId1,
        ratingDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readRatingDataPlanNodeId2,
        ratingDataPaths,
        dwio::common::FileFormat::PARQUET);

    cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
    std::shared_ptr<OutputStat> userStats =
        std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
    Source userSrc =
        Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
    cataLog.addSource(std::make_shared<Source>(userSrc));

    cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
    std::shared_ptr<OutputStat> movieStats =
        std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
    Source movieSrc =
        Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
    cataLog.addSource(std::make_shared<Source>(movieSrc));

    cataLog.addNodeIdRelationName(readRatingDataPlanNodeId1, "movie_rating");
    cataLog.addNodeIdRelationName(readRatingDataPlanNodeId2, "movie_rating");
    std::shared_ptr<OutputStat> ratingStats =
        std::make_shared<OutputStat>(OutputStat(ratingNumRows, ratingNumCols));
    Source ratingSrc1 =
        Source(readRatingDataPlanNodeId1, Source::Type::FILE, ratingStats);
    cataLog.addSource(std::make_shared<Source>(ratingSrc1));
    Source ratingSrc2 =
        Source(readRatingDataPlanNodeId2, Source::Type::FILE, ratingStats);
    cataLog.addSource(std::make_shared<Source>(ratingSrc2));
  } else if (queryType.find("q2") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    if (queryOptType.empty() || queryOptType == "" ||
        queryOptType == "mlq2-fusion" || queryOptType == "mlq2-mul2join") {
      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .project(
                          {"m_movie_id",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count"})
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 5, false)
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "u_user_id",
                  "u_age",
                  "u_occupation",
                  "gender_encoder(u_gender) as u_gender_encoded",
                  "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_score",
                  "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features",
                  "llm_ffnn_interest_scaler(transform(array_constructor(u_age, u_occupation), x-> CAST(X as REAL)))  AS u_interest_features",
              })
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "m_trending_features",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                   "u_final_interest_features",
                   "mt_relevance_score"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "mt_relevance_score",
                   "trending_prediction"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                   "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                   "gender_embedding(u_gender_encoded) as u_gender_embed",
                   "mt_relevance_score",
                   "trending_prediction",
                   "user_interest_prediction"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "trending_prediction",
                   "user_interest_prediction",
                   "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out",
                   "concat(u_age_embed, u_occupation_embed, u_gender_embed) as categorical_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "trending_prediction",
                   "user_interest_prediction",
                   "concat(bottom_mlp_out, categorical_features) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "trending_prediction",
                   "user_interest_prediction",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"})
              .filter("trending_prediction = 1")
              .filter("user_interest_prediction = 1");
    }
    if (queryOptType == "decomposition_pushdown") {
      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .project(
                          {"m_movie_id",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count"})
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
              })
              .project({
                  "m_movie_id",
                  "mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
                  "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 5, false)
                      .project(
                          {"u_user_id",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                           "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                           "gender_embedding(gender_encoder(u_gender)) as u_gender_embed"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "gender_encoder(u_gender) as u_gender_encoded",
                   "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features",
                   "llm_ffnn_interest_scaler(transform(array_constructor(u_age, u_occupation), x-> CAST(X as REAL)))  AS u_interest_features",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "m_trending_features",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                   "u_final_interest_features",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .filter("trending_prediction = 1")
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "u_gender_encoded",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .filter("user_interest_prediction = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "concat(bottom_mlp_out, u_age_embed, u_occupation_embed, u_gender_embed) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"});

    } else if (queryOptType.find("optimized") != std::string::npos) {
      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .project(
                          {"m_movie_id",
                           "m_popularity",
                           "m_vote_average",
                           "m_vote_count",
                           "llm_ffnn_minmax_scaler(transform(array_constructor(m_popularity, m_vote_average, m_vote_count), x-> CAST(X as REAL)))  AS m_trending_features"})
                      .project({
                          "m_movie_id",
                          "m_popularity",
                          "m_vote_average",
                          "m_vote_count",
                          "argmax(softmax(mat_vector_add3_6(mat_mul3_5(relu(mat_vector_add3_4(mat_mul3_3(relu(mat_vector_add3_2(mat_mul3_1(m_trending_features)))))))))) AS trending_prediction",
                      })
                      .filter("trending_prediction = 1")
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count"})
              .project({
                  "m_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_score",
                  "mt_movie_id",
                  "m_popularity",
                  "m_vote_average",
                  "m_vote_count",
              })
              .project(
                  {"m_movie_id",
                   "mt_relevance_score",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "relu(mat_vector_add11_2(mat_mul11_1(mt_relevance_score))) as bottom_mlp_out"});
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 5, false)
                      .project(
                          {"u_user_id",
                           "u_age",
                           "u_gender",
                           "u_occupation",
                           "age_embedding(age_encoder(convert_int_array(u_age))) as u_age_embed",
                           "occupation_embedding(occupation_encoder(convert_int_array(u_occupation))) as u_occupation_embed",
                           "gender_embedding(gender_encoder(u_gender)) as u_gender_embed"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "mt_movie_id",
                   "m_popularity",
                   "m_vote_average",
                   "m_vote_count",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "transform(array_constructor(if (u_gender = 'M', 1, 0)), x->Cast(x AS real)) as u_gender",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "bottom_mlp_out",
                   "llm_ffnn_interest_scaler(transform(array_constructor(u_age, u_occupation), x-> CAST(X as REAL)))  AS u_interest_features",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "m_movie_id",
                   "concat(u_gender, u_interest_features, mt_relevance_score) as u_final_interest_features",
                   "mt_relevance_score",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed",
                   "bottom_mlp_out"})
              .project(
                  {"u_user_id",
                   "u_age",
                   "u_occupation",
                   "m_movie_id",
                   "argmax(softmax(mat_vector_add9_4(mat_mul9_3(relu(mat_vector_add9_2(mat_mul9_1(u_final_interest_features))))))) AS user_interest_prediction",
                   "bottom_mlp_out",
                   "u_age_embed",
                   "u_occupation_embed",
                   "u_gender_embed"})
              .filter("user_interest_prediction = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "concat(bottom_mlp_out, u_age_embed, u_occupation_embed, u_gender_embed) as top_mlp_input"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "relu(mat_vector_add12_6(mat_mul12_5(relu(mat_vector_add12_4(mat_mul12_3(relu(mat_vector_add12_2(mat_mul12_1(top_mlp_input))))))))) as top_mlp_out"});
    }
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);

    cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
    std::shared_ptr<OutputStat> userStats =
        std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
    Source userSrc =
        Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
    cataLog.addSource(std::make_shared<Source>(userSrc));

    cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
    std::shared_ptr<OutputStat> movieStats =
        std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
    Source movieSrc =
        Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
    cataLog.addSource(std::make_shared<Source>(movieSrc));

    cataLog.addNodeIdRelationName(
        readMovieTagDataPlanNodeId, "movie_relevance_tag");
    std::shared_ptr<OutputStat> movieTagStats = std::make_shared<OutputStat>(
        OutputStat(movieTagNumRows, movieTagNumCols));
    Source movieTagSrc =
        Source(readMovieTagDataPlanNodeId, Source::Type::FILE, movieTagStats);
    cataLog.addSource(std::make_shared<Source>(movieTagSrc));

  } else if (queryType.find("q3") != std::string::npos) {
    PlanNodeId readMovieTagDataPlanNodeId;
    PlanNodeId readMovieTagDataPlanNodeId2;
    PlanNodeId readUserDataPlanNodeId;
    PlanNodeId readMovieDataPlanNodeId;
    if (queryOptType.empty() || queryOptType == "" ||
        queryOptType == "mlq3-fusion" || queryOptType == "mlq3-mul2join") {
      auto movieTagQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId2)
              .project(
                  {"mt_movie_id AS mt_movie_id1",
                   "relu(mat_vector_add20_4(mat_mul20_3(relu(mat_vector_add20_2(mat_mul20_1(mt_relevance_score)))))) AS mt_relevance_ir1"});

      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .limit(0, 1000, false)
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average",
                   "m_genres"});
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 50, false)
                      .project(
                          {"u_user_id",
                           "CAST (u_age AS REAL) AS u_age",
                           "if (u_gender = 'M', 1.0, 0.0) AS u_gender",
                           "CAST (u_occupation AS REAL) AS u_occupation"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_score",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "model_features",
                   "mt_relevance_score",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_score",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features",
                   "user_movie_interest_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir",
                   "user_movie_rating_pred",
                   "model_features",
                   "user_movie_interest_pred"})
              .nestedLoopJoin(
                  movieTagQueryPlan.planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id1",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_ir",
                   "mt_relevance_ir1",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id1",
                   "mt_movie_id",
                   "m_genres",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   "user_movie_interest_pred",
                   "user_movie_rating_pred",
                   "cosine_sim"})
              .filter("user_movie_interest_pred = 1")
              .filter("user_movie_rating_pred = 5")
              .filter("m_genres LIKE '\%Adventure\%'");

    } else if (queryOptType.find("mlq3-pushdown") != std::string::npos) {
      auto movieTagQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId2)
              .project(
                  {"mt_movie_id AS mt_movie_id1",
                   "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir1"});

      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .limit(0, 1000, false)
                      .filter("m_genres LIKE '\%Adventure\%'")
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_vote_average"})
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir",
                  "m_popularity",
                  "m_vote_average",
              })
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_ir",
                  "m_popularity",
                  "m_vote_average",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 50, false)
                      .project(
                          {"u_user_id",
                           "CAST (u_age AS REAL) AS u_age",
                           "if (u_gender = 'M', 1.0, 0.0) AS u_gender",
                           "CAST (u_occupation AS REAL) AS u_occupation"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "m_popularity",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "model_features",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .filter("user_movie_interest_pred = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features"})
              .filter("user_movie_rating_pred = 5")
              .nestedLoopJoin(
                  movieTagQueryPlan.planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "mt_movie_id1",
                   "mt_relevance_ir1"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim"});

    } else if (queryOptType.find("mlq3-optimized") != std::string::npos) {
      auto movieTagQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId2)
              .project(
                  {"mt_movie_id AS mt_movie_id1",
                   "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir1"});

      auto movieQueryPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieTagDataRowType, {}, "")
              .capturePlanNodeId(readMovieTagDataPlanNodeId)
              .hashJoin(
                  {"mt_movie_id"},
                  {"m_movie_id"},
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(movieDataRowType, {}, "")
                      .capturePlanNodeId(readMovieDataPlanNodeId)
                      .limit(0, 1000, false)
                      .filter("m_genres LIKE '\%Adventure\%'")
                      .planNode(),
                  "",
                  {"m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_score",
                   "m_popularity",
                   "m_genres",
                   "m_vote_average"})
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "relu(mat_vector_add10_4(mat_mul10_3(relu(mat_vector_add10_2(mat_mul10_1(mt_relevance_score)))))) AS mt_relevance_ir",
                  "m_popularity",
                  "m_genres",
                  "m_vote_average",
              })
              .project({
                  "m_movie_id",
                  "mt_movie_id",
                  "mt_relevance_ir",
                  "m_popularity",
                  "m_genres",
                  "m_vote_average",
              });
      queryPlan =
          movieQueryPlan
              .nestedLoopJoin(
                  PlanBuilder(planNodeIdGenerator, pool_.get())
                      .tableScan(userDataRowType, {}, "")
                      .capturePlanNodeId(readUserDataPlanNodeId)
                      .project(
                          {"u_user_id", "u_age", "u_gender", "u_occupation"})
                      .limit(0, 50, false)
                      .project(
                          {"u_user_id",
                           "CAST (u_age AS REAL) AS u_age",
                           "if (u_gender = 'M', 1.0, 0.0) AS u_gender",
                           "CAST (u_occupation AS REAL) AS u_occupation"})
                      .planNode(),
                  {"u_user_id",
                   "u_age",
                   "u_gender",
                   "u_occupation",
                   "m_movie_id",
                   "mt_movie_id",
                   "mt_relevance_ir",
                   "m_popularity",
                   "m_genres",
                   "m_vote_average"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "mt_movie_id",
                   "m_genres",
                   "mt_relevance_ir",
                   "transform(array_constructor(u_age, u_gender, u_occupation, m_popularity, m_vote_average), x->Cast(x AS real))   AS model_features"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "model_features",
                   "m_genres",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add15_6(mat_mul15_5(relu(mat_vector_add15_4(mat_mul15_3(relu(mat_vector_add15_2(mat_mul15_1(model_features))))))))) AS user_movie_interest_pred"})
              .filter("user_movie_interest_pred = 1")
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "m_genres",
                   //  "user_movie_interest_pred",
                   "mt_relevance_ir",
                   "argmax(mat_vector_add16_6(mat_mul16_5(relu(mat_vector_add16_4(mat_mul16_3(relu(mat_vector_add16_2(mat_mul16_1(model_features))))))))) AS user_movie_rating_pred",
                   "model_features"})
              .filter("user_movie_rating_pred = 5")
              .nestedLoopJoin(
                  movieTagQueryPlan.planNode(),
                  {"u_user_id",
                   "m_movie_id",
                   "mt_relevance_ir",
                   "mt_movie_id1",
                   "mt_relevance_ir1"})
              .project(
                  {"u_user_id",
                   "m_movie_id",
                   "cosine_similarity_q3(mt_relevance_ir, mt_relevance_ir1) as cosine_sim "});
    }
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieTagDataPlanNodeId2,
        movieTagDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readMovieDataPlanNodeId,
        movieDataPaths,
        dwio::common::FileFormat::PARQUET);
    cataLog.setIdAddressMap(
        readUserDataPlanNodeId,
        userDataPaths,
        dwio::common::FileFormat::PARQUET);

    cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
    std::shared_ptr<OutputStat> userStats =
        std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
    Source userSrc =
        Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
    cataLog.addSource(std::make_shared<Source>(userSrc));

    cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
    std::shared_ptr<OutputStat> movieStats =
        std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
    Source movieSrc =
        Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
    cataLog.addSource(std::make_shared<Source>(movieSrc));

    cataLog.addNodeIdRelationName(
        readMovieTagDataPlanNodeId, "movie_relevance_tag");
    std::shared_ptr<OutputStat> movieTagStats = std::make_shared<OutputStat>(
        OutputStat(movieTagNumRows, movieTagNumCols));
    Source movieTagSrc =
        Source(readMovieTagDataPlanNodeId, Source::Type::FILE, movieTagStats);
    cataLog.addSource(std::make_shared<Source>(movieTagSrc));

    cataLog.addNodeIdRelationName(
        readMovieTagDataPlanNodeId2, "movie_relevance_tag");
    std::shared_ptr<OutputStat> movieTagStats2 = std::make_shared<OutputStat>(
        OutputStat(movieTagNumRows, movieTagNumCols));
    Source movieTagSrc2 =
        Source(readMovieTagDataPlanNodeId2, Source::Type::FILE, movieTagStats2);
    cataLog.addSource(std::make_shared<Source>(movieTagSrc2));
  }

  return queryPlan;
};

std::vector<std::string> sampleUserMovieFilterExpr(
    std::string filterTable,
    int randomSeed = -1) {
  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  if (randomSeed != -1) {
    timestampSeed = randomSeed;
  }

  RandomSampler randomSampler = RandomSampler(timestampSeed);

  std::vector<std::string> ratingTimestampFilterExprs = {
      // range  max : 1046454590 | min : 956703932
      "r_timestamp <= 964152800",
      "r_timestamp >= 964152816",
      "r_timestamp < 974687965",
      "r_timestamp >= 975768738",
      "r_timestamp < 967588077",
  };

  std::vector<std::string> userGenderFilterExprs = {
      "u_gender = 'M'",
      "u_gender = 'F'",
  };

  std::vector<std::string> userAgeFilterExprs = {
      "u_age > 30",
      "u_age < 50",
      "u_age >= 20",
      "u_age <= 60",
      "u_age = 40",
      "u_age > 15",
      "u_age < 25",
      "u_age >= 55",
      "u_age <= 35",
      "u_age = 65",
  };

  std::vector<std::string> userOccupationFilterExprs = {
      "u_occupation = 10",
      "u_occupation < 5",
      "u_occupation > 15",
      "u_occupation = 0",
      "u_occupation >= 7",
      "u_occupation <= 3",
      "u_occupation = 12",
      "u_occupation < 20",
      "u_occupation > 8",
      "u_occupation = 18",
  };

  std::vector<std::string> userZipCodeFilterExprs = {
      "u_zipcode = '94043'",
      "u_zipcode = '94301'",
      "u_zipcode = '94305'",
      "u_zipcode = '94306'",
      "u_zipcode = '94309'",
      "u_zipcode = '80212'",
      "u_zipcode = '80213'",
      "u_zipcode = '80219'",
      "u_zipcode = '12301'",
      "u_zipcode = '40201'",
  };

  std::vector<std::string> movieGenresFilterExprs = {
      "m_genres LIKE '\%Action\%'",
      "m_genres LIKE '\%Comedy\%'",
      "m_genres LIKE '\%Drama\%'",
      "m_genres LIKE '\%Horror\%'",
      "m_genres LIKE '\%Sci-Fi\%'",
      "m_genres LIKE '\%Romance\%'",
      "m_genres LIKE '\%Adventure\%'",
      "m_genres LIKE '\%Thriller\%'",
      "m_genres LIKE '\%Fantasy\%'",
      "m_genres LIKE '\%Documentary\%'",
  };

  std::vector<std::string> movieSpokenLanguageFilterExprs = {
      "m_spoken_languages LIKE '\%English\%'",
      "m_spoken_languages LIKE '\%French\%'",
      "m_spoken_languages LIKE '\%German\%'",
      "m_spoken_languages LIKE '\%Japanese\%'",
      "m_spoken_languages LIKE '\%Spanish\%'",
      "m_spoken_languages LIKE '\%Italian\%'",
      "m_spoken_languages LIKE '\%Korean\%'",
      "m_spoken_languages LIKE '\%Mandarin\%'",
  };

  std::vector<std::string> moviePopularityFilterExprs = {
      "m_popularity > 5.0",
      "m_popularity < 3.0",
      "m_popularity >= 7.5",
      "m_popularity <= 2.5",
      "m_popularity = 8.0",
      "m_popularity > 4.0",
      "m_popularity < 6.0",
      "m_popularity >= 9.0",
      "m_popularity <= 1.0",
      "m_popularity = 3.5",

  };

  std::vector<std::string> movieVoteAverageFilterExprs = {
      "m_vote_average > 3.0",
      "m_vote_average < 2.0",
      "m_vote_average >= 4.5",
      "m_vote_average <= 1.5",
      "m_vote_average = 5.0",
      "m_vote_average > 1.0",
      "m_vote_average < 4.0",
      "m_vote_average >= 2.5",
      "m_vote_average <= 3.5",
      "m_vote_average = 3.0",
  };

  std::vector<std::string> movieVoteCountFilterExprs = {
      "m_vote_count > 500",
      "m_vote_count < 100",
      "m_vote_count >= 750",
      "m_vote_count <= 250",
      "m_vote_count = 300",
      "m_vote_count > 200",
      "m_vote_count < 600",
      "m_vote_count >= 1000",
      "m_vote_count <= 50",
      "m_vote_count = 150",
  };

  std::vector<std::vector<std::string>> predefinedUserFilterExprs = {
      userGenderFilterExprs,
      userAgeFilterExprs,
      userOccupationFilterExprs,
      userZipCodeFilterExprs,
  };

  std::vector<std::vector<std::string>> predefinedMovieFilterExprs = {
      movieGenresFilterExprs,
      movieSpokenLanguageFilterExprs,
      moviePopularityFilterExprs,
      movieVoteAverageFilterExprs,
      movieVoteCountFilterExprs,
  };

  std::vector<std::vector<std::string>> combinedFilterExprSets;

  std::vector<std::string> sampledFilterExprs;
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, timestampSeed);
  std::vector<std::vector<std::string>> sampleFilterPool;
  if (filterTable == "user") {
    randomGenerator.setIntRange(1, 3);
    sampleFilterPool = predefinedUserFilterExprs;
  } else if (filterTable == "movie") {
    randomGenerator.setIntRange(1, 3);
    sampleFilterPool = predefinedMovieFilterExprs;
  } else if (filterTable == "movie_user") {
    randomGenerator.setIntRange(1, 3);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        predefinedUserFilterExprs.begin(),
        predefinedUserFilterExprs.end());
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        predefinedMovieFilterExprs.begin(),
        predefinedMovieFilterExprs.end());
    sampleFilterPool = combinedFilterExprSets;
  } else if (filterTable == "user_movie_genres") {
    randomGenerator.setIntRange(1, 3);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        predefinedUserFilterExprs.begin(),
        predefinedUserFilterExprs.end());
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(), movieGenresFilterExprs);
    sampleFilterPool = combinedFilterExprSets;
  } else if (filterTable == "template1") {
    randomGenerator.setIntRange(0, 2);
    std::vector<std::vector<std::string>> template3FilterExprs = {
        movieGenresFilterExprs, movieSpokenLanguageFilterExprs};
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        template3FilterExprs.begin(),
        template3FilterExprs.end());
    sampleFilterPool = combinedFilterExprSets;
  } else if (filterTable == "template2") {
    randomGenerator.setIntRange(0, 2);
    std::vector<std::vector<std::string>> template2FilterExprs = {
        movieGenresFilterExprs, movieSpokenLanguageFilterExprs};
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        template2FilterExprs.begin(),
        template2FilterExprs.end());
    sampleFilterPool = combinedFilterExprSets;
  } else if (filterTable == "template3") {
    randomGenerator.setIntRange(0, 2);
    std::vector<std::vector<std::string>> template3FilterExprs = {
        movieGenresFilterExprs, movieSpokenLanguageFilterExprs};
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        template3FilterExprs.begin(),
        template3FilterExprs.end());
    sampleFilterPool = combinedFilterExprSets;
  } else if (filterTable == "age_gender_occupation_genre") {
    randomGenerator.setIntRange(1, 3);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(),
        predefinedUserFilterExprs.begin(),
        predefinedUserFilterExprs.end() - 1);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(), movieGenresFilterExprs);
    sampleFilterPool = combinedFilterExprSets;
  } else if (filterTable == "genre_rating") {
    randomGenerator.setIntRange(1, 3);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(), movieGenresFilterExprs);
    combinedFilterExprSets.insert(
        combinedFilterExprSets.end(), ratingTimestampFilterExprs);
    sampleFilterPool = combinedFilterExprSets;
  } else {
    throw std::invalid_argument(
        "Invalid table for sampling filter expression: " + filterTable);
  }

  std::set<std::vector<std::string>> usedExprSets;
  while (sampledFilterExprs.size() < randomGenerator.genRandomIntValue()) {
    auto pickedExprSets = randomSampler.sampleFromSets(1, sampleFilterPool)[0];
    if (usedExprSets.find(pickedExprSets) == usedExprSets.end()) {
      auto sampledFilterExpr =
          randomSampler.sampleFromSets(1, pickedExprSets)[0];
      sampledFilterExprs.push_back(sampledFilterExpr);
      usedExprSets.insert(pickedExprSets);
    }
  }

  return sampledFilterExprs;
};

std::vector<std::string> sampleTPCxAIFilterExpr(
    const std::string& filterTable,
    int randomSeed = -1) {
  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  if (randomSeed != -1) {
    timestampSeed = randomSeed;
  }

  std::vector<std::string> departmentFilterExprs = {
      "department = 'DSD GROCERY'",
      "department = 'IMPULSE MERCHANDISE'",
      "department = 'PERSONAL CARE'",
      "department = 'GROCERY DRY GOODS'",
      "department = 'PHARMACY OTC'",
      "department = 'PRODUCE'",
      "department = 'FINANCIAL SERVICES'",
      "department = 'MENS WEAR'",
      "department = 'DAIRY'",
      "department = 'AUTOMOTIVE'",
      "department = 'ELECTRONICS'",
  };
  std::vector<std::string> customerBirthDayFilterExprs = {
      // range from 1 to 31
      "c_birth_day < 15",
      "c_birth_day > 20",
      "c_birth_day >= 10",
      "c_birth_day <= 25",
      "c_birth_day <= 5",
  };
  std::vector<std::string> customerBirthCountryFilterExprs = {
      "c_birth_country = 'QATAR'",
      "c_birth_country = 'UZBEKISTAN'",
      "c_birth_country = 'SAINT HELENA'",
      "c_birth_country = 'ECUADOR'",
      "c_birth_country = 'MONGOLIA'",
      "c_birth_country = 'NORFOLK ISLAND'",
      "c_birth_country = 'HONDURAS'",
      "c_birth_country = 'SAUDI ARABIA'",
      "c_birth_country = 'PARAGUAY'",
      "c_birth_country = 'CYPRUS'",
      "c_birth_country = 'LEBANON'",
      "c_birth_country = 'NEW CALEDONIA'",
      "c_birth_country = 'ANGOLA'",
      "c_birth_country = 'MAYOTTE'",
      "c_birth_country = 'GUINEA-BISSAU'",
  };
  std::vector<std::string> orderWeekDayFilterExprs = {
      // range from 0 to 6
      "weekday < 2",
      "weekday > 3",
      "weekday >= 1",
      "weekday <= 5"};
  std::vector<std::string> orderTimeFilterExprs = {
      // range from 2012-01-02 to 2013-12-29
      "date < cast('2012-06-01 00:00:00' as timestamp)",
      "date > cast('2012-07-01 00:00:00' as timestamp)",
      "date >= cast('2013-08-01 00:00:00' as timestamp)",
      "date <= cast('2013-09-01 00:00:00' as timestamp)",
  };

  std::vector<std::string> lineitemPriceFilterExprs = {
      // range from 0.1 to 17.06
      "price < 3.0",
      "price > 6.0",
      "price >= 9.0",
      "price <= 12.0",
      "price >= 15.0",
  };
  std::vector<std::string> lineitemQuantityFilterExprs = {
      // range from 1 to 7
      "quantity < 7",
      "quantity > 3",
      "quantity >= 4",
      "quantity <= 5",
      "quantity = 6",
      "quantity = 2",
  };
  std::vector<std::string> financialTransactionsAmountFilterExprs = {
      // range from 0.01 to 15039.84
      "amount < 1000.0",
      "amount > 5000.0",
      "amount >= 10000.0",
      "amount <= 2000.0",
  };
  std::vector<std::string> financialTransactionsTimeFilterExprs = {
      // range from "2012-01-01 00:14:00" to "2013-12-30 23:16:00
      "time < cast('2012-06-01 00:00:00' as timestamp)",
      "time > cast('2012-07-01 00:00:00' as timestamp)",
      "time >= cast('2013-08-01 00:00:00' as timestamp)",
      "time <= cast('2013-09-01 00:00:00' as timestamp)"};
  std::vector<std::string> storeFilterExprs = {
      // range from 1 to 11
      "store_id = 1",
      "store_id = 2",
      "store_id = 3",
      "store_id = 4",
      "store_id = 5",
      "store_id = 6",
      "store_id = 7",
      "store_id = 8",
      "store_id = 9",
      "store_id = 10",
      "store_id = 11"};
  std::vector<std::string> numWeekFilterExprs = {
      // range from 104 to 155
      "num_of_week < 120",
      "num_of_week > 130",
      "num_of_week >= 140",
      "num_of_week <= 150",
  };
  std::vector<std::string> productIDFilterExprs = {
      // range from 1 to 706
      "product_id < 100",
      // "product_id < 200",
      // "product_id <= 300",
      // "product_id >= 400",
      // "product_id >= 500",
      "product_id >= 600",
  };

  std::vector<std::string> idReviewFilterExprs = {
      // range from 0 to 13432
      "id < 1243",
      "id >= 5834",
      "id <= 10341",
      "id > 2587",
      "id >= 9476",
  };

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, timestampSeed);
  std::vector<std::vector<std::string>> sampleFilterPool;
  if (filterTable.find("department") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), departmentFilterExprs);
  }
  if (filterTable.find("birthDay") != std::string::npos) {
    sampleFilterPool.insert(
        sampleFilterPool.end(), customerBirthDayFilterExprs);
  }
  if (filterTable.find("birthCountry") != std::string::npos) {
    sampleFilterPool.insert(
        sampleFilterPool.end(), customerBirthCountryFilterExprs);
  }
  if (filterTable.find("weekday") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), orderWeekDayFilterExprs);
  }
  if (filterTable.find("orderTime") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), orderTimeFilterExprs);
  }
  if (filterTable.find("price") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), lineitemPriceFilterExprs);
  }
  if (filterTable.find("quantity") != std::string::npos) {
    sampleFilterPool.insert(
        sampleFilterPool.end(), lineitemQuantityFilterExprs);
  }
  if (filterTable.find("amount") != std::string::npos) {
    sampleFilterPool.insert(
        sampleFilterPool.end(), financialTransactionsAmountFilterExprs);
  }
  if (filterTable.find("transactionTime") != std::string::npos) {
    sampleFilterPool.insert(
        sampleFilterPool.end(), financialTransactionsTimeFilterExprs);
  }
  if (filterTable.find("store") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), storeFilterExprs);
  }
  if (filterTable.find("numWeek") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), numWeekFilterExprs);
  }
  if (filterTable.find("product") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), productIDFilterExprs);
  }
  if (filterTable.find("idReview") != std::string::npos) {
    sampleFilterPool.insert(sampleFilterPool.end(), idReviewFilterExprs);
  }

  RandomSampler randomSampler = RandomSampler(timestampSeed);
  std::set<std::vector<std::string>> usedExprSets;
  std::vector<std::string> sampledFilterExprs;
  while (sampledFilterExprs.size() < randomGenerator.genRandomIntValue()) {
    auto pickedExprSets = randomSampler.sampleFromSets(1, sampleFilterPool)[0];
    if (usedExprSets.find(pickedExprSets) == usedExprSets.end()) {
      auto sampledFilterExpr =
          randomSampler.sampleFromSets(1, pickedExprSets)[0];
      sampledFilterExprs.push_back(sampledFilterExpr);
      usedExprSets.insert(pickedExprSets);
    }
  }

  return sampledFilterExprs;
}
// Function to write a string to a file
void writeStringToFile(const std::string& str, const std::string& filename) {
  // Create the folder if it does not exist
  std::filesystem::path folderPath =
      std::filesystem::path(filename).parent_path();
  if (!std::filesystem::exists(folderPath)) {
    std::filesystem::create_directories(folderPath);
  }
  // Open the file in write mode
  std::ofstream outfile(filename);

  // Check if the file opened successfully
  if (outfile.is_open()) {
    // Write the string to the file
    outfile << str;

    // Close the file
    outfile.close();
  } else {
    std::cerr << "Error: Could not open the file for writing." << std::endl;
  }
};

void outputAugmentedQueryPlan(
    CataLog& cataLog,
    PlanBuilder& plan,
    std::string outputPath = "") {
  auto serializedPlan = plan.planNode()->serialize();
  if (outputPath == "") {
    outputPath = "/home/velox/velox/optimizer/tests/serializedQueryPlan.json";
  }
  augmentSerializedPlan(serializedPlan, cataLog);
  writeStringToFile(folly::toJson(serializedPlan), outputPath);
};

void outputStructuredQueryPlan(PlanBuilder& plan) {
  auto structuredPlan = plan.planNode()->toString(true, true);
  writeStringToFile(
      structuredPlan,
      "/home/velox/velox/optimizer/tests/structuredQueryPlan.txt");
}

void deleteFilesInFolder(const std::string& folderPath) {
  try {
    // Check if the folder exists and is a directory
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
      std::cerr << "Error: Folder does not exist or is not a directory."
                << std::endl;
      return;
    }

    // Iterate through the files in the folder
    for (const auto& entry : fs::directory_iterator(folderPath)) {
      if (fs::is_regular_file(entry.path())) {
        // Delete the file
        fs::remove(entry.path());
        // std::cout << "Deleted file: " << entry.path() << std::endl;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception occurred: " << e.what() << std::endl;
  }
};
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& rowVectors,
    std::shared_ptr<memory::MemoryPool>& pool) {
  if (rowVectors.empty()) {
    return nullptr; // Nothing to merge
  }

  // Check schema consistency and get the number of columns.
  size_t numColumns = rowVectors[0]->childrenSize();
  for (const auto& rowVector : rowVectors) {
    VELOX_CHECK_EQ(rowVector->childrenSize(), numColumns, "Schema mismatch.");
  }

  // Total number of rows in the merged RowVector.
  vector_size_t totalRows = 0;
  for (const auto& rowVector : rowVectors) {
    totalRows += rowVector->size();
  }

  // Prepare new child vectors for the merged RowVector.
  std::vector<VectorPtr> mergedColumns(numColumns);
  for (size_t colIdx = 0; colIdx < numColumns; ++colIdx) {
    // Determine the type of the column.
    auto columnType = rowVectors[0]->childAt(colIdx)->type();

    // Create a new FlatVector to hold all data for this column.
    auto flatVector = BaseVector::create(columnType, totalRows, pool.get());

    // Fill the new FlatVector with data from each RowVector.
    vector_size_t offset = 0;
    for (const auto& rowVector : rowVectors) {
      auto sourceColumn = rowVector->childAt(colIdx);
      auto numRows = sourceColumn->size();

      flatVector->copy(sourceColumn.get(), offset, 0, numRows);
      offset += numRows;
    }

    mergedColumns[colIdx] = flatVector;
  }

  // Create the merged RowVector.
  return std::make_shared<RowVector>(
      pool.get(),
      rowVectors[0]->type(), // Use the type of the first RowVector.
      BufferPtr(nullptr), // No nulls buffer for the whole RowVector.
      totalRows,
      std::move(mergedColumns));
}

PlanBuilder rewriteQuery(
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    PlanBuilder& plan,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
    int verbose,
    std::string rewriteStrategt = "random") {
  VectorMaker maker{pool_.get()};
  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  // Create ruleManager
  RuleManager ruleManager;
  // Create planState
  PlanState planState(ruleManager);

  RandomGenerator randomGenerator = RandomGenerator(0, 1, timestampSeed);
  RandomSampler randomSampler = RandomSampler(timestampSeed);
  // randomly apple 1 to 3 actions
  randomGenerator.setIntRange(1, 8);

  auto planNode = plan.planNode();
  planState.getPossibleActions(planNode, cataLog);

  std::pair<std::string, std::string> selectedAction;

  if (verbose >= 2) {
    planState.showAllActions();
  }

  std::vector<std::pair<std::string, std::string>> listOfAppliedActions;

  for (int i = 0; i < randomGenerator.genRandomIntValue(); i++) {
    // if (true) {
    // if (randomGenerator.genRandomFloatValue() > 0.2) {
    if (true) {
      // Get the logical plan
      auto planNode = plan.planNode();
      planState.getPossibleActions(planNode, cataLog);
      std::vector<std::pair<std::string, std::string>> availableActions;
      for (auto entry : planState.actionsPair) {
        std::string targetExprStr = entry.first;
        auto optimizationRules = entry.second;

        for (auto action : optimizationRules) {
          if (rewriteStrategt == "pushdown" &&
              action != "MLDecompositionPushdownRewriteAction") {
            // if pushdown strategy, only select pushdown actions
            continue;
          }
          availableActions.push_back(std::make_pair(targetExprStr, action));
        }
      }

      if (availableActions.size() == 0) {
        // if no available actions, break
        break;
      }

      std::pair<std::string, std::string> selectedAction =
          randomSampler.sampleFromSets<std::pair<std::string, std::string>>(
              1, availableActions)[0];
      if (verbose >= 2) {
        std::cout << "[INFO] Selected action: " << selectedAction.first << ": "
                  << selectedAction.second << std::endl;
      }

      listOfAppliedActions.push_back(selectedAction);
      planState.takeAction(
          planNode,
          nullptr,
          maker,
          plan,
          pool_,
          planNodeIdGenerator,
          {selectedAction},
          cataLog);
    } else {
      break;
    }
  }

  if (verbose >= 2) {
    std::cout << "[INFO] List of applied actions:" << std::endl;
    std::cout << "====================================" << std::endl;
    size_t i = 0;
    for (auto action : listOfAppliedActions) {
      std::cout << i++ << ". " << action.first << ": " << action.second
                << std::endl;
    }
  }
  return plan;
}

PlanBuilder arbitraryQueryRewrite(
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    PlanBuilder& plan,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
    int verbose) {
  VectorMaker maker{pool_.get()};
  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  // Create ruleManager
  RuleManager ruleManager;
  // Create planState
  PlanState planState(ruleManager);
  RandomSampler randomSampler = RandomSampler(timestampSeed);

  auto planNode = plan.planNode();
  planState.getPossibleActions(planNode, cataLog);

  std::pair<std::string, std::string> selectedAction;

  if (verbose >= 2) {
    planState.showAllActions();
  }

  std::vector<std::pair<std::string, std::string>> listOfAppliedActions;

  while (planState.actionsPair.size() > 0) {
    auto planNode = plan.planNode();
    std::vector<std::pair<std::string, std::string>> availableActions;
    for (auto entry : planState.actionsPair) {
      std::string targetExprStr = entry.first;
      auto optimizationRules = entry.second;
      for (auto action : optimizationRules) {
        availableActions.push_back(std::make_pair(targetExprStr, action));
      }
    }
    if (availableActions.size() == 0) {
      // if no available actions, break
      break;
    }
    std::pair<std::string, std::string> selectedAction =
        randomSampler.sampleFromSets<std::pair<std::string, std::string>>(
            1, availableActions)[0];
    listOfAppliedActions.push_back(selectedAction);
    if (verbose >= 2) {
      std::cout << "[DEBUG] query plan: " << planNode->toString(true, true)
                << std::endl;
      std::cout << "[INFO] Selected action: " << selectedAction.first << ": "
                << selectedAction.second << std::endl;
    }
    planState.takeAction(
        planNode,
        nullptr,
        maker,
        plan,
        pool_,
        planNodeIdGenerator,
        {selectedAction},
        cataLog);
    planState.getPossibleActions(planNode, cataLog);
  }

  if (verbose >= 2) {
    std::cout << "[INFO] List of applied actions:" << std::endl;
    std::cout << "====================================" << std::endl;
    size_t i = 0;
    for (auto action : listOfAppliedActions) {
      std::cout << i++ << ". " << action.first << ": " << action.second
                << std::endl;
    }
  }

  return plan;
}

PlanBuilder heuristicQueryRewrite(
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    PlanBuilder& plan,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
    int verbose) {
  VectorMaker maker{pool_.get()};
  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  // Create ruleManager
  RuleManager ruleManager;
  // Create planState
  PlanState planState(ruleManager);
  RandomSampler randomSampler = RandomSampler(timestampSeed);

  std::pair<std::string, std::string> selectedAction;

  if (verbose >= 2) {
    planState.showAllActions();
  }

  std::vector<std::pair<std::string, std::string>> listOfAppliedActions;

  auto planNode = plan.planNode();
  planState.getPossibleActions(planNode, cataLog);
  std::vector<std::pair<std::string, std::string>> pushDownRules =
      planState.getActionPairsWithRules(
          {"MLDecompositionPushdownRewriteAction",
           "MultiLayerUDF2TorchNNRewriteAction"});
  while (pushDownRules.size() > 0) {
    auto planNode = plan.planNode();
    std::pair<std::string, std::string> selectedAction =
        randomSampler.sampleFromSets<std::pair<std::string, std::string>>(
            1, pushDownRules)[0];
    listOfAppliedActions.push_back(selectedAction);
    if (verbose >= 2) {
      std::cout << "[DEBUG] query plan: " << planNode->toString(true, true)
                << std::endl;
      std::cout << "[INFO] Selected action: " << selectedAction.first << ": "
                << selectedAction.second << std::endl;
    }
    planState.takeAction(
        planNode,
        nullptr,
        maker,
        plan,
        pool_,
        planNodeIdGenerator,
        {selectedAction},
        cataLog);
    planNode = plan.planNode();
    planState.getPossibleActions(planNode, cataLog);
    pushDownRules = planState.getActionPairsWithRules(
        {"MLDecompositionPushdownRewriteAction",
         "MultiLayerUDF2TorchNNRewriteAction"});
  }

  if (verbose >= 2) {
    std::cout << "[INFO] List of applied actions:" << std::endl;
    std::cout << "====================================" << std::endl;
    size_t i = 0;
    for (auto action : listOfAppliedActions) {
      std::cout << i++ << ". " << action.first << ": " << action.second
                << std::endl;
    }
  }

  return plan;
}

std::vector<std::vector<int>> readModelStructureFromFile(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error opening model structure file: " << filename
              << std::endl;
    return {};
  }

  std::vector<std::vector<int>> all_lines;
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::vector<int> values;
    int num;

    // Read each integer in the line
    while (iss >> num) {
      values.push_back(num);
    }

    // Store the line of integers in the main vector
    all_lines.push_back(values);
  }

  file.close();
  return all_lines;
}

std::vector<std::vector<std::string>> readModelKernelStrFromFile(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Error opening model kernel file: " << filename << std::endl;
    return {};
  }

  std::vector<std::vector<std::string>> all_lines;
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::vector<std::string> values;
    std::string word;

    while (iss >> word) {
      values.push_back(word);
    }

    all_lines.push_back(values);
  }

  file.close();
  return all_lines;
}

void generateDummyData(
    std::string workload,
    std::vector<int> numberOfTuples,
    std::vector<int> dummyFeatureSizes,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    std::shared_ptr<memory::MemoryPool> rootPool,
    int dataBatchSize = 256,
    std::string dataPath = "") {
  if (workload == "ml") {
    VectorMaker maker{pool_.get()};
    // it should contains the numbers: # of users, # of movies, # of relevance
    // tag
    assert(numberOfTuples.size() == 3 && dummyFeatureSizes.size() == 2);
    int numUsers = numberOfTuples[0];
    int numMovies = numberOfTuples[1];
    int numRelevanceTags = numberOfTuples[2];
    int userFeatureSize = dummyFeatureSizes[0];
    int movieFeatureSize = dummyFeatureSizes[1];

    std::string tableStatsPath =
        "/home/velox/velox/optimizer/tests/tableStats.txt";
    remove(tableStatsPath.c_str());

    std::vector<std::vector<int>> userModelStructures =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
    std::vector<std::vector<int>> movieModelStructures =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/movie_dummy_model_structure.txt");
    std::vector<std::vector<int>> tagModelStructures =
        readModelStructureFromFile(
            "/home/velox/velox/optimizer/tests/tag_dummy_model_structure.txt");

    RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
    RandomSampler randomSampler = RandomSampler(0);
    // sample user data
    std::vector<int> userIDs = randomGenerator.genIntRange(0, numUsers);
    std::vector<std::string> userGender =
        randomSampler.sampleFromSets<std::string>(numUsers, {"M", "F"});
    std::vector<int> userAge = randomGenerator.gen1DInt(numUsers, 10, 70);
    std::vector<int> userOccupation = randomGenerator.gen1DInt(numUsers, 0, 20);
    std::vector<std::string> zipCode = {
        "94043",
        "94301",
        "94305",
        "94306",
        "94309",
        "80212",
        "80213",
        "80219",
        "12301",
        "40201"};
    std::vector<std::string> userZipcode =
        randomSampler.sampleFromSets<std::string>(numUsers, zipCode);
    std::vector<std::vector<float>> userFeatures =
        randomGenerator.genFloat2dVector(numUsers, userFeatureSize);

    cataLog.addNumericalColMinMax("u_user_id", 0, numUsers);
    cataLog.addNumericalColMinMax("u_age", 10, 70);
    cataLog.addNumericalColMinMax("u_occupation", 0, 20);
    cataLog.addCategoricalColVals("u_zipcode", zipCode);

    auto userDataRowVector = maker.rowVector(
        {"u_user_id",
         "u_gender",
         "u_age",
         "u_occupation",
         "u_zipcode",
         "u_features"},
        {maker.flatVector<int>(userIDs, INTEGER()),
         maker.flatVector<std::string>(userGender, VARCHAR()),
         maker.flatVector<int>(userAge, INTEGER()),
         maker.flatVector<int>(userOccupation, INTEGER()),
         maker.flatVector<std::string>(userZipcode, VARCHAR()),
         maker.arrayVector<float>(userFeatures, REAL())});

    // output the histogram for the user data
    cataLog.outputHistogramForData(
        userDataRowVector, "user", 50, tableStatsPath);

    MovieTitleGenerator movieTitleGenerator = MovieTitleGenerator(0);
    std::vector<int> movieIDs = randomGenerator.genIntRange(0, numMovies);
    std::vector<std::string> movieTitle =
        movieTitleGenerator.generateBatchRandomTitles(numMovies);
    std::vector<std::string> genres = {
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"};
    std::vector<std::string> movieGenres =
        randomSampler.sampleFromSets<std::string>(numMovies, genres);
    std::vector<std::string> spokenLanguage = {
        "English",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Mandarin",
        "Spanish"};
    std::vector<std::string> movieSpokenLanguage =
        randomSampler.sampleFromSets<std::string>(numMovies, spokenLanguage);
    randomGenerator.setFloatRange(0, 10);
    std::vector<float> moviePopularity =
        randomGenerator.genFloat1dVector(numMovies);
    randomGenerator.setFloatRange(0, 5);
    std::vector<float> movieVoteAverage =
        randomGenerator.genFloat1dVector(numMovies);
    std::vector<int> movieVoteCount =
        randomGenerator.gen1DInt(numMovies, 0, 5000);
    std::vector<std::vector<float>> movieFeatures =
        randomGenerator.genFloat2dVector(numMovies, movieFeatureSize);

    cataLog.addNumericalColMinMax("m_movie_id", 0, numMovies);
    cataLog.addNumericalColMinMax("m_popularity", 0, 10);
    cataLog.addNumericalColMinMax("m_vote_average", 0, 5);
    cataLog.addNumericalColMinMax("m_vote_count", 0, 5000);
    cataLog.addCategoricalColVals("m_genres", genres);
    cataLog.addCategoricalColVals("m_spoken_languages", spokenLanguage);

    auto movieDataRowVector = maker.rowVector(
        {"m_movie_id",
         "m_title",
         "m_genres",
         "m_spoken_languages",
         "m_popularity",
         "m_vote_average",
         "m_vote_count",
         "m_features"},
        {maker.flatVector<int>(movieIDs, INTEGER()),
         maker.flatVector<std::string>(movieTitle, VARCHAR()),
         maker.flatVector<std::string>(movieGenres, VARCHAR()),
         maker.flatVector<std::string>(movieSpokenLanguage, VARCHAR()),
         maker.flatVector<float>(moviePopularity, REAL()),
         maker.flatVector<float>(movieVoteAverage, REAL()),
         maker.flatVector<int>(movieVoteCount, INTEGER()),
         maker.arrayVector<float>(movieFeatures, REAL())});

    // output the histogram for the movie data
    cataLog.outputHistogramForData(
        movieDataRowVector, "movie", 50, tableStatsPath);

    randomGenerator.setFloatRange(0, 1);
    std::vector<std::vector<float>> movieRelevanceTags =
        randomGenerator.genFloat2dVector(numMovies, numRelevanceTags);

    cataLog.addNumericalColMinMax("mt_movie_id", 0, numMovies);

    auto movieRelevanceTagRowVector = maker.rowVector(
        {"mt_movie_id", "mt_relevance_score"},
        {maker.flatVector<int>(movieIDs, INTEGER()),
         maker.arrayVector<float>(movieRelevanceTags, REAL())});

    // output the histogram for the movie relevance tag data
    cataLog.outputHistogramForData(
        movieRelevanceTagRowVector, "movie_relevance_tag", 50, tableStatsPath);

    std::pair<int, int> userStats =
        std::make_pair(numUsers, userDataRowVector->childrenSize());
    std::pair<int, int> movieStats =
        std::make_pair(numMovies, movieDataRowVector->childrenSize());
    std::pair<int, int> movieRelevanceTagStats =
        std::make_pair(numMovies, movieRelevanceTagRowVector->childrenSize());

    if (dataPath.empty()) {
      // Save the data to temprary files
      // user
      std::vector<std::shared_ptr<TempFilePath>> userFilePaths =
          splitRowVectorIntoBatchFiles(userDataRowVector, dataBatchSize);
      cataLog.registerDataSrc(
          "user",
          userFilePaths,
          asRowType(userDataRowVector->type()),
          userStats);
      // movie
      std::vector<std::shared_ptr<TempFilePath>> movieFilePaths =
          splitRowVectorIntoBatchFiles(movieDataRowVector, dataBatchSize);
      cataLog.registerDataSrc(
          "movie",
          movieFilePaths,
          asRowType(movieDataRowVector->type()),
          movieStats);
      // movie_relevance_tag
      std::vector<std::shared_ptr<TempFilePath>> movieRelevanceTagFilePaths =
          splitRowVectorIntoBatchFiles(
              movieRelevanceTagRowVector, dataBatchSize);
      cataLog.registerDataSrc(
          "movie_relevance_tag",
          movieRelevanceTagFilePaths,
          asRowType(movieRelevanceTagRowVector->type()),
          movieRelevanceTagStats);
    } else {
      // Save the data to the specified path
      // user
      std::vector<std::string> userFilePaths =
          splitRowVectorsIntoBatchParquetFiles(
              userDataRowVector,
              dataBatchSize,
              fmt::format("{}/user", dataPath),
              rootPool);
      cataLog.registerDataSrc(
          "user",
          userFilePaths,
          dwio::common::FileFormat::PARQUET,
          asRowType(userDataRowVector->type()),
          userStats);
      // movie
      std::vector<std::string> movieFilePaths =
          splitRowVectorsIntoBatchParquetFiles(
              movieDataRowVector,
              dataBatchSize,
              fmt::format("{}/movie", dataPath),
              rootPool);
      cataLog.registerDataSrc(
          "movie",
          movieFilePaths,
          dwio::common::FileFormat::PARQUET,
          asRowType(movieDataRowVector->type()),
          movieStats);
      // movie_relevance_tag
      std::vector<std::string> movieRelevanceTagFilePaths =
          splitRowVectorsIntoBatchParquetFiles(
              movieRelevanceTagRowVector,
              dataBatchSize,
              fmt::format("{}/movie_relevance_tag", dataPath),
              rootPool);
      cataLog.registerDataSrc(
          "movie_relevance_tag",
          movieRelevanceTagFilePaths,
          dwio::common::FileFormat::PARQUET,
          asRowType(movieRelevanceTagRowVector->type()),
          movieRelevanceTagStats);
    }
  }
}

void randomly_sample_trees(
    int numTrees,
    int maxDepth,
    const std::string& outputPath) {
  const std::string sourceFolder =
      fmt::format("resources/model/xgboost_10000_{}_netsdb", maxDepth);

  namespace fs = std::filesystem;

  // Check if source directory exists
  if (!fs::exists(sourceFolder) || !fs::is_directory(sourceFolder)) {
    std::cerr << "Source folder does not exist: " << sourceFolder << std::endl;
    return;
  }

  if (fs::exists(outputPath)) {
    for (const auto& entry : fs::directory_iterator(outputPath)) {
      fs::remove_all(entry); // Works for both files and subdirectories
    }
  } else {
    fs::create_directories(outputPath);
  }

  // Collect all files in the source folder
  std::vector<fs::path> treeFiles;
  for (const auto& entry : fs::directory_iterator(sourceFolder)) {
    if (entry.is_regular_file()) {
      treeFiles.push_back(entry.path());
    }
  }

  if (treeFiles.empty()) {
    std::cerr << "No tree files found in " << sourceFolder << std::endl;
    return;
  }

  // Shuffle and take the first numTrees files
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(treeFiles.begin(), treeFiles.end(), gen);

  int count = std::min(numTrees, static_cast<int>(treeFiles.size()));
  for (int i = 0; i < count; ++i) {
    const fs::path& sourceFile = treeFiles[i];
    fs::path destinationFile = fs::path(outputPath) / sourceFile.filename();
    fs::copy_file(
        sourceFile, destinationFile, fs::copy_options::overwrite_existing);
  }

  std::cout << "Copied " << count << " tree(s) to " << outputPath << std::endl;
}

void registerDFModel(
    std::vector<int> units,
    CataLog& catalog,
    int& modelGroupId_,
    bool hasArgmax = false) {
  int numTrees = units[0];
  int maxDepth = units[1];

  // std::string modelFilePath =
  //     fmt::format("resources/model/decision_forest_{}_{}", numTrees,
  //     maxDepth);
  std::string modelFilePath = "resources/model/decision_forest_tmp";
  randomly_sample_trees(numTrees, maxDepth, modelFilePath);
  int numCols = 4;
  exec::registerVectorFunction(
      "decision_forest_predict",
      ForestPrediction::signatures(),
      std::make_unique<ForestPrediction>(modelFilePath, numCols, true));
  return;
}

std::string registerNNModel(
    std::vector<int> units,
    CataLog& catalog,
    int& modelGroupId_,
    bool hasArgmax = false) {
  // use input size as random seed
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, units[0]);
  int modelGroupId = modelGroupId_++;
  int functionId = 0;
  int numberOfLayers = units.size() - 1;

  optimization::registerVectorFunction(
      "relu", Relu::signatures(), std::make_unique<Relu>(), {}, true, catalog);
  optimization::registerVectorFunction(
      "softmax",
      Softmax::signatures(),
      std::make_unique<Softmax>(),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "argmax",
      Argmax::signatures(),
      std::make_unique<Argmax>(),
      {},
      true,
      catalog);
  optimization::registerVectorFunction(
      "sigmoid",
      Sigmoid::signatures(),
      std::make_unique<Sigmoid>(),
      {},
      true,
      catalog);
  std::string modelComputationStr = "{}";
  int lastSize = units[0];

  for (int i = 1; i < units.size(); i++) {
    int layerSize = units[i];
    std::vector<std::vector<float>> weights =
        randomGenerator.genFloat2dVector(lastSize, layerSize);
    std::vector<std::vector<float>> bias =
        randomGenerator.genFloat2dVector(1, layerSize);
    std::string matMulName =
        fmt::format("mat_mul{}_{}", modelGroupId_, functionId++);
    optimization::registerVectorFunction(
        matMulName,
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(
            std::move(flattenVectorToPointer(weights)), lastSize, layerSize),
        {},
        true,
        catalog);
    std::string matVectorAddName =
        fmt::format("mat_vector_add{}_{}", modelGroupId_, functionId++);
    optimization::registerVectorFunction(
        matVectorAddName,
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(
            std::move(flattenVectorToPointer(bias)), layerSize),
        {},
        true,
        catalog);
    modelComputationStr =
        matVectorAddName + "(" + matMulName + "(" + modelComputationStr + "))";
    if (i != units.size() - 1) {
      modelComputationStr = "relu(" + modelComputationStr + ")";
    } else {
      if (units.size() > 1) {
        modelComputationStr = "softmax(" + modelComputationStr + ")";
      } else {
        modelComputationStr = "sigmoid(" + modelComputationStr + ")";
      }
    }
    lastSize = layerSize;
  }
  if (hasArgmax) {
    modelComputationStr = "argmax(" + modelComputationStr + ")";
  }
  return modelComputationStr;
}

PlanBuilder setupProfileQueryPlan(
    std::string mode,
    std::string queryTemplate,
    int& modelGroupId_,
    CataLog& cataLog,
    std::shared_ptr<memory::MemoryPool> pool_,
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
    int randomSeed = -1) {
  bool generateFilter = stringToBool(getEnvVar("CD_PROFILE_W_FILTER"));

  unsigned timestampSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  if (randomSeed != -1) {
    timestampSeed = randomSeed;
  }
  RandomGenerator randomGenerator = RandomGenerator(-1, 1, timestampSeed);
  randomGenerator.setIntRange(0, 1);
  PlanBuilder myPlan;

  std::vector<std::vector<int>> userModelStructures =
      readModelStructureFromFile(
          "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
  std::vector<std::vector<int>> movieModelStructures =
      readModelStructureFromFile(
          "/home/velox/velox/optimizer/tests/movie_dummy_model_structure.txt");
  std::vector<std::vector<int>> tagModelStructures = readModelStructureFromFile(
      "/home/velox/velox/optimizer/tests/tag_dummy_model_structure.txt");

  if (mode == "ml") {
    RowTypePtr userDataRowType = cataLog.getRegisteredDataSrcSchema("user");
    RowTypePtr movieDataRowType = cataLog.getRegisteredDataSrcSchema("movie");
    RowTypePtr movieRelevanceTagRowType =
        cataLog.getRegisteredDataSrcSchema("movie_relevance_tag");

    dwio::common::FileFormat userFileFormat =
        cataLog.getRegisteredDataSrcFormat("user");
    dwio::common::FileFormat movieFileFormat =
        cataLog.getRegisteredDataSrcFormat("movie");
    dwio::common::FileFormat movieRelevanceTagFileFormat =
        cataLog.getRegisteredDataSrcFormat("movie_relevance_tag");

    std::vector<std::string> userFilePaths =
        cataLog.getRegisteredDataSrcFiles("user");
    std::vector<std::string> movieFilePaths =
        cataLog.getRegisteredDataSrcFiles("movie");
    std::vector<std::string> movieRelevanceTagFilePaths =
        cataLog.getRegisteredDataSrcFiles("movie_relevance_tag");

    std::pair<int, int> userStats = cataLog.getRegisteredDataSrcStats("user");
    int userNumRows = userStats.first;
    int userNumCols = userStats.second;
    std::pair<int, int> movieStats = cataLog.getRegisteredDataSrcStats("movie");
    int movieNumRows = movieStats.first;
    int movieNumCols = movieStats.second;
    std::pair<int, int> movieRelevanceTagStats =
        cataLog.getRegisteredDataSrcStats("movie_relevance_tag");
    int movieRelevanceTagNumRows = movieRelevanceTagStats.first;
    int movieRelevanceTagNumCols = movieRelevanceTagStats.second;

    if (queryTemplate == "" || queryTemplate == "user") {
      std::string userModel1ComputExpr = registerNNModel(
          userModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readUserDataPlanNodeId;
      myPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .tableScan(userDataRowType, {}, "")
                   .capturePlanNodeId(readUserDataPlanNodeId)
                   .project(
                       {"u_user_id",
                        "u_age",
                        "u_gender",
                        "u_occupation",
                        "u_zipcode",
                        "u_features"})
                   .project(
                       {"u_user_id",
                        "u_age",
                        "u_gender",
                        "u_occupation",
                        "u_zipcode",
                        fmt::format(userModel1ComputExpr, "u_features")});
      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("user", timestampSeed);
        for (auto expr : filterExpr) {
          myPlan = myPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      cataLog.setIdAddressMap(
          readUserDataPlanNodeId, userFilePaths, userFileFormat);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));
    } else if (queryTemplate == "movie") {
      std::string movieModel1ComputExpr = registerNNModel(
          movieModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readMovieDataPlanNodeId;
      myPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .tableScan(movieDataRowType, {}, "")
                   .capturePlanNodeId(readMovieDataPlanNodeId)
                   .project(
                       {"m_movie_id",
                        "m_title",
                        "m_genres",
                        "m_spoken_languages",
                        "m_popularity",
                        "m_vote_average",
                        "m_vote_count",
                        "m_features"})
                   .project(
                       {"m_movie_id",
                        "m_title",
                        "m_genres",
                        "m_spoken_languages",
                        "m_popularity",
                        "m_vote_average",
                        "m_vote_count",
                        "m_features",
                        fmt::format(movieModel1ComputExpr, "m_features")});

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("movie", timestampSeed);
        for (auto expr : filterExpr) {
          myPlan = myPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId, movieFilePaths, movieFileFormat);

      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      std::shared_ptr<OutputStat> movieStats =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      cataLog.addSource(std::make_shared<Source>(movieSrc));
    } else if (queryTemplate == "movie_tag") {
      std::string tagModel1ComputExpr = registerNNModel(
          tagModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readMovieRelevanceTagDataPlanNodeId;
      myPlan =
          PlanBuilder(planNodeIdGenerator, pool_.get())
              .tableScan(movieRelevanceTagRowType, {}, "")
              .capturePlanNodeId(readMovieRelevanceTagDataPlanNodeId)
              .project(
                  {"mt_movie_id",
                   "mt_relevance_score",
                   fmt::format(tagModel1ComputExpr, "mt_relevance_score")});

      cataLog.setIdAddressMap(
          readMovieRelevanceTagDataPlanNodeId,
          movieRelevanceTagFilePaths,
          movieRelevanceTagFileFormat);

      cataLog.addNodeIdRelationName(
          readMovieRelevanceTagDataPlanNodeId, "movie_relevance_tag");
      std::shared_ptr<OutputStat> movieRelevanceTagStats =
          std::make_shared<OutputStat>(
              OutputStat(movieRelevanceTagNumRows, movieRelevanceTagNumCols));
      Source movieRelevanceTagSrc = Source(
          readMovieRelevanceTagDataPlanNodeId,
          Source::Type::FILE,
          movieRelevanceTagStats);
      cataLog.addSource(std::make_shared<Source>(movieRelevanceTagSrc));
    } else if (queryTemplate == "movie_user" || queryTemplate == "user_movie") {
      std::string userModel1ComputExpr = registerNNModel(
          userModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readUserDataPlanNodeId;
      auto userPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .tableScan(userDataRowType, {}, "")
                          .capturePlanNodeId(readUserDataPlanNodeId)
                          .project(
                              {"u_user_id",
                               "u_age",
                               "u_gender",
                               "u_occupation",
                               "u_zipcode",
                               "u_features"});

      std::string movieModel1ComputExpr = registerNNModel(
          movieModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readMovieDataPlanNodeId;
      auto moviePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                           .tableScan(movieDataRowType, {}, "")
                           .capturePlanNodeId(readMovieDataPlanNodeId)
                           .project(
                               {"m_movie_id",
                                "m_title",
                                "m_genres",
                                "m_spoken_languages",
                                "m_popularity",
                                "m_vote_average",
                                "m_vote_count",
                                "m_features"});
      RandomGenerator numModelGenerator = RandomGenerator(-1, 1, timestampSeed);
      numModelGenerator.setIntRange(0, 2);
      int numModel = numModelGenerator.genRandomIntValue();
      myPlan = userPlan.nestedLoopJoin(
          moviePlan.planNode(),
          {"u_user_id",
           "u_age",
           "u_gender",
           "u_occupation",
           "u_zipcode",
           "u_features",
           "m_movie_id",
           "m_title",
           "m_genres",
           "m_spoken_languages",
           "m_popularity",
           "m_vote_average",
           "m_vote_count",
           "m_features"});

      if (numModel == 1) {
        myPlan = myPlan.project(
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             fmt::format(userModel1ComputExpr, "u_features")});
      } else if (numModel == 2) {
        myPlan = myPlan.project(
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             fmt::format(userModel1ComputExpr, "u_features"),
             fmt::format(movieModel1ComputExpr, "m_features")});
      }

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("movie_user", timestampSeed);
        for (auto expr : filterExpr) {
          myPlan = myPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      cataLog.setIdAddressMap(
          readUserDataPlanNodeId, userFilePaths, userFileFormat);
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId, movieFilePaths, movieFileFormat);
      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");

      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));
      std::shared_ptr<OutputStat> movieStats =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      cataLog.addSource(std::make_shared<Source>(movieSrc));
    } else if (
        queryTemplate == "movie_user_tag" ||
        queryTemplate == "user_movie_tag") {
      std::string userModel1ComputExpr = registerNNModel(
          userModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readUserDataPlanNodeId;
      auto userPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .tableScan(userDataRowType, {}, "")
                          .capturePlanNodeId(readUserDataPlanNodeId)
                          .project(
                              {"u_user_id",
                               "u_age",
                               "u_gender",
                               "u_occupation",
                               "u_zipcode",
                               "u_features"});

      std::string movieModel1ComputExpr = registerNNModel(
          movieModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readMovieDataPlanNodeId;
      auto moviePlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                           .tableScan(movieDataRowType, {}, "")
                           .capturePlanNodeId(readMovieDataPlanNodeId)
                           .project({
                               "m_movie_id",
                               "m_title",
                               "m_genres",
                               "m_spoken_languages",
                               "m_popularity",
                               "m_vote_average",
                               "m_vote_count",
                               "m_features",
                           });

      std::string tagModel1ComputExpr = registerNNModel(
          tagModelStructures[0],
          cataLog,
          modelGroupId_,
          randomGenerator.genRandomIntValue());

      core::PlanNodeId readMovieRelevanceTagDataPlanNodeId;
      auto tagPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .tableScan(movieRelevanceTagRowType, {}, "")
                         .capturePlanNodeId(readMovieRelevanceTagDataPlanNodeId)
                         .project({
                             "mt_movie_id",
                             "mt_relevance_score",
                         });

      auto movieTagPlan = tagPlan.hashJoin(
          {"mt_movie_id"},
          {"m_movie_id"},
          moviePlan.planNode(),
          "",
          {"m_movie_id",
           "m_title",
           "m_genres",
           "m_spoken_languages",
           "m_popularity",
           "m_vote_average",
           "m_vote_count",
           "m_features",
           "mt_relevance_score"});

      RandomGenerator numModelGenerator = RandomGenerator(-1, 1, timestampSeed);
      numModelGenerator.setIntRange(0, 3);
      int numModel = numModelGenerator.genRandomIntValue();
      myPlan = userPlan.nestedLoopJoin(
          movieTagPlan.planNode(),
          {"u_user_id",
           "u_age",
           "u_gender",
           "u_occupation",
           "u_zipcode",
           "u_features",
           "m_movie_id",
           "m_title",
           "m_genres",
           "m_spoken_languages",
           "m_popularity",
           "m_vote_average",
           "m_vote_count",
           "m_features",
           "mt_relevance_score"});

      if (numModel == 1) {
        myPlan = myPlan.project(
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             "mt_relevance_score",
             fmt::format(userModel1ComputExpr, "u_features")});
      } else if (numModel == 2) {
        myPlan = myPlan.project(
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             "mt_relevance_score",
             fmt::format(userModel1ComputExpr, "u_features"),
             fmt::format(movieModel1ComputExpr, "m_features")});
      } else if (numModel == 3) {
        myPlan = myPlan.project(
            {"u_user_id",
             "u_age",
             "u_gender",
             "u_occupation",
             "u_zipcode",
             "u_features",
             "m_movie_id",
             "m_title",
             "m_genres",
             "m_spoken_languages",
             "m_popularity",
             "m_vote_average",
             "m_vote_count",
             "m_features",
             "mt_relevance_score",
             fmt::format(userModel1ComputExpr, "u_features"),
             fmt::format(movieModel1ComputExpr, "m_features"),
             fmt::format(tagModel1ComputExpr, "mt_relevance_score")});
      }

      if (generateFilter) {
        std::vector<std::string> filterExpr =
            sampleUserMovieFilterExpr("movie_user", timestampSeed);
        for (auto expr : filterExpr) {
          myPlan = myPlan.filter(expr);
        }
        // myPlan = myPlan.filter(filterExpr);
      }

      cataLog.setIdAddressMap(
          readUserDataPlanNodeId, userFilePaths, userFileFormat);
      cataLog.setIdAddressMap(
          readMovieDataPlanNodeId, movieFilePaths, movieFileFormat);
      cataLog.setIdAddressMap(
          readMovieRelevanceTagDataPlanNodeId,
          movieRelevanceTagFilePaths,
          movieRelevanceTagFileFormat);

      cataLog.addNodeIdRelationName(readUserDataPlanNodeId, "user");
      cataLog.addNodeIdRelationName(readMovieDataPlanNodeId, "movie");
      cataLog.addNodeIdRelationName(
          readMovieRelevanceTagDataPlanNodeId, "movie_relevance_tag");
      std::shared_ptr<OutputStat> userStats =
          std::make_shared<OutputStat>(OutputStat(userNumRows, userNumCols));
      Source userSrc =
          Source(readUserDataPlanNodeId, Source::Type::FILE, userStats);
      cataLog.addSource(std::make_shared<Source>(userSrc));
      std::shared_ptr<OutputStat> movieStats =
          std::make_shared<OutputStat>(OutputStat(movieNumRows, movieNumCols));
      Source movieSrc =
          Source(readMovieDataPlanNodeId, Source::Type::FILE, movieStats);
      cataLog.addSource(std::make_shared<Source>(movieSrc));
      std::shared_ptr<OutputStat> movieRelevanceTagStats =
          std::make_shared<OutputStat>(
              OutputStat(movieRelevanceTagNumRows, movieRelevanceTagNumCols));
      Source movieRelevanceTagSrc = Source(
          readMovieRelevanceTagDataPlanNodeId,
          Source::Type::FILE,
          movieRelevanceTagStats);
      cataLog.addSource(std::make_shared<Source>(movieRelevanceTagSrc));
    } else {
      throw std::runtime_error(
          fmt::format("Non-supported queryTemplate: {}", queryTemplate));
    }
  } else {
    throw std::runtime_error(fmt::format("Non-supported model: {}", mode));
  }

  return myPlan;
}

void checkValidProfileQueryGenerationSetting(
    std::vector<int> numberOfTuples,
    std::vector<int> dummyFeatureSizes,
    std::string queryTemplate) {
  int numRelevanceTags = numberOfTuples[2];
  int userFeatureSize = dummyFeatureSizes[0];
  int movieFeatureSize = dummyFeatureSizes[1];

  std::vector<std::vector<int>> userModelStructures =
      readModelStructureFromFile(
          "/home/velox/velox/optimizer/tests/user_dummy_model_structure.txt");
  std::vector<std::vector<int>> movieModelStructures =
      readModelStructureFromFile(
          "/home/velox/velox/optimizer/tests/movie_dummy_model_structure.txt");
  std::vector<std::vector<int>> tagModelStructures = readModelStructureFromFile(
      "/home/velox/velox/optimizer/tests/tag_dummy_model_structure.txt");

  if (queryTemplate.find("user") != std::string::npos &&
      userFeatureSize != userModelStructures[0][0])
    checkOrAbort(
        userModelStructures[0][0],
        userFeatureSize,
        "[registerProfileModel-userModel]");
  if (queryTemplate.find("movie") != std::string::npos &&
      movieFeatureSize != movieModelStructures[0][0])
    checkOrAbort(
        movieModelStructures[0][0],
        movieFeatureSize,
        "[registerProfileModel-movieModel]");
  if (queryTemplate.find("movie") != std::string::npos &&
      queryTemplate.find("user") != std::string::npos &&
      numRelevanceTags != tagModelStructures[0][0])
    checkOrAbort(
        tagModelStructures[0][0],
        numRelevanceTags,
        "[registerProfileModel-movieTagModel]");
}