// ml_functions/tests/ImbridgeQueriesTest.cpp

// #include "velox/ml_functions/imbridge/RegisterImbridgeFunctions.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/program_options.hpp>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
//#include <torch/torch.h>
#include <random>
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/dwio/parquet/writer/Writer.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
// #include "velox/ml_functions/BatchNorm.h"
// #include "velox/ml_functions/ComplexLayer.h"
// #include "velox/ml_functions/Concat.h"
// #include "velox/ml_functions/CosineSimilarity.h"
// #include "velox/ml_functions/Dropout.h"
// #include "velox/ml_functions/Embedding.h"
// #include "velox/ml_functions/Encoder.h"
// #include "velox/ml_functions/SequencePooling.h"
// #include "velox/ml_functions/UtilFunction.h"
// #include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"

// using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;
// using namespace facebook::velox::ml_functions::imbridge;

DEFINE_string(
    query_template,
    "creditcard",
    "Which query template to run {expedia, flights, creditcard}");

std::vector<std::string> getFilePathsFromDir(const std::string& dirPath) {
  std::vector<std::string> filePaths;
  for (auto const& dirEntry : fs::directory_iterator(dirPath)) {
    if (!dirEntry.is_regular_file()) {
      continue;
    }
    // Ignore hidden files.
    if (dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    // auto dataFile = CustomTempFilePath::create(dirEntry.path());
    filePaths.push_back(dirEntry.path());
  }
  return filePaths;
}

class ImbridgeQueriesTest : public HiveConnectorTestBase {
    public : 
        ImbridgeQueriesTest(){
             // Register Presto scalar functions.
            functions::prestosql::registerAllScalarFunctions();

            // Register Presto aggregate functions.
            aggregate::prestosql::registerAllAggregateFunctions();

            // Register type resolver with DuckDB SQL parser.
            parse::registerTypeResolver();

            // HiveConnectorTestBase::SetUp();
            parquet::registerParquetReaderFactory();

            const std::string kHiveConnectorId = "test-hive";

            auto hiveConnector =
                connector::getConnectorFactory(
                    connector::hive::HiveConnectorFactory::kHiveConnectorName)
                    ->newConnector(
                        kHiveConnectorId, std::make_shared<core::MemConfig>());
            connector::registerConnector(hiveConnector);

        }

        ~ImbridgeQueriesTest() {}
        
        void run(std::string queryTemplate);


        void TestBody() override {}

         void SetUp() override {
             HiveConnectorTestBase::SetUp(); // Call base class setup if needed
         }

        void TearDown() override {
            HiveConnectorTestBase::TearDown(); // Call base class teardown if needed
        }

        std::shared_ptr<folly::Executor> executor_{
        std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency())};
    std::shared_ptr<core::QueryCtx> queryCtx_{
        std::make_shared<core::QueryCtx>(executor_.get())};

        std::shared_ptr<memory::MemoryPool> pool_{
            memory::MemoryManager::getInstance()->addLeafPool()};
        VectorMaker maker{pool_.get()};

        static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
            while (!task->isFinished()) {
            usleep(1000); // 0.01 second.
            }
        }
};

void ImbridgeQueriesTest::run(std::string queryTemplate) {
    // Shared plan node id generator
    auto planNodeIdGenerator =
        std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodePtr plan;
    std::function<void(exec::Task*)> addSplits;

    if (queryTemplate == "creditcard") {
      // Define row type
      auto CreditCardType = ROW(
          {"time", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
           "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
           "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
           "amount", "class"},
          {DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(),
           DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(),
           DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(),
           DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), DOUBLE(), BIGINT()});

      // Build plan
      core::PlanNodeId readCCNodeId;
      plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                 .tableScan(CreditCardType, {}, "")
                 .capturePlanNodeId(readCCNodeId)
                 .project({
                     "v1 as fv1", "v2 as fv2", "v3 as fv3",
                     "time",
                     "v1_minmax_scaler(transform(array_constructor(v1),    x -> CAST(x AS REAL))) as v1",
                     "v2_minmax_scaler(transform(array_constructor(v2),    x -> CAST(x AS REAL))) as v2",
                     "v3_minmax_scaler(transform(array_constructor(v3),    x -> CAST(x AS REAL))) as v3",
                     "class"
                 })
                 .project({
                     "concat(v1, v2, v3, v4, v5, v6, v7, v8, v9,"
                     "v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,"
                     "v20, v21, v22, v23, v24, v25, v26, v27, v28, amount) as u_features",
                     "fv1", "fv2", "fv3"
                 })
                 .project({
                     "decision_forest_predict(u_features) as prediction_result",
                     "fv1", "fv2", "fv3"
                 })
                 .filter("fv1 > 1 AND fv2 < 0.27 AND fv3 > 0.3")
                 .planNode();

          // Prepare splits for creditcard parquet files
    const char* env = std::getenv("CD_DATA_DIR_PREFIX");
    std::string dataDir = env ? env
                              : "/home/cactusdb/resources/data/parquet/creditcard/";
    std::string filePath = fmt::format("file:{}creditcard", dataDir);
    auto creditSplits = HiveConnectorTestBase::makeHiveConnectorSplits(
        getFilePathsFromDir(filePath),
        // /*numSplits=*/4,  // match your 4 parquet splits
        dwio::common::FileFormat::PARQUET);

    // Add splits exactly once per node
    bool noMoreSplits = false;
    auto addSplits = [&](exec::Task* task) {
      if (!noMoreSplits) {
        for (auto& split : creditSplits) {
          task->addSplit(readCCNodeId, exec::Split(std::move(split)));
        }
        task->noMoreSplits(readCCNodeId);
      }
      noMoreSplits = true;
    };
    } else {
      std::cerr << "Unsupported template in this build: " << queryTemplate << std::endl;
      return;
    }

    // Execute
    CursorParameters params;
    params.queryCtx = queryCtx_;
    params.planNode = plan;
    params.maxDrivers = 1;
    auto [cursor, results] = readCursor(params, addSplits);
    waitForFinishedDrivers(cursor->task());

    std::cout << "CreditCard plan executed: " << results.size() << " batches." << std::endl;
  }


int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  ImbridgeQueriesTest demo;
  demo.run(FLAGS_query_template);
  return 0;
}
