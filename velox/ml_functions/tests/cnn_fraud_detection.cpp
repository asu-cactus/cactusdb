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

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

class cnn_fraud_detection : public HiveConnectorTestBase {
    public : 
        cnn_fraud_detection(){
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
                    ->newConnector(
                        kHiveConnectorId, std::make_shared<core::MemConfig>());
            connector::registerConnector(hiveConnector);

        }

        ~cnn_fraud_detection() {}

        std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> runtestidmeta();
        std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> runtestidlabel();
        void runjointest(std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>>
 idmeta_data, std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>>
 idlabel_data);

        void TestBody() override {}

         void SetUp() override {
             HiveConnectorTestBase::SetUp(); // Call base class setup if needed
         }

        void TearDown() override {
            HiveConnectorTestBase::TearDown(); // Call base class teardown if needed
        }

        std::shared_ptr<memory::MemoryPool> pool_{
            memory::MemoryManager::getInstance()->addLeafPool()};
        VectorMaker maker{pool_.get()};

        int numSamples = 10;
    
};

std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> cnn_fraud_detection::runtestidmeta(){
    // First Part 
    // Read Data


    // idmeta
    //id	                        name	    address	                                                birthday	gender	ethnicity	class	issue_date	expire_date	height	weight	eye_color	hair_color	is_donor	is_veteran	license_number
    //generated.photos_v3_0081593	Jack Martin	624 Superstition Boulevard, Apache Junction, AZ 85120	11/23/93	M	    white	    B	    5/9/20	    5/9/25	    5'-10''	142 lb	BRO	        BRO	        FALSE	    FALSE	    D90236684

    auto inputRowType =
      ROW({"id",
           "name",
           "address",
           "birthday",
           "gender",
           "ethnicity",	
           "class",	
           "issue_date",	
           "expire_date",	
           "height",
           "weight",
           "eye_color",
           "hair_color",
           "is_donor",
           "is_veteran",
           "license_number",
           },
          {VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           BOOLEAN(),
           BOOLEAN(),
           VARCHAR()
           });
    
    auto plan =
      PlanBuilder(pool_.get())
          .tableScan(inputRowType, {}, "")
          .project(
            {"id",
           "name",
           "address",
           "birthday",
           "gender",
           "ethnicity",	
           "class",	
           "issue_date",	
           "expire_date",	
           "height",
           "weight",
           "eye_color",
           "hair_color",
           "is_donor",
           "is_veteran",
           "license_number"})
            // .limit(0, numSamples, false)
          .planNode();

    std::shared_ptr<folly::Executor> executor =
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency());
    std::shared_ptr<core::QueryCtx> queryCtx =
        std::make_shared<core::QueryCtx>(executor.get());

    std::unordered_map<std::string, std::string> configs = {
        {std::string(
            connector::hive::HiveConfig::kFileColumnNamesReadAsLowerCase),
        "true"}};
    queryCtx->setConnectorSessionOverridesUnsafe(
        kHiveConnectorId, std::move(configs));
    const int numSplitsPerFile = 1;
    CursorParameters params;
    params.queryCtx = queryCtx;
    params.planNode = plan;

    bool noMoreSplits = false;
    auto addSplits = [&](exec::Task* task) {
        if (!noMoreSplits) {
        auto const splits = HiveConnectorTestBase::makeHiveConnectorSplits(
            {"file:../../../../data/idmeta.parquet"},
            numSplitsPerFile,
            dwio::common::FileFormat::PARQUET);
        for (const auto& split : splits) {
            task->addSplit("0", exec::Split(split));
        }
        task->noMoreSplits("0");
        }
        noMoreSplits = true;
  };

  auto result = readCursor(params, addSplits);
  auto data = result.second;

    // std::cout << "[INFO] loaded data: \n"
    //           << data[0]->toString(0, data[0]->size()) << std::endl;

  // idlabel
  // id	                    isfraud	fraudpattern	                srcvalue	srcfontstyle	srcfontsize	srcfontcolor	srcbbox	desvalue	desfontstyle	desfontsize	desfontcolor	desbbox	srcname	srcregionvalue	srcregionfontstyle	srcregionfontsize	srcregionfontcolor	srcregionbbox	srcshift	desname	desregionvalue	desregionfontstyle	desregionfontsize	desregionfontcolor	desregionbbox	desshift

  std::cout << "\n" << typeid(data).name() << "\n";
  return data;
  

}

std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> cnn_fraud_detection::runtestidlabel(){
    // First Part 
    // Read Data


    // idmeta
    //id	                        name	    address	                                                birthday	gender	ethnicity	class	issue_date	expire_date	height	weight	eye_color	hair_color	is_donor	is_veteran	license_number
    //generated.photos_v3_0081593	Jack Martin	624 Superstition Boulevard, Apache Junction, AZ 85120	11/23/93	M	    white	    B	    5/9/20	    5/9/25	    5'-10''	142 lb	BRO	        BRO	        FALSE	    FALSE	    D90236684

    auto inputRowType =
      ROW({"id",	"isfraud",	"fraudpattern",	"srcvalue",	"srcfontstyle",	"srcfontsize",	"srcfontcolor",	"srcbbox",	"desvalue",	"desfontstyle",	"desfontsize",	"desfontcolor",	"desbbox",	"srcname",	"srcregionvalue",	"srcregionfontstyle",	"srcregionfontsize",	"srcregionfontcolor",	"srcregionbbox",	"srcshift",	"desname",	"desregionvalue",	"desregionfontstyle",	"desregionfontsize",	"desregionfontcolor",	"desregionbbox",	"desshift"
           },
          {VARCHAR(),
           BOOLEAN(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           DOUBLE(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           DOUBLE(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           DOUBLE(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR(),
           DOUBLE(),
           VARCHAR(),
           VARCHAR(),
           VARCHAR()
           });
    
    auto plan =
      PlanBuilder(pool_.get())
          .tableScan(inputRowType, {}, "")
          .project(
            {"id",	"isfraud",	"fraudpattern",	"srcvalue",	"srcfontstyle",	"srcfontsize",	"srcfontcolor",	"srcbbox",	"desvalue",	"desfontstyle",	"desfontsize",	"desfontcolor",	"desbbox",	"srcname",	"srcregionvalue",	"srcregionfontstyle",	"srcregionfontsize",	"srcregionfontcolor",	"srcregionbbox",	"srcshift",	"desname",	"desregionvalue",	"desregionfontstyle",	"desregionfontsize",	"desregionfontcolor",	"desregionbbox",	"desshift"})
            // .limit(0, numSamples, false)
          .planNode();

    std::shared_ptr<folly::Executor> executor =
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency());
    std::shared_ptr<core::QueryCtx> queryCtx =
        std::make_shared<core::QueryCtx>(executor.get());

    std::unordered_map<std::string, std::string> configs = {
        {std::string(
            connector::hive::HiveConfig::kFileColumnNamesReadAsLowerCase),
        "true"}};
    queryCtx->setConnectorSessionOverridesUnsafe(
        kHiveConnectorId, std::move(configs));
    const int numSplitsPerFile = 1;
    CursorParameters params;
    params.queryCtx = queryCtx;
    params.planNode = plan;

    bool noMoreSplits = false;
    auto addSplits = [&](exec::Task* task) {
        if (!noMoreSplits) {
        auto const splits = HiveConnectorTestBase::makeHiveConnectorSplits(
            {"file:../../../../data/idlabel.parquet"},
            numSplitsPerFile,
            dwio::common::FileFormat::PARQUET);
        for (const auto& split : splits) {
            task->addSplit("0", exec::Split(split));
        }
        task->noMoreSplits("0");
        }
        noMoreSplits = true;
  };

  auto result = readCursor(params, addSplits);
  auto data = result.second;

    // std::cout << "[INFO] loaded data: \n"
    //           << data[0]->toString(0, data[0]->size()) << std::endl;

  // idlabel
  // id	                    isfraud	fraudpattern	                srcvalue	srcfontstyle	srcfontsize	srcfontcolor	srcbbox	desvalue	desfontstyle	desfontsize	desfontcolor	desbbox	srcname	srcregionvalue	srcregionfontstyle	srcregionfontsize	srcregionfontcolor	srcregionbbox	srcshift	desname	desregionvalue	desregionfontstyle	desregionfontsize	desregionfontcolor	desregionbbox	desshift
  std::cout << "\n" << typeid(data).name() << "\n";
  return data;

}

 void cnn_fraud_detection::runjointest(std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>>
 idmeta_data, std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>>
 idlabel_data){
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator,pool_.get())
                    .values(idmeta_data)
                    .mergeJoin(
                        {"id"},
                        {"id_label"},
                        PlanBuilder(planNodeIdGenerator,pool_.get())
                            .values(idlabel_data)
                            .project({"id as id_label", "isfraud"})
                            .planNode(),
                        "",
                        {"id", "name", "isfraud"},
                        core::JoinType::kLeft)
                    .planNode();

    auto executor = std::make_shared<folly::CPUThreadPoolExecutor>(
            std::thread::hardware_concurrency());
    auto queryCtx = std::make_shared<core::QueryCtx>(executor.get());

    // Step 4: Set up Cursor Parameters
    CursorParameters params;
    params.queryCtx = queryCtx;
    params.planNode = plan;

    // Step 5: Execute the plan
    auto [cursor, results] = readCursor(params, [](Task* /*task*/) {});
    
    // Step 6: Print results
    for (const auto& vector : results) {
        std::cout << vector->toString(0, vector->size()) << std::endl;
    }

    // std::cout << "[INFO] loaded data: \n"
    //           << finalScore->toString(0, finalScore->size()) << std::endl;

}





int main(int argc, char** argv){
    std::cout << "Experiment Started.";
    folly::init(&argc, &argv, false);
    memory::MemoryManager::initialize({});
    cnn_fraud_detection demo;
    std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> idmeta_data = demo.runtestidmeta();
    std::vector<std::shared_ptr<facebook::velox::RowVector>, std::allocator<std::shared_ptr<facebook::velox::RowVector>>> idlabel_data = demo.runtestidlabel();
    demo.runjointest(idmeta_data, idlabel_data);
    return 0;
}