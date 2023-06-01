#include <folly/init/Init.h>

#include "velox/common/memory/Memory.h"
#include "velox/exec/Operator.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/BaseVector.h"
#include "velox/core/Expressions.h"
#include "velox/expression/Expr.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/FunctionRegistry.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"

#include "velox/exec/Task.h"

// Test utilities, for convenience and conciseness.
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include <algorithm>
#include <iostream>

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include "velox/exec/tests/utils/TempDirectoryPath.h"



#include "folly/experimental/EventCount.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/DataSink.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PartitionedOutputBufferManager.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::connector::hive;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::memory;



using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;


using exec::test::HiveConnectorTestBase;

// Create a new memory pool to in this example.
auto pool_ = memory::addDefaultLeafMemoryPool();
std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};


class AddVectors: public exec::VectorFunction {
public:
    AddVectors() {}

    void apply(
	const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    
        auto arg1 = args[0]->as<FlatVector<int64_t>>();
	auto arg2 = args[1]->as<FlatVector<int64_t>>();
	auto size = arg1->size();
	auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
	for (auto i = 0; i < size; ++i) {
	    result->set(i, arg1->valueAt(i) + arg2->valueAt(i));
	}
        output = result;
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
		     .returnType("BIGINT")
	         .argumentType("BIGINT")
		     .argumentType("BIGINT")
		     .build()};
    
    }    

};


class AddVectorToConstant: public exec::VectorFunction {
public:
    AddVectorToConstant(FlatVectorPtr<int64_t> vec, int size) {
        vec_ = vec;
	size_ = size;
    }
     
    FlatVectorPtr<int64_t> vec_;
    int size_;

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        auto arg1 = args[0]->as<FlatVector<int64_t>>();
        auto size = arg1->size();
        auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
        for (auto i = 0; i < size; ++i) {
            result->set(i, arg1->valueAt(i) + vec_->valueAt(i));
        }
        output = result;
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("BIGINT")
                     .argumentType("BIGINT")
                     .build()};

    }

};

void writeToFile(
    const std::string& filePath,
    const std::vector<RowVectorPtr>& vectors,
    std::shared_ptr<dwrf::Config> config) {
          std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::defaultMemoryManager().addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
  facebook::velox::dwrf::WriterOptions options;
  options.config = config;
  options.schema = vectors[0]->type();
  auto sink =
      std::make_unique<facebook::velox::dwio::common::LocalFileSink>(filePath);
  auto childPool = rootPool_->addAggregateChild("HiveConnectorTestBase.Writer");
  facebook::velox::dwrf::Writer writer{options, std::move(sink), *childPool};
  for (size_t i = 0; i < vectors.size(); ++i) {
    writer.write(vectors[i]);
  }
  writer.close();
}

static std::string makeTaskId(const std::string& prefix, int num) {
  return fmt::format("local://{}-{}", prefix, num);
}

std::shared_ptr<Task> makeTask(
    const std::string& taskId,
    std::shared_ptr<const core::PlanNode> planNode,
    int destination,
    Consumer consumer = nullptr,
    int64_t maxMemory = kMaxMemory) {
    std::unordered_map<std::string, std::string> configSettings_;
  auto queryCtx = std::make_shared<core::QueryCtx>(
      executor_.get(), std::make_shared<core::MemConfig>(configSettings_));
  queryCtx->testingOverrideMemoryPool(
      memory::defaultMemoryManager().addRootPool(
          queryCtx->queryId(), maxMemory));
  core::PlanFragment planFragment{planNode};
  return std::make_shared<Task>(
      taskId,
      std::move(planFragment),
      destination,
      std::move(queryCtx),
      std::move(consumer));
}

// std::vector<RowVectorPtr> makeVectors(int count, int rowsPerVector) {
//   std::vector<RowVectorPtr> vectors;
//   for (int i = 0; i < count; ++i) {
//     auto vector = std::dynamic_pointer_cast<RowVector>(
//         BatchMaker::createBatch(rowType_, rowsPerVector, *pool_));
//     vectors.push_back(vector);
//   }
//   return vectors;
// }

void addHiveSplits(
    std::shared_ptr<Task> task,
    const std::vector<std::shared_ptr<TempFilePath>>& filePaths) {
  for (auto& filePath : filePaths) {
    auto split = exec::Split(
        std::make_shared<HiveConnectorSplit>(
            kHiveConnectorId,
            "file:" + filePath->path,
            facebook::velox::dwio::common::FileFormat::DWRF),
        -1);
    task->addSplit("0", std::move(split));
    VLOG(1) << filePath->path << "\n";
  }
  task->noMoreSplits("0");
}

void addRemoteSplits(
    std::shared_ptr<Task> task,
    const std::vector<std::string>& remoteTaskIds) {
  for (auto& taskId : remoteTaskIds) {
    auto split =
        exec::Split(std::make_shared<RemoteConnectorSplit>(taskId), -1);
    task->addSplit("0", std::move(split));
  }
  task->noMoreSplits("0");
}

// void setupSources(int filePathCount, int rowsPerVector) {
//   auto filePaths_ = makeFilePaths(filePathCount);
//   auto vectors_ = makeVectors(filePaths_.size(), rowsPerVector);
//   for (int i = 0; i < filePaths_.size(); i++) {
//     writeToFile(filePaths_[i]->path, vectors_[i]);
//   }
//   createDuckDbTable(vectors_);
// }

RowTypePtr rowType_{
    ROW({"c0", "c1", "c2", "c3", "c4", "c5"},
        {BIGINT(), INTEGER(), SMALLINT(), REAL(), DOUBLE(), VARCHAR()})};


int main(int argc, char** argv) {
  std::unordered_map<std::string, std::string> configSettings_;
  std::vector<std::shared_ptr<TempFilePath>> filePaths_;
  std::vector<RowVectorPtr> vectors_;
   folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();
   
   const std::string kHiveConnectorId = "test-hive";
   auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(kHiveConnectorId, nullptr);
  connector::registerConnector(hiveConnector);

   filesystems::registerLocalFileSystem();
   dwrf::registerDwrfReaderFactory();

  VectorMaker maker{pool_.get()};
  auto c1 = maker.flatVector({0, 1, 2, 3, 4});
  auto c2 = maker.flatVector({1, 2, 3, 4, 5});
  auto inputRowVector = maker.rowVector({"c1", "c2"}, {c1, c2});
  auto file = TempFilePath::create();
  writeToFile(file->path, {inputRowVector}, std::make_shared<facebook::velox::dwrf::Config>());
  filePaths_.push_back(file);


  std::vector<std::shared_ptr<Task>> tasks;
  auto leafTaskId = makeTaskId("leaf", 0);
  auto Plan = exec::test::PlanBuilder()
                         .tableScan(std::dynamic_pointer_cast<const RowType>(inputRowVector->type()))
                         .planNode();

    auto leafTask = makeTask(leafTaskId, Plan, 0);
    tasks.push_back(leafTask);
    // Task::start(leafTask, 2);
    addHiveSplits(leafTask, filePaths_);
  
  auto result0 = tasks[0].get()->next();
  std::cout << "Results for Query 1:" << result0->toString() << std::endl;
  std::cout << result0->toString(0, result0->size()) << std::endl;




  // next() starts execution using the client thread. The loop pumps output
  // vectors out of the task (there are none in this query fragment).
  // while (auto result = writeTask->next())
  //   std::cout << result->toString(0, result->size()) << std::endl;

  //   // hive.writeToFile(file->path, {data});
  //   std::string filePath = "/home/local/ASUAD/qlin36/velox/_build/release/velox/examples/iris.orc";
  //   auto leafTaskId = makeTaskId("leaf", 0);

  //   auto leafPlan =
  //     PlanBuilder()
  //         .tableScan(std::dynamic_pointer_cast<const RowType>(data->type()))
  //         .planFragment();
  //   auto leafTask = std::make_shared<exec::Task>(leafTaskId, leafPlan, 0, queryCtx_);
  //   // auto leafTask = makeTask(leafTaskId, leafPlan, 0);
  //   // Task::start(leafTask, 1);

  //   leafTask.get()->addSplit(
  //     "0", exec::Split(HiveConnectorTestBase::makeHiveConnectorSplit(file->path)));

  //   leafTask.get()->noMoreSplits("0");

  //   auto results = leafTask->next();
  //   std::cout << "Results for:" << results->toString() << std::endl;
  //   std::cout << results->toString(0, results->size()) << std::endl;
  
    // addHiveSplits(leafTask, filePaths_);

    // for (auto& filePath : fs::directory_iterator(tempDir->path)) {
    // auto connectorSplit = std::make_shared<connector::hive::HiveConnectorSplit>(
    //     kHiveConnectorId,
    //     "file:" + filePath.path().string(),
    //     dwio::common::FileFormat::DWRF);
    // // Wrap it in a `Split` object and add to the task. We need to specify to
    // // which operator we're adding the split (that's why we captured the
    // // TableScan's id above). Here we could pump subsequent split/files into the
    // // TableScan.
    // readTask->addSplit(scanNodeId, exec::Split{connectorSplit});
    // }
    // readTask->noMoreSplits(scanNodeId);

    // // Read output vectors and print them.
    // while (auto result = readTask->next()) {
    //     LOG(INFO) << "Vector available after processing (scan + sort):";
    //     for (vector_size_t i = 0; i < result->size(); ++i) {
    //     LOG(INFO) << result->toString(i);
    //     }
    // }
    //    const std::string kHiveConnectorId = "test-hive";

//   // Create a new connector instance from the connector factory and register
//   // it:


//     // To be able to read local files, we need to register the local file
//     // filesystem. We also need to register the dwrf reader factory as well as a
//     // write protocol, in this case commit is not required:
//     filesystems::registerLocalFileSystem();
//     dwrf::registerDwrfReaderFactory();



//     auto inputRowType = ROW({{"my_col", BIGINT()}});
//     const size_t vectorSize = 10;

//     // Create a base flat vector and fill it with consecutive integers, then
//     // shuffle them.
//     auto vector = BaseVector::create(BIGINT(), vectorSize, pool_.get());
//     auto rawValues = vector->values()->asMutable<int64_t>();

//     std::iota(rawValues, rawValues + vectorSize, 0); // 0, 1, 2, 3, ...
//     std::random_device rd;
//     std::mt19937 g(rd());
//     std::shuffle(rawValues, rawValues + vectorSize, g);

//     // Wrap the vector (column) in a RowVector.
//     auto rowVector = std::make_shared<RowVector>(
//         pool_.get(), // pool where allocations will be made.
//         inputRowType, // input row type (defined above).
//         BufferPtr(nullptr), // no nulls on this example.
//         vectorSize, // length of the vectors.
//         std::vector<VectorPtr>{vector}); // the input vector data.

//     auto writerPlanFragment =
//       exec::test::PlanBuilder()
//           .values({rowVector})
//           .tableWrite(
//               inputRowType->names(),
//               std::make_shared<core::InsertTableHandle>(
//                   kHiveConnectorId,
//                   HiveConnectorTestBase::makeHiveInsertTableHandle(
//                       inputRowType->names(),
//                       inputRowType->children(),
//                       {},
//                       HiveConnectorTestBase::makeLocationHandle(
//                           tempDir->path))),
//               connector::CommitStrategy::kNoCommit)
//           .planFragment();


//     auto writeTask = std::make_shared<exec::Task>(
//       "my_write_task",
//       writerPlanFragment,
//       /*destination=*/0,
//       std::make_shared<core::QueryCtx>(executor_.get()));

//     while (auto result = writeTask->next())
//     ;

//     core::PlanNodeId scanNodeId;
//     auto readPlanFragment = exec::test::PlanBuilder()
//                                 .tableScan(inputRowType)
//                                 .capturePlanNodeId(scanNodeId)
//                                 .orderBy({"my_col"}, /*isPartial=*/false)
//                                 .planFragment();

//     // Create the reader task.
//     auto readTask = std::make_shared<exec::Task>(
//         "my_read_task",
//         readPlanFragment,
//         /*destination=*/0,
//         std::make_shared<core::QueryCtx>(executor_.get()));


    
   // VectorMaker is a test utility that helps you build vectors. Shouldn't be
   // in production.
  //  VectorMaker maker{pool_.get()};
  //  auto myVec = maker.flatVector<int64_t>({1, 10, 100, 1000, 10000});

  //  exec::registerVectorFunction(
  //     "vec_add",
  //     AddVectors::signatures(),
  //     std::make_unique<AddVectors>());

  //  exec::registerVectorFunction(
  //     "vec_add_to_constant",
  //     AddVectorToConstant::signatures(),
  //     std::make_unique<AddVectorToConstant>(myVec, 5));

  // // From now on we will create a query plan and input dataset, execute it, an
  // // assert that the output results contain the dataset properly duplicated.



  // // Create an input dataset containing two unnamed columns (INTEGER and
  // // VARCHAR), and 5 records:
 
  // auto col1 = maker.flatVector({0, 1, 2, 3, 4});
  // auto col2 = maker.flatVector({1, 2, 3, 4, 5});
  // auto inputRowVector = maker.rowVector({"col1", "col2"}, {col1, col2});

   
  // // Create a query plan containing a ValuesNode (to let you pump input datasets
  // // directly into the operator chain), and our custom plan node.
  // auto plan1 = PlanBuilder()
  //                 .values({inputRowVector})
	// 	          .filter("vec_add(col1,col2) > 5")
  //                 .planFragment();

  // auto task1 = std::make_shared<exec::Task>("task1", plan1, 0, queryCtx_);
  // // Execute the plan above
  // auto result1 = task1->next();
  // std::cout << "Results for Query 1:" << result1->toString() << std::endl;
  // std::cout << result1->toString(0, result1->size()) << std::endl;
  
  // auto plan2 = PlanBuilder()
  //                 .values({inputRowVector})
  //                 .project({"vec_add_to_constant(col1)"})
  //                 .planFragment();

  // auto task2 = std::make_shared<exec::Task>("task2", plan2, 0, queryCtx_);
  // // Execute the plan above
  // auto result2 = task2->next();
  // std::cout << "Results for Query 2:" << result2->toString() << std::endl;
  // std::cout << result2->toString(0, result2->size()) << std::endl;

  return 0;
}