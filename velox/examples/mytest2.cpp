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

#include "velox/exec/LocalPlanner.h"
#include "velox/exec/Task.h"
#include "velox/exec/FilterProject.h"
#include "velox/optimizer/Optimizer.h"

#include "velox/exec/tests/HashJoinTest.cpp"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/parse/PlanNodeIdGenerator.h"

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
  auto exprset = context.exprSet();
   std::cout << exprset->toString(false /*compact*/) << std::endl;
    std::cout << exprset->expr(0)->toString(false /*compact*/) << std::endl;
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
        auto exprset = context.exprSet();
        std::cout << exprset->toString(false /*compact*/) << std::endl;
        std::cout << exprset->expr(0)->toString(false /*compact*/) << std::endl;
        
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

    int getSize() const override {
      return size_;
    }
    static exec::VectorFunctionMetadata metadata() {
    return {true /* supportsFlattening */};
  }

};




class MatrixMultiply: public MLFunction {
public:
    MatrixMultiply(float* weights, int shape) {
        weights_ = weights;
        shape_ = shape;
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
        // main logic
        // auto values = args[0]->values()->asMutable<float>();
        // Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(weights_, shape_/20, 20);
        //       Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(values, 20, args[0]->size()/20);
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        
        // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 * m2;
        
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0;
        // auto exprset = context.exprSet();
        // std::cout << exprset->toString(false /*compact*/) << std::endl;
        // std::cout << exprset->expr(0)->toString(false /*compact*/) << std::endl;
        
        std::cout << "This is mat_mul" << std::endl;
        std::cout << "\n" << std::endl;
        std::cout << "my shape is: " << std::endl;
        std::cout << shape_ << std::endl;

        int result_size = shape_;

        auto arg1 = args[0]->as<FlatVector<float>>();
        auto size = arg1->size();
        for (auto j = 0; j < size; ++j) {
          std::cout<< "\n" << arg1->valueAt(j);
        }

        auto result = BaseVector::create<FlatVector<float>>(type, result_size, context.pool());
        for (auto i = 0; i < result_size; ++i) {
            result->set(i, 1.0);
        }
        output = result;
        std::cout<< "\n" << output->size();
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("REAL")
                     .argumentType("REAL")
                     .build()};
    }


    // getters for metadata to be used by optimiser
    float* getTensor(){
        return weights_;
    }

    // get shape
    int getShape() const override{
        return shape_;
    }

private:
    float* weights_;
    int shape_;

};

class Optimizer {
  public:
  Optimizer(std::shared_ptr<core::QueryCtx> queryCtx){
    queryCtx_ = queryCtx;
  }

    void traverse(const core::PlanFragment& planFragment){
      auto task_op = std::make_shared<exec::Task>("task_op", planFragment, 0, queryCtx_);
      auto drivers = task_op->op();
      auto operators = drivers[0]->operators();
      for (const auto& op : operators) {
        auto fp = dynamic_cast<FilterProject*>(op);
        if (fp) {
            std::cout << "The planNodeId is: "<< op->planNodeId() << std::endl;
            std::cout << "The operatorType is: "<< op->operatorType() << std::endl;
            std::cout << "\n" << std::endl;
            std::cout << "The expression tree: " << std::endl;
            const std::unique_ptr<ExprSet>& exprs = fp->getExprs();

            std::cout << exprs->toString(false /*compact*/) << std::endl;
            // Use exprs as needed.
        }
        else {
          std::cout << "The planNodeId is: "<< op->planNodeId() << std::endl;
          std::cout << "The operatorType is: "<< op->operatorType() << std::endl;
          std::cout << "\n" << std::endl;
        }
      }

    }

  ~Optimizer() {
  }

  private:
  std::shared_ptr<core::QueryCtx> queryCtx_;

};


int main(int argc, char** argv) {

   folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();

    
   // VectorMaker is a test utility that helps you build vectors. Shouldn't be
   // in production.
   VectorMaker maker{pool_.get()};
   auto myVec = maker.flatVector<int64_t>({1, 10, 100, 1000, 10000});

   exec::registerVectorFunction(
      "vec_add",
      AddVectors::signatures(),
      std::make_unique<AddVectors>());

  exec::registerVectorFunction(
      "vec_add_2",
      AddVectors::signatures(),
      std::make_unique<AddVectors>());
    
  exec::registerVectorFunction(
      "vec_add_3",
      AddVectors::signatures(),
      std::make_unique<AddVectors>());


    exec::registerVectorFunction(
        "vec_add_to_constant",
        AddVectorToConstant::signatures(),
        std::make_unique<AddVectorToConstant>(myVec, 5),
        AddVectorToConstant::metadata());



BufferPtr values = AlignedBuffer::allocate<float>(10, pool_.get());
float* weights = values->asMutable<float>();

  exec::registerVectorFunction(
  "mat_mul",
  MatrixMultiply::signatures(),
  std::make_unique<MatrixMultiply>(weights,5));

  exec::registerVectorFunction(
  "mat_mul_2",
  MatrixMultiply::signatures(),
  std::make_unique<MatrixMultiply>(weights,10));

  // auto col11 = BaseVector::create<FlatVector<float>>(REAL(), 10, pool_.get());
  // auto col22 = BaseVector::create<FlatVector<float>>(REAL(), 10, pool_.get());
  // for(int i=0; i < 10; i++){
	//   weights[i] = i*10;
  // 	col11->set(i, i*5);
	//   col22->set(i, i*2);
  // } 
  
  // auto inputRowVector1 = maker.rowVector({"col11", "col22"}, {col11, col22});


  // auto myPlan = exec::test::PlanBuilder()
  //                 .values({inputRowVector1})
  //                 .filter("col11 > 5.0")
  //                 .project({"mat_mul(col11)"})
	// 	              .planFragment();

  // auto op = Optimizer(queryCtx_);
  // op.traverse(myPlan);


  // auto task3 = std::make_shared<exec::Task>("task3", myPlan, 0, queryCtx_);
  // // Execute the plan above
  // auto result3 = task3->next();
  // std::cout << "Results for Query 1:" << result3->toString() << std::endl;
  // std::cout << result3->toString(0, result3->size()) << std::endl;
  // From now on we will create a query plan and input dataset, execute it, an
  // assert that the output results contain the dataset properly duplicated.



  // Create an input dataset containing two unnamed columns (INTEGER and
  // VARCHAR), and 5 records:
 
  auto col1 = maker.flatVector({0, 1, 2, 3, 4});
  auto col2 = maker.flatVector({1, 2, 2, 4, 5});
  auto inputRowVector = maker.rowVector({"col1", "col2"}, {col1, col2});

  auto row = maker.flatVector({0, 0, 1, 1});
  auto col = maker.flatVector({0, 1, 0, 1});
  auto va = maker.flatVector({1, 2, 3, 4});
  auto vb = maker.flatVector({11, 12, 13, 14});
  // {1,2 plus {11,12
  //  3,4}      13,14}
  auto inputRowVectorJoinA = maker.rowVector({"rowa", "cola", "valuea"}, {row, col, va});
  auto inputRowVectorJoinB = maker.rowVector({"rowb", "colb", "valueb"}, {row, col, vb});
   
  // Create a query plan containing a ValuesNode (to let you pump input datasets
  // directly into the operator chain), and our custom plan node.
  // auto plan1 = PlanBuilder()
  //                 .values({inputRowVector})
	// 	          .filter("vec_add(col1,col2) > 5")
  //                 .planFragment();

  // auto task1 = std::make_shared<exec::Task>("task1", plan1, 0, queryCtx_);
  // // Execute the plan above
  // auto result1 = task1->next();
  // std::cout << "Results for Query 1:" << result1->toString() << std::endl;
  // std::cout << result1->toString(0, result1->size()) << std::endl;
    facebook::velox::serializer::presto::PrestoVectorSerde::
        registerVectorSerde();

     filesystems::registerLocalFileSystem();
   dwrf::registerDwrfReaderFactory();

  DuckDbQueryRunner duckDbQueryRunner_;
  HashJoinBuilder(*pool_, duckDbQueryRunner_, executor_.get())
      .numDrivers(1)
      .keyTypes({BIGINT()})
      .probeVectors(1600, 5)
      .buildVectors(1500, 5)
      .referenceQuery(
          "SELECT t_k0, t_data, u_k0, u_data FROM t, u WHERE t.t_k0 = u.u_k0")
      .run();

  
auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
core::PlanNodeId nationScanId;
core::PlanNodeId regionScanId;
auto planjoin = PlanBuilder(planNodeIdGenerator)
           .values({inputRowVectorJoinA})
           .capturePlanNodeId(nationScanId)
           .hashJoin(
               {"cola"},
               {"rowb"},
               PlanBuilder(planNodeIdGenerator)
                   .values({inputRowVectorJoinB})
                   .capturePlanNodeId(regionScanId)
                   .planNode(),
               "", // extra filter
               {"rowa","colb", "valuea", "valueb"})
          //  .singleAggregation({"rowa","colb"}, {"sum()"})
           .planNode();

auto nationCnt = AssertQueryBuilder(planjoin).copyResults(pool_.get());

std::cout << std::endl
          << "> number of nations per region in TPC-H: "
          << nationCnt->toString() << std::endl;
std::cout << nationCnt->toString(0, 10) << std::endl;



  auto plan2 = PlanBuilder()
                  .values({inputRowVector}).filter("vec_add(col1,col2) > 5")
                  .project({"vec_add_to_constant(vec_add_3(vec_add_2(vec_add(col1,col2), col2), col2))"})
                  .planFragment();

  auto task2 = std::make_shared<exec::Task>("task2", plan2, 0, queryCtx_);
  // Execute the plan above
  auto result2 = task2->next();
  std::cout << "Results for Query 2:" << result2->toString() << std::endl;
  std::cout << result2->toString(0, result2->size()) << std::endl;

  return 0;
}