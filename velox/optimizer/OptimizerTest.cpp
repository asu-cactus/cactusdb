#include <iostream>
#include <folly/init/Init.h>
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
#include "velox/optimizer/Optimizer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;
using namespace facebook::velox::optimizer;

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

    int getSize() const override {
      return size_;
    }
    static exec::VectorFunctionMetadata metadata() {
    return {true /* supportsFlattening */};
  }

};

class VectorPlus: public exec::VectorFunction {
public:
    VectorPlus() {
    }

     
    FlatVectorPtr<int64_t> vec_1;
    FlatVectorPtr<int64_t> vec_2;


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
            result->set(i, arg1->valueAt(i) * arg2->valueAt(i));
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

    int getSize() const override {
      return 0;
    }
    static exec::VectorFunctionMetadata metadata() {
    return {true /* supportsFlattening */};
  }

};


auto pool_ = memory::addDefaultLeafMemoryPool();
std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

int main(int argc, char** argv) {
    
    folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();

   VectorMaker maker{pool_.get()};
   auto myVec = maker.flatVector<int64_t>({1, 10, 100, 1000, 10000});

    exec::registerVectorFunction(
        "vec_add_to_constant",
        AddVectorToConstant::signatures(),
        std::make_unique<AddVectorToConstant>(myVec, 5),
        AddVectorToConstant::metadata());

  exec::registerVectorFunction(
        "vec_plus",
        VectorPlus::signatures(),
        std::make_unique<VectorPlus>(),
        VectorPlus::metadata());
  // auto col1 = maker.flatVector({0, 1, 2, 3, 4});
  // auto col2 = maker.flatVector({1, 2, 3, 4, 5});
  // auto inputRowVector = maker.rowVector({"col1", "col2"}, {col1, col2});

  auto row = maker.flatVector({0, 0, 1, 1});
  auto col = maker.flatVector({0, 1, 0, 1});
  auto va = maker.flatVector({1, 2, 3, 4});
  auto vb = maker.flatVector({11, 12, 13, 14});
  // {1,2 plus {11,12
  //  3,4}      13,14}
  auto inputRowVectorJoinA = maker.rowVector({"a_row", "a_col", "a_value"}, {row, col, va});
  auto inputRowVectorJoinB = maker.rowVector({"b_row", "b_col", "b_value"}, {row, col, vb});

  auto inputRowVectorJoin = maker.rowVector({"a_row", "a_col", "a_value", "b_row", "b_col", "b_value"}, {row, col, va, row, col, vb});
  auto JoinT = maker.rowVector({"table_a", "table_b"},{inputRowVectorJoinA, inputRowVectorJoinB});
  //two tables means nestjoin or other format of plan

  auto myPlan = exec::test::PlanBuilder()
                  .values({JoinT})
                  .project({"vec_plus(table_a.a_value, table_b.b_value) AS result"})
		            .planFragment();

  // auto task1 = std::make_shared<exec::Task>("task1", myPlan, 0, queryCtx_);
  // // Execute the plan above
  // auto result1 = task1->next();
  // std::cout << "Results for Query 1:" << result1->toString() << std::endl;
  // std::cout << result1->toString(0, result1->size()) << std::endl;
  core::PlanNodeId Id;
  auto myPlan2 = exec::test::PlanBuilder()
                  .values({JoinT})
                  .project({"vec_plus(table_a.a_value, table_b.b_value) AS result"})
		            .planNode();

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId AId;
  core::PlanNodeId BId;
  auto planjoin = exec::test::PlanBuilder(planNodeIdGenerator)
            .values({JoinT})
            .project({"table_a.a_col", "table_a.a_row","table_a.a_value"})
            .capturePlanNodeId(AId)
            .hashJoin(
                {"a_col"},
                {"b_row"},
                exec::test::PlanBuilder(planNodeIdGenerator)
                    .values({JoinT})
                    .project({"table_b.b_col", "table_b.b_row","table_b.b_value"})
                    .capturePlanNodeId(BId)
                    .planNode(),
                "", // extra filter
                {"a_row","b_col", "a_value", "b_value"})
            .project({"a_row", "b_col","a_value * b_value AS mp"})
            .singleAggregation({"a_row","b_col"}, {"sum(mp) AS result"})
            .project({"result"})
            .planNode();

  // auto taskj = std::make_shared<exec::Task>("taskj", planjoin, 0, queryCtx_);
  // auto res = taskj->next();
  // std::cout << "Results for Query 1:" << res->toString() << std::endl;
  // std::cout << res->toString(0, res->size()) << std::endl;
auto oldplan = myPlan2->sources()[0];
auto source = planjoin->sources()[0]->sources()[0]->sources()[0]->sources()[0]->sources()[0];
source = oldplan;

  auto res2 = AssertQueryBuilder(myPlan2).copyResults(pool_.get());
  std::cout << "Results for Query 3:" << res2->toString() << std::endl;
  std::cout << res2->toString(0, res2->size()) << std::endl;


  auto res = AssertQueryBuilder(planjoin).copyResults(pool_.get());
  std::cout << "Results for Query 2:" << res->toString() << std::endl;
  std::cout << res->toString(0, res->size()) << std::endl;

  // auto newplan = std::shared_ptr<const PlanNode>;

  // newplan->sources()[0] = myPlan2->sources()[0];

  // planjoin->sources()[0]->sources()[0]->sources()[0]->sources()[0]->sources()[0] = const_cast<std::shared_ptr<const facebook::velox::core::PlanNode>&>(myPlan2->sources()[0]);
  // auto newplan = planjoin;

  // auto res = AssertQueryBuilder(planjoin).copyResults(pool_.get());
  // std::cout << "Results for Query 2:" << res->toString() << std::endl;
  // std::cout << res->toString(0, res->size()) << std::endl;
  // myPlan2->sources()[0] = planjoin->sources()[0];

  // facebook::velox::optimizer::Optimizer op(queryCtx_);
  // auto optimizedPlan = op.op(myPlan);
  // op.traverse(myPlan);

  // auto res = AssertQueryBuilder(optimizedPlan).copyResults(pool_.get());
  // std::cout << "Results for Query 3:" << res->toString() << std::endl;
  // std::cout << res->toString(0, res->size()) << std::endl;



  // std::cout << "Results for Query:" << results->toString() << std::endl;
  // std::cout << results->toString(0, results->size()) << std::endl;
}