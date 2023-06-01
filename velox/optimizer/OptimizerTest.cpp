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

using namespace facebook::velox;
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

  auto col1 = maker.flatVector({0, 1, 2, 3, 4});
  auto col2 = maker.flatVector({1, 2, 3, 4, 5});
  auto inputRowVector = maker.rowVector({"col1", "col2"}, {col1, col2});

  auto myPlan = exec::test::PlanBuilder()
                  .values({inputRowVector})
                  .filter("col1 > 2")
                  .project({"vec_add_to_constant(col1)"})
		            .planFragment();

  facebook::velox::optimizer::Optimizer op(queryCtx_);
  op.traverse(myPlan);


  auto task3 = std::make_shared<exec::Task>("task3", myPlan, 0, queryCtx_);
  // Execute the plan above
  auto result3 = task3->next();
  std::cout << "Results for Query 1:" << result3->toString() << std::endl;
  std::cout << result3->toString(0, result3->size()) << std::endl;
}