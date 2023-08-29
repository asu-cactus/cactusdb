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
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

// #include <Eigen/Dense>
// #include <cblas.h>
// #include <chrono>
// #include <torch/torch.h>
// #include "velox/ml_functions/DNNBuilder.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>

#include "velox/exec/FilterProject.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;
using namespace facebook::velox::optimizer;

using exec::test::HiveConnectorTestBase;

constexpr int64_t KB = 1024L;
constexpr int64_t MB = 1024L * KB;
constexpr int64_t GB = 1024L * MB;

// class AddVectorToConstant: public exec::VectorFunction {
// public:
//     AddVectorToConstant(FlatVectorPtr<int64_t> vec, int size) {
//         vec_ = vec;
// 	      size_ = size;
//     }

     
//     FlatVectorPtr<int64_t> vec_;
//     int size_;


//     void apply(
//         const SelectivityVector& rows,
//         std::vector<VectorPtr>& args,
//         const TypePtr& type,
//         exec::EvalCtx& context,
//         VectorPtr& output) const override {

//         auto arg1 = args[0]->as<FlatVector<int64_t>>();
//         auto size = arg1->size();
//         auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
        
        
//         for (auto i = 0; i < size; ++i) {
//             result->set(i, arg1->valueAt(i) + vec_->valueAt(i));
//         }
//         output = result;
//     }

//     static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
//         return {exec::FunctionSignatureBuilder()
//                      .returnType("BIGINT")
//                      .argumentType("BIGINT")
//                      .build()};

//     }

//     int getSize() const override {
//       return size_;
//     }
//     static exec::VectorFunctionMetadata metadata() {
//     return {true /* supportsFlattening */};
//   }

// };

// class VectorPlus: public exec::VectorFunction {
// public:
//     VectorPlus() {
//     }

     
//     FlatVectorPtr<int64_t> vec_1;
//     FlatVectorPtr<int64_t> vec_2;


//     void apply(
//         const SelectivityVector& rows,
//         std::vector<VectorPtr>& args,
//         const TypePtr& type,
//         exec::EvalCtx& context,
//         VectorPtr& output) const override {

//         auto arg = args[0]->as<FlatVector<int64_t>>();

//         if (arg == nullptr) {
//           auto arg1 = args[0]->wrappedVector()->as<FlatVector<int64_t>>();;

//           auto arg2 = args[1]->as<FlatVector<int64_t>>();
//           auto size = arg1->size();
//           auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
        
        
//           for (auto i = 0; i < size; ++i) {
//               result->set(i, arg1->valueAt(i) * arg2->valueAt(i));
//           }
//           output = result;
//         }
//         else {
//           auto arg1 = std::move(arg);
//           auto arg2 = args[1]->as<FlatVector<int64_t>>();
//           auto size = arg1->size();
//           auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
        
        
//           for (auto i = 0; i < size; ++i) {
//               result->set(i, arg1->valueAt(i) * arg2->valueAt(i));
//           }
//           output = result;
//         }
//     }

//     static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
//         return {exec::FunctionSignatureBuilder()
//                      .returnType("BIGINT")
//                      .argumentType("BIGINT")
//                      .argumentType("BIGINT")
//                      .build()};

//     }

//     int getSize() const override {
//       return 0;
//     }
//     static exec::VectorFunctionMetadata metadata() {
//     return {true /* supportsFlattening */};
//   }

// };

// class VectorMut: public exec::VectorFunction {
// public:
//     VectorMut() {
//     }



//     void apply(
//         const SelectivityVector& rows,
//         std::vector<VectorPtr>& args,
//         const TypePtr& type,
//         exec::EvalCtx& context,
//         VectorPtr& output) const override {

//         auto arg = args[0]->as<FlatVector<int64_t>>();

//         if (arg == nullptr) {
//           auto arg1 = args[0]->wrappedVector()->as<FlatVector<int64_t>>();;

//           auto arg2 = args[1]->as<FlatVector<int64_t>>();
//           auto size = arg1->size();
//           auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
        
        
//           for (auto i = 0; i < size; ++i) {
//               result->set(i, arg1->valueAt(i) * arg2->valueAt(i));
//           }
//           output = result;
//         }
//         else {
//           auto arg1 = std::move(arg);
//           auto arg2 = args[1]->as<FlatVector<int64_t>>();
//           auto size = arg1->size();
//           auto result = BaseVector::create<FlatVector<int64_t>>(type, size, context.pool());
        
        
//           for (auto i = 0; i < size; ++i) {
//             if (i==0) {
//               result->set(i, 37);
//             }
//             else if(i==1) {
//               result->set(i, 40);
//             }
//             else if(i==2) {
//               result->set(i, 85);
//             }
//             else {
//               result->set(i, 92);
//             }
//           }
//           output = result;
//         }
//     }

//     static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
//         return {exec::FunctionSignatureBuilder()
//                      .returnType("BIGINT")
//                      .argumentType("BIGINT")
//                      .argumentType("BIGINT")
//                      .build()};

//     }

//     int getSize() const override {
//       return 0;
//     }
//     static exec::VectorFunctionMetadata metadata() {
//     return {true /* supportsFlattening */};
//   }

// };

// class Flat: public exec::VectorFunction {
// public:
//     Flat() {
//     }

     
//     FlatVectorPtr<int64_t> vec_1;
//     FlatVectorPtr<int64_t> vec_2;


//     void apply(
//         const SelectivityVector& rows,
//         std::vector<VectorPtr>& args,
//         const TypePtr& type,
//         exec::EvalCtx& context,
//         VectorPtr& output) const override {

//         auto input_elements = args[0]->as<ArrayVector>()->elements();
//         float* input_values = input_elements->values()->asMutable<float>();
//         int size = input_elements->size();
//         auto result = BaseVector::create<FlatVector<float>>(type, size, context.pool());
        
        
//           for (auto i = 0; i < size; ++i) {
//               result->set(i, input_values[i]);
//           }
//           output = result;
        
//     }

//     static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
//         return {exec::FunctionSignatureBuilder()
//                      .returnType("REAL")
//                      .argumentType("array(REAL)")
//                      .build()};

//     }

//     int getSize() const override {
//       return 0;
//     }
//     static exec::VectorFunctionMetadata metadata() {
//     return {true /* supportsFlattening */};
//   }

// };

class ADP: public exec::VectorFunction {
public:
    ADP() {
    }

     
    FlatVectorPtr<int64_t> vec_1;
    FlatVectorPtr<int64_t> vec_2;


    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
        

        auto col = args[0]->as<FlatVector<int32_t>>();
        auto row = args[1]->as<FlatVector<int32_t>>();
        auto res = args[2]->as<FlatVector<float>>();
        auto colnum = col->valueAt(0)+1;
        auto size = res->size();
        auto rownum = size / colnum;

      
          
        std::vector<std::vector<float>> result(rownum, std::vector<float>(colnum));
        for (int i = 0; i < size; ++i) {
              result[row->valueAt(i)][col->valueAt(i)] = res->valueAt(i);
        }  
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(result, REAL());
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("array(REAL)")
                     .argumentType("INTEGER")
                     .argumentType("INTEGER")
                     .argumentType("REAL")
                     .build()};

    }

    int getSize() const override {
      return 0;
    }
    static exec::VectorFunctionMetadata metadata() {
    return {true /* supportsFlattening */};
  }

};

std::string throwAggregateFunctionDoesntExist(const std::string& name) {
  std::stringstream error;
  error << "Aggregate function doesn't exist: " << name << ".";
  if (exec::aggregateFunctions().empty()) {
    error << " Registry of aggregate functions is empty. "
             "Make sure to register some aggregate functions.";
  }
  VELOX_USER_FAIL(error.str());
}

std::string toString(
    const std::string& name,
    const std::vector<TypePtr>& types) {
  std::ostringstream signature;
  signature << name << "(";
  for (auto i = 0; i < types.size(); i++) {
    if (i > 0) {
      signature << ", ";
    }
    signature << types[i]->toString();
  }
  signature << ")";
  return signature.str();
}

// std::string toString(
//     const std::vector<std::shared_ptr<facebook::velox::exec::AggregateFunctionSignature>>&
//         signatures) {
//   std::stringstream out;
//   for (auto i = 0; i < signatures.size(); ++i) {
//     if (i > 0) {
//       out << ", ";
//     }
//     out << signatures[i]->toString();
//   }
//   return out.str();
// }

std::string throwAggregateFunctionSignatureNotSupported(
    const std::string& name,
    const std::vector<TypePtr>& types,
    const std::vector<std::shared_ptr<facebook::velox::exec::AggregateFunctionSignature>>&
        signatures) {
  std::stringstream error;
  error << "Aggregate function signature is not supported: "
        << toString(name, types)
        << ". Supported signatures: " << toString(signatures) << ".";
  VELOX_USER_FAIL(error.str());
}

TypePtr resolveAggregateType(
    const std::string& aggregateName,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& rawInputTypes,
    bool nullOnFailure) {
  if (auto signatures = exec::getAggregateFunctionSignatures(aggregateName)) {
    for (const auto& signature : signatures.value()) {
      exec::SignatureBinder binder(*signature, rawInputTypes);
      if (binder.tryBind()) {
        return binder.tryResolveType(
            exec::isPartialOutput(step) ? signature->intermediateType()
                                        : signature->returnType());
      }
    }

    if (nullOnFailure) {
      return nullptr;
    }

    throwAggregateFunctionSignatureNotSupported(
        aggregateName, rawInputTypes, signatures.value());
  }

  if (nullOnFailure) {
    return nullptr;
  }

  throwAggregateFunctionDoesntExist(aggregateName);
  return nullptr;
}

class AggregateTypeResolver {
 public:
  explicit AggregateTypeResolver(core::AggregationNode::Step step)
      : step_(step), previousHook_(core::Expressions::getResolverHook()) {
    core::Expressions::setTypeResolverHook(
        [&](const auto& inputs, const auto& expr, bool nullOnFailure) {
          return resolveType(inputs, expr, nullOnFailure);
        });
  }

  ~AggregateTypeResolver() {
    core::Expressions::setTypeResolverHook(previousHook_);
  }

  void setResultType(const TypePtr& type) {
    resultType_ = type;
  }

 private:
  TypePtr resolveType(
      const std::vector<core::TypedExprPtr>& inputs,
      const std::shared_ptr<const core::CallExpr>& expr,
      bool nullOnFailure) const {
    if (resultType_) {
      return resultType_;
    }

    std::vector<TypePtr> types;
    for (auto& input : inputs) {
      types.push_back(input->type());
    }

    auto functionName = expr->getFunctionName();

    // Use raw input types (if available) to resolve intermediate and final
    // result types.
    if (exec::isRawInput(step_)) {
      return resolveAggregateType(functionName, step_, types, nullOnFailure);
    }

    if (!nullOnFailure) {
      VELOX_USER_FAIL(
          "Cannot resolve aggregation function return type without raw input types: {}",
          functionName);
    }
    return nullptr;
  }

  const core::AggregationNode::Step step_;
  const core::Expressions::TypeResolverHook previousHook_;
  TypePtr resultType_;
};


class optimizerBuilder : public exec::test::PlanBuilder {


public:
  memory::MemoryPool* pool_;

  core::PlanFragment planFragment(std::shared_ptr<const core::PlanNode> planNode_) const {
  return core::PlanFragment{planNode_};
}
 void capturePlanNodeId(core::PlanNodeId& id) {
    VELOX_CHECK_NOT_NULL(planNode_);
    id = planNode_->id();
  }
  core::TypedExprPtr inferTypes(
    const std::shared_ptr<const core::IExpr>& untypedExpr,
    core::PlanNodePtr source) {
  return core::Expressions::inferTypes(
      untypedExpr, source->outputType(), pool_);
}

  core::PlanNodePtr& project_O(const std::vector<std::string>& projections, core::PlanNodePtr source) {
    std::vector<core::TypedExprPtr> expressions;
    std::vector<std::string> projectNames;
    
    for (auto i = 0; i < projections.size(); ++i) {
      auto untypedExpr = parse::parseExpr(projections[i], options_);
      expressions.push_back(inferTypes(untypedExpr, source));
      if (untypedExpr->alias().has_value()) {
        projectNames.push_back(untypedExpr->alias().value());
      } else if (
          auto fieldExpr = dynamic_cast<const core::FieldAccessExpr*>(untypedExpr.get())) {
        projectNames.push_back(fieldExpr->getFieldName());
      } else {
        projectNames.push_back(fmt::format("p{}", i));
      }
    }
    
    planNode_ = std::make_shared<core::ProjectNode>(
        nextPlanNodeId(),
        std::move(projectNames),
        std::move(expressions),
        source);
    
    return planNode_;
  }

  // core::PlanNodePtr& hashjoin_O
  RowTypePtr concat(const RowTypePtr& a, const RowTypePtr& b) {
  std::vector<std::string> names = a->names();
  std::vector<TypePtr> types = a->children();
  names.insert(names.end(), b->names().begin(), b->names().end());
  types.insert(types.end(), b->children().begin(), b->children().end());
  return ROW(std::move(names), std::move(types));
}

RowTypePtr extract(
    const RowTypePtr& type,
    const std::vector<std::string>& childNames) {
  std::vector<std::string> names = childNames;

  std::vector<TypePtr> types;
  types.reserve(childNames.size());
  for (const auto& name : childNames) {
    types.emplace_back(type->findChild(name));
  }
  return ROW(std::move(names), std::move(types));
}

core::TypedExprPtr parseExpr(
    const std::string& text,
    RowTypePtr& rowType,
    const parse::ParseOptions& options,
    memory::MemoryPool* pool) {
  auto untyped = parse::parseExpr(text, options);
  return core::Expressions::inferTypes(untyped, rowType, pool);
}

std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields(
    const RowTypePtr& inputType,
    const std::vector<std::string>& names) {
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields;
  for (const auto& name : names) {
    fields.push_back(field(inputType, name));
  }
  return fields;
}

std::shared_ptr<const core::FieldAccessTypedExpr> field(
    const RowTypePtr& inputType,
    const std::string& name) {
  auto index = inputType->getChildIdx(name);
  return field(inputType, index);
}

std::shared_ptr<const core::FieldAccessTypedExpr> field(
    const RowTypePtr& inputType,
    column_index_t index) {
  auto name = inputType->names()[index];
  auto type = inputType->childAt(index);
  return std::make_shared<core::FieldAccessTypedExpr>(type, name);
}

  core::PlanNodePtr& hashjoin_O(
    const std::vector<std::string>& leftKeys,
    const std::vector<std::string>& rightKeys,
    const core::PlanNodePtr& build,
    const std::string& filter,
    const std::vector<std::string>& outputLayout,
    core::PlanNodePtr source,
    core::JoinType joinType,
    bool nullAware) {
    VELOX_CHECK_EQ(leftKeys.size(), rightKeys.size());

    auto leftType = source->outputType();
    auto rightType = build->outputType();
    auto resultType = concat(leftType, rightType);
    core::TypedExprPtr filterExpr;
    if (!filter.empty()) {
      filterExpr = parseExpr(filter, resultType, options_, pool_);
    }

    RowTypePtr outputType;
    if (isLeftSemiProjectJoin(joinType) || isRightSemiProjectJoin(joinType)) {
      std::vector<std::string> names = outputLayout;

      // Last column in 'outputLayout' must be a boolean 'match'.
      std::vector<TypePtr> types;
      types.reserve(outputLayout.size());
      for (auto i = 0; i < outputLayout.size() - 1; ++i) {
        types.emplace_back(resultType->findChild(outputLayout[i]));
      }
      types.emplace_back(BOOLEAN());

      outputType = ROW(std::move(names), std::move(types));
    } else {
      outputType = extract(resultType, outputLayout);
    }

    auto leftKeyFields = fields(leftType, leftKeys);
    auto rightKeyFields = fields(rightType, rightKeys);

    planNode_ = std::make_shared<core::HashJoinNode>(
        nextPlanNodeId(),
        joinType,
        nullAware,
        leftKeyFields,
        rightKeyFields,
        std::move(filterExpr),
        std::move(source),
        build,
        outputType);
    return planNode_;
  }

std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> fields(const std::vector<std::string>& names) {
  return fields(planNode_->outputType(), names);
}

  struct ExpressionsAndNames {
    std::vector<std::shared_ptr<const core::CallTypedExpr>> expressions;
    std::vector<std::string> names;
  };

ExpressionsAndNames createAggregateExpressionsAndNames(
    const std::vector<std::string>& aggregates,
    core::AggregationNode::Step step,
    const std::vector<TypePtr>& resultTypes,
    core::PlanNodePtr source) {
  AggregateTypeResolver resolver(step);
  std::vector<std::shared_ptr<const core::CallTypedExpr>> exprs;
  std::vector<std::string> names;
  exprs.reserve(aggregates.size());
  names.reserve(aggregates.size());
  for (auto i = 0; i < aggregates.size(); i++) {
    auto& agg = aggregates[i];
    if (i < resultTypes.size()) {
      resolver.setResultType(resultTypes[i]);
    }

    auto untypedExpr = parse::parseExpr(agg, options_);

    auto expr = std::dynamic_pointer_cast<const core::CallTypedExpr>(
        inferTypes(untypedExpr, source));
    exprs.emplace_back(expr);

    if (untypedExpr->alias().has_value()) {
      names.push_back(untypedExpr->alias().value());
    } else {
      names.push_back(fmt::format("a{}", i));
    }
  }

  return {exprs, names};
}

std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> createAggregateMasks(
    size_t numAggregates,
    const std::vector<std::string>& masks,
    const RowTypePtr& inputType) {
  std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>> maskExprs(
      numAggregates);
  if (masks.empty()) {
    return maskExprs;
  }

  VELOX_CHECK_EQ(numAggregates, masks.size());
  for (auto i = 0; i < numAggregates; ++i) {
    if (!masks[i].empty()) {
      maskExprs[i] = field(inputType, masks[i]);
    }
  }

  return maskExprs;
}

core::PlanNodePtr& aggregation_O(
    const std::vector<std::string>& groupingKeys,
    const std::vector<std::string>& preGroupedKeys,
    const std::vector<std::string>& aggregates,
    const std::vector<std::string>& masks,
    core::AggregationNode::Step step,
    bool ignoreNullKeys,
    core::PlanNodePtr source,
    const std::vector<TypePtr>& resultTypes) {
  auto numAggregates = aggregates.size();
  auto aggregatesAndNames =
      createAggregateExpressionsAndNames(aggregates, step, resultTypes, source);
  planNode_ = std::make_shared<core::AggregationNode>(
      nextPlanNodeId(),
      step,
      fields(source->outputType(), groupingKeys),
      fields(source->outputType(), preGroupedKeys),
      aggregatesAndNames.names,
      aggregatesAndNames.expressions,
      createAggregateMasks(numAggregates, masks, source->outputType()),
      ignoreNullKeys,
      source);
  return planNode_;
}

core::PlanNodePtr& values_O(
    const std::vector<RowVectorPtr>& values,
    bool parallelizable,
    size_t repeatTimes) {
  auto valuesCopy = values;
  planNode_ = std::make_shared<core::ValuesNode>(
      nextPlanNodeId(), std::move(valuesCopy), parallelizable, repeatTimes);
  return planNode_;
}

core::PlanNodePtr& filter_O(const std::string& filter, core::PlanNodePtr source) {
  RowTypePtr outputType = source->outputType();
  planNode_ = std::make_shared<core::FilterNode>(
      nextPlanNodeId(),
      parseExpr(filter, outputType, options_, pool_),
      source);
  return planNode_;
}

};




auto pool_ = memory::addDefaultLeafMemoryPool();
std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<core::QueryCtx> newQueryCtx(
      int64_t memoryCapacity) {
    
    std::unordered_map<std::string, std::shared_ptr<Config>> configs;
    std::shared_ptr<MemoryPool> pool = memory::defaultMemoryManager().addRootPool(
        "", memoryCapacity, MemoryReclaimer::create());
   std::unordered_map<std::string, std::string> myMapWithValues = {{core::QueryConfig::kSpillEnabled, "true"}, 
                                      {core::QueryConfig::kJoinSpillEnabled, "true"},  
                                      {core::QueryConfig::kJoinSpillMemoryThreshold, "1"},
                                       {core::QueryConfig::kSpillableReservationGrowthPct, "1"},
                                       {core::QueryConfig::kSpillPartitionBits, "1"}
                                      };
    auto queryCtx = std::make_shared<core::QueryCtx>(
        executor_.get(),
        myMapWithValues,
        configs,
        memory::MemoryAllocator::getInstance(),
        std::move(pool));
    return queryCtx;
  }

core::PlanNodePtr& Optest(core::PlanNodePtr source, RowVectorPtr data, FlatVectorPtr<float> weights, 
int output_size, int size, std::vector<std::string> str){
  int input_size = size / output_size;
   exec::registerVectorFunction(
    "adapter",
    ADP::signatures(),
    std::make_unique<ADP>()
  );

    exec::registerVectorFunction(
    "mat_mul_s",
    MatrixMultiply_s::signatures(),
    std::make_unique<MatrixMultiply_s>(2, 5)
  );

  VectorMaker maker{pool_.get()};

  optimizerBuilder opBuilder;
  std::string x_var = "x"; 
  std::string w_var = "w"; 
  auto plan_1 = opBuilder.values_O({data}, false, 1);

  auto plan_2 = opBuilder.project_O({fmt::format("{}_col", x_var), fmt::format("{}_row", x_var),fmt::format("{}", x_var)}, plan_1);

  auto results_2 = exec::test::AssertQueryBuilder(plan_2).copyResults(pool_.get());
  std::cout << "2 Results:" << results_2->toString() << std::endl;
  std::cout << results_2->toString(0, results_2->size()) << std::endl;


  std::vector<int> weighsrowVector;
  std::vector<int> weighscolVector;
  for (int i=0; i < size; i++) {
    int rowIndex = i / output_size;
    int colIndex = i % output_size;
    weighsrowVector.push_back(rowIndex);
    weighscolVector.push_back(colIndex);
  }

std::vector<std::vector<float>> weightsArray;

for (int i = 0; i < input_size; i++) {
    std::vector<float> weightsArrayrow;
    for (int j = 0; j < output_size; j++) {
        int index = i * output_size + j;
        if (index < size) {
            weightsArrayrow.push_back(weights->valueAt(index));
        }
    }
    weightsArray.push_back(weightsArrayrow);
}
  auto weightsArrayVector = maker.arrayVector<float>(weightsArray, REAL());

std::vector<std::vector<float>> weightsArray_3;

for (int i = 0; i < output_size; i++) {
    std::vector<float> weightsArrayrow_3;
    for (int j = 0; j < input_size; j++) {
        int index = i + output_size * j;
        if (index < size) {
            weightsArrayrow_3.push_back(weights->valueAt(index));
        }
    }
    weightsArray_3.push_back(weightsArrayrow_3);
}
  auto weightsArrayVector_3 = maker.arrayVector<float>(weightsArray_3, REAL());

  std::vector<float> w0 = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
  std::vector<float> w1 = {100, 110, 120, 130, 140, 150, 160, 170, 180, 190};
  std::vector<float> w2 = {200, 210, 220, 230, 240, 250, 260, 270, 280, 290};
  std::vector<float> w3 = {300, 310, 320, 330, 340, 350, 360, 370, 380, 390};
  std::vector<float> w4 = {400, 410, 420, 430, 440, 450, 460, 470, 480, 490};

  std::vector<std::vector<float>> weightsArray_4;
  weightsArray_4.push_back(w0);
  weightsArray_4.push_back(w1);
  weightsArray_4.push_back(w2);
  weightsArray_4.push_back(w3);
  weightsArray_4.push_back(w4);

auto weightsArrayVector_4 = maker.arrayVector<float>(weightsArray_4, REAL());

  std::vector<std::vector<int>> weighsrowVectors;
  for (int i = 0; i < input_size; i++) {
    std::vector<int> weighsrow;
    for (int j = 0; j < output_size; j++) {
        weighsrow.push_back(i);
    }
    weighsrowVectors.push_back(weighsrow);
}
  auto weighsrowArrayVector = maker.arrayVector<int32_t>(weighsrowVectors, INTEGER());
  std::vector<std::vector<int>> weighscolVectors;
  for (int i = 0; i < input_size; i++) {
    std::vector<int> weighscol;
    for (int j = 0; j < output_size; j++) {
        weighscol.push_back(j);
    }
    weighscolVectors.push_back(weighscol);
}
  auto weighscolArrayVector = maker.arrayVector<int32_t>(weighscolVectors, INTEGER()); 
  // auto weighsrowVectors = maker.arrayVector<float>(weighsrowVector, REAL());
  // auto weighscolVectors = maker.arrayVector<float>(weighscolVector, REAL());
  auto w_row_3 = maker.flatVector({0, 0, 0, 0, 0});
  auto w_col_3 = maker.flatVector({0, 1, 2, 3, 4});

  auto w_col_4 = maker.flatVector({0, 0, 0, 0, 0});
  auto w_row_4 = maker.flatVector({0, 1, 2, 3, 4});


  auto weighsrowVectors_f = maker.flatVector(weighsrowVector);
  auto weighscolVectors_f = maker.flatVector(weighscolVector);
  auto inputRowVector_dense_weighs = maker.rowVector({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, {weights, weighsrowVectors_f, weighscolVectors_f});
  auto inputRowVector_dense_weighs_a = maker.rowVector({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, {weightsArrayVector, weighsrowArrayVector, weighscolArrayVector});
  auto inputRowVector_dense_weighs_3 = maker.rowVector({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, {weightsArrayVector_3, w_row_3, w_col_3});
  auto inputRowVector_dense_weighs_4 = maker.rowVector({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, {weightsArrayVector_4, w_row_4, w_col_4});

  auto plan_3 = opBuilder.values_O({inputRowVector_dense_weighs_4}, false, 1);
  auto plan_4 = opBuilder.project_O({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, plan_3);

  auto results_4 = exec::test::AssertQueryBuilder(plan_4).copyResults(pool_.get());
  std::cout << "4 Results:" << results_4->toString() << std::endl;
  std::cout << results_4->toString(0, results_4->size()) << std::endl;


  auto plan_5 = opBuilder.hashjoin_O({fmt::format("{}_col", x_var)}, {fmt::format("{}_row", w_var)}, plan_4, "", {fmt::format("{}_col", x_var), fmt::format("{}_row", x_var),fmt::format("{}_col", w_var), 
fmt::format("{}", x_var), fmt::format("{}", w_var)}, plan_2, core::JoinType::kInner, false);
  auto plan_61 = opBuilder.project_O({fmt::format("{}_row", x_var), fmt::format("{}_col", w_var), fmt::format("{}", x_var), fmt::format("{}", w_var)}, plan_5);
  
  auto results_5 = exec::test::AssertQueryBuilder(plan_61).copyResults(pool_.get());
  std::cout << "5 Results:" << results_5->toString() << std::endl;
  std::cout << results_5->toString(0, results_5->size()) << std::endl;

  auto plan_6 = opBuilder.project_O({fmt::format("{}_row", x_var), fmt::format("{}_col", w_var), "mat_mul_s(x, w) AS mp"}, plan_61);
  
  auto results_6 = exec::test::AssertQueryBuilder(plan_6).copyResults(pool_.get());
  std::cout << "6 Results:" << results_6->toString() << std::endl;
  std::cout << results_6->toString(0, results_6->size()) << std::endl;
  
  auto plan_7 = opBuilder.aggregation_O({fmt::format("{}_col", w_var),fmt::format("{}_row", x_var)}, {}, {"array_sum(mp) AS result"}, 
{}, core::AggregationNode::Step::kSingle, false, plan_6, {});
  
  auto results_7 = exec::test::AssertQueryBuilder(plan_7).copyResults(pool_.get());
  std::cout << "7 Results:" << results_7->toString() << std::endl;
  std::cout << results_7->toString(0, results_7->size()) << std::endl;

  auto plan_8 = opBuilder.project_O({fmt::format("adapter({}_col, {}_row, result) AS res", w_var, x_var)}, plan_7);
  // auto plan_9 = opBuilder.filter_O({"res IS NOT NULL"}, plan_8);
  // auto plan_10 = opBuilder.project_O({"res"}, plan_9);

  auto results_8 = exec::test::AssertQueryBuilder(plan_8).copyResults(pool_.get());
  std::cout << "8 Results:" << results_8->toString() << std::endl;
  std::cout << results_8->toString(0, results_8->size()) << std::endl;

  std::string searchString = fmt::format("mat_mul({})", x_var);
  std::string replaceString = "res";

  std::size_t found = str[0].find(searchString);
  while (found != std::string::npos) {
      str[0].replace(found, searchString.length(), replaceString);
      found = str[0].find(searchString, found + replaceString.length());
  }
  
  static core::PlanNodePtr plan_9 = opBuilder.project_O({str[0]}, plan_8); // TODO: get the rest expression
  

  return plan_9;
};

const std::shared_ptr<exec::VectorFunction> findName(const std::shared_ptr<exec::Expr>& expr) {
    if (expr->name() == "mat_mul0") {
        return expr->vectorFunction();
    }

    for (const auto& input : expr->inputs()) {
        const auto result = findName(input);
        if (result) {
            return result;
        }
    }

    return nullptr;
}

void test_optimizer(int argc, char** argv) {
  folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();

   VectorMaker maker{pool_.get()};
  int output_size = 5;
  int input_size = 10;
  int num_features = 3;
  int size = output_size*input_size;
  
  auto weights = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 

  auto bias = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  bias->set(i, i % output_size);
  } 
  
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_features; i++){
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());

  std::vector<int> rowVector;
  std::vector<int> colVector;
  for (int i=0; i < num_features*input_size; i++) {
    int rowIndex = i / input_size;
    int colIndex = i % input_size;
    rowVector.push_back(rowIndex);
    colVector.push_back(colIndex);
  }
  std::vector<std::vector<int>> rowVectors;
  for (int i = 0; i < num_features; i++) {
    std::vector<int> row;
    for (int j = 0; j < input_size; j++) {
        row.push_back(i);
    }
    rowVectors.push_back(row);
}
  auto rowArrayVector = maker.arrayVector<int32_t>(rowVectors, INTEGER());
  std::vector<std::vector<int>> colVectors;
  for (int i = 0; i < num_features; i++) {
    std::vector<int> col;
    for (int j = 0; j < input_size; j++) {
        col.push_back(j);
    }
    colVectors.push_back(col);
}
  auto colArrayVector = maker.arrayVector<int32_t>(colVectors, INTEGER()); 

  auto rowVectors_f = maker.flatVector(rowVector);
  auto colVectors_f = maker.flatVector(colVector);
  // auto rowVectors = maker.arrayVector<int_32_t>({rowVector}, INTEGER());
  // auto colVectors = maker.arrayVector<int_32_t>({colVector}, INTEGER());
  auto x_row_3 = maker.flatVector({0, 1, 2});
  auto x_col_3= maker.flatVector({0, 0, 0});

  auto x_row_4 = maker.flatVector({0, 0, 0, 0, 0});
  auto x_col_4= maker.flatVector({0, 1, 2, 3, 4});


  std::vector<float> v0 = {1, 0, 0, 1, 0, 2};
  std::vector<float> v1 = {1, 0, 2, 3, 4, 6};
  std::vector<float> v2 = {1, 0, 4, 5, 8, 10};
  std::vector<float> v3 = {1, 0, 6, 7, 12, 14};
  std::vector<float> v4 = {1, 0, 8, 9, 16, 18};

  std::vector<std::vector<float>> featureArrayVector_4;
  featureArrayVector_4.push_back(v0);
  featureArrayVector_4.push_back(v1);
  featureArrayVector_4.push_back(v2);
  featureArrayVector_4.push_back(v3);
  featureArrayVector_4.push_back(v4);


auto featureArrayVectors_4 = maker.arrayVector<float>(featureArrayVector_4, REAL());


  auto inputRowVector_dense = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVector, rowVectors_f, colVectors_f});
  auto inputRowVector_dense_a = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVector, rowArrayVector, colArrayVector});
  auto inputRowVector_dense_a_3 = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVector, x_row_3, x_col_3});
  auto inputRowVector_dense_a_4 = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVectors_4, x_row_4, x_col_4});

  std::vector<float> flattenedVector;
  for (const auto& featureVector : featureVectors) {
      flattenedVector.insert(flattenedVector.end(), featureVector.begin(), featureVector.end());
  }

  auto featureArrayVectorFlat = maker.flatVector(flattenedVector);
  auto inputRowVector_dense_flat = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVectorFlat, rowVectors_f, colVectors_f});


  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  exec::registerVectorFunction(
    "mat_add",
    MatrixAddition::signatures(),
    std::make_unique<MatrixAddition>(bias->values()->asMutable<float>(), output_size)
  );  

  exec::registerVectorFunction(
    "relu",
    Relu::signatures(),
    std::make_unique<Relu>()
  );

  auto planbuilder = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector_dense_a_3})
                  .project({"relu(mat_add(mat_mul(x)))"})
		              .planBuild();

  auto myDensePlan = planbuilder.planNode();

  auto results_dense = exec::test::AssertQueryBuilder(myDensePlan).copyResults(pool_.get());
  std::cout << "Dense Results:" << results_dense->toString() << std::endl;
  std::cout << results_dense->toString(0, results_dense->size()) << std::endl;

  // auto newplan = Optest(myDensePlan, inputRowVector_dense_flat, output_size, size, weights);
  
  auto nodeid = myDensePlan->id();
  auto str = planbuilder.findExprStrings(nodeid);

  auto myDensePlanF = planbuilder.planFragment();
  core::PlanNodePtr newplan = nullptr;
  Optimizer op(queryCtx_);
  auto ops = op.traverse(myDensePlanF);

  for (const auto& op : ops) {
  // TODO: Add more logic to determine if the operator should be insert to candidates.
  // Here only check the type of an operator.
  auto fp = dynamic_cast<exec::FilterProject*>(op);
  if (fp) {
    std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
    std::cout << "The operatorType is: " << op->operatorType() << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "The expression tree: " << std::endl;
    const std::unique_ptr<exec::ExprSet>& exprs = fp->getExprs();

    std::shared_ptr<exec::VectorFunction> result = nullptr;
    for (const auto& expr : exprs->exprs()) {
        result = findName(expr);
        if (result) {
            break;
        }
    }

    if (result) {
        auto call = std::dynamic_pointer_cast<MatrixMultiply>(result);
        newplan = Optest(myDensePlan, inputRowVector_dense_a_4, weights, 
        call->getDims()[1], call->getDims()[0]*call->getDims()[1], str);
    } else {
        std::cout << "No layer with the desired name found." << std::endl;
    }

    // auto call = std::dynamic_pointer_cast<MLFunction>(exprs->exprs()[0]->vectorFunction());//inputs_ contains mat_add
    
    std::cout << exprs->toString(false /*compact*/) << std::endl;

    // Use exprs as needed.
  } else {
    std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
    std::cout << "The operatorType is: " << op->operatorType() << std::endl;
    std::cout << "\n" << std::endl;
  }
}





  auto results_dense_rep = exec::test::AssertQueryBuilder(newplan).copyResults(pool_.get());
  std::cout << "Rep Dense Results:" << results_dense_rep->toString() << std::endl;
  std::cout << results_dense_rep->toString(0, results_dense_rep->size()) << std::endl;
}

FlatVectorPtr<float> get_tensor(VectorMaker& m, std::ifstream& file, int size, int lines){
  std::cout << "Loading tensor of size " << size << std::endl;
  FlatVectorPtr<float> tensor = m.flatVector<float>(size);
  int index = 0;
  std::string line;
  while (lines--) { // Read a line from the file
      std::getline(file, line);
      std::istringstream iss(line); // Create an input string stream from the line
    
      std::string numberStr;
      while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
          float number = std::stof(numberStr);    // Convert the string to float
          tensor->set(index++, number);
      }
  }
  return tensor;
}
//core::PlanNodePtr source,
core::PlanNodePtr& getNewPlan(RowVectorPtr values, RowVectorPtr weights, std::vector<std::string> str){
  optimizerBuilder opBuilder;
     exec::registerVectorFunction(
    "mat_mul_s",
    MatrixMultiply_s::signatures(),
    std::make_unique<MatrixMultiply_s>(196, 1024)
  );

  auto plan_a = opBuilder.values_O({values}, false, 1);
  auto plan_a1 = opBuilder.project_O({"v", "v_row", "v_col"}, plan_a);


  auto results_a1 = exec::test::AssertQueryBuilder(plan_a1).copyResults(pool_.get());
  std::cout << "a1 values Results:" << results_a1->toString() << std::endl;
  std::cout << results_a1->toString(0, results_a1->size()) << std::endl;

  auto plan_b = opBuilder.values_O({weights}, false, 1);
  auto plan_b1 = opBuilder.project_O({"w", "w_row", "w_col"}, plan_b);

  auto results_b1 = exec::test::AssertQueryBuilder(plan_b1).copyResults(pool_.get());
  std::cout << "b1 weights Results:" << results_b1->toString() << std::endl;
  std::cout << results_b1->toString(0, results_b1->size()) << std::endl;

  auto plan_c = opBuilder.hashjoin_O({"v_col"}, {"w_row"}, plan_b1, "", {"v_row", "w_col", "v", "w"}, plan_a1, core::JoinType::kInner, false);
  
  auto results_c = exec::test::AssertQueryBuilder(plan_c).copyResults(pool_.get());
  std::cout << "hashjoin Results:" << results_c->toString() << std::endl;
  std::cout << results_c->toString(0, results_c->size()) << std::endl;

  auto plan_d = opBuilder.project_O({"v_row", "w_col", "mat_mul_s(v, w) AS mp"}, plan_c);
  
  auto results_d = exec::test::AssertQueryBuilder(plan_d).copyResults(pool_.get());
  std::cout << "map Results:" << results_d->toString() << std::endl;
  std::cout << results_d->toString(0, results_d->size()) << std::endl;
  
  auto plan_e = opBuilder.aggregation_O({"w_col","v_row"}, {}, {"array_sum(mp) AS result"}, 
{}, core::AggregationNode::Step::kSingle, false, plan_d, {});
  
  auto results_e = exec::test::AssertQueryBuilder(plan_e).copyResults(pool_.get());
  std::cout << "aggregate Results:" << results_e->toString() << std::endl;
  std::cout << results_e->toString(0, results_e->size()) << std::endl;

  // auto plan_8 = opBuilder.project_O({fmt::format("adapter({}_col, {}_row, result) AS res", w_var, x_var)}, plan_7);
  // auto plan_9 = opBuilder.filter_O({"res IS NOT NULL"}, plan_8);
  // auto plan_10 = opBuilder.project_O({"res"}, plan_9);

  // auto results_8 = exec::test::AssertQueryBuilder(plan_8).copyResults(pool_.get());
  // std::cout << "8 Results:" << results_8->toString() << std::endl;
  // std::cout << results_8->toString(0, results_8->size()) << std::endl;

  std::string searchString = "mat_mul0(x)";
  std::string replaceString = "result";

  std::size_t found = str[0].find(searchString);
  while (found != std::string::npos) {
      str[0].replace(found, searchString.length(), replaceString);
      found = str[0].find(searchString, found + replaceString.length());
  }
  
  auto plan_f = opBuilder.project_O({str[0]}, plan_e); // TODO: get the rest expression

  auto results_f = exec::test::AssertQueryBuilder(plan_f).copyResults(pool_.get());
  std::cout << "f Results:" << results_f->toString() << std::endl;
  std::cout << results_f->toString(0, results_f->size()) << std::endl;
  

  return plan_f;
}

RowVectorPtr createBlock_w(int input_size, int output_size, FlatVectorPtr<float> weights){
  int size = input_size*output_size;
  VectorMaker maker{pool_.get()};

  auto w_col = maker.flatVector({0, 0, 0, 0});//split to 4 parts
  auto w_row = maker.flatVector({0, 1, 2, 3});
  int weight_block_size = size / 4;
  std::vector<std::vector<float>> weightsArray;

  for (int i = 0; i < 4; i++) {
      std::vector<float> weightsArraySingle;
      for (int j = 0; j < weight_block_size; j++) {
          int index = i * weight_block_size + j;
          if (index < size) {
              weightsArraySingle.push_back(weights->valueAt(index));
          }
      }
      weightsArray.push_back(weightsArraySingle);
  }
  auto weightsArrayVector = maker.arrayVector<float>(weightsArray, REAL());

    // FlatVectorPtr<float> w_index = maker.flatVector<float>(784);
    // for(int i=0; i < 784; i++)
    //   w_index->set(i, i*1.0);

  
  return maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector, w_row, w_col});
}

RowVectorPtr createBlock_w_oom(int input_size, int output_size, FlatVectorPtr<float> weights, VectorMaker& maker){
  int size = input_size*output_size;

  auto w_col = maker.flatVector({0, 0, 0, 0});//split to 4 parts
  auto w_row = maker.flatVector({0, 1, 2, 3});
  int weight_block_size = size / 4;
  std::vector<std::vector<float>> weightsArray;

  for (int i = 0; i < 4; i++) {
      std::vector<float> weightsArraySingle;
      for (int j = 0; j < weight_block_size; j++) {
          int index = i * weight_block_size + j;
          if (index < size) {
              weightsArraySingle.push_back(weights->valueAt(index));
          }
      }
      weightsArray.push_back(weightsArraySingle);
  }
  auto weightsArrayVector = maker.arrayVector<float>(weightsArray, REAL());

    // FlatVectorPtr<float> w_index = maker.flatVector<float>(784);
    // for(int i=0; i < 784; i++)
    //   w_index->set(i, i*1.0);

  
  return maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector, w_row, w_col});
}

RowVectorPtr createBlock_v(int input_size, int output_size, FlatVectorPtr<float> values){
  int size = input_size*output_size;
  VectorMaker maker{pool_.get()};

  auto v_row = maker.flatVector({0, 0, 0, 0});//split to 4 parts
  auto v_col = maker.flatVector({0, 1, 2, 3});
  int values_block_size = size / 4;
  std::vector<std::vector<float>> valuesArray;

  for (int i = 0; i < 4; i++) {
      std::vector<float> valuesArraySingle;
      for (int j = 0; j < values_block_size; j++) {
          int index = i * values_block_size + j;
          if (index < size) {
              valuesArraySingle.push_back(values->valueAt(index));
          }
      }
      valuesArray.push_back(valuesArraySingle);
  }
  auto valuesArrayVector = maker.arrayVector<float>(valuesArray, REAL());

  // FlatVectorPtr<float> v_index = maker.flatVector<float>(1000);
  //   for(int i=0; i < 1000; i++)
  //     v_index->set(i, i*1.0);

  
  return maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector, v_row, v_col});
}

std::vector<RowVectorPtr> createBlock_v_oom(int input_size, int output_size, FlatVectorPtr<float> values, VectorMaker& maker){
  int size = input_size*output_size;


  auto v_row = maker.flatVector({0, 0, 0, 0});//split to 4 parts
  auto v_col = maker.flatVector({0, 1, 2, 3});
  int values_block_size = size / 4;
  std::vector<std::vector<float>> valuesArray;
  std::vector<RowVectorPtr> ve;
  for (int i = 0; i < 4; i++) {
      std::vector<float> valuesArraySingle;
      for (int j = 0; j < values_block_size; j++) {
          int index = i * values_block_size + j;
          if (index < size) {
              valuesArraySingle.push_back(values->valueAt(index));
          }
      }
      valuesArray.push_back(valuesArraySingle);
  }
  for (auto singleblock : valuesArray){
      auto valuevector = maker.flatVector<float>(singleblock, REAL());
      
      ve.push_back(maker.rowVector({"v", "v_row", "v_col"}, {valuevector, v_row, v_col}));
  }
  
  auto valuesArrayVector = maker.arrayVector<float>(valuesArray, REAL());

  // FlatVectorPtr<float> v_index = maker.flatVector<float>(1000);
  //   for(int i=0; i < 1000; i++)
  //     v_index->set(i, i*1.0);

  
  return ve;
}


core::PlanNodePtr optimizing(PlanBuilder& planbuilder, std::shared_ptr<core::QueryCtx> queryCtx_, FlatVectorPtr<float> weights, FlatVectorPtr<float> input){
  auto nodeid = planbuilder.planNode()->id();
  auto str = planbuilder.findExprStrings(nodeid);

  auto myOptimizdPlanF = planbuilder.planFragment();
  core::PlanNodePtr newplan = nullptr;
  Optimizer op(queryCtx_);
  auto ops = op.traverse(myOptimizdPlanF);

  for (const auto& op : ops) {
  // TODO: Add more logic to determine if the operator should be insert to candidates.
  // Here only check the type of an operator.
  auto fp = dynamic_cast<exec::FilterProject*>(op);
  if (fp) {
    std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
    std::cout << "The operatorType is: " << op->operatorType() << std::endl;
    std::cout << "\n" << std::endl;
    std::cout << "The expression tree: " << std::endl;
    const std::unique_ptr<exec::ExprSet>& exprs = fp->getExprs();

    std::shared_ptr<exec::VectorFunction> result = nullptr;
    for (const auto& expr : exprs->exprs()) {
        result = findName(expr);
        if (result) {
            break;
        }
    }

    if (result) {
        auto call = std::dynamic_pointer_cast<MatrixMultiply>(result);
        auto w_block = createBlock_w(call->getDims()[0], call->getDims()[1], weights);
        auto v_block = createBlock_v(1000, 784, input);
        newplan = getNewPlan(v_block, w_block, str);
    } else {
        std::cout << "No layer with the desired name found." << std::endl;
    }

    // auto call = std::dynamic_pointer_cast<MLFunction>(exprs->exprs()[0]->vectorFunction());//inputs_ contains mat_add
    
    std::cout << exprs->toString(false /*compact*/) << std::endl;

    // Use exprs as needed.
  } else {
    std::cout << "The planNodeId is: " << op->planNodeId() << std::endl;
    std::cout << "The operatorType is: " << op->operatorType() << std::endl;
    std::cout << "\n" << std::endl;
  }
  
}

return newplan;
}


void test_mnist_optimizer(int argc, char** argv, int flag){
  
  folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();
   VectorMaker maker{pool_.get()};
    int input_size = 784; // num_features
    int layer1_size = 1024; // num units in hidden layer 1
    int layer2_size = 10;
    int num_samples = 0;
    if (flag == 1){
      num_samples = 1000;}
    else{
      num_samples = 2000;}

   
    std::ifstream weights_file("/home/local/ASUAD/qlin36/w1024.txt"); 
    std::ifstream bias_file("/home/local/ASUAD/qlin36/b1024.txt"); 
    std::ifstream test_file("/home/local/ASUAD/qlin36/x_test_large.txt"); 


    FlatVectorPtr<float> weights_1 = get_tensor(maker, weights_file, layer1_size * input_size, input_size);
    FlatVectorPtr<float> bias_1 = get_tensor(maker, bias_file, layer1_size, 1);
    FlatVectorPtr<float> weights_2 = get_tensor(maker, weights_file, layer2_size * layer1_size, layer1_size);
    FlatVectorPtr<float> bias_2 = get_tensor(maker, bias_file, layer2_size, 1);
    weights_file.close();
    bias_file.close();

    float* bias_1_values = bias_1->values()->asMutable<float>();
    float* bias_2_values = bias_2->values()->asMutable<float>();

    FlatVectorPtr<float> bias_1_mat = maker.flatVector<float>(num_samples * layer1_size);
    for(int i=0; i < bias_1_mat->size(); i++)
      bias_1_mat->set(i, bias_1_values[i%layer1_size]);
    
    FlatVectorPtr<float> bias_2_mat = maker.flatVector<float>(num_samples * layer2_size);
    for(int i=0; i < bias_2_mat->size(); i++)
      bias_2_mat->set(i, bias_2_values[i%layer2_size]);

    FlatVectorPtr<float> input = get_tensor(maker, test_file, input_size * num_samples, num_samples);
    float* data = input->values()->asMutable<float>();

    std::vector<std::vector<float>> featureVectors;
    for(int i=0, cursor = 0; i < num_samples; i++, cursor += input_size){
      std::vector<float> featureVector(data + cursor, data + cursor + input_size);
      featureVectors.push_back(featureVector);
    }

    auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());

    auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

    std::string compute =  NNBuilder()
                          .denseLayer(layer1_size ,input_size, weights_1->values()->asMutable<float>(), 
                            bias_1_mat->values()->asMutable<float>(), NNBuilder::RELU)
                          .denseLayer(layer2_size ,layer1_size, weights_2->values()->asMutable<float>(), 
                            bias_2_mat->values()->asMutable<float>(), NNBuilder::SOFTMAX)
                          .build();

  
    std::cout << compute << std::endl; // softmax5(mat_add4(mat_mul3(relu2(mat_add1(mat_mul0({}))))))

    auto planbuilder = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({fmt::format(compute, "x")}) 
		              .planBuild();

  auto myDensePlan = planbuilder.planNode();
  auto results_dense = exec::test::AssertQueryBuilder(myDensePlan).copyResults(pool_.get());
  std::cout << "Dense Results:" << results_dense->toString() << std::endl;
  std::cout << results_dense->toString(0, results_dense->size()) << std::endl;

  auto myOptimizedPlan = optimizing(planbuilder, queryCtx_, weights_1, input);
  auto results_Optimized = exec::test::AssertQueryBuilder(myOptimizedPlan).copyResults(pool_.get());
  std::cout << "Optimized Results:" << results_Optimized->toString() << std::endl;
  std::cout << results_Optimized->toString(0, results_Optimized->size()) << std::endl;

}
static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {

  while (!task->isFinished()) {     
    usleep(1000); // 0.01 second.
  }
}

// void writeToFile(
//     const std::string& filePath,
//     const std::vector<RowVectorPtr>& vectors,
//     std::shared_ptr<dwrf::Config> config) {
//   facebook::velox::dwrf::WriterOptions options;
//   options.config = config;
//   options.schema = vectors[0]->type();
//   auto sink =
//       std::make_unique<facebook::velox::dwio::common::LocalFileSink>(filePath);
//   auto childPool = rootPool_->addAggregateChild("HiveConnectorTestBase.Writer");
//   facebook::velox::dwrf::Writer writer{options, std::move(sink), *childPool};
//   for (size_t i = 0; i < vectors.size(); ++i) {
//     writer.write(vectors[i]);
//   }
//   writer.close();
// }
class MyFileTest : public HiveConnectorTestBase {
  public:
  MyFileTest(){
    SetUp();
  }
  ~MyFileTest() {
  }

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TestBody() override {}

};

core::PlanFragment test_oom_optimizer(PlanBuilder& planbuilder, std::shared_ptr<memory::MemoryPool> pool, core::PlanNodeId& id){
  auto mat_mul = std::dynamic_pointer_cast<MatrixMultiply>(exec::getVectorFunction("mat_mul", {ARRAY(REAL())}, {}));
  std::ifstream weights_file(mat_mul->getWeightsFile()); 
  VectorMaker m{pool.get()};
  FlatVectorPtr<float> weight = get_tensor(m, weights_file, 500000, 1000);
  auto weights = createBlock_w_oom(1000, 500, weight, m);

  std::ifstream test_file("/home/local/ASUAD/qlin36/x_test_large.txt"); 
  FlatVectorPtr<float> inputs = get_tensor(m, test_file, 6000000, 6000);
  auto input = createBlock_v_oom(6000, 1000, inputs, m);

  optimizerBuilder opBuilder;
     exec::registerVectorFunction(
    "mat_mul_s",
    MatrixMultiply_s::signatures(),
    std::make_unique<MatrixMultiply_s>(250, 500)
  );

  auto plan_a = opBuilder.values_O({input}, false, 1);
  opBuilder.capturePlanNodeId(id);

  auto plan_a1 = opBuilder.project_O({"v", "v_row", "v_col"}, plan_a);

  auto plan_b = opBuilder.values_O({weights}, false, 1);
  auto plan_b1 = opBuilder.project_O({"w", "w_row", "w_col"}, plan_b);

  auto plan_c = opBuilder.hashjoin_O({"v_col"}, {"w_row"}, plan_b1, "", {"v_row", "w_col", "v", "w"}, plan_a1, core::JoinType::kInner, false);


  auto plan_d = opBuilder.project_O({"v_row", "w_col", "mat_mul_s(v, w) AS mp"}, plan_c);
  
  
  auto plan_e = opBuilder.aggregation_O({"w_col","v_row"}, {}, {"array_sum(mp) AS result"}, 
{}, core::AggregationNode::Step::kSingle, false, plan_d, {});

  auto planf = opBuilder.planFragment(plan_e);

  auto file = TempFilePath::create();
  MyFileTest myfile;
  myfile.writeToFile(file->path, {input});
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  int num_splits = 20;
  for(int i=0; i < num_splits; i++)
    paths.push_back(file);
  auto hiveSplits = myfile.makeHiveConnectorSplits(paths);
 
  int concurrency = 48;
  boost::interprocess::interprocess_semaphore semaphore(concurrency);

  auto task = exec::Task::create("1", planf, 0, queryCtx_, 
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            semaphore.post();
          }
          return exec::BlockingReason::kNotBlocked;
  });

  // Create 2 hive splits and add them to task
  task->start(task, concurrency);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(id, exec::Split(std::move(split)));
  }
  task->noMoreSplits(id);
  std::cout << std::endl;
  // Start task with 2 as maximum drivers and wait for execution to finish
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  return planf;
}

void test_mnist_oom_error(int argc, char** argv){
  
  folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();
   std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 38 * MB)};
  //  auto childPool = rootPool->addLeafChild("leaf");
   VectorMaker maker{pool_.get()};
  // VectorMaker maker{childPool.get()};

  int input_size = 1000;
  int output_size = 500;
  int num_samples = 6000;
  // ( 6000 * 1000 x 1000 * 500 )
  int size = output_size * input_size;
  
  auto weights = maker.flatVector<float>(size);

  for(int i=0; i < size; i++){
    weights->set(i, i*2);
  } 
  // register Vector Function
  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  // Create input
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_samples; i++){ 
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
  
  // Create Plan
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p0;
  auto planOOM = exec::test::PlanBuilder(planNodeIdGenerator)
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
		              .project({"mat_mul(x)"})
                  .planBuild();

  auto plan0 = planOOM.planFragment();
  // queryCtx_->testingOverrideConfigUnsafe(
  //     {{core::QueryConfig::kPreferredOutputBatchRows, "400"}, {core::QueryConfig::kPreferredOutputBatchBytes, "2000000"},  {core::QueryConfig::kMaxOutputBatchRows, "300"}});
  // Create task
  
  // std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 39 * MB)};
  // auto childPool = rootPool->addLeafChild("leaf");
  queryCtx_->testingOverrideMemoryPool(rootPool);
  
  queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});
  
  auto file = TempFilePath::create();
  MyFileTest myfile;
  myfile.writeToFile(file->path, {inputRowVector});
  
  std::vector<std::shared_ptr<TempFilePath>> paths;
  int num_splits = 1;
  for(int i=0; i < num_splits; i++)
    paths.push_back(file);
  auto hiveSplits = myfile.makeHiveConnectorSplits(paths);
  // auto hiveSplits =  myfile.makeHiveConnectorSplits(file->path, 4, dwio::common::FileFormat::DWRF);
 
  int concurrency = 8;
  boost::interprocess::interprocess_semaphore semaphore(concurrency);

  auto task = exec::Task::create("0", plan0, 0, queryCtx_, 
        [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
          if(result){
            semaphore.post();
          }
          return exec::BlockingReason::kNotBlocked;
  });

  // Create 2 hive splits and add them to task
  task->start(task, concurrency);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p0, exec::Split(std::move(split)));
  }
  task->noMoreSplits(p0);
  std::cout << std::endl;
  // Start task with 2 as maximum drivers and wait for execution to finish
 
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
}

void test_oom_success(int argc, char** argv){
  
  folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();
   VectorMaker maker{pool_.get()};

  int input_size = 1000;
  int output_size = 500;
  int num_samples = 6000;
  // ( 6000 * 1000 x 1000 * 500 )
  int size = output_size * input_size;
  
  // auto weights = maker.flatVector<float>(size);

  // for(int i=0; i < size; i++){
  //   weights->set(i, i*2);
  // } 
  // // register Vector Function
  // exec::registerVectorFunction(
  //   "mat_mul",
  //   MatrixMultiply::signatures(),
  //   std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  // );

  // // Create input
  // std::vector<std::vector<float>> featureVectors;
  // for(int i=0; i < num_samples; i++){ 
  //   std::vector<float> featureVector;
  //   for(int j=0; j < input_size; j++){
  //     featureVector.push_back(i*j);
  //   }
  //   featureVectors.push_back(featureVector);
  // }
  
  // auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  // auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});
  
  // // Create Plan
  // auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  // core::PlanNodeId p0;
  // auto planOOM = exec::test::PlanBuilder(planNodeIdGenerator)
  //                 .tableScan(asRowType(inputRowVector->type()))
  //                 .capturePlanNodeId(p0)
	// 	              .project({"mat_mul(x)"})
  //                 .planBuild();

  // auto plan0 = planOOM.planFragment();
  // queryCtx_->testingOverrideConfigUnsafe(
  //     {{core::QueryConfig::kPreferredOutputBatchRows, "400"}, {core::QueryConfig::kPreferredOutputBatchBytes, "2000000"},  {core::QueryConfig::kMaxOutputBatchRows, "300"}});
  // Create task
  


  std::shared_ptr<memory::MemoryPool> rootPool{memory::defaultMemoryManager().addRootPool("root", 102 * MB)}; // 280 pass for 4 threads, 40 for 1 thread
  auto childPool = rootPool->addLeafChild("leaf");
  queryCtx_->testingOverrideMemoryPool(rootPool);
  
  // queryCtx_->testingOverrideConfigUnsafe({{core::QueryConfig::kSpillEnabled, "false"}});
  
  // VectorMaker m{childPool.get()};
  // auto weight = createBlock_w_oom(1000, 500, weights, m);

  // auto inputs = maker.flatVector<float>(6000000);

  // for(int i=0; i < 6000000; i++){
  //   inputs->set(i, i*2);
  // } 
  // auto input = createBlock_v_oom(6000, 1000, inputs, m);

  auto v_row = maker.flatVector({0, 0, 0, 0});//split to 4 parts
  auto v_col = maker.flatVector({0, 1, 2, 3});
  int values_block_size = 6000000 / 4;
  std::vector<std::vector<float>> valuesArray;

  for (int i = 0; i < 4; i++) {
      std::vector<float> valuesArraySingle;
      for (int j = 0; j < values_block_size; j++) {
          // int index = i * values_block_size + j;
          // if (index < 6000000) {
              valuesArraySingle.push_back(i*values_block_size+j);
          // }
      }
      valuesArray.push_back(valuesArraySingle);
  }

  auto valuesArrayVector = maker.arrayVector<float>(valuesArray, REAL());
  auto input = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector, v_row, v_col});
  auto valuesArrayVector1 = maker.arrayVector<float>({valuesArray[0]}, REAL());
  auto valuesArrayVector2 = maker.arrayVector<float>({valuesArray[1]}, REAL());
  auto valuesArrayVector3 = maker.arrayVector<float>({valuesArray[2]}, REAL());
  auto valuesArrayVector4 = maker.arrayVector<float>({valuesArray[3]}, REAL());

  // auto input = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector, v_row, v_col});
  // auto input1 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector1, v_row, v_col});
  // auto input2 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector2, v_row, v_col});
  // auto input3 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector3, v_row, v_col});
  // auto input4 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector4, v_row, v_col});

  auto input1 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector1, maker.flatVector({0}), maker.flatVector({0})});
  auto input2 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector2, maker.flatVector({0}), maker.flatVector({1})});
  auto input3 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector3, maker.flatVector({0}), maker.flatVector({2})});
  auto input4 = maker.rowVector({"v", "v_row", "v_col"}, {valuesArrayVector4, maker.flatVector({0}), maker.flatVector({3})});

  auto w_col = maker.flatVector({0, 0, 0, 0});//split to 4 parts
  auto w_row = maker.flatVector({0, 1, 2, 3});
  int weight_block_size = 500000 / 4;
  std::vector<std::vector<float>> weightsArray;

  for (int i = 0; i < 4; i++) {
      std::vector<float> weightsArraySingle;
      for (int j = 0; j < weight_block_size; j++) {
          // int index = i * weight_block_size + j;
          // if (index < 500000) {
              weightsArraySingle.push_back(i*weight_block_size+j);
          
      }
      weightsArray.push_back(weightsArraySingle);
  }
  auto weightsArrayVector = maker.arrayVector<float>(weightsArray, REAL());
  auto weightsArrayVector1 = maker.arrayVector<float>({weightsArray[0]}, REAL());
  auto weightsArrayVector2 = maker.arrayVector<float>({weightsArray[1]}, REAL());
  auto weightsArrayVector3 = maker.arrayVector<float>({weightsArray[2]}, REAL());
  auto weightsArrayVector4 = maker.arrayVector<float>({weightsArray[3]}, REAL());
  
  auto weightb = maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector, w_row, w_col});
  auto weightb1 = maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector1, maker.flatVector({0}), maker.flatVector({0})});
  auto weightb2 = maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector2, maker.flatVector({1}), maker.flatVector({0})});
  auto weightb3 = maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector3, maker.flatVector({2}), maker.flatVector({0})});
  auto weightb4 = maker.rowVector({"w", "w_row", "w_col"}, {weightsArrayVector4, maker.flatVector({3}), maker.flatVector({0})});

  auto weight_block = maker.flatVector<float>(size);

  for(int i=0; i < size; i++){
    weight_block->set(i, i*2);
  } 

  exec::registerVectorFunction(
    "mat_mul_b",
    MatrixMultiply_b::signatures(),
    std::make_unique<MatrixMultiply_b>(250, 500, weight_block->values()->asMutable<float>())
  );

  auto planNodeIdGenerator2 = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId p2;
  core::PlanNodeId p3;
  auto planOpt = exec::test::PlanBuilder(planNodeIdGenerator2)
                  .tableScan(asRowType(input->type()))
                  // .values({input})
                  .capturePlanNodeId(p2)
                  .hashJoin(
                      {"v_col"},
                      {"w_row"},
                    exec::test::PlanBuilder(planNodeIdGenerator2)
                   .tableScan(asRowType(weightb->type()))
                  // .values({weightb})
                   .capturePlanNodeId(p3)
                   .planNode(),
                    "", // extra filter
                    {"v_row", "w_col", "v", "w"})
                  .project({"v_row", "w_col", "mat_mul_b(v, w) AS mp"})
                  .singleAggregation({"w_col","v_row"}, {"array_sum(mp) AS result"})
                  .planBuild();

  //  core::PlanNodeId p4;
  // auto plantest = exec::test::PlanBuilder(planNodeIdGenerator2)
  //                 .tableScan(asRowType(input->type()))
  //                 .capturePlanNodeId(p4)
  //                 .project({"v", "v_row", "v_col"})
  //                 .planNode();

  auto plant = planOpt.planNode();
  auto plan2 = planOpt.planFragment();

  MyFileTest myfile;
  auto file1 = TempFilePath::create();
  myfile.writeToFile(file1->path, {input1});
    auto file2 = TempFilePath::create();
  myfile.writeToFile(file2->path, {input2});
    auto file3 = TempFilePath::create();
  myfile.writeToFile(file3->path, {input3});
    auto file4 = TempFilePath::create();
  myfile.writeToFile(file4->path, {input4});

  auto file0 = TempFilePath::create();
  myfile.writeToFile(file0->path, {input});


  auto file5 = TempFilePath::create();
  myfile.writeToFile(file5->path, {weightb1});
    auto file6 = TempFilePath::create();
  myfile.writeToFile(file6->path, {weightb2});
    auto file7 = TempFilePath::create();
  myfile.writeToFile(file7->path, {weightb3});
    auto file8 = TempFilePath::create();
  myfile.writeToFile(file8->path, {weightb4});

  auto file9 = TempFilePath::create();
  myfile.writeToFile(file9->path, {weightb});

  // auto file1 = TempFilePath::create();
  // myfile.writeToFile(file1->path, {input});
  // auto file2 = TempFilePath::create();
  // myfile.writeToFile(file2->path, {weightb});


  std::vector<std::shared_ptr<TempFilePath>> paths;
  std::vector<std::shared_ptr<TempFilePath>> paths2;
  std::vector<std::shared_ptr<TempFilePath>> paths3;

  paths3.push_back(file1);

    paths.push_back(file1);
    paths.push_back(file2);
    paths.push_back(file3);
    paths.push_back(file4);

    paths2.push_back(file5);
    paths2.push_back(file6);
    paths2.push_back(file7);
    paths2.push_back(file8);



  // auto hiveSplits0 =  myfile.makeHiveConnectorSplits(file0->path, 4, dwio::common::FileFormat::DWRF);
  auto hiveSplits = myfile.makeHiveConnectorSplits(paths);
  // auto hiveSplits1 = myfile.makeHiveConnectorSplits(paths3);
  auto hiveSplits2 = myfile.makeHiveConnectorSplits(paths2);

  // boost::interprocess::interprocess_semaphore semaphore(1);
  // auto task1 = exec::Task::create("0", plan2, 0, queryCtx_, 
  //     [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
  //       if(result){
  //         semaphore.post();
  //       }
  //       return exec::BlockingReason::kNotBlocked;
  // });
  // task1->setSpillDirectory(spillDirectory->path);
  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  // task1->start(task1, 1);

  // for(auto& split : hiveSplits) {
  //   semaphore.wait();
  //   std::cout << split->toString() << std::endl;
  //   task1->addSplit(p2, exec::Split(std::move(split)));
  // }
  // for(auto& split : hiveSplits2) {
  //   semaphore.wait();
  //   std::cout << split->toString() << std::endl;
  //   task1->addSplit(p3, exec::Split(std::move(split)));
  // }
  // task1->noMoreSplits(p2);
  // task1->noMoreSplits(p3);
  
  // waitForFinishedDrivers(task1);

  //   auto stats = task1->taskStats().pipelineStats;
  // for(auto stat : stats){
  //   for(auto ops : stat.operatorStats){
  //     std::cout << ops.spilledBytes << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // std::cout << "Time for Test (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;


  boost::interprocess::interprocess_semaphore semaphore(8);
  auto task = exec::Task::create("0", plan2, 0, queryCtx_, 
      [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
        if(result){
          semaphore.post();
        }
        return exec::BlockingReason::kNotBlocked;
  });

  task->start(task, 4);
  std::cout << "Hive splits:" << std::endl;
  for(auto& split : hiveSplits) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p2, exec::Split(std::move(split)));
  }
  for(auto& split : hiveSplits2) {
    semaphore.wait();
    std::cout << split->toString() << std::endl;
    task->addSplit(p3, exec::Split(std::move(split)));
  }
  task->noMoreSplits(p2);
  task->noMoreSplits(p3);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  waitForFinishedDrivers(task);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

  // Used AssertQueryBuilder
  // DuckDbQueryRunner duckDbQueryRunner_;
  // auto results_a1 = exec::test::AssertQueryBuilder(plant, duckDbQueryRunner_)
  // // exec::test::AssertQueryBuilder(plant, duckDbQueryRunner_)
  // .splits(p2, myfile.makeHiveConnectorSplits(paths))
  // .splits(p3, myfile.makeHiveConnectorSplits(paths2))
  // .copyResults(pool_.get());
  // std::cout << "a1 values Results:" << results_a1->toString() << std::endl;
  // std::cout << results_a1->toString(0, results_a1->size()) << std::endl;

  // DuckDbQueryRunner duckDbQueryRunner_;
  // auto results_a1 = exec::test::AssertQueryBuilder(plantest, duckDbQueryRunner_)
  // // exec::test::AssertQueryBuilder(plant, duckDbQueryRunner_)
  // .splits(p4, myfile.makeHiveConnectorSplits(paths))
  // .copyResults(pool_.get());
  // std::cout << "test values Results:" << results_a1->toString() << std::endl;
  // std::cout << results_a1->toString(0, results_a1->size()) << std::endl;

  // auto plan2 = planOpt.planFragment();
  // MyFileTest myfile;
  // auto file = TempFilePath::create();
  // myfile.writeToFile(file->path, {input});
  // std::vector<std::shared_ptr<TempFilePath>> paths;
  // for(int i=0; i < 20; i++)
  //   paths.push_back(file);
  // // for (auto single:input){
  // //   auto file = TempFilePath::create();
  // //   myfile.writeToFile(file->path, {single});
  // //   paths.push_back(file);
  // // }
  // // auto file2 = TempFilePath::create();
  // // myfile.writeToFile(file2->path, {weight});
  
  // // std::vector<std::shared_ptr<TempFilePath>> paths2;
  // auto hiveSplits = myfile.makeHiveConnectorSplits(paths);
  // // for(int i=0; i < num_splits; i++)
  // //   paths2.push_back(file2);
  // // auto hiveSplits2 = myfile.makeHiveConnectorSplits(paths2);

 
  // int concurrency = 48;
  // boost::interprocess::interprocess_semaphore semaphore(concurrency);

  // auto task = exec::Task::create("0", plan2, 0, queryCtx_, 
  //       [&semaphore](RowVectorPtr result, ContinueFuture* /*unused*/) {
  //         if(result){
  //           semaphore.post();
  //         }
  //         return exec::BlockingReason::kNotBlocked;
  // });

  // // Create 2 hive splits and add them to task
  // task->start(task, concurrency);
  // std::cout << "Hive splits:" << std::endl;
  // for(auto& split : hiveSplits) {
  //   semaphore.wait();
  //   std::cout << split->toString() << std::endl;
  //   task->addSplit(p2, exec::Split(std::move(split)));
  // }
  // task->noMoreSplits(p2);
  // std::cout << std::endl;
  // // Start task with 2 as maximum drivers and wait for execution to finish
  // //  while (auto result = task->next()) {
  // //   LOG(INFO) << "Vector available after processing (scan + sort):";
  // //   for (vector_size_t i = 0; i < result->size(); ++i) {
  // //     LOG(INFO) << result->toString(i);
  // //   }
  // // }
  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  // waitForFinishedDrivers(task);
  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // std::cout << "Total time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
  
}


int main(int argc, char** argv) {
    // test_optimizer(argc, argv);
    // test_mnist_optimizer(argc, argv, 1);
    // test_mnist_oom_error(argc, argv);
    test_oom_success(argc, argv);






















//    auto myVec = maker.flatVector<int64_t>({1, 10, 100, 1000, 10000});

//     exec::registerVectorFunction(
//         "vec_add_to_constant",
//         AddVectorToConstant::signatures(),
//         std::make_unique<AddVectorToConstant>(myVec, 5),
//         AddVectorToConstant::metadata());

//   exec::registerVectorFunction(
//         "vec_plus",
//         VectorMut::signatures(),
//         std::make_unique<VectorMut>(),
//         VectorMut::metadata());

//   exec::registerVectorFunction(
//       "vec_plus2",
//       VectorPlus::signatures(),
//       std::make_unique<VectorPlus>(),
//       VectorPlus::metadata());

//   exec::registerVectorFunction(
//       "vec_plus3",
//       VectorPlus::signatures(),
//       std::make_unique<VectorPlus>(),
//       VectorPlus::metadata());
//   // auto col1 = maker.flatVector({0, 1, 2, 3, 4});
//   // auto col2 = maker.flatVector({1, 2, 3, 4, 5});
//   // auto inputRowVector = maker.rowVector({"col1", "col2"}, {col1, col2});

//   auto row = maker.flatVector({0, 0, 1, 1});
//   auto col = maker.flatVector({0, 1, 0, 1});
//   auto va = maker.flatVector({1, 2, 3, 4});
//   auto vb = maker.flatVector({11, 12, 13, 14});
//   // {1,2 plus {11,12
//   //  3,4}      13,14}
//   auto inputRowVectorJoinA = maker.rowVector({"a_row", "a_col", "a_value"}, {row, col, va});
//   auto inputRowVectorJoinB = maker.rowVector({"b_row", "b_col", "b_value"}, {row, col, vb});

//   auto inputRowVectorJoin = maker.rowVector({"a_row", "a_col", "a_value", "b_row", "b_col", "b_value"}, {row, col, va, row, col, vb});
//   auto JoinT = maker.rowVector({"table_a", "table_b"},{inputRowVectorJoinA, inputRowVectorJoinB});
//   //two tables means nestjoin or other format of plan

//   auto myPlan = exec::test::PlanBuilder()
//                   .values({JoinT})
//                   .project({"vec_plus(table_a.a_value, table_b.b_value) AS result"})
// 		            .planFragment();

//   // auto task1 = std::make_shared<exec::Task>("task1", myPlan, 0, queryCtx_);
//   // // Execute the plan above
//   // auto result1 = task1->next();
//   // std::cout << "Results for Query 1:" << result1->toString() << std::endl;
//   // std::cout << result1->toString(0, result1->size()) << std::endl;
//   core::PlanNodeId Id;
//   auto myPlan2 = exec::test::PlanBuilder()
//                   .values({JoinT})
//                   .project({"vec_plus3(vec_plus2(vec_plus(table_a.a_value, table_b.b_value), table_b.b_value), table_b.b_value)"})
// 		            .planNode();

//   auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
//   core::PlanNodeId AId;
//   core::PlanNodeId BId;
//   auto planjoin = exec::test::PlanBuilder(planNodeIdGenerator)
//             .values({JoinT})
//             .project({"table_a.a_col", "table_a.a_row","table_a.a_value"})
//             // .capturePlanNodeId(AId)
//             .hashJoin(
//                 {"a_col"},
//                 {"b_row"},
//                 exec::test::PlanBuilder(planNodeIdGenerator)
//                     .values({JoinT})
//                     .project({"table_b.b_col", "table_b.b_row","table_b.b_value"})
//                     // .capturePlanNodeId(BId)
//                     .planNode(),
//                 "", // extra filter
//                 {"a_row","b_col", "a_value", "b_value"})
//             .project({"a_row", "b_col","a_value * b_value AS mp"})
//             .singleAggregation({"a_row","b_col"}, {"sum(mp) AS result"})
//             .project({"result"})
//             .planNode();

//   // auto taskj = std::make_shared<exec::Task>("taskj", planjoin, 0, queryCtx_);
//   // auto res = taskj->next();
//   // std::cout << "Results for Query 1:" << res->toString() << std::endl;
//   // std::cout << res->toString(0, res->size()) << std::endl;


//     auto myPlan3 = exec::test::PlanBuilder()
//                   .values({JoinT})
//                   .project({"vec_plus3(vec_plus2(vec_plus(table_a.a_value, table_b.b_value), table_b.b_value), table_b.b_value)"})
// 		            .planNode();


// auto oldplan = myPlan2->sources()[0];
// int oldPlanId = std::stoi(oldplan->id());
// auto generator = std::make_shared<core::PlanNodeIdGenerator>(oldPlanId + 1);
// optimizerBuilder opbuilder;
// opbuilder.setPlanNodeIdGenerator(generator);



// // should add source()[1] to solve 

// // optimizerBuilder opbuilder;
// auto plana = opbuilder.project_O({"table_a.a_col", "table_a.a_row","table_a.a_value"}, oldplan);

// auto plan_b1 = opbuilder.values_O({JoinT}, false, 1);
// auto plan_b = opbuilder.project_O({"table_b.b_col", "table_b.b_row","table_b.b_value"}, plan_b1);

// auto planb = opbuilder.hashjoin_O({"a_col"}, {"b_row"}, plan_b, "", {"a_row","b_col", 
// "a_value", "b_value"}, plana, core::JoinType::kInner, false);

// auto planc = opbuilder.project_O({"a_row", "b_col", "a_value * b_value AS mp"}, planb);
// auto pland = opbuilder.aggregation_O({"b_col","a_row"}, {}, {"sum(mp) AS result"}, 
// {}, core::AggregationNode::Step::kSingle, false, planc, {});

// auto plan_d2 = opbuilder.project_O({"a_row","b_col", "result"}, pland);

// auto plan_e1 = opbuilder.values_O({JoinT}, false, 1);
// auto plan_e2 = opbuilder.project_O({"table_b.b_col", "table_b.b_row","table_b.b_value"}, plan_e1);
// auto plan_e3 = opbuilder.hashjoin_O({"a_row","b_col"}, {"b_row","b_col"}, plan_e2, "", 
// {"b_value", "result"}, plan_d2, core::JoinType::kInner, false);

// auto plan_e = opbuilder.project_O({"b_value", "result"}, plan_e3);
// auto plan_e4 = opbuilder.project_O({"vec_plus3(vec_plus2(result, b_value), b_value)"}, plan_e);

// // auto plane = opbuilder.project_O({"vec_plus3(vec_plus2(result, b_value), b_value)"}, plan_e4);
// // auto source = planjoin->sources()[0]->sources()[0]->sources()[0]->sources()[0]->sources()[0];
// // source = oldplan;

//   auto res2 = AssertQueryBuilder(myPlan2).copyResults(pool_.get());
//   std::cout << "Results for Query 3:" << res2->toString() << std::endl;
//   std::cout << res2->toString(0, res2->size()) << std::endl;

//   auto res4 = AssertQueryBuilder(plan_e4).copyResults(pool_.get());
//   std::cout << "Results for Query 4:" << res4->toString() << std::endl;
//   std::cout << res4->toString(0, res4->size()) << std::endl;


//   auto res = AssertQueryBuilder(planjoin).copyResults(pool_.get());
//   std::cout << "Results for Query 2:" << res->toString() << std::endl;
//   std::cout << res->toString(0, res->size()) << std::endl;

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
  

  // exec::registerVectorFunction(
  //   "flat",
  //   Flat::signatures(),
  //   std::make_unique<Flat>()
  // );

  // auto Plan_1 = myDensePlan->sources()[0];
  // int Plan_1_Id = std::stoi(Plan_1->id());
  // auto generator_1 = std::make_shared<core::PlanNodeIdGenerator>(Plan_1_Id + 1);
  // optimizerBuilder opBuilder;
  // opBuilder.setPlanNodeIdGenerator(generator_1);

  // auto myflatPlan = exec::test::PlanBuilder(pool_.get())
  //                 .values({inputRowVector_dense})
  //                 .project({"Flat(x)"})
	// 	              .planNode();

  // auto results_flat = exec::test::AssertQueryBuilder(myflatPlan).copyResults(pool_.get());
  // std::cout << "flat Dense Results:" << results_flat ->toString() << std::endl;
  // std::cout << results_flat ->toString(0, results_flat ->size()) << std::endl;

//     exec::registerVectorFunction(
//     "adaptive",
//     ADP::signatures(),
//     std::make_unique<ADP>()
//   );

//   optimizerBuilder opBuilder;
//   auto plan_1 = opBuilder.values_O({inputRowVector_dense_flat}, false, 1);

//   auto plan_2 = opBuilder.project_O({"x_col", "x_row","x"}, plan_1);

//   std::vector<int> weighsrowVector;
//   std::vector<int> weighscolVector;
//   for (int i=0; i < size; i++) {
//     int rowIndex = i / output_size;
//     int colIndex = i % output_size;
//     weighsrowVector.push_back(rowIndex);
//     weighscolVector.push_back(colIndex);
//   }

//   // auto weighsrowVectors = maker.arrayVector<float>(weighsrowVector, REAL());
//   // auto weighscolVectors = maker.arrayVector<float>(weighscolVector, REAL());
//   auto weighsrowVectors = maker.flatVector(weighsrowVector);
//   auto weighscolVectors = maker.flatVector(weighscolVector);
//   auto inputRowVector_dense_weighs = maker.rowVector({"w", "w_row", "w_col"}, {weights, weighsrowVectors, weighscolVectors});

//   auto plan_3 = opBuilder.values_O({inputRowVector_dense_weighs}, false, 1);
//   auto plan_4 = opBuilder.project_O({"w", "w_row", "w_col"}, plan_3);

//   auto plan_5 = opBuilder.hashjoin_O({"x_col"}, {"w_row"}, plan_4, "", {"x_row","w_col", 
// "x", "w"}, plan_2, core::JoinType::kInner, false);

//   auto plan_6 = opBuilder.project_O({"x_row", "w_col", "x * w AS mp"}, plan_5);
//   auto plan_7 = opBuilder.aggregation_O({"w_col","x_row"}, {}, {"sum(mp) AS result"}, 
// {}, core::AggregationNode::Step::kSingle, false, plan_6, {});
  
//   auto plan_8 = opBuilder.project_O({"adaptive(w_col, x_row, result) AS res"}, plan_7);
//   // auto plan_9 = opBuilder.filter_O({"res IS NOT NULL"}, plan_8);
//   // auto plan_10 = opBuilder.project_O({"res"}, plan_9);
//   auto plan_9 = opBuilder.project_O({"relu(mat_add(res))"}, plan_8);
  
//   // auto results_dense_rep = exec::test::AssertQueryBuilder(plan_9).copyResults(pool_.get());
//   // std::cout << "Rep Dense Results:" << results_dense_rep->toString() << std::endl;
//   // std::cout << results_dense_rep->toString(0, results_dense_rep->size()) << std::endl;


}