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

// #include <Eigen/Dense>
// #include <cblas.h>
// #include <chrono>
// #include <torch/torch.h>
#include "velox/ml_functions/DNNBuilder.h"
#include <fstream>
#include <sstream>

#include "velox/exec/FilterProject.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::test;
using namespace facebook::velox::optimizer;

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

std::string toString(
    const std::vector<std::shared_ptr<facebook::velox::exec::AggregateFunctionSignature>>&
        signatures) {
  std::stringstream out;
  for (auto i = 0; i < signatures.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << signatures[i]->toString();
  }
  return out.str();
}

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

core::PlanNodePtr& Optest(core::PlanNodePtr source, RowVectorPtr data, int output_size, int size, FlatVectorPtr<float> weights){
   exec::registerVectorFunction(
    "adaptive",
    ADP::signatures(),
    std::make_unique<ADP>()
  );
  VectorMaker maker{pool_.get()};

  optimizerBuilder opBuilder;
  std::string x_var = "x"; 
  std::string w_var = "w"; 
  auto plan_1 = opBuilder.values_O({data}, false, 1);

  auto plan_2 = opBuilder.project_O({fmt::format("{}_col", x_var), fmt::format("{}_row", x_var),fmt::format("{}", x_var)}, plan_1);

  std::vector<int> weighsrowVector;
  std::vector<int> weighscolVector;
  for (int i=0; i < size; i++) {
    int rowIndex = i / output_size;
    int colIndex = i % output_size;
    weighsrowVector.push_back(rowIndex);
    weighscolVector.push_back(colIndex);
  }

  // auto weighsrowVectors = maker.arrayVector<float>(weighsrowVector, REAL());
  // auto weighscolVectors = maker.arrayVector<float>(weighscolVector, REAL());
  auto weighsrowVectors = maker.flatVector(weighsrowVector);
  auto weighscolVectors = maker.flatVector(weighscolVector);
  auto inputRowVector_dense_weighs = maker.rowVector({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, {weights, weighsrowVectors, weighscolVectors});

  auto plan_3 = opBuilder.values_O({inputRowVector_dense_weighs}, false, 1);
  auto plan_4 = opBuilder.project_O({fmt::format("{}", w_var), fmt::format("{}_row", w_var), fmt::format("{}_col", w_var)}, plan_3);

  auto plan_5 = opBuilder.hashjoin_O({fmt::format("{}_col", x_var)}, {fmt::format("{}_row", w_var)}, plan_4, "", {fmt::format("{}_row", x_var),fmt::format("{}_col", w_var), 
fmt::format("{}", x_var), fmt::format("{}", w_var)}, plan_2, core::JoinType::kInner, false);

  auto plan_6 = opBuilder.project_O({fmt::format("{}_row", x_var), fmt::format("{}_col", w_var), fmt::format("{} * {} AS mp", x_var, w_var)}, plan_5);
  auto plan_7 = opBuilder.aggregation_O({fmt::format("{}_col", w_var),fmt::format("{}_row", x_var)}, {}, {"sum(mp) AS result"}, 
{}, core::AggregationNode::Step::kSingle, false, plan_6, {});
  
  auto plan_8 = opBuilder.project_O({fmt::format("adaptive({}_col, {}_row, result) AS res", w_var, x_var)}, plan_7);
  // auto plan_9 = opBuilder.filter_O({"res IS NOT NULL"}, plan_8);
  // auto plan_10 = opBuilder.project_O({"res"}, plan_9);
  static core::PlanNodePtr plan_9 = opBuilder.project_O({"relu(mat_add(res))"}, plan_8); // TODO: get the rest expression


  return plan_9;
};

int main(int argc, char** argv) {
    
    folly::init(&argc, &argv, false);

   functions::prestosql::registerAllScalarFunctions();
   aggregate::prestosql::registerAllAggregateFunctions();

   parse::registerTypeResolver();

   VectorMaker maker{pool_.get()};
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

  auto rowVectors = maker.flatVector(rowVector);
  auto colVectors = maker.flatVector(colVector);
  // auto rowVectors = maker.arrayVector<int_32_t>({rowVector}, INTEGER());
  // auto colVectors = maker.arrayVector<int_32_t>({colVector}, INTEGER());
  auto inputRowVector_dense = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVector, rowVectors, colVectors});

  std::vector<float> flattenedVector;
  for (const auto& featureVector : featureVectors) {
      flattenedVector.insert(flattenedVector.end(), featureVector.begin(), featureVector.end());
  }

  auto featureArrayVectorFlat = maker.flatVector(flattenedVector);
  auto inputRowVector_dense_flat = maker.rowVector({"x", "x_row", "x_col"}, {featureArrayVectorFlat, rowVectors, colVectors});


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
                  .values({inputRowVector_dense})
                  .project({"relu(mat_add(mat_mul(x)))"})
		              .planBuild();

  auto myDensePlan = planbuilder.planNode();

  auto results_dense = exec::test::AssertQueryBuilder(myDensePlan).copyResults(pool_.get());
  std::cout << "Dense Results:" << results_dense->toString() << std::endl;
  std::cout << results_dense->toString(0, results_dense->size()) << std::endl;

  auto newplan = Optest(myDensePlan, inputRowVector_dense_flat, output_size, size, weights);
  
  auto nodeid = myDensePlan->id();
  auto str = planbuilder.findExprStrings(nodeid);

  auto myDensePlanF = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector_dense})
                  .project({"relu(mat_add(mat_mul(x)))"})
		              .planFragment();

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
    auto call = std::dynamic_pointer_cast<MLFunction>(exprs->exprs()[0]->vectorFunction());//inputs_ contains mat_add
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