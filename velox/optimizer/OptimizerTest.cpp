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

  exec::registerVectorFunction(
      "vec_plus2",
      VectorPlus::signatures(),
      std::make_unique<VectorPlus>(),
      VectorPlus::metadata());

  exec::registerVectorFunction(
      "vec_plus3",
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
                  .project({"vec_plus3(vec_plus2(vec_plus(table_a.a_value, table_b.b_value), table_b.b_value), table_b.b_value)"})
		            .planNode();

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId AId;
  core::PlanNodeId BId;
  auto planjoin = exec::test::PlanBuilder(planNodeIdGenerator)
            .values({JoinT})
            .project({"table_a.a_col", "table_a.a_row","table_a.a_value"})
            // .capturePlanNodeId(AId)
            .hashJoin(
                {"a_col"},
                {"b_row"},
                exec::test::PlanBuilder(planNodeIdGenerator)
                    .values({JoinT})
                    .project({"table_b.b_col", "table_b.b_row","table_b.b_value"})
                    // .capturePlanNodeId(BId)
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


    auto myPlan3 = exec::test::PlanBuilder()
                  .values({JoinT})
                  .project({"vec_plus3(vec_plus2(vec_plus(table_a.a_value, table_b.b_value), table_b.b_value), table_b.b_value)"})
		            .planNode();


auto oldplan = myPlan2->sources()[0];
int oldPlanId = std::stoi(oldplan->id());
auto generator = std::make_shared<core::PlanNodeIdGenerator>(oldPlanId + 1);
optimizerBuilder opbuilder;
opbuilder.setPlanNodeIdGenerator(generator);



// should add source()[1] to solve 

// optimizerBuilder opbuilder;
auto plana = opbuilder.project_O({"table_a.a_col", "table_a.a_row","table_a.a_value"}, oldplan);
auto plan_b = opbuilder.project_O({"table_b.b_col", "table_b.b_row","table_b.b_value"}, oldplan);
auto planb = opbuilder.hashjoin_O({"a_col"}, {"b_row"}, plan_b, "", {"a_row","b_col", "a_value", "b_value"}, plana, core::JoinType::kInner, false);
auto planc = opbuilder.project_O({"a_row", "b_col","a_value * b_value AS mp"}, planb);
auto pland = opbuilder.aggregation_O({"a_row","b_col"}, {}, {"sum(mp) AS result"}, {}, core::AggregationNode::Step::kSingle, false, planc, {});

// auto source = planjoin->sources()[0]->sources()[0]->sources()[0]->sources()[0]->sources()[0];
// source = oldplan;

  auto res2 = AssertQueryBuilder(myPlan2).copyResults(pool_.get());
  std::cout << "Results for Query 3:" << res2->toString() << std::endl;
  std::cout << res2->toString(0, res2->size()) << std::endl;

  auto res4 = AssertQueryBuilder(pland).copyResults(pool_.get());
  std::cout << "Results for Query 4:" << res4->toString() << std::endl;
  std::cout << res4->toString(0, res4->size()) << std::endl;


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