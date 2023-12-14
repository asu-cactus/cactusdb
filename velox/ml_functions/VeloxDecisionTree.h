#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/functions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/ml_functions/DecisionTree.h"

using namespace std;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml{

class VeloxDecisionTree : public OpaqueType {


public:
    
    VeloxDecisionTree() : OpaqueType(std::type_index(typeid(Tree))) {}

    std::string toString() const override {
    
        return name();
    
    }

    const char * name() const override {
    
        return "velox_decision_tree";
    
    }
    
};


class VeloxTreePrediction : public exec::VectorFunction {

public:

  VeloxTreePrediction() {}

  void apply(

      const SelectivityVector& rows,

      std::vector<VectorPtr>& args,

      const TypePtr& type,

      exec::EvalCtx& context,

      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    BaseVector* left = args[0].get();

    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);

    auto decodedLeftArray = leftHolder.get();
    
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);

    auto decodedRightArray = rightHolder.get();

    auto baseRightArray = rightHolder->base()->as<FlatVector<TreePtr>>();

    float* input1Values = baseLeftArray->values()->asMutable<float>();

    TreePtr* input2Values = baseRightArray->values()->asMutable<TreePtr>();;

    std::vector<float> results;

    for (int i = 0; i < rows.size(); i++) {

      results.push_back(i);

    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("velox_decision_tree")
                .returnType("REAL")
                .build()};
  }

  static std::string getName() {
    return "velox_tree_predict";
  }

};

}
