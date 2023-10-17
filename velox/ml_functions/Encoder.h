#pragma once
#include <fmt/format.h>
#include <iostream>
#include "functions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

// Implementation of embedding layer where the embedding is stored as a 2-D
// array: numEmbedding*embeddingDims, lookup takes a int vector as indices

class IntEncoder : public exec::VectorFunction {
 public:
  IntEncoder(std::unordered_map<int, int> mapping) {
    mapping_ = mapping;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    auto inputVector = args[0]->as<ArrayVector>()->elements();
    int* inputValues = inputVector->values()->asMutable<int>();
    int numInputs = rows.size();

    auto indicesRowVector = args[0];

    auto arrayVector = args[0]->as<ArrayVector>();

    // TODO: current only consider 1 element case
    std::vector<std::vector<int>> result(numInputs, std::vector<int>(1));
    for (int i = 0; i < numInputs; i++) {
      result[i][0] = mapping_.at(inputValues[i]);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<int>(result, INTEGER());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(INTEGER)")
                .returnType("array(INTEGER)")
                .build()};
  }


  static std::string getName() {
    return "encoder";
  };


 private:
  std::unordered_map<int, int> mapping_;
};