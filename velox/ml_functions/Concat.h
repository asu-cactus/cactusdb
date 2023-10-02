#pragma once
#include <Eigen/Dense>
#include <cmath>
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

class Concat : public exec::VectorFunction {
 public:
  Concat(int input1Dims, int input2Dims) {
    input1Dims_ = input1Dims;
    input2Dims_ = input2Dims;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input1Elements = args[0]->as<ArrayVector>()->elements();
    auto input2Elements = args[1]->as<ArrayVector>()->elements();
    float* input1Values = input1Elements->values()->asMutable<float>();
    float* input2Values = input2Elements->values()->asMutable<float>();

    std::vector<std::vector<float>> results;

    for (int i = 0; i < rows.size(); i++) {
      std::vector<float> concatenatedVector(input1Dims_ + input2Dims_);
      // copy the 1st array
      std::memcpy(
          concatenatedVector.data(),
          input1Values + i * input1Dims_,
          input1Dims_ * sizeof(float));
      // copy the 2nd array
      std::memcpy(
          concatenatedVector.data() + input1Dims_,
          input2Values + i * input2Dims_,
          input2Dims_ * sizeof(float));
      results.push_back(concatenatedVector);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }
  static std::string getName() {
    return "concat";
  }

 private:
  int input1Dims_;
  int input2Dims_;
};