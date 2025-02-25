/*
 * Copyright (c) 2025 ASU Cactus Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "BaseFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

class Concat : public MLFunction {
 public:
  Concat(int input1Dims, int input2Dims) {
    input1Dims_ = input1Dims;
    input2Dims_ = input2Dims;
    LOG(ERROR)
        << "[ERROR UDF-CONCAT] Bug exists in the apply function when decoding the right input arrays of filtered rows. Use built-in concat instead!";
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    // Decoder is required to handle address error, reference code:
    // ArrayIntersectExcept.cpp
    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray =
        decodedRightArray->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();
    // std::cout << "[DEBUG]: rows.size(): " << rows.size()
    //           << " # selected: " << rows.countSelected() << std::endl;
    // std::cout << "[DEBUG]: size of Elements: " << baseLeftArray->size() << ",
    // "
    //           << baseRightArray->size() << std::endl;
    // std::cout << "[DEBUG] input1Dims_: " << input1Dims_
    //           << ", input2Dims_: " << input2Dims_ << std::endl;

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

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int input1Dims_;
  int input2Dims_;
};