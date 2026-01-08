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

class PositionEncoding : public MLFunction {
 public:
  PositionEncoding(int inputDims) {
    inputDims_ = inputDims;
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
    BaseVector* input = args[0].get();

    exec::LocalDecodedVector inputHolder(context, *input, rows);
    auto decodedInputArray = inputHolder.get();
    auto baseInputArray =
        decodedInputArray->base()->as<ArrayVector>()->elements();

    float* inputValues = baseInputArray->values()->asMutable<float>();

    auto numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m(inputValues, numInput, inputDims_);

    std::vector<std::vector<float>> results;

    if (inputDims_ % 2 != 0) {
      throw std::runtime_error(fmt::format(
          "Position Encoding Dims has to be a even number, current value is: {}",
          inputDims_));
    }
    for (int i = 0; i < numInput; i++) {
      for (int j = 0; j < inputDims_ / 2; j++) {
        float angle = i / std::pow(10000.0, 2.0 * j / inputDims_);
        int dataShift = i * inputDims_ + j * 2;
        inputValues[dataShift] += std::sin(angle);
        inputValues[dataShift + 1] += std::cos(angle);
      }
      std::vector<float> row(m.row(i).data(), m.row(i).data() + m.cols());
      results.push_back(row);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "position_encoding";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // Computes the operation cost using a legacy static cost model.
    // This method is retained for compatibility and reference purposes.
    // NOTE: Query optimization uses an ML-based cost model
    // (optimizer/query2vec), and this function is not invoked in that process.

    // TODO: Implement a static cost estimation method for the specified kernel.
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int inputDims_;
};