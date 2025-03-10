/**
 * @file
 * @brief Implementation of a position encoding function for machine learning.
 * @copyright Copyright (c) 2025 ASU Cactus Lab.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
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

/**
 * @class PositionEncoding
 * @brief Implements a position encoding function for machine learning tasks.
 */
class PositionEncoding : public MLFunction {
 public:
  /**
   * @brief Constructor for PositionEncoding.
   * @param inputDims The dimensionality of the input vectors.
   */
  PositionEncoding(int inputDims) {
    inputDims_ = inputDims;
  }

  /**
   * @brief Applies the position encoding function to the input data.
   * @param rows Selectivity vector indicating which rows to process.
   * @param args Vector of input arguments.
   * @param type Type of the output vector.
   * @param context Evaluation context.
   * @param output Output vector to store the results.
   */
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

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "position_encoding";
  }

  /**
   * @brief Returns the tensor associated with the function.
   * @return Pointer to the tensor.
   */
  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  /**
   * @brief Estimates the cost of the function.
   * @param inputDims Dimensions of the input.
   * @return Cost estimate.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int inputDims_; ///< The dimensionality of the input vectors.
};
