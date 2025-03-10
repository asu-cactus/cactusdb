/**
 * @file
 * @brief Implementation of a dropout layer for machine learning.
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
 * @class Dropout
 * @brief Implements a dropout layer for machine learning, which randomly sets input values to zero during training.
 */
class Dropout : public MLFunction {
 public:
  /**
   * @brief Constructor for Dropout.
   * @param p The probability of dropping out an input value (setting it to zero).
   */
  Dropout(float p) {
    p_ = p;
    // std::random_device device;
    // std::mt19937 gen(device());
    // std::bernoulli_distribution coin_flip(0.5);
    // bool outcome = coin_flip(gen);
  }

  /**
   * @brief Applies the dropout function to the input data.
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution bernoulli(p_);

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto inputFeatures = args[0]->as<ArrayVector>()->elements();
    float* inputValues = inputFeatures->values()->asMutable<float>();

    int inputSize = inputFeatures->size();
    int numInput = args[0]->size();
    int numFeatures = inputSize / numInput;

    // float* result[inputSize];
    std::vector<std::vector<float>> result(
        numInput, std::vector<float>(numFeatures));

    for (int i = 0; i < numInput; i++) {
      for (int j = 0; j < numFeatures; j++) {
        bool outcome = bernoulli(gen);
        if (outcome) {
          result[i][j] = 0;
        } else {
          result[i][j] = inputValues[i * numFeatures + j];
        }
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
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
    return "dropout";
  };

  /**
   * @brief Returns the tensor associated with the function.
   * @return Pointer to the tensor.
   */
  float* getTensor() const override {
    // FIXME
    return nullptr;
  }

  /**
   * @brief Sets the dropout probability.
   * @param p The probability of dropping out an input value (setting it to zero).
   */
  void setWeight(float p) {
    p_ = p;
  }

 private:
  float p_; ///< The probability of dropping out an input value.
  std::mt19937 gen_; ///< Random number generator.
  std::bernoulli_distribution bernoulli_; ///< Bernoulli distribution for dropout.
};
