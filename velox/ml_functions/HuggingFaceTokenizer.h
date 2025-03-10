/**
 * @file
 * @brief Implementation of a Hugging Face tokenizer for machine learning tasks.
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

#include <fmt/format.h>
#include <tokenizers_cpp.h>
#include <iostream>
#include "BaseFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using tokenizers::Tokenizer;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

/**
 * @class HuggingFaceTokenizer
 * @brief Implements a machine learning function for tokenizing text using a Hugging Face tokenizer.
 */
class HuggingFaceTokenizer : public MLFunction {
 public:
  /**
   * @brief Constructor for HuggingFaceTokenizer.
   * @param pathToTokenizer Path to the tokenizer configuration file.
   */
  HuggingFaceTokenizer(std::string pathToTokenizer) {
    pathToTokenizer_ = pathToTokenizer;
    auto blob = LoadBytesFromFile(pathToTokenizer_);
    tokenizer_ = Tokenizer::FromBlobJSON(blob);
  }

  /**
   * @brief Applies the tokenizer function to the input data.
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

    // Read string input
    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();
    int numInputs = rows.size();

    std::vector<std::vector<int>> result;
    for (int i = 0; i < numInputs; i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::vector<int> ids = tokenizer_->Encode(val);

      result.push_back(ids);
    }
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<int>(result, INTEGER());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("array(INTEGER)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "hf_tokenizer";
  };

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
  std::string pathToTokenizer_; ///< Path to the tokenizer configuration file.
  std::unique_ptr<Tokenizer> tokenizer_; ///< Pointer to the tokenizer instance.
};
