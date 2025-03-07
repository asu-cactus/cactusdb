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
#include <fmt/format.h>
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

// Implementation of embedding layer where the embedding is stored as a 2-D
// array: numEmbedding*embeddingDims, lookup takes a int vector as indices

class IntEncoder : public MLFunction {
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
    BaseVector::ensureWritable(rows, ARRAY(INTEGER()), context.pool(), output);

    // Decode the input argument.

    auto arrayVector = args[0]->as<ArrayVector>();
    auto elementsVector = arrayVector->elements()->asFlatVector<int>();

    // Map to store result rows.
    auto numInputs = rows.size();
    std::vector<std::vector<int>> result(numInputs);

    // Process only the selected rows.
    rows.applyToSelected([&](int row) {
      // Decode the array element for this row.
      auto userIdBeforeEncode = elementsVector->valueAt(row);

      // Check if the userId exists in the mapping.
      auto it = mapping_.find(userIdBeforeEncode);
      if (it != mapping_.end()) {
        // If found, set the result.
        result[row] = {it->second};
      } else {
        // Handle missing keys if necessary.
        result[row] = {-1};
        LOG(WARNING) << "[WARNING] Missing key: " << userIdBeforeEncode
                     << " mapping size: " << mapping_.size() << std::endl;
      }
    });

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

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::unordered_map<int, int> mapping_;
};

class StringEncoder : public MLFunction {
 public:
  StringEncoder(std::unordered_map<std::string, int> mapping) {
    mapping_ = mapping;
  }

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

    std::vector<std::vector<int>> result(numInputs);

    rows.applyToSelected([&](int row) {
      StringView val = decodedStringInput->valueAt<StringView>(row);
      auto it = mapping_.find(val.getString());
      if (it != mapping_.end()) {
        result[row] = {it->second};
      } else {
        // Handle missing keys if necessary
        result[row] = {-1};
        LOG(WARNING) << "[WARNING] Missing key: " << val.getString()
                     << std::endl;
      }
    });

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<int>(result, INTEGER());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("array(INTEGER)")
                .build()};
  }

  static std::string getName() {
    return "encoder_string";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::unordered_map<std::string, int> mapping_;
};

class StringVariadicEncoder : public MLFunction {
 public:
  StringVariadicEncoder(std::unordered_map<std::string, int> mapping) {
    mapping_ = std::unordered_map<std::string, int>(mapping);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto arrayVector = args[0]->as<ArrayVector>();
    auto elementsVector = arrayVector->elements()->asFlatVector<StringView>();
    auto numRows = rows.size();

    std::vector<std::vector<int>> result(numRows);

    rows.applyToSelected([&](vector_size_t row) {
      int numElements = arrayVector->sizeAt(row);
      int offset = arrayVector->offsetAt(row);

      std::vector<int> indices;
      indices.reserve(numElements);

      for (int j = 0; j < numElements; ++j) {
        // Safely decode each string
        StringView val = elementsVector->valueAt(offset + j);

        auto it = mapping_.find(val.getString());
        if (it != mapping_.end()) {
          indices.push_back(it->second);
        } else {
          // Handle missing keys if necessary
          indices.push_back(0); // Or some default value
          std::cout << "[ERROR] Missing key: " << val.getString() << std::endl;
        }
      }
      result[row] = indices;
    });

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<int>(result, INTEGER());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(VARCHAR)")
                .returnType("array(INTEGER)")
                .build()};
  }

  static std::string getName() {
    return "encoder_string_variadic";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::unordered_map<std::string, int> mapping_;
};

class MultiHotNormalizedEncoder : public MLFunction {
 public:
  MultiHotNormalizedEncoder(int size) {
    size_ = size;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto indicesRowVector = args[0];
    auto arrayVector = indicesRowVector->as<ArrayVector>();

    auto indicesVector = arrayVector->elements();
    int* indicesValues = indicesVector->values()->asMutable<int>();
    int numInputs = rows.size();

    std::vector<std::vector<float>> encoding(
        numInputs, std::vector<float>(size_, 0));

    for (int i = 0; i < numInputs; i++) {
      int numSubIndices = arrayVector->sizeAt(i);
      int indicesOffset = arrayVector->offsetAt(i);
      float value = 1.0 / numSubIndices;
      for (int j = 0; j < numSubIndices; j++) {
        int embedIndex = indicesValues[indicesOffset + j];
        encoding[i][embedIndex] = value;
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(encoding, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(INTEGER)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "multi_hot_norm_encoder";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int size_;
};