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

class OneHotEncoder : public MLFunction {
 public:
  OneHotEncoder(std::unordered_map<std::string, int> mapping) {
    mapping_ = std::unordered_map<std::string, int>(mapping);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    auto inputVector = args[0]->asFlatVector<StringView>();
    auto numRows = rows.size();

    // Determine fixed one-hot length
    int numCategories = mapping_.size();

    std::vector<std::vector<int>> result(
        numRows, std::vector<int>(numCategories, 0));

    rows.applyToSelected([&](vector_size_t row) {
      StringView val = inputVector->valueAt(row);
      std::string raw = val.getString();

      std::stringstream ss(raw);
      std::string token;
      while (std::getline(ss, token, '|')) {
        // Optionally trim and capitalize/lowercase here
        auto it = mapping_.find(token);
        if (it != mapping_.end()) {
          int index = it->second - 1; // Assuming mapping values are 1-based
          if (index >= 0 && index < numCategories) {
            result[row][index] = 1;
          }
        } else {
          std::cout << "[ERROR] Unknown genre: " << token << std::endl;
        }
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
    return "one_hot_encoder";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
    // return 0;
  }

 private:
  std::unordered_map<std::string, int> mapping_;
};

class TokenFreqVector : public MLFunction {
 public:
  explicit TokenFreqVector(int vocabSize) : vocabSize_(vocabSize) {}

  /// Signature must match VectorFunction
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx& context,
      VectorPtr& result) const override {
    BaseVector::ensureWritable(rows, outputType, context.pool(), result);
    auto arrayVec = args[0]->as<ArrayVector>();
    auto elements = arrayVec->elements()->asFlatVector<int>();
    const auto& offsets = arrayVec->rawOffsets();
    const auto& sizes = arrayVec->rawSizes();

    int numRows = rows.size();
    std::vector<std::vector<float>> buffer(
        numRows, std::vector<float>(vocabSize_, 0.0f));

    rows.applyToSelected([&](vector_size_t row) {
      int offset = offsets[row];
      int size = sizes[row];
      for (int i = 0; i < size; ++i) {
        int tokenId = elements->valueAt(offset + i);
        if (tokenId >= 0 && tokenId < vocabSize_) {
          buffer[row][tokenId] += 1.0f;
        }
      }
    });
    rows.applyToSelected([&](vector_size_t row) {
      float sum = 0.0f;
      for (int i = 0; i < vocabSize_; ++i) {
        sum += buffer[row][i];
      }
      if (sum > 0.0f) {
        for (int i = 0; i < vocabSize_; ++i) {
          buffer[row][i] /= sum;
        }
      }
    });

    // Build the output ArrayVector<REAL>
    VectorMaker maker{context.pool()};
    result = maker.arrayVector<float>(buffer, REAL());
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  static std::string name() {
    return "extract_tf_features";
  }

  /// Declare signature: array(INTEGER) -> array(REAL)
  static std::vector<std::shared_ptr<FunctionSignature>> signatures() {
    return {FunctionSignatureBuilder()
                .argumentType("array(INTEGER)")
                .returnType("array(REAL)")
                .build()};
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
    // return 0;
  }

 private:
  const int vocabSize_;
};

class RatingMapToArray : public MLFunction {
 public:
  /// numItems should be 3706 for your use case
  explicit RatingMapToArray(int numItems) : numItems_(numItems) {}

  /// Must match VectorFunction signature exactly
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      EvalCtx& context,
      VectorPtr& result) const override {
    BaseVector::ensureWritable(rows, outputType, context.pool(), result);

    // Input is a MapVector<INTEGER,INTEGER>
    auto mapVec = args[0]->as<MapVector>();
    auto keys = mapVec->mapKeys()->asFlatVector<int>();
    auto vals = mapVec->mapValues()->asFlatVector<int>();
    const auto& offsets = mapVec->rawOffsets();
    const auto& sizes = mapVec->rawSizes();

    int numRows = rows.size();
    // buffer[row][i] will hold rating for movie (i+1), default 0
    std::vector<std::vector<float>> buffer(
        numRows, std::vector<float>(numItems_, 0.0f));

    rows.applyToSelected([&](vector_size_t row) {
      int off = offsets[row];
      int sz = sizes[row];
      for (int i = 0; i < sz; ++i) {
        int movieId = keys->valueAt(off + i);
        int rating = vals->valueAt(off + i);
        int idx = movieId - 1;
        if (idx >= 0 && idx < numItems_) {
          buffer[row][idx] = static_cast<float>(rating);
        }
      }
    });

    VectorMaker maker{context.pool()};
    result = maker.arrayVector<float>(buffer, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("map(INTEGER,INTEGER)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "map_to_array";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
    // return 0;
  }

 private:
  const int numItems_;
};

class OneHotEncoderInt : public MLFunction {
 public:
  explicit OneHotEncoderInt(
      std::vector<std::pair<int64_t,int>> mapping) {
    int maxPos = 0;
    for (auto& p : mapping) {
      valueToIndex_[p.first] = p.second;
      maxPos = std::max(maxPos, p.second);
    }
    numCategories_ = maxPos + 1;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // 1) Prepare result vector
    BaseVector::ensureWritable(rows, outputType, context.pool(), result);

    // 2) Decode input BIGINT column
    exec::LocalDecodedVector decoded(context, *args[0], rows);
    auto decodedVec = decoded.get();

    // 3) Build a [rows.size() × numCategories_] zero buffer
    auto totalRows = rows.size();
    std::vector<std::vector<float>> buffer(
        totalRows, std::vector<float>(numCategories_, 0.0f));

    // 4) Fill in the 1.0f for each selected row
    rows.applyToSelected([&](vector_size_t row) {
      int64_t cat = decodedVec->valueAt<int64_t>(row);
      auto it     = valueToIndex_.find(cat);
      if (it != valueToIndex_.end()) {
        buffer[row][it->second] = 1.0f;
      }
    });

    // 5) **FIXED**: pass REAL() (the element type), not the array type
    VectorMaker maker{context.pool()};
    result = maker.arrayVector<float>(buffer, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
      exec::FunctionSignatureBuilder()
        .argumentType("BIGINT")
        .returnType("ARRAY(REAL)")
        .build()
    };
  }

  static std::string name() { return "one_hot_int"; }

  float* getTensor() const override { return nullptr; }

  CostEstimate getCost(std::vector<int> inputDims) override {
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int numCategories_;
  std::unordered_map<int64_t,int> valueToIndex_;
};

class OneHotEncoderString : public MLFunction {
 public:
  explicit OneHotEncoderString(
      std::vector<std::pair<std::string, int>> mapping) {
    int maxPos = 0;
    for (const auto& p : mapping) {
      valueToIndex_[p.first] = p.second;
      maxPos = std::max(maxPos, p.second);
    }
    numCategories_ = maxPos + 1;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    // 1) Prepare result vector
    BaseVector::ensureWritable(rows, outputType, context.pool(), result);

    // 2) Decode input VARCHAR column
    exec::LocalDecodedVector decoded(context, *args[0], rows);
    auto decodedVec = decoded.get();

    // 3) Build a [rows.size() × numCategories_] zero buffer
    auto totalRows = rows.size();
    std::vector<std::vector<float>> buffer(
        totalRows, std::vector<float>(numCategories_, 0.0f));

    // 4) Fill in the 1.0f for each selected row
    rows.applyToSelected([&](vector_size_t row) {
      auto strView = decodedVec->valueAt<StringView>(row);
      std::string key(strView.data(), strView.size());
      auto it = valueToIndex_.find(key);
      if (it != valueToIndex_.end()) {
        buffer[row][it->second] = 1.0f;
      }
    });

    // 5) Construct result
    VectorMaker maker{context.pool()};
    result = maker.arrayVector<float>(buffer, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
      exec::FunctionSignatureBuilder()
        .argumentType("VARCHAR")
        .returnType("ARRAY(REAL)")
        .build()
    };
  }

  static std::string name() { return "one_hot_string"; }

  float* getTensor() const override { return nullptr; }

  CostEstimate getCost(std::vector<int> inputDims) override {
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int numCategories_;
  std::unordered_map<std::string, int> valueToIndex_;
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