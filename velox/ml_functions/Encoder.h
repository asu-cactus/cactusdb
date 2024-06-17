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

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims){
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

    std::vector<std::vector<int>> result(numInputs, std::vector<int>(1));
    for (int i = 0; i < numInputs; i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      result[i][0] = mapping_.at(val.data());
    }
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

  CostEstimate getCost(std::vector<int> inputDims){
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::unordered_map<std::string, int> mapping_;
};

class StringVariadicEncoder : public MLFunction {
 public:
  StringVariadicEncoder(std::unordered_map<std::string, int> mapping) {
    mapping_ = mapping;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto arrayVector = args[0]->as<ArrayVector>();

    // Read string input
    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();
    int numInputs = rows.size();

    StringView* valVector =
        arrayVector->elements()->values()->asMutable<StringView>();
    std::vector<std::vector<int>> result;
    for (int i = 0; i < numInputs; i++) {
      int numSubIndices = arrayVector->sizeAt(i);
      int indicesOffset = arrayVector->offsetAt(i);
      std::vector<int> indices;
      for (int j = 0; j < numSubIndices; j++) {
        StringView val = valVector[indicesOffset + j];
        indices.push_back(mapping_.at(val.data()));
      }
      result.push_back(indices);
    }
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

  CostEstimate getCost(std::vector<int> inputDims){
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

    
      
      std::vector<std::vector<float>> encoding(numInputs, std::vector<float>(size_, 0));

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

  CostEstimate getCost(std::vector<int> inputDims){
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int size_;
};