#pragma once
#include <iostream>
#include "functions.h"
#include <Eigen/Dense>
#include <cmath>
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

class ChangeRating : public MLFunction {
 public:
  ChangeRating() {
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input = args[0];
    int* inputValues = input->as<FlatVector<int>>()->values()->asMutable<int>();

    // auto inputFeatures = args[0]->as<ArrayVector>()->elements();
    // int* inputValues = inputFeatures->values()->asMutable<int>();

    int inputSize = rows.size();

    std::vector<int> result(rows.size());

    for (int i = 0; i < inputSize; i++) {
        result[i] = (inputValues[i] >= 3) ? 1 : 0;      
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(result, INTEGER());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .returnType("INTEGER")
                .build()};
  }

  static std::string getName() {
    return "change_rating";
  };

  float* getTensor() const override {
    // FIXME
    return nullptr;
  }

  void setWeight() {
    //
  }


 private:
};

class ConvertToIntArray : public MLFunction {
 public:
  ConvertToIntArray() {
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input = args[0];
    int* inputValues = input->as<FlatVector<int>>()->values()->asMutable<int>();

    int inputSize = rows.size();

    std::vector<std::vector<int>> result(rows.size(), std::vector<int>(1));

    for (int i = 0; i < inputSize; i++) {
        result[i][0] = inputValues[i];
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<int>(result, INTEGER());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .returnType("array(INTEGER)")
                .build()};
  }

  static std::string getName() {
    return "convert_int_array";
  };

  float* getTensor() const override {
    // FIXME
    return nullptr;
  }

  void setWeight() {
    //
  }


 private:
};

class ConvertToFloatArray : public MLFunction {
 public:
  ConvertToFloatArray() {
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input = args[0];
    float* inputValues = input->as<FlatVector<float>>()->values()->asMutable<float>();

    int inputSize = rows.size();

    std::vector<std::vector<float>> result(rows.size(), std::vector<float>(1));

    for (int i = 0; i < inputSize; i++) {
        result[i][0] = inputValues[i];
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("REAL")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "convert_float_array";
  };

  float* getTensor() const override {
    // FIXME
    return nullptr;
  }

  void setWeight() {
    //
  }


 private:
};

class ConvertDoubleToFloatArray : public MLFunction {
 public:
  ConvertDoubleToFloatArray() {
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input = args[0];
    double* inputValues = input->as<FlatVector<double>>()->values()->asMutable<double>();

    int inputSize = rows.size();

    std::vector<std::vector<float>> result(rows.size(), std::vector<float>(1));

    for (int i = 0; i < inputSize; i++) {
        result[i][0] = inputValues[i];
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("DOUBLE")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "convert_double_to_float_array";
  };

  float* getTensor() const override {
    // FIXME
    return nullptr;
  }

  void setWeight() {
    //
  }


 private:
};

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}