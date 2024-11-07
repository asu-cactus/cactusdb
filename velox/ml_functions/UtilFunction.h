#pragma once
#include <iostream>
#include "functions.h"
#include <Eigen/Dense>
#include <filesystem>
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

class ConvertDoubleArrayToFloatArray : public MLFunction {
 public:
  ConvertDoubleArrayToFloatArray() {
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
    
    double* inputValues = baseInputArray->values()->asMutable<double>();

    // There is a tricky thing here, rows.size() return the number of raw inputs
    // while rows.countSelected() return the number of selected rows (after filtering)
    // the second one should be used for computation and the numElements is mapped to the
    // rows.countSelected() instead of rows.size(). If using the following code, the size
    // of the result vector should be mapped to the rows.size() otherwise the returned 
    // vector won't be aligned with the selected inputs. Another workaround is to use
    // rows.applyToSelected() to iterate over the selected rows for computation, which is
    // temporarily marked as #TODO.

    int numRawInput = rows.size();
    int numInput = rows.countSelected();
    int numElements = baseInputArray->size();
    int sizeOfArray = numElements / numInput;

    std::vector<std::vector<float>> result(numRawInput, std::vector<float>(sizeOfArray));
    int processedIndex = 0;
    for (int i = 0; i < numRawInput; i++) {
      if (!rows.isValid(i)) {
        // Skip invalid rows
        continue;
      }
      // inputValues only has the length equal to the number of selected rows, so we another
      // index to access the inputValues, which is processedIndex
      std::transform(inputValues + processedIndex * sizeOfArray, inputValues + (processedIndex + 1) * sizeOfArray, result[i].begin(),
                       [](double val) { return static_cast<float>(val); });
      processedIndex++;
    }

    VectorMaker maker{context.pool()};
    // output = maker.arrayVector<float>(result, REAL());
    auto localResult = maker.arrayVector<float>(result, REAL());
    context.moveOrCopyResult(localResult, rows, output);

    /*
    auto  arrayResult = output->as<ArrayVector>();
    auto sizes = arrayResult->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = arrayResult->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();
    auto elementsResult = arrayResult->elements();
    rows.applyToSelected([&](vector_size_t row) {
          rawSizes[row] = numArgs;
          rawOffsets[row] = offset;

          targetRows.setValid(offset, true);
          toSourceRow[offset] = row;

          offset += numArgs;
    }); 
    */
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(DOUBLE)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "convert_double_array_to_float_array";
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

bool stringToBool(const std::string& str) {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return (lowerStr == "true");
}

std::string getEnvVar(std::string const& key) {
  char const* val = getenv(key.c_str());
  return val == NULL ? std::string() : std::string(val);
}

void readDataStats(const std::string& path, int& numRows, int& numCols) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string line;
  std::getline(file, line);
  numRows = std::stoi(line);
  std::getline(file, line);
  numCols = std::stoi(line);
}

template <typename T>
T* flattenVectorToPointer(const std::vector<std::vector<T>>& vec2D, size_t& totalSize) {
    // Calculate total size in one pass
    totalSize = 0;
    for (const auto& row : vec2D) {
        totalSize += row.size();
    }

    // Allocate memory for the flattened array
    T* flatArray = new T[totalSize];

    // Flatten the 2D vector into the 1D array
    T* ptr = flatArray;
    for (const auto& row : vec2D) {
        std::copy(row.begin(), row.end(), ptr);
        ptr += row.size();
    }

    return flatArray;
}

// Overloaded version without totalSize (default)
template <typename T>
T* flattenVectorToPointer(const std::vector<std::vector<T>>& vec2D) {
    size_t totalSize = 0;  // A local variable to hold the size if not provided by the caller
    return flattenVectorToPointer(vec2D, totalSize);
}