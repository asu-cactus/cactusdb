/**
 * @file
 * @brief Implementation of utility functions and classes for machine learning tasks.
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
#include <filesystem>
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
 * @class ChangeRating
 * @brief Implements a function to change ratings to binary values.
 */
class ChangeRating : public MLFunction {
 public:
  /**
   * @brief Default constructor for ChangeRating.
   */
  ChangeRating() {}

  /**
   * @brief Applies the function to change ratings to binary values.
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

    auto input = args[0];
    int* inputValues = input->as<FlatVector<int>>()->values()->asMutable<int>();

    int inputSize = rows.size();

    std::vector<int> result(rows.size());

    for (int i = 0; i < inputSize; i++) {
      result[i] = (inputValues[i] >= 3) ? 1 : 0;
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(result, INTEGER());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .returnType("INTEGER")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "change_rating";
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
   * @brief Sets the weight for the function.
   */
  void setWeight() {
    //
  }
};

/**
 * @class ConvertToIntArray
 * @brief Implements a function to convert an integer vector to an integer array.
 */
class ConvertToIntArray : public MLFunction {
 public:
  /**
   * @brief Default constructor for ConvertToIntArray.
   */
  ConvertToIntArray() {}

  /**
   * @brief Applies the function to convert an integer vector to an integer array.
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

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .returnType("array(INTEGER)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "convert_int_array";
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
   * @brief Sets the weight for the function.
   */
  void setWeight() {
    //
  }
};

/**
 * @class ConvertToFloatArray
 * @brief Implements a function to convert a float vector to a float array.
 */
class ConvertToFloatArray : public MLFunction {
 public:
  /**
   * @brief Default constructor for ConvertToFloatArray.
   */
  ConvertToFloatArray() {}

  /**
   * @brief Applies the function to convert a float vector to a float array.
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

    auto input = args[0];
    float* inputValues =
        input->as<FlatVector<float>>()->values()->asMutable<float>();

    int inputSize = rows.size();

    std::vector<std::vector<float>> result(rows.size(), std::vector<float>(1));

    for (int i = 0; i < inputSize; i++) {
      result[i][0] = inputValues[i];
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
                .argumentType("REAL")
                .returnType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "convert_float_array";
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
   * @brief Sets the weight for the function.
   */
  void setWeight() {
    //
  }
};

/**
 * @class ConvertDoubleToFloatArray
 * @brief Implements a function to convert a double vector to a float array.
 */
class ConvertDoubleToFloatArray : public MLFunction {
 public:
  /**
   * @brief Default constructor for ConvertDoubleToFloatArray.
   */
  ConvertDoubleToFloatArray() {}

  /**
   * @brief Applies the function to convert a double vector to a float array.
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

    auto input = args[0];
    double* inputValues =
        input->as<FlatVector<double>>()->values()->asMutable<double>();

    int inputSize = rows.size();

    std::vector<std::vector<float>> result(rows.size(), std::vector<float>(1));

    for (int i = 0; i < inputSize; i++) {
      result[i][0] = inputValues[i];
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
                .argumentType("DOUBLE")
                .returnType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "convert_double_to_float_array";
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
   * @brief Sets the weight for the function.
   */
  void setWeight() {
    //
  }
};

/**
 * @class ConvertDoubleArrayToFloatArray
 * @brief Implements a function to convert a double array to a float array.
 */
class ConvertDoubleArrayToFloatArray : public MLFunction {
 public:
  /**
   * @brief Default constructor for ConvertDoubleArrayToFloatArray.
   */
  ConvertDoubleArrayToFloatArray() {}

  /**
   * @brief Applies the function to convert a double array to a float array.
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

    double* inputValues = baseInputArray->values()->asMutable<double>();

    int numRawInput = rows.size();
    int numInput = rows.countSelected();
    int numElements = baseInputArray->size();
    int sizeOfArray = numElements / numInput;

    std::vector<std::vector<float>> result(
        numRawInput, std::vector<float>(sizeOfArray));
    int processedIndex = 0;
    for (int i = 0; i < numRawInput; i++) {
      if (!rows.isValid(i)) {
        // Skip invalid rows
        continue;
      }
      std::transform(
          inputValues + processedIndex * sizeOfArray,
          inputValues + (processedIndex + 1) * sizeOfArray,
          result[i].begin(),
          [](double val) { return static_cast<float>(val); });
      processedIndex++;
    }

    VectorMaker maker{context.pool()};
    auto localResult = maker.arrayVector<float>(result, REAL());
    context.moveOrCopyResult(localResult, rows, output);
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(DOUBLE)")
                .returnType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "convert_double_array_to_float_array";
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
   * @brief Sets the weight for the function.
   */
  void setWeight() {
    //
  }
};

/**
 * @brief Loads bytes from a file into a string.
 * @param path Path to the file.
 * @return A string containing the file's contents.
 */
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

/**
 * @brief Converts a string to a boolean value.
 * @param str The string to convert.
 * @return The boolean value corresponding to the string.
 */
bool stringToBool(const std::string& str) {
  std::string lowerStr = str;
  std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
  return (lowerStr == "true");
}

/**
 * @brief Retrieves the value of an environment variable.
 * @param key The name of the environment variable.
 * @return The value of the environment variable, or an empty string if not found.
 */
std::string getEnvVar(std::string const& key) {
  char const* val = getenv(key.c_str());
  return val == NULL ? std::string() : std::string(val);
}

/**
 * @brief Reads the number of rows and columns from a data statistics file.
 * @param path Path to the data statistics file.
 * @param numRows Reference to store the number of rows.
 * @param numCols Reference to store the number of columns.
 */
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

/**
 * @brief Flattens a 2D vector into a 1D array.
 * @tparam T The type of elements in the vector.
 * @param vec2D The 2D vector to flatten.
 * @param totalSize Reference to store the total size of the flattened array.
 * @return A pointer to the flattened array.
 */
template <typename T>
T* flattenVectorToPointer(
    const std::vector<std::vector<T>>& vec2D,
    size_t& totalSize) {
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

/**
 * @brief Flattens a 2D vector into a 1D array (overloaded version without totalSize).
 * @tparam T The type of elements in the vector.
 * @param vec2D The 2D vector to flatten.
 * @return A pointer to the flattened array.
 */
template <typename T>
T* flattenVectorToPointer(const std::vector<std::vector<T>>& vec2D) {
  size_t totalSize =
      0; // A local variable to hold the size if not provided by the caller
  return flattenVectorToPointer(vec2D, totalSize);
}

/**
 * @brief Counts the number of words in a string.
 * @param input The input string.
 * @return The number of words in the string.
 */
int countWords(const std::string& input) {
  std::istringstream stream(input);
  std::string word;
  int count = 0;

  while (stream >> word) {
    ++count;
  }

  return count;
}
