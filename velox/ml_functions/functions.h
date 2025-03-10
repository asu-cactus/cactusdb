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

/**
 * @file functions.h
 * @brief Header file containing various machine learning functions and utilities.
 *
 * This file defines a collection of classes and functions for performing
 * machine learning operations such as matrix multiplication, addition,
 * activation functions, and more. It also includes utility functions for
 * cost estimation and tensor manipulation.
 */

#pragma once
#include <torch/torch.h>
#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include "BaseFunction.h"
#include "BatchNorm.h"
#include "ChatGPT.h"
#include "ComplexLayer.h"
#include "Concat.h"
#include "CosineSimilarity.h"
#include "DecisionForest.h"
#include "DecisionTree.h"
#include "DotProduct.h"
#include "Dropout.h"
#include "Embedding.h"
#include "Encoder.h"
#include "HuggingFaceServerless.h"
#include "HuggingFaceTokenizer.h"
#include "PositionEncoding.h"
#include "RAG.h"
#include "SequencePooling.h"
#include "XGBoost.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

/*
    TODO
    1. conv2d - done
    2. max pooling - done
    3. flatten - not required
    4. batch normalization
    5. padding
    6. concatenate
    7. embedding
    8. transformer -> exiting libraries, encoder, decoder, how to decompoose it
   into atomic linear algebra
    // focus on weight
    9. GRU -> not interesting
*/

// TODO: Refactor
// class MLFunction : public exec::VectorFunction {
//  public:
//   virtual ~MLFunction() = default;

//   virtual float* getTensor() const = 0;

//   virtual std::vector<int> getDims() {
//     return dims;
//   }

//   virtual std::string getFuncName() {
//     return "";
//   }

//   virtual int getNumDims() {
//     return dims.size();
//   }

//   virtual CostEstimate getCost(std::vector<int> inputDims) {
//     return CostEstimate(0, inputDims[0], inputDims[1]);
//   }

//  protected:
//   std::vector<int> dims;
//   double getWeightedCost(std::string name, float cost) {
//     std::vector<double> coefficient =
//         UdfCostCoefficient::getInstance().getCoefficient(name);
//     // FIXME
//     return 0;
//     // return coefficient[0] * cost;
//   }
//   std::vector<double> getCoefficientVector(std::string name) {
//     return UdfCostCoefficient::getInstance().getCoefficient(name);
//   }
// };

/**
 * @class MatrixMultiply
 * @brief Class for performing matrix multiplication.
 *
 * This class implements matrix multiplication and provides methods to apply the
 * operation, retrieve tensor data, and estimate computational cost.
 */
class MatrixMultiply : public MLFunction {
 public:
  /**
   * @brief Constructor for MatrixMultiply.
   * @param weights A pointer to the weight matrix.
   * @param num_rows The number of rows in the weight matrix.
   * @param num_cols The number of columns in the weight matrix.
   */
  MatrixMultiply(float* weights, int num_rows, int num_cols) {
    // Create a deep copy of the weights.
    weights_ = new float[num_rows * num_cols];
    std::memcpy(weights_, weights, num_rows * num_cols * sizeof(float));
    dims.push_back(num_rows);
    dims.push_back(num_cols);
  }

  /**
   * @brief Constructor for MatrixMultiply.
   * @param weightsFile The file containing the weight matrix.
   * @param num_rows The number of rows in the weight matrix.
   * @param num_cols The number of columns in the weight matrix.
   */
  MatrixMultiply(std::string weightsFile, int num_rows, int num_cols) {
    weightsFile_ = weightsFile;
    dims.push_back(num_rows);
    dims.push_back(num_cols);
  }

  /**
   * @brief Apply the matrix multiplication operation.
   * @param rows The selectivity vector indicating which rows to process.
   * @param args The input arguments.
   * @param outputType The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    bool use_gpu = false;
    if (args.size() == 2) {
      // An optional parameter can be passed to enable the GPU for matrix multiplication.
      use_gpu = args[1]->as<ConstantVector<bool>>()->valueAt(0);
    }
    if (use_gpu) {
      // TODO: Implement GPU matrix multiplication.
      throw std::runtime_error(
          "GPU implementation of Matrix Multiplication is not implemented.");
    } else {
      // Ensure output vector is writable.
      context.ensureWritable(rows, outputType, output);
      output->clearNulls(rows);
      auto arrayOutput = output->as<ArrayVector>();
      auto sizes = arrayOutput->mutableSizes(rows.end());
      auto rawSizes = sizes->asMutable<int32_t>();
      auto offsets = arrayOutput->mutableOffsets(rows.end());
      auto rawOffsets = offsets->asMutable<int32_t>();

      // Initialize sizes and offsets to zero.
      std::fill(rawSizes, rawSizes + rows.end(), 0);
      std::fill(rawOffsets, rawOffsets + rows.end(), 0);

      auto elementsOutput = arrayOutput->elements();
      auto elementsPool = context.pool();

      // Perform matrix multiplication logic.
      exec::DecodedArgs decodedArgs(rows, args, context);
      auto decodedInput = decodedArgs.at(0);
      auto inputArray = decodedInput->base()->as<ArrayVector>();
      auto inputElements = inputArray->elements();
      float* inputValues = inputElements->values()->asMutable<float>();
      auto inputOffsets = inputArray->rawOffsets();
      auto inputSizes = inputArray->rawSizes();

      // The map between the row index in the input data and the row index in the output data.
      std::map<vector_size_t, vector_size_t> rowMap;
      // For efficient check.
      std::unordered_set<vector_size_t> uniqueRawIndexeSet;
      // For iterating over the insert ordering.
      std::vector<vector_size_t> uniqueRawIndexeVector;
      vector_size_t numUniqueRows = 0;
      rows.applyToSelected([&](vector_size_t row) {
        auto mappedIndexInRowData = decodedInput->index(row);
        if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
            uniqueRawIndexeSet.end()) {
          // Add it.
          rowMap[row] = numUniqueRows;
          uniqueRawIndexeSet.insert(mappedIndexInRowData);
          uniqueRawIndexeVector.push_back(mappedIndexInRowData);
          ++numUniqueRows;
        } else {
          // Already added.
          rowMap[row] = rowMap[mappedIndexInRowData];
        }
      });

      int numInputMatrixRows = numUniqueRows;
      Eigen::MatrixXf inputMatrix(numInputMatrixRows, dims[0]);
      int rowIndex = 0;
      for (auto rawIndex : uniqueRawIndexeVector) {
        Eigen::Map<const Eigen::VectorXf> rowVector(
            inputValues + inputOffsets[rawIndex], dims[0]);
        inputMatrix.row(rowIndex++) = rowVector;
      }

      Eigen::Map<
          Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          weightMatrix(weights_, dims[0], dims[1]);
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
          resultMatrix = inputMatrix * weightMatrix;

      // Append results to the output vector.
      auto baseOffset = elementsOutput->size();
      elementsOutput->resize(baseOffset + rows.end() * dims[1]);

      float* outputValues = elementsOutput->values()->asMutable<float>();
      vector_size_t outputOffset = 0;
      rows.applyToSelected([&](vector_size_t row) {
        if (rowMap.find(row) == rowMap.end()) {
          throw std::runtime_error(
              "Mapped index not found for the result matrix.");
        }
        auto mappedIndexInResultMatrix = rowMap[row];
        rawOffsets[row] = outputOffset;
        rawSizes[row] = dims[1];
        std::memcpy(
            outputValues + outputOffset,
            resultMatrix.row(mappedIndexInResultMatrix).data(),
            dims[1] * sizeof(float));

        outputOffset += dims[1];
      });
      arrayOutput->setElements(elementsOutput);
    }
  }

  /**
   * @brief Get the function signatures for matrix multiplication.
   * @return A vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .returnType("array(REAL)")
            .argumentType("array(REAL)")
            .build(),
        // Supports an additional flag: use_gpu.
        exec::FunctionSignatureBuilder()
            .returnType("array(REAL)")
            .argumentType("array(REAL)")
            .argumentType("BOOLEAN")
            .build()};
  }

  /**
   * @brief Get the tensor data associated with this function.
   * @return A pointer to the tensor data.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  static std::string getName() {
    return "mat_mul";
  };

  /**
   * @brief Get the weights file associated with this function.
   * @return A string representing the weights file path.
   */
  std::string getWeightsFile() {
    return weightsFile_;
  }

  /**
   * @brief Set the weights for this function.
   * @param weights A pointer to the weight matrix.
   */
  void setWeights(float* weights) {
    weights_ = weights;
  }

  /**
   * @brief Estimate the computational cost of the function.
   * @param inputDims The dimensions of the input data.
   * @return A CostEstimate object representing the estimated cost.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    std::vector<double> coefficientVector = getCoefficientVector(getName());
    int factor1 = inputDims[0];
    int factor2 = dims[0];
    int factor3 = dims[1];
    float cost = coefficientVector[0] * factor1 * factor2 * factor3 +
        coefficientVector[1] * factor1 + coefficientVector[2] * factor2 +
        coefficientVector[3] * factor3;
    return CostEstimate(cost, inputDims[0], dims[1]);
  }

 private:
  float* weights_; ///< Pointer to the weight matrix.
  std::string weightsFile_; ///< Path to the weights file.
};

/**
 * @class MatrixMultiply_b
 * @brief Class for performing blocked matrix multiplication.
 *
 * This class implements matrix multiplication in a blocked manner, which is useful
 * for optimizing performance on large matrices.
 */
class MatrixMultiply_b : public MLFunction {
 public:
  /**
   * @brief Constructor for MatrixMultiply_b.
   * @param num_rows The number of rows in the matrix.
   * @param num_cols The number of columns in the matrix.
   * @param num_samples The number of samples.
   * @param blocks The number of blocks for partitioning the matrix.
   */
  MatrixMultiply_b(int num_rows, int num_cols, int num_samples, int blocks) {
    dims.push_back(num_rows);
    dims.push_back(num_cols);
    dims.push_back(num_samples);
    dims.push_back(blocks);
  }

  /**
   * @brief Apply the blocked matrix multiplication operation.
   * @param rows The selectivity vector indicating which rows to process.
   * @param args The input arguments.
   * @param type The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    VectorMaker maker{context.pool()};

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = rightHolder->base()->as<ArrayVector>()->elements();

    float* input_values_v = baseLeftArray->values()->asMutable<float>();
    float* input_values_w = baseRightArray->values()->asMutable<float>();

    // auto varrayVector = std::make_shared<ArrayVector<float>>();
    // const int elements_v_per_row = 1500000; //6000*250
    // const int elements_w_per_row = 125000; // 250*500

    // std::vector<std::vector<float>> result(1,
    // std::vector<float>(dims[1]*dims[2])); //6000*500
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m1(input_values_v, dims[2], dims[0]); // 3*2
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m2(input_values_w, dims[0], dims[1]); // 2*5
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m =
        m1 * m2; // 3*5

    // for (int i = 0; i < m.rows(); ++i) {
    //         for (int j = 0; j < m.cols(); ++j) {
    //             result[0][i * dims[1] + j] = m(i, j);
    //     }
    // }
    // m = m.reshaped(1, m.size());
    // std::cout << "shape: " << m.rows() << "," <<m.cols() << std::endl;
    std::vector<std::vector<float>> result;
    for (int i = 0; i < m.rows(); i++) {
      std::vector<float> row(m.row(i).data(), m.row(i).data() + m.cols());
      result.push_back(row);
    }
    auto baseVector = maker.arrayVector<float>(result, REAL());
    auto arrayOfArrays = maker.arrayVector({0}, baseVector);
    output = arrayOfArrays;
  }

  /**
   * @brief Get the function signatures for blocked matrix multiplication.
   * @return A vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("array(array(REAL))")
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .build()};
  }

  /**
   * @brief Get the tensor data associated with this function.
   * @return A pointer to the tensor data.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  static std::string getName() {
    return "mat_mul_block";
  };

 private:
  float* weights_; ///< Pointer to the weight matrix.
};


/**
 * @class MatrixMultiply_h
 * @brief Class for performing matrix multiplication with a hierarchical approach.
 *
 * This class implements matrix multiplication using a hierarchical approach,
 * which is useful for optimizing performance on large matrices by breaking
 * the computation into smaller blocks.
 */
class MatrixMultiply_h : public MLFunction {
 public:
  /**
   * @brief Constructor for MatrixMultiply_h.
   * @param num_rows The number of rows in the matrix.
   * @param num_cols The number of columns in the matrix.
   * @param block_size The size of each block for hierarchical computation.
   */
  MatrixMultiply_h(int num_rows, int num_cols, int block_size) {
    dims.push_back(num_rows);
    dims.push_back(num_cols);
    dims.push_back(block_size);
  }

  /**
   * @brief Apply the hierarchical matrix multiplication operation.
   * @param rows The selectivity vector indicating which rows to process.
   * @param args The input arguments.
   * @param outputType The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, outputType, context.pool(), output);
    output->clearNulls(rows);
    auto arrayOutput = output->as<ArrayVector>();
    auto sizes = arrayOutput->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = arrayOutput->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();

    // Initialize sizes and offsets to zero.
    std::fill(rawSizes, rawSizes + rows.end(), 0);
    std::fill(rawOffsets, rawOffsets + rows.end(), 0);

    auto elementsOutput = arrayOutput->elements();
    auto elementsPool = context.pool();
    VectorMaker maker{context.pool()};

    // Validate input arguments
    VELOX_CHECK_EQ(
        args.size(), 2, "Blocked-based matrix multiply requires 2 inputs");

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto numInputs = rows.size();
    auto decodedInput1 = decodedArgs.at(0);
    auto decodedInput2 = decodedArgs.at(1);
    auto input1Array = decodedInput1->base()->as<ArrayVector>();
    auto input2Array = decodedInput2->base()->as<ArrayVector>();
    auto input1Elements = input1Array->elements();
    auto input1Offsets = input1Array->rawOffsets();
    auto input1Sizes = input1Array->rawSizes();
    auto input2Elements = input2Array->elements();

    float* input1Values = input1Elements->values()->asMutable<float>();
    float* input2Values = input2Elements->values()->asMutable<float>();

    int currentBlockSize = (input2Elements->size() < (dims[0] * dims[2]))
        ? input2Elements->size() / dims[0]
        : dims[2];
    int input1MatrixNumRow = input1Elements->size() / dims[0];

    std::map<vector_size_t, vector_size_t> rowMap;
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput1->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // Add it.
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
      } else {
        // Already added.
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, dims[0]);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          input1Values + input1Offsets[rawIndex], dims[0]);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        weightMatrix(input2Values, dims[0], currentBlockSize);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMatrix = inputMatrix * weightMatrix;

    auto baseOffset = elementsOutput->size();
    elementsOutput->resize(baseOffset + rows.end() * currentBlockSize);

    float* outputValues = elementsOutput->values()->asMutable<float>();

    vector_size_t outputOffset = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (rowMap.find(row) == rowMap.end()) {
        throw std::runtime_error(
            "Mapped index not found for the result matrix.");
      }
      auto mappedIndexInResultMatrix = rowMap[row];
      rawOffsets[row] = outputOffset;
      rawSizes[row] = currentBlockSize;
      std::memcpy(
          outputValues + outputOffset,
          resultMatrix.row(mappedIndexInResultMatrix).data(),
          currentBlockSize * sizeof(float));
      outputOffset += currentBlockSize;
    });
    arrayOutput->setElements(elementsOutput);
  }

  /**
   * @brief Get the function signatures for hierarchical matrix multiplication.
   * @return A vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("array(REAL)")
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .build()};
  }

  /**
   * @brief Get the tensor data associated with this function.
   * @return A pointer to the tensor data.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  static std::string getName() {
    return "mat_mul_h";
  };

  /**
   * @brief Estimate the computational cost of the function.
   * @param inputDims The dimensions of the input data.
   * @return A CostEstimate object representing the estimated cost.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    std::vector<double> coefficientVector = getCoefficientVector("mat_mul");
    int factor1 = inputDims[0];
    int factor2 = inputDims[1];
    int factor3 = dims[2];
    float cost = coefficientVector[0] * factor1 * factor2 * factor3 +
        coefficientVector[1] * factor1 + coefficientVector[2] * factor2 +
        coefficientVector[3] * factor3;
    return CostEstimate(cost, inputDims[0], dims[2]);
  }

 private:
  float* weights_; ///< Pointer to the weight matrix.
};
/**
 * @class MatrixMultiply_Block
 * @brief Class for performing blocked matrix multiplication.
 *
 * This class implements matrix multiplication using a blocked approach,
 * which is useful for optimizing performance on large matrices by breaking
 * the computation into smaller blocks.
 */
class MatrixMultiply_Block : public MLFunction {
 public:
  /**
   * @brief Constructor for MatrixMultiply_Block.
   * @param num_rows The number of rows in the matrix.
   * @param num_cols The number of columns in the matrix.
   * @param num_samples The number of samples.
   * @param blocks The number of blocks for partitioning the matrix.
   */
  MatrixMultiply_Block(
      int num_rows,
      int num_cols,
      int num_samples,
      int blocks) {
    dims.push_back(num_rows);
    dims.push_back(num_cols);
    dims.push_back(num_samples);
    dims.push_back(blocks);
  }

  /**
   * @brief Apply the blocked matrix multiplication operation.
   * @param rows The selectivity vector indicating which rows to process.
   * @param args The input arguments.
   * @param type The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    auto elementType =
        ArrayType(std::make_shared<ArrayType>(ArrayType(REAL())));
    BaseVector::ensureWritable(
        rows, std::make_shared<ArrayType>(elementType), context.pool(), output);
    VectorMaker maker{context.pool()};

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = rightHolder->base()->as<ArrayVector>()->elements();

    float* input_values_v = baseLeftArray->values()->asMutable<float>();
    float* input_values_w = baseRightArray->values()->asMutable<float>();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m1(input_values_v, dims[2], dims[0]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m2(input_values_w, dims[0], dims[1]);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m =
        m1 * m2;

    std::vector<std::vector<float>> result;
    for (int i = 0; i < m.rows(); i++) {
      std::vector<float> row(m.row(i).data(), m.row(i).data() + m.cols());
      result.push_back(row);
    }
    auto baseVector = maker.arrayVector<float>(result, REAL());
    auto arrayOfArrays = maker.arrayVector({0}, baseVector);
    output = arrayOfArrays;
  }

  /**
   * @brief Get the function signatures for blocked matrix multiplication.
   * @return A vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("array(array(REAL))")
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .build()};
  }

  /**
   * @brief Get the tensor data associated with this function.
   * @return A pointer to the tensor data.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  static std::string getName() {
    return "mat_mul_block";
  };

 private:
  float* weights_; ///< Pointer to the weight matrix.
};
/**
 * @class MatrixAddition
 * @brief Class for performing matrix addition.
 *
 * This class implements matrix addition, which adds two matrices element-wise.
 * It supports both in-place addition and addition with a weight matrix.
 */
class MatrixAddition : public MLFunction {
 public:
  /**
   * @brief Constructor for MatrixAddition.
   * @param weights A pointer to the weight matrix.
   * @param num_cols The number of columns in the matrix.
   */
  MatrixAddition(float* weights, int num_cols) {
    weights_ = weights;
    dims.push_back(num_cols);
  }

  /**
   * @brief Constructor for MatrixAddition.
   * @param weightsFile The file containing the weight matrix.
   * @param num_cols The number of columns in the matrix.
   */
  MatrixAddition(std::string weightsFile, int num_cols) {
    weightsFile_ = weightsFile;
    dims.push_back(num_cols);
  }

  /**
   * @brief Apply the matrix addition operation.
   * @param rows The selectivity vector indicating which rows to process.
   * @param args The input arguments.
   * @param type The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m1(input_values, rows.size(), dims[0]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m2(weights_, rows.size(), dims[0]);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m =
        m1 + m2;

    std::vector<std::vector<float>> result;
    for (int i = 0; i < m.rows(); i++) {
      std::vector<float> row(m.row(i).data(), m.row(i).data() + m.cols());
      result.push_back(row);
    }
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
  }

  /**
   * @brief Get the function signatures for matrix addition.
   * @return A vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("array(REAL)")
                .argumentType("array(REAL)")
                .build()};
  }

  /**
   * @brief Get the tensor data associated with this function.
   * @return A pointer to the tensor data.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Get the name of the function.
   * @return A string representing the function name.
   */
  static std::string getName() {
    return "mat_add";
  };

  /**
   * @brief Get the weights file associated with this function.
   * @return A string representing the weights file path.
   */
  std::string getWeightsFile() {
    return weightsFile_;
  }

  /**
   * @brief Set the weights for this function.
   * @param weights A pointer to the weight matrix.
   */
  void setWeights(float* weights) {
    weights_ = weights;
  }

  /**
   * @brief Estimate the computational cost of the function.
   * @param inputDims The dimensions of the input data.
   * @return A CostEstimate object representing the estimated cost.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    std::vector<double> coefficientVector = getCoefficientVector(getName());
    float cost = coefficientVector[0] * inputDims[0] * inputDims[1];
    return CostEstimate(cost, inputDims[0], inputDims[1]);
  }

 private:
  float* weights_; ///< Pointer to the weight matrix.
  std::string weightsFile_; ///< Path to the weights file.
};
/**
 * @class MatrixVectorAddition
 * @brief A class that performs matrix-vector addition, inheriting from MLFunction.
 *
 * This class provides functionality to add a vector to each row of a matrix.
 * It supports initialization with either a raw array of weights or a file containing weights.
 * The `apply` method performs the matrix-vector addition operation and writes the result to the output vector.
 */
class MatrixVectorAddition : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the class with a raw array of weights.
     *
     * @param weights A pointer to a float array containing the weights (vector values).
     * @param num_cols The number of columns in the matrix (and size of the vector).
     */
    MatrixVectorAddition(float* weights, int num_cols) {
        // Create a deep copy of the weights
        weights_ = new float[num_cols];
        std::memcpy(weights_, weights, num_cols * sizeof(float));
        dims.push_back(num_cols);
    }

    /**
     * @brief Constructor that initializes the class with a file containing weights.
     *
     * @param weightsFile The path to the file containing the weights.
     * @param num_cols The number of columns in the matrix (and size of the vector).
     */
    MatrixVectorAddition(std::string weightsFile, int num_cols) {
        weightsFile_ = weightsFile;
        dims.push_back(num_cols);
    }

    /**
     * @brief Applies the matrix-vector addition operation.
     *
     * This method performs the matrix-vector addition for the selected rows and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input matrix).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    output->clearNulls(rows);
    auto arrayOutput = output->as<ArrayVector>();
    auto sizes = arrayOutput->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = arrayOutput->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();

    // Initialize sizes and offsets to zero.
    std::fill(rawSizes, rawSizes + rows.end(), 0);
    std::fill(rawOffsets, rawOffsets + rows.end(), 0);
    auto elementsOutput = arrayOutput->elements();
    auto elementsPool = context.pool();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();
    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::map<vector_size_t, vector_size_t> rowMap;
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, dims[0]);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], dims[0]);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        vectorMatrix(weights_, 1, dims[0]);

    inputMatrix.rowwise() += vectorMatrix.row(0);

    auto baseOffset = elementsOutput->size();
    elementsOutput->resize(baseOffset + rows.end() * dims[0]);
    float* outputValues = elementsOutput->values()->asMutable<float>();

    vector_size_t outputOffset = 0;

    rows.applyToSelected([&](vector_size_t row) {
      if (rowMap.find(row) == rowMap.end()) {
        throw std::runtime_error(
            "Mapped index not found for the result matrix.");
      }
      auto mappedIndexInResultMatrix = rowMap[row];
      rawOffsets[row] = outputOffset;
      rawSizes[row] = dims[0];

      std::memcpy(
          outputValues + outputOffset,
          inputMatrix.row(mappedIndexInResultMatrix).data(),
          dims[0] * sizeof(float));

      outputOffset += dims[0];
    });
    arrayOutput->setElements(elementsOutput);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("array(REAL)")
                .argumentType("array(REAL)")
                .build()};
  }

  float* getTensor() const override {
    return weights_;
  }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor (weights) associated with this function.
     *
     * @return A pointer to the float array containing the weights.
     */
    float* getTensor() const override {
        return weights_;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    };

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string.
     */
    static std::string getName() {
        return "mat_add";
    };

    /**
     * @brief Returns the path to the weights file.
     *
     * @return The path to the weights file as a string.
     */
    std::string getWeightsFile() {
        return weightsFile_;
    }

    /**
     * @brief Sets the weights for the function.
     *
     * @param weights A pointer to a float array containing the new weights.
     */
    void setWeights(float* weights) {
        weights_ = weights;
    }

private:
    float* weights_;          ///< Pointer to the weights (vector values).
    std::string weightsFile_; ///< Path to the file containing the weights.
    std::vector<int> dims;    ///< Dimensions of the matrix (e.g., number of columns).
};
/**
 * @class Sigmoid
 * @brief A class that implements the Sigmoid activation function, inheriting from MLFunction.
 *
 * The Sigmoid function maps input values to a range between 0 and 1. This class provides functionality
 * to apply the Sigmoid function element-wise to an input array and produce an output array.
 */
class Sigmoid : public MLFunction {
public:
    /**
     * @brief Default constructor.
     */
    Sigmoid() {}

    /**
     * @brief Computes the Sigmoid function for a single input value.
     *
     * @param x The input value.
     * @return The result of the Sigmoid function: 1.0f / (1.0f + exp(-x)).
     */
    static float sigmoidFunction(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * @brief Applies the Sigmoid function to the input array.
     *
     * This method processes the input array, applies the Sigmoid function element-wise, and stores the result
     * in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();

    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::vector<std::vector<float>> result(numRows);

    rows.applyToSelected([&](vector_size_t i) {
      size_t mappedIndexInRowData = decodedInput->index(i);
      size_t dataSize = inputSizes[mappedIndexInRowData];
      size_t dataOffset = inputOffsets[mappedIndexInRowData];
      std::vector<float> rowResult(dataSize);
      std::transform(
          inputValues + dataOffset,
          inputValues + dataOffset + dataSize,
          rowResult.data(),
          sigmoidFunction);
      result[i] = rowResult;
    });
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for Sigmoid).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("sigmoid").
     */
    static std::string getName() {
        return "sigmoid";
    }

    /**
     * @brief Estimates the computational cost of applying the Sigmoid function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        std::vector<double> coefficientVector = getCoefficientVector(getName());
        float cost = coefficientVector[0] * inputDims[0] * inputDims[1];
        return CostEstimate(cost, inputDims[0], inputDims[1]);
    }
};
/**
 * @class Relu
 * @brief A class that implements the Rectified Linear Unit (ReLU) activation function, inheriting from MLFunction.
 *
 * The ReLU function returns the input value if it is positive; otherwise, it returns 0. This class provides
 * functionality to apply the ReLU function element-wise to an input array and produce an output array.
 */
class Relu : public MLFunction {
public:
    /**
     * @brief Default constructor.
     */
    Relu() {}

    /**
     * @brief Computes the ReLU function for a single input value.
     *
     * @param x The input value.
     * @return The result of the ReLU function: max(0, x).
     */
    static float reluFunction(float x) {
        return (x > 0.0f) ? x : 0.0f;
    }

    /**
     * @brief Applies the ReLU function to the input array.
     *
     * This method processes the input array, applies the ReLU function element-wise, and stores the result
     * in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();

    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::vector<std::vector<float>> result(numRows);

    rows.applyToSelected([&](vector_size_t i) {
      size_t mappedIndexInRowData = decodedInput->index(i);
      size_t dataSize = inputSizes[mappedIndexInRowData];
      size_t dataOffset = inputOffsets[mappedIndexInRowData];
      std::vector<float> rowResult(dataSize);
      std::transform(
          inputValues + dataOffset,
          inputValues + dataOffset + dataSize,
          rowResult.data(),
          reluFunction);
      result[i] = rowResult;
    });
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for ReLU).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("relu").
     */
    static std::string getName() {
        return "relu";
    }

    /**
     * @brief Estimates the computational cost of applying the ReLU function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        std::vector<double> coefficientVector = getCoefficientVector(getName());
        float cost = coefficientVector[0] * inputDims[0] * inputDims[1];
        return CostEstimate(cost, inputDims[0], inputDims[1]);
    }
};

/**
 * @class Softmax
 * @brief A class that implements the Softmax activation function, inheriting from MLFunction.
 *
 * The Softmax function converts a vector of values into a probability distribution, where the values sum to 1.
 * This class provides functionality to apply the Softmax function to an input array and produce an output array.
 */
class Softmax : public MLFunction {
public:
    /**
     * @brief Default constructor.
     */
    Softmax() {}

    /**
     * @brief Applies the Softmax function to the input array.
     *
     * This method processes the input array, applies the Softmax function, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    output->clearNulls(rows);
    auto arrayOutput = output->as<ArrayVector>();
    auto sizes = arrayOutput->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = arrayOutput->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();

    // Initialize sizes and offsets to zero.
    std::fill(rawSizes, rawSizes + rows.end(), 0);
    std::fill(rawOffsets, rawOffsets + rows.end(), 0);
    auto elementsOutput = arrayOutput->elements();
    auto elementsPool = context.pool();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();
    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::map<vector_size_t, vector_size_t> rowMap;
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    int numCols;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
        numCols = inputSizes[mappedIndexInRowData];
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, numCols);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], numCols);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    Eigen::ArrayXXf exp = inputMatrix.array().exp();
    Eigen::ArrayXXf sum = exp.rowwise().sum();
    for (int i = 0; i < exp.rows(); i++) {
      exp.row(i) /= sum(i);
    }

    auto baseOffset = elementsOutput->size();
    elementsOutput->resize(baseOffset + rows.end() * numCols);
    float* outputValues = elementsOutput->values()->asMutable<float>();
    vector_size_t outputOffset = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (rowMap.find(row) == rowMap.end()) {
        throw std::runtime_error(
            "Mapped index not found for the result matrix.");
      }
      auto mappedIndexInResultMatrix = rowMap[row];
      rawOffsets[row] = outputOffset;
      rawSizes[row] = numCols;

      std::memcpy(
          outputValues + outputOffset,
          exp.row(mappedIndexInResultMatrix).data(),
          numCols * sizeof(float));

      outputOffset += numCols;
    });

    arrayOutput->setElements(elementsOutput);
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for Softmax).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("softmax").
     */
    static std::string getName() {
        return "softmax";
    }

    /**
     * @brief Estimates the computational cost of applying the Softmax function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        std::vector<double> coefficientVector = getCoefficientVector(getName());
        float cost = coefficientVector[0] * inputDims[0] * inputDims[1];
        return CostEstimate(cost, inputDims[0], inputDims[1]);
    }
};
/**
 * @class Argmax
 * @brief A class that implements the Argmax function, inheriting from MLFunction.
 *
 * The Argmax function returns the index of the maximum value in a vector. This class provides functionality
 * to apply the Argmax function to an input array and produce an output array of indices.
 */
class Argmax : public MLFunction {
public:
    /**
     * @brief Default constructor.
     */
    Argmax() {}

    /**
     * @brief Applies the Argmax function to the input array.
     *
     * This method processes the input array, applies the Argmax function, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    auto arrayOutput = output->asFlatVector<int>();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();

    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::map<vector_size_t, vector_size_t> rowMap;
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    int numCols;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
        numCols = inputSizes[mappedIndexInRowData];
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, numCols);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], numCols);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    std::map<vector_size_t, vector_size_t> argmaxMap;
    for (int i = 0; i < inputMatrix.rows(); i++) {
      Eigen::Index maxRow, maxCol;
      inputMatrix.row(i).maxCoeff(&maxRow, &maxCol);
      argmaxMap[i] = maxCol;
    }

    int* outputValues = arrayOutput->mutableRawValues<int>();
    vector_size_t outputOffset = 0;
    std::unordered_map<int, int> valueCounts;
    rows.applyToSelected([&](vector_size_t row) {
      if (rowMap.find(row) == rowMap.end()) {
        throw std::runtime_error(
            "Mapped index not found for the result matrix.");
      }
      auto mappedIndexInResultMatrix = rowMap[row];
      outputValues[row] = argmaxMap[mappedIndexInResultMatrix];
      valueCounts[outputValues[row]]++;
    });

    for (const auto& pair : valueCounts) {
      LOG(INFO) << "[INFO] Label Distributions: Key: " << pair.first
                << ", Value: " << pair.second << std::endl;
    }
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("INTEGER")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for Argmax).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("argmax").
     */
    static std::string getName() {
        return "argmax";
    }

    /**
     * @brief Estimates the computational cost of applying the Argmax function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        std::vector<double> coefficientVector = getCoefficientVector(getName());
        float cost = coefficientVector[0] * inputDims[0] * inputDims[1];
        return CostEstimate(cost, inputDims[0], inputDims[1]);
    }
};
/**
 * @class MinMaxScaler
 * @brief A class that implements Min-Max scaling, inheriting from MLFunction.
 *
 * Min-Max scaling normalizes input data to a specified range (typically [0, 1]) using the formula:
 * \[
 * X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
 * \]
 * This class supports initialization with either raw arrays of min/max values or a file containing these values.
 */
class MinMaxScaler : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the scaler with raw arrays of min and max values.
     *
     * @param scalerMinValues A pointer to a float array containing the minimum values for each feature.
     * @param scalerMaxValues A pointer to a float array containing the maximum values for each feature.
     * @param numCols The number of features (columns) in the input data.
     */
    MinMaxScaler(float* scalerMinValues, float* scalerMaxValues, int numCols) {
        scalerMinValues_ = new float[numCols];
        scalerMaxValues_ = new float[numCols];
        std::memcpy(scalerMinValues_, scalerMinValues, numCols * sizeof(float));
        std::memcpy(scalerMaxValues_, scalerMaxValues, numCols * sizeof(float));
        numCols_ = numCols;
    }

    /**
     * @brief Constructor that initializes the scaler with a file containing min and max values.
     *
     * @param minMaxScalerDataPath The path to the file containing min and max values.
     */
    MinMaxScaler(std::string minMaxScalerDataPath) {
    std::vector<float> scalerMinVector;
    std::vector<float> scalerMaxVector;

    if (!std::filesystem::exists(minMaxScalerDataPath)) {
      throw std::runtime_error("File not found: " + minMaxScalerDataPath);
    }
    std::ifstream file(minMaxScalerDataPath);
    std::string line;
    // Read each line from the file
    int lineCount = 0;
    while (std::getline(file, line)) {
      std::istringstream iss(line); // Create a string stream from the line
      float value;

      // Read each value from the line
      // First line should be min values
      // Second line should be max values
      while (iss >> value) {
        if (lineCount == 0) {
          scalerMinVector.push_back(value); // Store the value in tempValues
        } else if (lineCount == 1) {
          scalerMaxVector.push_back(value); // Store the value in tempValues
        } else {
          throw std::runtime_error(
              "Invalid file format, parsed lineCount: " +
              std::to_string(lineCount));
        }
      }
      lineCount++;
    }
    file.close(); // Close the file
    // the size should be equal
    assert(scalerMinVector.size() == scalerMaxVector.size());
    numCols_ = scalerMinVector.size();

    scalerMinValues_ = new float[numCols_];
    scalerMaxValues_ = new float[numCols_];
    std::memcpy(
        scalerMinValues_, scalerMinVector.data(), numCols_ * sizeof(float));
    std::memcpy(
        scalerMaxValues_, scalerMaxVector.data(), numCols_ * sizeof(float));
    }

    /**
     * @brief Applies Min-Max scaling to the input array.
     *
     * This method processes the input array, applies Min-Max scaling, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    output->clearNulls(rows);
    auto arrayOutput = output->as<ArrayVector>();
    auto sizes = arrayOutput->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = arrayOutput->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();

    // Initialize sizes and offsets to zero.
    std::fill(rawSizes, rawSizes + rows.end(), 0);
    std::fill(rawOffsets, rawOffsets + rows.end(), 0);
    auto elementsOutput = arrayOutput->elements();
    auto elementsPool = context.pool();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();
    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::map<vector_size_t, vector_size_t> rowMap;
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    int numCols = numCols_;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, numCols);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], numCols);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        minVals(scalerMinValues_, 1, numCols);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        maxVals(scalerMaxValues_, 1, numCols);
    Eigen::MatrixXf resultMatrix =
        (inputMatrix.rowwise() - minVals.row(0)).array().rowwise() /
        (maxVals.row(0) - minVals.row(0)).array();

    auto baseOffset = elementsOutput->size();
    elementsOutput->resize(baseOffset + rows.end() * numCols);
    float* outputValues = elementsOutput->values()->asMutable<float>();
    vector_size_t outputOffset = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (rowMap.find(row) == rowMap.end()) {
        throw std::runtime_error(
            "Mapped index not found for the result matrix.");
      }
      auto mappedIndexInResultMatrix = rowMap[row];
      rawOffsets[row] = outputOffset;
      rawSizes[row] = numCols;

      std::memcpy(
          outputValues + outputOffset,
          resultMatrix.row(mappedIndexInResultMatrix).data(),
          numCols * sizeof(float));

      outputOffset += numCols;
    });
    arrayOutput->setElements(elementsOutput);
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for MinMaxScaler).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("min_max_scaler").
     */
    static std::string getName() {
        return "min_max_scaler";
    }

    /**
     * @brief Estimates the computational cost of applying Min-Max scaling.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        std::vector<double> coefficientVector = getCoefficientVector(getName());
        float cost = coefficientVector[0] * inputDims[0] * inputDims[1];
        return CostEstimate(cost, inputDims[0], inputDims[1]);
    }

private:
    float* scalerMinValues_; ///< Pointer to the array of minimum values.
    float* scalerMaxValues_; ///< Pointer to the array of maximum values.
    int numCols_;            ///< Number of features (columns) in the input data.
};
/**
 * @class TorchDNN2Level
 * @brief A class that implements a 2-level deep neural network using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a 2-level dense neural network with ReLU activation and softmax output.
 */
class TorchDNN2Level : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the neural network with weights and biases.
     *
     * @param weights A pointer to an array of pointers to weight matrices.
     * @param bias A pointer to an array of pointers to bias vectors.
     * @param dimensions A vector containing the dimensions of the neural network layers.
     */
    TorchDNN2Level(float** weights, float** bias, std::vector<int> dimensions) {
        this->weights = weights;
        this->bias = bias;
        dims = dimensions;
    }

    /**
     * @brief Applies the 2-level neural network to the input array.
     *
     * This method processes the input array, applies the neural network, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    torch::nn::Linear dense1(dims[0], dims[1]);
    torch::nn::Linear dense2(dims[1], dims[2]);
    torch::nn::ReLU relu;

    torch::Tensor weightTensor1 =
        torch::from_blob(weights[0], {dims[0], dims[1]}).t();
    torch::Tensor weightTensor2 =
        torch::from_blob(weights[1], {dims[1], dims[2]}).t();
    torch::Tensor bias1 = torch::from_blob(bias[0], {dims[1]});
    torch::Tensor bias2 = torch::from_blob(bias[1], {dims[2]});

    dense1->weight.set_data(weightTensor1);
    dense2->weight.set_data(weightTensor2);
    dense1->bias.set_data(bias1);
    dense2->bias.set_data(bias2);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();
    int input_size = input_elements->size();

    torch::Tensor input =
        torch::from_blob(input_values, {rows.size(), dims[0]});

    torch::Tensor layer1_output = dense1->forward(input);
    torch::Tensor reluOutput = relu->forward(layer1_output);
    torch::Tensor layer2_output = dense2->forward(reluOutput);
    torch::Tensor softmax_output =
        torch::nn::functional::softmax(layer2_output, 1);
    float* data = softmax_output.data_ptr<float>();

    std::vector<std::vector<float>> results;
    for (int i = 0; i < rows.size(); ++i) {
      // std::vector<float> result;
      std::vector<float> result(data + i * dims[2], data + (i + 1) * dims[2]);
      // for (int j = 0; j < dims[2]; ++j) {
      //     result.push_back(data[i*dims[2] + j]);
      // }
      results.push_back(result);
    }
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for TorchDNN2Level).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the weights of the neural network.
     *
     * @return A pointer to an array of pointers to weight matrices.
     */
    float** getWeights() const {
        return weights;
    }

    /**
     * @brief Returns the biases of the neural network.
     *
     * @return A pointer to an array of pointers to bias vectors.
     */
    float** getBias() const {
        return bias;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("torch_dnn").
     */
    static std::string getName() {
        return "torch_dnn";
    }

    /**
     * @brief Estimates the computational cost of applying the neural network.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        float cost = getWeightedCost(
            getName(), inputDims[0] * inputDims[1] * dims[0] * dims[1]);
        return CostEstimate(cost, inputDims[0], inputDims[1]);
    }

private:
    float** weights; ///< Pointer to an array of pointers to weight matrices.
    float** bias;    ///< Pointer to an array of pointers to bias vectors.
    std::vector<int> dims; ///< Dimensions of the neural network layers.
};
/**
 * @class TorchDNN
 * @brief A class that implements a deep neural network using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a multi-layer neural network with ReLU activations and softmax output.
 */
class TorchDNN : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the neural network with weights, biases, and layer dimensions.
     *
     * @param weights A vector of pointers to weight matrices for each layer.
     * @param bias A vector of pointers to bias vectors for each layer.
     * @param dimensions A vector containing the dimensions of the neural network layers.
     */
    TorchDNN(
        std::vector<float*> weights,
        std::vector<float*> bias,
        std::vector<int> dimensions) {
        this->weights = weights;
        this->bias = bias;
        dims = dimensions;
    }

    /**
     * @brief Applies the neural network to the input array.
     *
     * This method processes the input array, applies the neural network, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    std::vector<torch::nn::Linear> dense_layers;
    std::vector<torch::Tensor> weights_tensors;
    std::vector<torch::Tensor> bias_tensors;
    std::vector<torch::nn::ReLU> relus;

    // Create layers
    for (int i = 0; i < dims.size() - 1; ++i) {
      dense_layers.push_back(torch::nn::Linear(dims[i], dims[i + 1]));
      weights_tensors.push_back(
          torch::from_blob(weights[i], {dims[i], dims[i + 1]}).t());
      bias_tensors.push_back(torch::from_blob(bias[i], {dims[i + 1]}));
      relus.push_back(torch::nn::ReLU());
    }

    // Set weights and biases
    for (int i = 0; i < dense_layers.size(); ++i) {
      dense_layers[i]->weight.set_data(weights_tensors[i]);
      dense_layers[i]->bias.set_data(bias_tensors[i]);
    }

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();
    torch::Tensor input =
        torch::from_blob(input_values, {rows.size(), dims[0]});

    torch::Tensor output_tensor = input;
    for (int i = 0; i < dense_layers.size(); ++i) {
      output_tensor = dense_layers[i]->forward(output_tensor);
      output_tensor = relus[i]->forward(output_tensor);
    }

    // Softmax output
    output_tensor = torch::nn::functional::softmax(output_tensor, 1);
    float* data = output_tensor.data_ptr<float>();

    // Prepare results
    std::vector<std::vector<float>> results;
    for (int i = 0; i < rows.size(); ++i) {
      std::vector<float> result(
          data + i * dims.back(), data + (i + 1) * dims.back());
      results.push_back(result);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for TorchDNN).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the weights of the neural network.
     *
     * @return A vector of pointers to weight matrices.
     */
    const std::vector<float*>& getWeights() const {
        return weights;
    }

    /**
     * @brief Returns the biases of the neural network.
     *
     * @return A vector of pointers to bias vectors.
     */
    const std::vector<float*>& getBias() const {
        return bias;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("torchnn").
     */
    static std::string getName() {
        return "torchnn";
    }

    /**
     * @brief Estimates the computational cost of applying the neural network.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
    std::vector<double> coefficientVector = getCoefficientVector(getName());
    uint64_t factor1 = inputDims[0] * dims[0] * dims[1];
    uint64_t factor2 = inputDims[0] * dims[1] * dims[2];
    uint64_t factor3 = dims[0] * dims[1];
    uint64_t factor4 = dims[1] * dims[2];
    float cost = coefficientVector[0] * factor1 +
        coefficientVector[1] * factor2 + coefficientVector[2] * factor3 +
        coefficientVector[3] * factor4 + coefficientVector[4] * inputDims[0] +
        coefficientVector[5] * dims[0] + coefficientVector[6] * dims[1] +
        coefficientVector[7] * dims[2];
    // LOG(INFO) << fmt::format("[DEBUG] 4 values: {}, {}, {}, {}",inputDims[0],
    // inputDims[1], dims[0], dims[1]); LOG(INFO) << fmt::format("[DEBUG] coeff:
    // {}",coefficientVector); LOG(INFO) << fmt::format("[DEBUG] Cost
    // Computation: {}, {}, {}, {}, {}, {}, {}, {}", coefficientVector[0] *
    // factor1, coefficientVector[1] * factor2, coefficientVector[2] * factor3
    //             , coefficientVector[3] * factor4,  coefficientVector[4] *
    //             inputDims[0] , coefficientVector[5] * dims[0],
    //             coefficientVector[6] * dims[1] , coefficientVector[7] *
    //             dims[2]);
    // LOG(INFO) << fmt::format("[DEBUG] compute debug: {}, {}, {}, {}, {}",
    // inputDims[0], inputDims[0]*dims[1], inputDims[0]*dims[1]*dims[2],
    // factor2, coefficientVector[1] * factor2);

    return CostEstimate(cost, inputDims[0], dims[2]);
    }

private:
    std::vector<float*> weights; ///< Vector of pointers to weight matrices.
    std::vector<float*> bias;    ///< Vector of pointers to bias vectors.
    std::vector<int> dims;       ///< Dimensions of the neural network layers.
};
/**
 * @namespace velox::dl
 * @brief Namespace for deep learning-related utilities and kernels.
 */
namespace velox::dl {

/**
 * @enum KernelType
 * @brief Enumeration of kernel types used in deep learning operations.
 */
enum class KernelType {
  MatMul,   ///< Matrix multiplication kernel.
  MatAdd,   ///< Matrix addition kernel.
  ReLU,     ///< Rectified Linear Unit activation kernel.
  Softmax,  ///< Softmax activation kernel.
  BatchNorm,///< Batch normalization kernel.
  Argmax,   ///< Argmax operation kernel.
  Sigmoid   ///< Sigmoid activation kernel.
};

/**
 * @brief Converts a KernelType enum value to its string representation.
 * @param kernelType The kernel type to convert.
 * @return A string representing the kernel type.
 */
std::string kernelTypeToString(KernelType kernelType) {
  switch (kernelType) {
    case KernelType::MatMul:
      return "MatMul";
    case KernelType::MatAdd:
      return "MatAdd";
    case KernelType::ReLU:
      return "ReLU";
    case KernelType::Softmax:
      return "Softmax";
    case KernelType::BatchNorm:
      return "BatchNorm";
    case KernelType::Argmax:
      return "Argmax";
    case KernelType::Sigmoid:
      return "Sigmoid";
    default:
      return "Unknown";
  }
}

/**
 * @brief Overloads the `<<` operator for KernelType.
 * @param os The output stream.
 * @param kernelType The kernel type to stream.
 * @return The output stream with the kernel type string representation.
 */
std::ostream& operator<<(std::ostream& os, KernelType kernelType) {
  switch (kernelType) {
    case KernelType::MatMul:
      return os << "MatMul";
    case KernelType::MatAdd:
      return os << "MatAdd";
    case KernelType::ReLU:
      return os << "ReLU";
    case KernelType::Softmax:
      return os << "Softmax";
    case KernelType::BatchNorm:
      return os << "BatchNorm";
    case KernelType::Argmax:
      return os << "Argmax";
    default:
      return os << "Unknown";
  }
}

} // namespace velox::dl
/**
 * @class TorchDNNV2
 * @brief A class that implements a configurable deep neural network using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a neural network with configurable layers (e.g., MatMul, ReLU, Softmax, etc.).
 */
class TorchDNNV2 : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the neural network with kernel types, weights, and dimensions.
     *
     * @param kernelTypes A vector of KernelType specifying the types of layers in the network.
     * @param weights A vector of pointers to weight matrices for each layer.
     * @param dimensions A vector containing the dimensions of the neural network layers.
     */
    TorchDNNV2(
        std::vector<velox::dl::KernelType> kernelTypes,
        std::vector<float*> weights,
        std::vector<int> dimensions) {
        this->weights = weights;
        dims = dimensions;
        kernelTypes_ = kernelTypes;
    int numOps = kernelTypes.size();
    int weightIdx = 0;
    hasArgmax_ = false;
    model_ = torch::nn::Sequential();
    if (2 * numOps != dims.size()) {
      throw std::runtime_error(fmt::format(
          "Mismatched number of  2*kernel types and dimensions: {} vs {}",
          2 * numOps,
          dims.size()));
    }
    assert(2 * numOps == dims.size());
    for (int i = 0; i < numOps; ++i) {
      if (kernelTypes[i] == velox::dl::KernelType::MatMul &&
          kernelTypes[i + 1] == velox::dl::KernelType::MatAdd) {
        auto denseLayer = torch::nn::Linear(dims[2 * i], dims[2 * i + 1]);
        denseLayer->weight.set_data(
            torch::from_blob(
                weights[weightIdx++], {dims[2 * i], dims[2 * i + 1]})
                .t());
        denseLayer->bias.set_data(
            torch::from_blob(weights[weightIdx++], {dims[2 * i + 1]}));
        model_->push_back(denseLayer);
      } else if (kernelTypes[i] == velox::dl::KernelType::MatAdd) {
        // Do nothing, which is handled by creating a Dense Layer in the above
        // code
      } else if (kernelTypes[i] == velox::dl::KernelType::BatchNorm) {
        auto batchNormLayer = torch::nn::BatchNorm1d(dims[2 * i]);
        batchNormLayer->weight.set_data(
            torch::from_blob(weights[weightIdx++], {dims[2 * i + 1]}));
        batchNormLayer->bias.set_data(
            torch::from_blob(weights[weightIdx++], {dims[2 * i + 1]}));
        model_->push_back(batchNormLayer);
      } else if (kernelTypes[i] == velox::dl::KernelType::ReLU) {
        model_->push_back(torch::nn::ReLU());
      } else if (kernelTypes[i] == velox::dl::KernelType::Sigmoid) {
        model_->push_back(torch::nn::Sigmoid());
      } else if (kernelTypes[i] == velox::dl::KernelType::Softmax) {
        model_->push_back(torch::nn::Softmax(1));
      } else if (kernelTypes[i] == velox::dl::KernelType::Argmax) {
        model_->push_back(LibTorchArgmaxKernel(1));
        hasArgmax_ = true;
      } else {
        throw std::runtime_error(fmt::format(
            "Unsupported kernel type of TorchDNNV2: {}", kernelTypes[i]));
      }
    }
    // enable evaluation mode, this is required for inference, otherwise some
    // module could failed, like dropout, batchnorm, etc.
    model_->eval();
    }

    /**
     * @brief Applies the neural network to the input array.
     *
     * This method processes the input array, applies the neural network, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    context.ensureWritable(rows, type, output);
    output->clearNulls(rows);

    // Perform matrix multiplication logic.
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    // The map between the row index in the input data and the row index in
    // the output data.
    std::map<vector_size_t, vector_size_t> rowMap;
    // for efficient check
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    // for iterating over the insert ordering
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, dims[0]);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], dims[0]);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    float* inputValues1 = inputMatrix.data();

    torch::Tensor input =
        torch::from_blob(inputValues1, {numUniqueRows, dims[0]});
    torch::Tensor output_tensor = input;

    output_tensor =
        const_cast<torch::nn::Sequential&>(model_)->forward(output_tensor);
    // Append results to the output vector.
    if (hasArgmax_) {
      auto arrayOutput = output->asFlatVector<int>();
      int* outputValues = arrayOutput->mutableRawValues<int>();
      auto int_tensor = output_tensor.to(torch::kInt);
      int* dataInt = int_tensor.data_ptr<int>();

      rows.applyToSelected([&](vector_size_t row) {
        if (rowMap.find(row) == rowMap.end()) {
          throw std::runtime_error(
              "Mapped index not found for the result matrix.");
        }
        auto mappedIndexInResultMatrix = rowMap[row];
        outputValues[row] = dataInt[mappedIndexInResultMatrix];
      });
    } else {
      auto arrayOutput = output->as<ArrayVector>();
      auto sizes = arrayOutput->mutableSizes(rows.end());
      auto rawSizes = sizes->asMutable<int32_t>();
      auto offsets = arrayOutput->mutableOffsets(rows.end());
      auto rawOffsets = offsets->asMutable<int32_t>();

      // Initialize sizes and offsets to zero.
      std::fill(rawSizes, rawSizes + rows.end(), 0);
      std::fill(rawOffsets, rawOffsets + rows.end(), 0);

      auto elementsOutput = arrayOutput->elements();
      auto elementsPool = context.pool();
      auto baseOffset = elementsOutput->size();
      elementsOutput->resize(baseOffset + rows.end() * dims.back());

      float* outputValues = elementsOutput->values()->asMutable<float>();
      vector_size_t outputOffset = 0;
      float* dataFloat = output_tensor.data_ptr<float>();

      rows.applyToSelected([&](vector_size_t row) {
        if (rowMap.find(row) == rowMap.end()) {
          throw std::runtime_error(
              "Mapped index not found for the result matrix.");
        }
        auto mappedIndexInResultMatrix = rowMap.at(row);
        rawOffsets[row] = outputOffset;
        rawSizes[row] = dims.back();
        std::memcpy(
            outputValues + outputOffset,
            dataFloat + mappedIndexInResultMatrix * dims.back(),
            dims.back() * sizeof(float));
        outputOffset += dims.back();
      });
      arrayOutput->setElements(elementsOutput);
    }
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {
            exec::FunctionSignatureBuilder()
                .returnType("array(REAL)")
                .argumentType("array(REAL)")
                .build(),
            exec::FunctionSignatureBuilder()
                .returnType("INTEGER")
                .argumentType("array(REAL)")
                .argumentType("INTEGER")
                .build(),
            exec::FunctionSignatureBuilder()
                .returnType("INTEGER")
                .argumentType("array(REAL)")
                .argumentType("BIGINT")
                .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for TorchDNNV2).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the weights of the neural network.
     *
     * @return A vector of pointers to weight matrices.
     */
    const std::vector<float*>& getWeights() const {
        return weights;
    }

    /**
     * @brief Returns the biases of the neural network.
     *
     * @return A vector of pointers to bias vectors.
     */
    const std::vector<float*>& getBias() const {
        return bias;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("complexTorchNN").
     */
    static std::string getName() {
        return "complexTorchNN";
    }

    /**
     * @brief Returns the kernel types used in the neural network.
     *
     * @return A vector of KernelType specifying the types of layers.
     */
    std::vector<velox::dl::KernelType> getKernelTypes() const {
        return kernelTypes_;
    }

    /**
     * @brief Estimates the computational cost of applying the neural network.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

private:
    std::vector<float*> weights; ///< Vector of pointers to weight matrices.
    std::vector<float*> bias;    ///< Vector of pointers to bias vectors.
    std::vector<velox::dl::KernelType> kernelTypes_; ///< Types of layers in the network.
    bool hasArgmax_; ///< Flag indicating if the network includes an Argmax layer.
    torch::nn::Sequential model_; ///< PyTorch sequential model representing the neural network.
};
/**
 * @class TorchDNNV2CUDA
 * @brief A class that implements a configurable deep neural network using PyTorch with CUDA support, inheriting from MLFunction.
 *
 * This class provides functionality to apply a neural network with configurable layers (e.g., MatMul, ReLU, Softmax, etc.) on a CUDA device.
 */
class TorchDNNV2CUDA : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the neural network with kernel types, weights, and dimensions.
     *
     * @param kernelTypes A vector of KernelType specifying the types of layers in the network.
     * @param weights A vector of pointers to weight matrices for each layer.
     * @param dimensions A vector containing the dimensions of the neural network layers.
     */
    TorchDNNV2CUDA(
        std::vector<velox::dl::KernelType> kernelTypes,
        std::vector<float*> weights,
        std::vector<int> dimensions) {
        device_ = "cuda:0"; // Initialize CUDA device.
        this->weights = weights;
        dims = dimensions;
        kernelTypes_ = kernelTypes;
    int numOps = kernelTypes.size();
    int weightIdx = 0;
    hasArgmax_ = false;
    model_ = torch::nn::Sequential();
    assert(2 * numOps == dims.size());
    for (int i = 0; i < numOps; ++i) {
      if (kernelTypes[i] == velox::dl::KernelType::MatMul &&
          kernelTypes[i + 1] == velox::dl::KernelType::MatAdd) {
        auto denseLayer = torch::nn::Linear(dims[2 * i], dims[2 * i + 1]);
        denseLayer->weight.set_data(
            torch::from_blob(
                weights[weightIdx++], {dims[2 * i], dims[2 * i + 1]})
                .t());
        denseLayer->bias.set_data(
            torch::from_blob(weights[weightIdx++], {dims[2 * i + 1]}));
        model_->push_back(denseLayer);
      } else if (kernelTypes[i] == velox::dl::KernelType::MatAdd) {
        // Do nothing, which is handled by creating a Dense Layer in the above
        // code
      } else if (kernelTypes[i] == velox::dl::KernelType::BatchNorm) {
        auto batchNormLayer = torch::nn::BatchNorm1d(dims[2 * i]);
        batchNormLayer->weight.set_data(
            torch::from_blob(weights[weightIdx++], {dims[2 * i + 1]}));
        batchNormLayer->bias.set_data(
            torch::from_blob(weights[weightIdx++], {dims[2 * i + 1]}));
        model_->push_back(batchNormLayer);
      } else if (kernelTypes[i] == velox::dl::KernelType::ReLU) {
        model_->push_back(torch::nn::ReLU());
      } else if (kernelTypes[i] == velox::dl::KernelType::Sigmoid) {
        model_->push_back(torch::nn::Sigmoid());
      } else if (kernelTypes[i] == velox::dl::KernelType::Softmax) {
        model_->push_back(torch::nn::Softmax(1));
      } else if (kernelTypes[i] == velox::dl::KernelType::Argmax) {
        model_->push_back(LibTorchArgmaxKernel(1));
        hasArgmax_ = true;
      } else {
        throw std::runtime_error(fmt::format(
            "Unsupported kernel type of TorchDNNV2: {}", kernelTypes[i]));
      }
    }
    // enable evaluation mode, this is required for inference, otherwise some
    // module could failed, like dropout, batchnorm, etc.
    model_->to(device_);
    model_->eval();
    }

    /**
     * @brief Applies the neural network to the input array using CUDA.
     *
     * This method processes the input array, applies the neural network on a CUDA device, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    context.ensureWritable(rows, type, output);
    output->clearNulls(rows);

    // Perform matrix multiplication logic.
    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    // The map between the row index in the input data and the row index in
    // the output data.
    std::map<vector_size_t, vector_size_t> rowMap;
    // for efficient check
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    // for iterating over the insert ordering
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, dims[0]);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], dims[0]);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    float* inputValues1 = inputMatrix.data();

    torch::Tensor input =
        torch::from_blob(inputValues1, {numUniqueRows, dims[0]});
    torch::Tensor output_tensor = input;
    output_tensor = output_tensor.to(device_);

    output_tensor =
        const_cast<torch::nn::Sequential&>(model_)->forward(output_tensor);
    output_tensor = output_tensor.to(torch::kCPU);
    // Append results to the output vector.

    if (hasArgmax_) {
      auto arrayOutput = output->asFlatVector<int>();
      int* outputValues = arrayOutput->mutableRawValues<int>();
      auto int_tensor = output_tensor.to(torch::kInt);
      int* dataInt = int_tensor.data_ptr<int>();

      rows.applyToSelected([&](vector_size_t row) {
        if (rowMap.find(row) == rowMap.end()) {
          throw std::runtime_error(
              "Mapped index not found for the result matrix.");
        }
        auto mappedIndexInResultMatrix = rowMap[row];
        outputValues[row] = dataInt[mappedIndexInResultMatrix];
      });
    } else {
      auto arrayOutput = output->as<ArrayVector>();
      auto sizes = arrayOutput->mutableSizes(rows.end());
      auto rawSizes = sizes->asMutable<int32_t>();
      auto offsets = arrayOutput->mutableOffsets(rows.end());
      auto rawOffsets = offsets->asMutable<int32_t>();

      // Initialize sizes and offsets to zero.
      std::fill(rawSizes, rawSizes + rows.end(), 0);
      std::fill(rawOffsets, rawOffsets + rows.end(), 0);

      auto elementsOutput = arrayOutput->elements();
      auto elementsPool = context.pool();
      auto baseOffset = elementsOutput->size();
      elementsOutput->resize(baseOffset + rows.end() * dims.back());
      float* outputValues = elementsOutput->values()->asMutable<float>();
      vector_size_t outputOffset = 0;

      float* dataFloat = output_tensor.data_ptr<float>();

      rows.applyToSelected([&](vector_size_t row) {
        if (rowMap.find(row) == rowMap.end()) {
          throw std::runtime_error(
              "Mapped index not found for the result matrix.");
        }
        auto mappedIndexInResultMatrix = rowMap[row];
        rawOffsets[row] = outputOffset;
        rawSizes[row] = dims.back();
        std::memcpy(
            outputValues + outputOffset,
            dataFloat + mappedIndexInResultMatrix * dims.back(),
            dims.back() * sizeof(float));
        outputOffset += dims.back();
      });
      arrayOutput->setElements(elementsOutput);
    }
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {
            exec::FunctionSignatureBuilder()
                .returnType("array(REAL)")
                .argumentType("array(REAL)")
                .build(),
            exec::FunctionSignatureBuilder()
                .returnType("INTEGER")
                .argumentType("array(REAL)")
                .argumentType("INTEGER")
                .build(),
            exec::FunctionSignatureBuilder()
                .returnType("INTEGER")
                .argumentType("array(REAL)")
                .argumentType("BIGINT")
                .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for TorchDNNV2CUDA).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the weights of the neural network.
     *
     * @return A vector of pointers to weight matrices.
     */
    const std::vector<float*>& getWeights() const {
        return weights;
    }

    /**
     * @brief Returns the biases of the neural network.
     *
     * @return A vector of pointers to bias vectors.
     */
    const std::vector<float*>& getBias() const {
        return bias;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("complexTorchNN_GPU").
     */
    static std::string getName() {
        return "complexTorchNN_GPU";
    }

    /**
     * @brief Returns the kernel types used in the neural network.
     *
     * @return A vector of KernelType specifying the types of layers.
     */
    std::vector<velox::dl::KernelType> getKernelTypes() const {
        return kernelTypes_;
    }

    /**
     * @brief Estimates the computational cost of applying the neural network.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

private:
    std::vector<float*> weights; ///< Vector of pointers to weight matrices.
    std::vector<float*> bias;    ///< Vector of pointers to bias vectors.
    std::vector<velox::dl::KernelType> kernelTypes_; ///< Types of layers in the network.
    bool hasArgmax_; ///< Flag indicating if the network includes an Argmax layer.
    std::string device_; ///< CUDA device identifier (e.g., "cuda:0").
    torch::nn::Sequential model_; ///< PyTorch sequential model representing the neural network.
};
/**
 * @class TorchDNNKernel
 * @brief A class that implements a single-layer neural network kernel using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a single-layer neural network with configurable kernel types (e.g., Dense, ReLU, Softmax).
 */
class TorchDNNKernel : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the neural network kernel with weights, biases, and dimensions.
     *
     * @param kernel The type of kernel (e.g., "Dense", "ReLU", "Softmax").
     * @param weights A pointer to the weight matrix.
     * @param bias A pointer to the bias vector.
     * @param dimensions A vector containing the dimensions of the neural network layer.
     */
    TorchDNNKernel(
        std::string kernel,
        float* weights,
        float* bias,
        std::vector<int> dimensions) {
        this->kernel = kernel;
        this->weights = weights;
        this->bias = bias;
        dims = dimensions;
    }

    /**
     * @brief Applies the neural network kernel to the input array.
     *
     * This method processes the input array, applies the neural network kernel, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    std::vector<torch::nn::Linear> dense_layers;
    std::vector<torch::Tensor> weights_tensors;
    std::vector<torch::Tensor> bias_tensors;
    std::vector<torch::nn::ReLU> relus;

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();
    torch::Tensor input =
        torch::from_blob(input_values, {rows.size(), dims[0]});
    torch::Tensor output_tensor = input;

    if (kernel == "Dense") {
      torch::nn::Linear denseLayer = torch::nn::Linear(dims[0], dims[1]);
      torch::Tensor weightsTensor =
          torch::from_blob(weights, {dims[0], dims[1]}).t();
      torch::Tensor biasTensor = torch::from_blob(bias, {dims[1]});
      denseLayer->weight.set_data(weightsTensor);
      denseLayer->bias.set_data(biasTensor);

      output_tensor = denseLayer->forward(output_tensor);
    } else if (kernel == "Relu") {
      torch::nn::ReLU reluLayer = torch::nn::ReLU();

      output_tensor = reluLayer->forward(output_tensor);
    } else if (kernel == "Softmax") {
      output_tensor = torch::nn::functional::softmax(output_tensor, 1);
    }

    // output_tensor = torch::nn::functional::softmax(output_tensor, 1);
    float* data = output_tensor.data_ptr<float>();

    // Prepare results
    std::vector<std::vector<float>> results;
    for (int i = 0; i < rows.size(); ++i) {
      std::vector<float> result(
          data + i * dims.back(), data + (i + 1) * dims.back());
      results.push_back(result);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for TorchDNNKernel).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the weights of the neural network kernel.
     *
     * @return A pointer to the weight matrix.
     */
    const float* getWeights() const {
        return weights;
    }

    /**
     * @brief Returns the biases of the neural network kernel.
     *
     * @return A pointer to the bias vector.
     */
    const float* getBias() const {
        return bias;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("torchnn_kernel").
     */
    static std::string getName() {
        return "torchnn_kernel";
    }

private:
    float* weights; ///< Pointer to the weight matrix.
    float* bias;    ///< Pointer to the bias vector.
    std::string kernel; ///< Type of kernel (e.g., "Dense", "ReLU", "Softmax").
    std::vector<int> dims; ///< Dimensions of the neural network layer.
};

/**
 * @class TorchDNN_Multi
 * @brief A class that implements a multi-layer deep neural network using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a multi-layer neural network with ReLU activations and softmax output.
 */
class TorchDNN_Multi : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the neural network with weights, biases, and layer dimensions.
     *
     * @param weights A vector of pointers to weight matrices for each layer.
     * @param bias A vector of pointers to bias vectors for each layer.
     * @param dimensions A vector containing the dimensions of the neural network layers.
     */
    TorchDNN_Multi(
        std::vector<float*> weights,
        std::vector<float*> bias,
        std::vector<int> dimensions) {
        this->weights = weights;
        this->bias = bias;
        dims = dimensions;
    }

    /**
     * @brief Applies the multi-layer neural network to the input array.
     *
     * This method processes the input array, applies the neural network, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    std::vector<torch::nn::Linear> dense_layers;
    std::vector<torch::Tensor> weights_tensors;
    std::vector<torch::Tensor> bias_tensors;
    std::vector<torch::nn::ReLU> relus;

    // Create layers
    for (int i = 0; i < dims.size() - 1; ++i) {
      dense_layers.push_back(torch::nn::Linear(dims[i], dims[i + 1]));
      weights_tensors.push_back(
          torch::from_blob(weights[i], {dims[i], dims[i + 1]}).t());
      bias_tensors.push_back(torch::from_blob(bias[i], {dims[i + 1]}));
      relus.push_back(torch::nn::ReLU());
    }

    // Set weights and biases
    for (int i = 0; i < dense_layers.size(); ++i) {
      dense_layers[i]->weight.set_data(weights_tensors[i]);
      dense_layers[i]->bias.set_data(bias_tensors[i]);
    }

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();
    torch::Tensor input =
        torch::from_blob(input_values, {rows.size(), dims[0]});

    torch::Tensor output_tensor = input;
    for (int i = 0; i < dense_layers.size(); ++i) {
      output_tensor = dense_layers[i]->forward(output_tensor);
      output_tensor = relus[i]->forward(output_tensor);
    }

    // Softmax output
    output_tensor = torch::nn::functional::softmax(output_tensor, 1);
    float* data = output_tensor.data_ptr<float>();

    // Prepare results
    std::vector<std::vector<float>> results;
    for (int i = 0; i < rows.size(); ++i) {
      std::vector<float> result(
          data + i * dims.back(), data + (i + 1) * dims.back());
      results.push_back(result);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for TorchDNN_Multi).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the weights of the neural network.
     *
     * @return A vector of pointers to weight matrices.
     */
    const std::vector<float*>& getWeights() const {
        return weights;
    }

    /**
     * @brief Returns the biases of the neural network.
     *
     * @return A vector of pointers to bias vectors.
     */
    const std::vector<float*>& getBias() const {
        return bias;
    }

private:
    std::vector<float*> weights; ///< Vector of pointers to weight matrices.
    std::vector<float*> bias;    ///< Vector of pointers to bias vectors.
    std::vector<int> dims;       ///< Dimensions of the neural network layers.
};
/**
 * @class Convolute
 * @brief A class for performing 2D convolution operations as part of a machine learning function.
 *
 * This class implements a 2D convolution operation using Eigen for matrix operations.
 * It supports multi-channel inputs and filters, and produces multi-channel outputs.
 */
class Convolute : public MLFunction {
 public:
  /**
   * @brief Constructs a new Convolute object.
   * @param weights A pointer to the filter weights.
   * @param dims_ An array of dimensions describing the filters and input:
   *              - dims_[0]: Number of filters.
   *              - dims_[1]: Filter height.
   *              - dims_[2]: Filter width.
   *              - dims_[3]: Number of input channels.
   *              - dims_[4]: Input height.
   *              - dims_[5]: Input width.
   */
  Convolute(float* weights, int* dims_) {
    weights_ = weights;
    for (int i = 0; i < 6; i++)
      dims.push_back(dims_[i]);
  }

  /**
   * @brief Applies the 2D convolution operation to the input data.
   * @param rows A SelectivityVector indicating which rows to process.
   * @param args A vector of input arguments (only the first argument is used).
   * @param type The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector where the results will be stored.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();

    int input_height = dims[4];
    int input_width = dims[5];
    int input_channel_size = input_height * input_width;
    int input_size = input_channel_size * dims[3];

    int filter_channel_size = dims[1] * dims[2];
    int filter_size = filter_channel_size * dims[3];

    int output_height = input_height - dims[1] + 1;
    int output_width = input_width - dims[2] + 1;

    std::vector<std::vector<float>> results(
        rows.size(),
        std::vector<float>(output_height * output_width * dims[0]));

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    for (int s = 0; s < rows.size(); s++) {
      // for each channel
      for (int c = 0; c < dims[3]; c++) {
        Eigen::Map<
            Eigen::
                Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            input(
                input_values + s * input_size + c * input_channel_size,
                input_height,
                input_width);
        // for every filter
        for (int f = 0; f < dims[0]; f++) {
          int filter_offset = f * output_height * output_width;
          Eigen::Map<Eigen::Matrix<
              float,
              Eigen::Dynamic,
              Eigen::Dynamic,
              Eigen::RowMajor>>
              kernel(
                  weights_ + f * filter_size + c * filter_channel_size,
                  dims[1],
                  dims[2]);
          for (int i = 0; i < output_height; ++i) {
            int offset = filter_offset + i * output_width;
            for (int j = 0; j < output_width; ++j) {
              results[s][offset + j] +=
                  (input.block(i, j, dims[1], dims[2]).cwiseProduct(kernel))
                      .sum();
            }
          }
        }
      }
    }

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Time for conv2d (sec) = "
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << std::endl;

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  /**
   * @brief Returns the function signatures for the convolution operation.
   * @return A vector of shared pointers to FunctionSignature objects.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("array(REAL)")
                .argumentType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the filter weights tensor.
   * @return A pointer to the filter weights.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Returns the name of the function.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return "conv2d";
  };

  /**
   * @brief Returns the name of the function.
   * @return A string representing the function name.
   */
  static std::string getName() {
    return "conv2d";
  };

 private:
  float* weights_; ///< Pointer to the filter weights.
  std::vector<int> dims; ///< Dimensions of the filters and input.
};
/**
 * @class TorchConvolute
 * @brief A class that implements a 2D convolution operation using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a 2D convolution operation to an input array using PyTorch.
 */
class TorchConvolute : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the convolution operation with weights and dimensions.
     *
     * @param weights A pointer to the weight matrix for the convolution.
     * @param dims_ An array containing the dimensions of the convolution operation.
     */
    TorchConvolute(float* weights, int* dims_) {
        weights_ = weights;
        for (int i = 0; i < 6; i++)
            dims.push_back(dims_[i]);
    }

    /**
     * @brief Applies the 2D convolution operation to the input array using PyTorch.
     *
     * This method processes the input array, applies the convolution, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();

    int input_height = dims[4];
    int input_width = dims[5];

    int output_height = input_height - dims[1] + 1;
    int output_width = input_width - dims[2] + 1;

    std::vector<std::vector<float>> results(
        rows.size(),
        std::vector<float>(output_height * output_width * dims[0]));

    torch::nn::Conv2d conv_layer(
        torch::nn::Conv2dOptions(dims[3], dims[0], {dims[1], dims[2]}));
    // torch::Tensor conv_weights = torch::tensor(weights_).view({dims[3],
    // dims[0], dims[1], dims[2]});

    // conv_layer->weight = torch::nn::parameter::Parameter (conv_weights);
    torch::Tensor input_data = torch::from_blob(
        input_values, {rows.size(), dims[3], input_height, input_width});

    torch::Tensor output_data = conv_layer(input_data);

    float* data = output_data.data_ptr<float>();

    int row_size = output_height * output_width * dims[0];

    for (int i = 0; i < rows.size(); ++i) {
      std::vector<float> result;
      for (int j = 0; j < row_size; ++j) {
        result.push_back(data[i * row_size + j]);
      }
      results.push_back(result);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to the weight matrix for the convolution.
     */
    float* getTensor() const override {
        return weights_;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("torchconv2d").
     */
    static std::string getName() {
        return "torchconv2d";
    }

private:
    float* weights_; ///< Pointer to the weight matrix for the convolution.
    std::vector<int> dims; ///< Dimensions of the convolution operation.
};

/**
 * @class TorchCNN
 * @brief A class that implements a convolutional neural network (CNN) using PyTorch, inheriting from MLFunction.
 *
 * This class provides functionality to apply a CNN to an input array using PyTorch.
 */
class TorchCNN : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the CNN with weights, biases, and dimensions.
     *
     * @param weights A pointer to the weight matrix for the convolution.
     * @param bias A pointer to the bias vector for the convolution.
     * @param dims_ An array containing the dimensions of the CNN.
     */
    TorchCNN(float* weights, float* bias, int* dims_) {
        weights_ = weights;
        bias_ = bias;
        for (int i = 0; i < 7; i++)
            dims.push_back(dims_[i]);
    }

    /**
     * @brief Applies the CNN to the input array using PyTorch.
     *
     * This method processes the input array, applies the CNN, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();

    int input_height = dims[4];
    int input_width = dims[5];

    int output_height = input_height - dims[1] + 1;
    int output_width = input_width - dims[2] + 1;

    int input_size = input_elements->size();
    // std::cout << "input_size:" << "," << input_size << std::endl;
    // std::cout << "input_values:" << "," << input_values[0] << "," <<
    // input_values[1] << "," << input_values[2080] << std::endl; std::cout <<
    // "row size" << "," << rows.size() << std::endl;

    std::vector<std::vector<float>> results(
        rows.size(),
        std::vector<float>(output_height * output_width * dims[0]));

    torch::nn::Conv2d conv_layer(
        torch::nn::Conv2dOptions(dims[0], dims[3], {dims[1], dims[2]})
            .bias(false));
    // torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(dims[3], dims[0],
    // {dims[1], dims[2]}));
    torch::Tensor conv_weights =
        torch::from_blob(weights_, {dims[3], dims[0], dims[1], dims[2]})
            .to(torch::kFloat);

    auto parameters = conv_layer->named_parameters();

    // Find and set the weight parameter
    for (auto& named_param : parameters) {
      if (named_param.key() == "weight") {
        named_param.value().data() = conv_weights;
        break;
      }
    }
    torch::Tensor input_data =
        torch::from_blob(
            input_values, {rows.size(), dims[3], input_height, input_width})
            .to(torch::kFloat);

    torch::Tensor output_data = conv_layer->forward(input_data);

    // Convert bias values to a tensor
    torch::Tensor bias_tensor = torch::from_blob(bias_, {dims[0]});
    if (conv_layer->bias.defined()) {
      output_data += bias_tensor;
    }

    // output_data = torch::relu(output_data);

    // output_data = torch::max_pool2d(output_data, {dims[6], dims[6]});

    float* data = output_data.data_ptr<float>();

    int row_size = output_height * output_width * dims[0];

    for (int i = 0; i < rows.size(); ++i) {
      std::vector<float> result;
      for (int j = 0; j < row_size; ++j) {
        result.push_back(data[i * row_size + j]);
      }
      results.push_back(result);
    }

    // for (auto entry: results) {
    //     for (int i =0; i < 1000; i++){
    //         std::cout << entry[i] << std::endl;
    //     }
    // }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to the weight matrix for the convolution.
     */
    float* getTensor() const override {
        return weights_;
    }

    /**
     * @brief Returns the weights of the CNN.
     *
     * @return A pointer to the weight matrix.
     */
    float* getWeights() const {
        return weights_;
    }

    /**
     * @brief Returns the biases of the CNN.
     *
     * @return A pointer to the bias vector.
     */
    float* getBias() const {
        return bias_;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("torchcnn").
     */
    static std::string getName() {
        return "torchcnn";
    }

private:
    float* weights_; ///< Pointer to the weight matrix for the convolution.
    float* bias_;    ///< Pointer to the bias vector for the convolution.
    std::vector<int> dims; ///< Dimensions of the CNN.
};
/**
 * @class VectorScalarAddition
 * @brief A class that implements vector-scalar addition, inheriting from MLFunction.
 *
 * This class provides functionality to add a scalar value to each element of a vector.
 */
class VectorScalarAddition : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the vector-scalar addition with weights and size.
     *
     * @param weights A pointer to the scalar values to add.
     * @param size The size of the vector.
     */
    VectorScalarAddition(float* weights, int size) {
        weights_ = weights;
        dims.push_back(size);
    }

    /**
     * @brief Applies vector-scalar addition to the input array.
     *
     * This method processes the input array, adds the scalar values, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();
    int num_cols = input_elements->size() / rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input(input_values, rows.size(), num_cols);
    // for each filter add bias
    for (int i = 0, step = num_cols / dims[0]; i < dims[0]; i++) {
      input.block(0, i * step, rows.size(), step).array() += weights_[i];
    }

    std::vector<std::vector<float>> results(
        input.rows(), std::vector<float>(input.cols()));
    for (int i = 0; i < input.rows(); ++i) {
      for (int j = 0; j < input.cols(); ++j) {
        results[i][j] = input(i, j);
      }
    }
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to the scalar values.
     */
    float* getTensor() const override {
        return weights_;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("vec_scal_add").
     */
    static std::string getName() {
        return "vec_scal_add";
    }

private:
    float* weights_; ///< Pointer to the scalar values to add.
};
/**
 * @class MaxPool
 * @brief A class that implements max pooling, inheriting from MLFunction.
 *
 * This class provides functionality to apply max pooling to an input array.
 */
class MaxPool : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the max pooling operation with dimensions.
     *
     * @param side The size of the pooling window.
     * @param rows The number of rows in the input.
     * @param cols The number of columns in the input.
     */
    MaxPool(int side, int rows, int cols) {
        dims.push_back(side);
        dims.push_back(rows);
        dims.push_back(cols);
    }

    /**
     * @brief Applies max pooling to the input array.
     *
     * This method processes the input array, applies max pooling, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto input_elements = args[0]->as<ArrayVector>()->elements();
    float* input_values = input_elements->values()->asMutable<float>();
    int num_cols = input_elements->size() / rows.size();
    int num_channels = num_cols / (dims[1] * dims[2]);
    int side = dims[0];
    int output_size = (dims[1] * dims[2]) / (side * side);
    int output_rows = dims[1] / side;
    int output_cols = dims[2] / side;
    // this can be done by using one big matrix but padding will not be possible
    // then this doesn't support padding yet but this makes it possible to add
    // it later
    std::vector<std::vector<float>> results(
        rows.size(), std::vector<float>(num_cols / (side * side)));
    // for each sample
    for (int s = 0; s < rows.size(); s++) {
      for (int c = 0; c < num_channels; c++) {
        Eigen::Map<
            Eigen::
                Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            input(
                input_values + s * num_cols + c * dims[1] * dims[2],
                dims[1],
                dims[2]);
        for (int i = 0; i < output_rows; i++) {
          for (int j = 0; j < output_cols; j++) {
            results[s][c * output_size + i * output_cols + j] =
                input.block(i * side, j * side, side, side).maxCoeff();
          }
        }
      }
    }

    // for (const auto& inner_vector : results) {
    //     // Iterate over each element in the inner vector
    //     for (const auto& element : inner_vector) {
    //         std::cout << element << std::endl;
    //     }
    // }

    // for(int i=0; i < 64; i++){

    //     for(int j=0; j < 144; j++){
    //         if(j % 12 == 0)
    //             std::cout << std::endl;
    //         std::cout << results[0][i*144 + j];
    //     }

    // }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .returnType("array(REAL)")
                    .argumentType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to an empty float array (no weights for MaxPool).
     */
    float* getTensor() const override {
        return new float[0];
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    std::string getFuncName() {
        return getName();
    }

    /**
     * @brief Static method to return the name of the function.
     *
     * @return The name of the function as a string ("max_pool").
     */
    static std::string getName() {
        return "max_pool";
    }

private:
    std::vector<int> dims; ///< Dimensions of the max pooling operation.
};
