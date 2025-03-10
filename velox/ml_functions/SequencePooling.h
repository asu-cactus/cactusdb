/**
 * @file
 * @brief Implementation of a sequence pooling function for machine learning.
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
#include <cstring>
#include <iostream>
#include <stdexcept>
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
 * @class SequencePooling
 * @brief Implements a sequence pooling function for machine learning.
 */
class SequencePooling : public MLFunction {
 public:
  /**
   * @brief Constructor for SequencePooling.
   * @param mode The pooling mode (MIN, MAX, or MEAN).
   * @param embeddingDims The dimensionality of the embeddings.
   */
  SequencePooling(std::string mode, int embeddingDims) {
    transform(mode.begin(), mode.end(), mode.begin(), ::toupper);

    if (mode != "MIN" && mode != "MAX" && mode != "MEAN") {
      throw std::runtime_error(
          "[Error]: The input mode: " + mode +
          " is not supported. Supported mode: MIN, MAX, MEAN");
    }
    mode_ = mode;
    embeddingDims_ = embeddingDims;
    dims.push_back(embeddingDims);
  }

  /**
   * @brief Applies the sequence pooling function to the input data.
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

    int numResultMatrixRows = numUniqueRows;
    Eigen::MatrixXf resultMatix(numResultMatrixRows, dims[0]);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      int numEmbeddingValues = inputSizes[rawIndex];
      // int valueOffset = inputOffsets[rawIndex];
      int numEmbeddingToCombie = numEmbeddingValues / embeddingDims_;
      if (numEmbeddingToCombie == 1) {
        Eigen::Map<const Eigen::VectorXf> rowVector(
            inputValues + inputOffsets[rawIndex], embeddingDims_);
        resultMatix.row(rowIndex) = rowVector;
      } else {
        Eigen::Map<
            Eigen::
                Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            varaidicEmbedding(
                inputValues + inputOffsets[rawIndex],
                numEmbeddingToCombie,
                embeddingDims_);

        Eigen::VectorXf mergedValues;
        if (mode_ == "MIN") {
          mergedValues = varaidicEmbedding.colwise().minCoeff();
        } else if (mode_ == "MAX") {
          mergedValues = varaidicEmbedding.colwise().maxCoeff();
        } else if (mode_ == "MEAN") {
          mergedValues = varaidicEmbedding.colwise().mean();
        }
        resultMatix.row(rowIndex) = mergedValues;
      }
      rowIndex++;
    }

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
          resultMatix.row(mappedIndexInResultMatrix).data(),
          dims[0] * sizeof(float));

      outputOffset += dims[0];
    });
    arrayOutput->setElements(elementsOutput);
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
    return "sequence_pooling";
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
  std::string mode_; ///< The pooling mode (MIN, MAX, or MEAN).
  int embeddingDims_; ///< The dimensionality of the embeddings.
};
