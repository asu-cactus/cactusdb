/**
 * @file
 * @brief Implementation of an embedding layer for machine learning.
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
 * @class Embedding
 * @brief Implements an embedding layer for machine learning, where embeddings are stored as a 2-D array.
 */
class Embedding : public MLFunction {
 public:
  /**
   * @brief Constructor for Embedding.
   * @param weights Pointer to the embedding weights.
   * @param numEmbeddings Number of embeddings.
   * @param embeddingDims Dimensionality of each embedding.
   */
  Embedding(float* weights, int numEmbeddings, int embeddingDims) {
    // Create a deep copy of the weights
    weights_ = new float[numEmbeddings * embeddingDims];
    std::memcpy(
        weights_, weights, numEmbeddings * embeddingDims * sizeof(float));
    // weights_ = std::move(weights);
    dims.push_back(numEmbeddings);
    dims.push_back(embeddingDims);
  }

  /**
   * @brief Constructor for Embedding.
   * @param weightsFile Path to the file containing the embedding weights.
   * @param numEmbeddings Number of embeddings.
   * @param embeddingDims Dimensionality of each embedding.
   */
  Embedding(std::string weightsFile, int numEmbeddings, int embeddingDims) {
    weightsFile_ = weightsFile;
    dims.push_back(numEmbeddings);
    dims.push_back(embeddingDims);
  }

  /**
   * @brief Applies the embedding function to the input data.
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
    auto input = decodedArgs.at(0);
    auto arrayVector = input->base()->as<ArrayVector>();

    auto indicesVector = arrayVector->elements();
    int* indicesValues = indicesVector->values()->asMutable<int>();
    int numInputs = rows.size();
    // You can also use sizeof(*arrayVector->rawSizes()) to compute the size of
    // a single entry in BufferPtr

    int numEmbeddingToRetireve = 0;
    rows.applyToSelected([&](vector_size_t row) {
      int numSubIndices = arrayVector->sizeAt(row);
      numEmbeddingToRetireve += numSubIndices;
    });

    auto baseOffset = elementsOutput->size();
    // here, we need to resize it to the number of embeddings we need to
    // retrieve
    elementsOutput->resize(baseOffset + numEmbeddingToRetireve * dims[1]);
    float* outputValues = elementsOutput->values()->asMutable<float>();

    vector_size_t outputOffset = 0;
    rows.applyToSelected([&](vector_size_t row) {
      int numSubIndices = arrayVector->sizeAt(row);
      int indicesOffset = arrayVector->offsetAt(row);
      rawOffsets[row] = outputOffset;
      rawSizes[row] = numSubIndices * dims[1];
      for (int i = 0; i < numSubIndices; i++) {
        // Support of variadic indexes
        int embedIndex = indicesValues[indicesOffset + i];
        if (embedIndex >= dims[0]) {
          throw std::runtime_error(fmt::format(
              "[Embedding] Index out of bounds: {} >= {}",
              embedIndex,
              dims[0]));
        }
        std::memcpy(
            outputValues + outputOffset,
            weights_ + embedIndex * dims[1],
            dims[1] * sizeof(float));
        outputOffset += dims[1];
      }
    });
    arrayOutput->setElements(elementsOutput);
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(INTEGER)")
                .returnType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the tensor associated with the function.
   * @return Pointer to the tensor.
   */
  float* getTensor() const override {
    return weights_;
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "embedding";
  };

  /**
   * @brief Returns the path to the weights file.
   * @return Path to the weights file.
   */
  std::string getWeightsFile() {
    return weightsFile_;
  }

  /**
   * @brief Sets the embedding weights.
   * @param weights Pointer to the embedding weights.
   */
  void setWeights(float* weights) {
    weights_ = weights;
  }

 private:
  float* weights_; ///< Pointer to the embedding weights.
  std::string weightsFile_; ///< Path to the file containing the embedding weights.
};
