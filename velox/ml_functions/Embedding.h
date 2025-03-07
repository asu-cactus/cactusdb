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

class Embedding : public MLFunction {
 public:
  Embedding(float* weights, int numEmbeddings, int embeddingDims) {
    // Create a deep copy of the weights
    weights_ = new float[numEmbeddings * embeddingDims];
    std::memcpy(
        weights_, weights, numEmbeddings * embeddingDims * sizeof(float));
    // weights_ = std::move(weights);
    dims.push_back(numEmbeddings);
    dims.push_back(embeddingDims);
  }

  Embedding(std::string weightsFile, int numEmbeddings, int embeddingDims) {
    weightsFile_ = weightsFile;
    dims.push_back(numEmbeddings);
    dims.push_back(embeddingDims);
  }

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

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(INTEGER)")
                .returnType("array(REAL)")
                .build()};
  }

  float* getTensor() const override {
    return weights_;
  }

  static std::string getName() {
    return "embedding";
  };

  std::string getWeightsFile() {
    return weightsFile_;
  }

  void setWeights(float* weights) {
    weights_ = weights;
  }

 private:
  float* weights_;
  std::string weightsFile_;
};