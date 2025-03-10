/**
 * @file
 * @brief Implementation of a Retrieval-Augmented Generation (RAG) function for machine learning.
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

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <cmath>
#include <iostream>
#include "BaseFunction.h"

/**
 * @brief Flattens a 2D vector into a 1D vector.
 * @param vec2D The 2D vector to flatten.
 * @return A 1D vector containing all elements of the input 2D vector.
 */
std::vector<float> flatten(const std::vector<std::vector<float>>& vec2D) {
  std::vector<float> flatVec;

  // Loop through each inner vector and add its elements to the flat vector
  for (const auto& innerVec : vec2D) {
    flatVec.insert(flatVec.end(), innerVec.begin(), innerVec.end());
  }

  return flatVec;
}

/**
 * @class RAG
 * @brief Implements a Retrieval-Augmented Generation (RAG) function for machine learning.
 */
class RAG : public MLFunction {
 public:
  /**
   * @brief Constructor for RAG.
   * @param document A vector of documents.
   * @param embedding A 2D vector of embeddings corresponding to the documents.
   * @param dimension The dimensionality of the embeddings.
   */
  RAG(std::vector<std::string> document,
      std::vector<std::vector<float>> embedding,
      int dimension) {
    // Create a deep copy of the weights
    document_ = document;
    embedding_ = embedding;
    dims.push_back(dimension);

    // Create the IndexFlatL2 index
    index_ = faiss::IndexFlatL2(dimension);
    faiss::IndexFlatL2 index(dimension); // call constructor
    int numDocument = document.size();
    assert(numDocument == embedding.size());

    weights_ = new float[numDocument * dimension];
    int dataIndex = 0;
    for (const auto& vec : embedding_) {
      std::copy(vec.begin(), vec.end(), weights_ + dataIndex);
      dataIndex += vec.size();
    }

    std::vector<float> flattened1DEmbedding = flatten(embedding_);
    index_.add(numDocument, weights_);
  }

  /**
   * @brief Applies the RAG function to the input data.
   * @param rows Selectivity vector indicating which rows to process.
   * @param args Vector of input arguments.
   * @param outputType Type of the output vector.
   * @param context Evaluation context.
   * @param output Output vector to store the results.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, outputType, context.pool(), output);
    auto arrayOutput = output->asFlatVector<StringView>();

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

    std::vector<std::string> uniqueResults(numUniqueRows);
    for (int i = 0; i < numUniqueRows; i++) {
      int index = uniqueRawIndexeVector[i];
      int k = 3;
      std::vector<faiss::idx_t> labels(k);
      std::vector<float> distances(k);
      index_.search(
          1,
          inputValues + inputOffsets[index],
          k,
          distances.data(),
          labels.data());
      uniqueResults[i] = document_[labels[0]];
    }

    std::vector<std::string> results(rows.size());
    rows.applyToSelected([&](vector_size_t row) {
      arrayOutput->set(row, StringView(uniqueResults[rowMap[row]]));
    });
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("VARCHAR")
                .argumentType("ARRAY(REAL)")
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
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "svd";
  };

  /**
   * @brief Returns the path to the weights file.
   * @return Path to the weights file.
   */
  std::string getWeightsFile() {
    return weightsFile_;
  }

  /**
   * @brief Sets the weights for the function.
   * @param weights Pointer to the weights array.
   */
  void setWeights(float* weights) {
    weights_ = weights;
  }

  /**
   * @brief Estimates the cost of the function.
   * @param inputDims Dimensions of the input.
   * @return Cost estimate.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    std::vector<double> coefficientVector = getCoefficientVector(getName());
    float cost = coefficientVector[0] * inputDims[0] * inputDims[1];

    return CostEstimate(cost, inputDims[0], inputDims[1]);
  }

 private:
  float* weights_; ///< Pointer to the weights array.
  std::vector<std::string> document_; ///< Vector of documents.
  std::vector<std::vector<float>> embedding_; ///< 2D vector of embeddings.
  std::string weightsFile_; ///< Path to the weights file.
  faiss::IndexFlatL2 index_; ///< FAISS index for nearest neighbor search.
};
