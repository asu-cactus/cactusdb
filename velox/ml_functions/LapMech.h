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
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "velox/ml_functions/BaseFunction.h"

using namespace facebook::velox;

// Implementation of embedding layer where the embedding is stored as a 2-D
// array: numEmbedding*embeddingDims, lookup takes a int vector as indices

class LapMech : public MLFunction {
 public:
  LapMech(CataLog* cataLog) {
    cataLog_ = cataLog;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    float eps;
    if (args.size() != 2) {
      LOG(FATAL)
          << "[ERROR TupleCounter:] args.size() != 2, an intermedaite state name is required";
    }

    // obtain the eps through the second argument
    exec::LocalDecodedVector decodedEpsHolder(context, *args[1], rows);
    auto decodedEps = decodedEpsHolder.get();
    if (args[1]->typeKind() == TypeKind::DOUBLE) {
      eps = decodedEps->valueAt<double>(0);
    } else if (args[1]->typeKind() == TypeKind::REAL) {
      eps = decodedEps->valueAt<float>(0);
    } else {
      LOG(FATAL) << "[ERROR LapMech:] eps is not a double or float";
    }

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
    int numCols = dims[0];
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
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> inputMatrix(numInputMatrixRows, numCols);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], numCols);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMatrix(numInputMatrixRows, numCols);

    int numTuplesAtIntermediateState =
        cataLog_->getIntermediateStateTupleCounter(intermediateStateName);

    // TODO: implement the logic of LapMech
    /* Below is the reference code for batch normalization

    for (int i = 0; i < numCols; i++) {
      Eigen::VectorXf colData = inputMatrix.col(i);
      float colMean = colData.mean();
      float colVariance =
          (colData.array() - colMean).square().sum() / (numInputMatrixRows - 1);

      resultMatrix.col(i) =
          (colData.array() - colMean) / sqrt(colVariance + eps_) * weights_[i] +
          bias_[i];
    }
    */

    auto baseOffset = elementsOutput->size();
    elementsOutput->resize(baseOffset + rows.end() * numCols);
    float* outputValues = elementsOutput->values()->asMutable<float>();
    vector_size_t outputOffset = baseOffset;
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
    });

    arrayOutput->setElements(elementsOutput);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .argumentType("array(REAL)")
            .argumentType("REAL")
            .returnType("array(REAL)")
            .build(),
        ,
        exec::FunctionSignatureBuilder()
            .argumentType("array(REAL)")
            .argumentType("DOUBLE")
            .returnType("array(REAL)")
            .build()}
  };
}

// TODO: add get and set for bias or we have a better way to store the two
// parameters in a single file
float*
getTensor() const override {
  return weights_;
}

float* getWeight() {
  return weights_;
}

float* getBias() {
  return bias_;
}

static std::string getName() {
  return "LapMech";
};

std::string getWeightsFile() {
  return weightsFile_;
}

void setWeights(float* weights) {
  weights_ = weights;
}

CostEstimate getCost(std::vector<int> inputDims) {
  // Compute the operation cost using a static cost model. Note: This is
    // currently not utilized for query optimization as we rely on an ML-based
    // model (optimizer/query2vec).
    // TODO: Implement a static cost estimation method for the specified kernel.
  return CostEstimate(0, inputDims[0], inputDims[1]);
}

private:
float* weights_;
float* bias_;
float eps_;
std::string weightsFile_;
std::string biasFile_;
CataLog* cataLog_;
}
;
