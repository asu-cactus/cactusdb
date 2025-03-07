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

class FullyConnectWithBatchNormAndRelu : public MLFunction {
 public:
  FullyConnectWithBatchNormAndRelu(
      // float** weights,
      float* NNWeights,
      float* NNBias,
      float* NormWeights,
      float* NormBias,
      float eps,
      int numInput,
      int numOutput) {
    this->NNWeights_ = NNWeights;
    this->NNBias_ = NNBias;
    this->NormWeights_ = NormWeights;
    this->NormBias_ = NormBias;
    this->eps_ = eps;
    this->dims.push_back(numInput);
    this->dims.push_back(numOutput);
  }

  float static relu_function(float x) {
    return (x > 0.0f) ? x : 0.0f;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto inputFeatures = args[0]->as<ArrayVector>()->elements();
    float* inputValues = inputFeatures->values()->asMutable<float>();
    int numInput = rows.size();

        Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        m1(inputValues, numInput, dims[0]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        nnWeight(this->NNWeights_, dims[0], dims[1]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        nnBias(this->NNBias_, 1, dims[1]);

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        result(numInput, dims[1]);
    result = m1 * nnWeight;
    // std::cout << fmt::format(
    //                  "[DEBUG] result shape: {}, {}, {}, {}",
    //                  inputFeatures->size(),
    //                  rows.size(),
    //                  result.rows(),
    //                  result.cols())
    //                  << std::endl;

    // Bias Addition
    result.rowwise() += nnBias.row(0);

    // Batch Normalization
    for (int i = 0; i < dims[1]; i++) {
      Eigen::VectorXf colData = result.col(i);
      float colMean = colData.mean();
      float colVariance =
          (colData.array() - colMean).square().sum() / (numInput - 1);

      result.col(i) = (colData.array() - colMean) / sqrt(colVariance + eps_) *
              NormWeights_[i] +
          NormBias_[i];
    }

    // Relu
    result = result.array().unaryExpr(&relu_function);

    // Convert from Eigen::Matrix to std::vector<std::vector<>>
    std::vector<std::vector<float>> resultVector;
    for (int rowIndex = 0; rowIndex < result.rows(); rowIndex++) {
      std::vector<float> row(
          result.row(rowIndex).data(),
          result.row(rowIndex).data() + result.cols());
      resultVector.push_back(row);
    }
    // std::cout << fmt::format(
    //                  "[DEBUG] result shape: {}, {}, {}, {}",
    //                  inputFeatures->size(),
    //                  rows.size(),
    //                  result.rows(),
    //                  result.cols())
    //           << std::endl;

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(resultVector, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  float* getTensor() const override {
    return NNWeights_;
  }

  static std::string getName() {
    return "fully_layer_with_batch_norm";
  };

  std::string getWeightsFile() {
    return weightsFile_;
  }

  void setWeights(float* weights) {
    // weights_ = weights;
  }

 private:
  float* NNWeights_;
  float* NNBias_;
  float* NormWeights_;
  float* NormBias_;
  float eps_;
  std::string weightsFile_;
  std::string biasFile_;
};