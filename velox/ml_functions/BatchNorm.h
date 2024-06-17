#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/functions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

// Implementation of embedding layer where the embedding is stored as a 2-D
// array: numEmbedding*embeddingDims, lookup takes a int vector as indices

class BatchNorm1D : public MLFunction {
 public:
  BatchNorm1D(float* weights, float* bias, int numDims, float eps = 1e-05) {
    // Create a deep copy of the weights
    weights_ = new float[numDims];
    bias_ = new float[numDims];
    std::memcpy(weights_, weights, numDims * sizeof(float));
    std::memcpy(bias_, bias, numDims * sizeof(float));
    // weights_ = weights;
    // bias_ = bias;
    eps_ = eps;
    dims.push_back(numDims);
  }

  // TODO: add support of loading from disk file
  // BatchNorm1D(std::string weightsFile, int numEmbeddings, int embeddingDims)
  // {
  //   weightsFile_ = weightsFile;
  //   dims.push_back(numEmbeddings);
  //   dims.push_back(embeddingDims);
  // }

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
        inputMatrix(inputValues, numInput, dims[0]);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        result(numInput, dims[0]);

    for (int i = 0; i < dims[0]; i++) {
      Eigen::VectorXf colData = inputMatrix.col(i);
      float colMean = colData.mean();
      float colVariance =
          (colData.array() - colMean).square().sum() / (numInput - 1);

      result.col(i) =
          (colData.array() - colMean) / sqrt(colVariance + eps_) * weights_[i] +
          bias_[i];
    }

    // Convert from Eigen::Matrix to std::vector<std::vector<>>
    std::vector<std::vector<float>> resultVector;
    for (int rowIndex = 0; rowIndex < result.rows(); rowIndex++) {
      std::vector<float> row(
          result.row(rowIndex).data(),
          result.row(rowIndex).data() + result.cols());
      resultVector.push_back(row);
    }

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
    return weights_;
  }

  static std::string getName() {
    return "batch_norm_1d";
  };

  std::string getWeightsFile() {
    return weightsFile_;
  }

  void setWeights(float* weights) {
    weights_ = weights;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  float* weights_;
  float* bias_;
  float eps_;
  std::string weightsFile_;
  std::string biasFile_;
};
