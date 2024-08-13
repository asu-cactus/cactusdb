#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "functions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

class SequencePooling : public MLFunction {
 public:
  SequencePooling(std::string mode, int embeddingDims) {
    transform(mode.begin(), mode.end(), mode.begin(), ::toupper);

    if (mode != "MIN" && mode != "MAX" && mode != "MEAN") {
      throw std::runtime_error(
          "[Error]: The input mode: " + mode +
          " is not supported. Supported mode: MIN, MAX, MEAN");
    }
    mode_ = mode;
    embeddingDims_ = embeddingDims;
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

    auto arrayVector = args[0]->as<ArrayVector>();
    std::vector<std::vector<float>> result;

    for (int i = 0; i < numInput; i++) {
      int numEmbeddingValues = arrayVector->sizeAt(i);
      int valueOffset = arrayVector->offsetAt(i);
      int numEmbeddingToCombie = numEmbeddingValues / embeddingDims_;
      std::vector<float> embeddingVector(embeddingDims_);
      if (numEmbeddingToCombie == 1) {
        // non-variadic, just copy
        std::memcpy(
            embeddingVector.data(),
            inputValues + valueOffset,
            embeddingDims_ * sizeof(float));
      } else {
        // variadic case, combine the values base on mode
        Eigen::Map<
            Eigen::
                Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            varaidicEmbedding(
                inputValues + valueOffset,
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

        std::memcpy(
            embeddingVector.data(),
            mergedValues.data(),
            embeddingDims_ * sizeof(float));
      }
      result.push_back(embeddingVector);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  // float* getTensor() const override {
  //   return weights_;
  // }

  static std::string getName() {
    return "sequence_pooling";
  };

  // std::string getWeightsFile() {
  //   return weightsFile_;
  // }

  // void setWeights(float* weights) {
  //   weights_ = weights;
  // }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims){
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::string mode_;
  int embeddingDims_;
};