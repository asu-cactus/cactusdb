#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
// #include "velox/exec/tests/utils/AssertQueryBuilder.h"
// #include "velox/exec/tests/utils/PlanBuilder.h"
// #include "velox/exec/tests/utils/TempDirectoryPath.h"
// // #include "velox/functions/lib/LambdaFunctionUtil.h"
// #include "velox/functions/lib/RowsTranslationUtil.h"
// #include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

class SVD : public MLFunction {
 public:
  SVD(float* bu,
      float* bi,
      float* pu,
      float* qi,
      int numUser,
      int numItem,
      int latentDims) {
    // Create a deep copy of the weights
    bu_ = new float[numUser];
    bi_ = new float[numItem];
    pu_ = new float[numUser * latentDims];
    qi_ = new float[numItem * latentDims];
    std::memcpy(bu_, bu, numUser * sizeof(float));
    std::memcpy(bi_, bi, numItem * sizeof(float));
    std::memcpy(pu_, pu, numUser * latentDims * sizeof(float));
    std::memcpy(qi_, qi, numItem * latentDims * sizeof(float));
    dims.push_back(numUser);
    dims.push_back(numItem);
    dims.push_back(latentDims);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, outputType, context.pool(), output);

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedUser = decodedArgs.at(0);
    auto decodedItem = decodedArgs.at(1);

    auto arrayOutput = output->asFlatVector<float>();
    float* outputValues = arrayOutput->mutableRawValues<float>();

    rows.applyToSelected([&](vector_size_t i) {
      auto userId = decodedUser->valueAt<int>(i);
      auto itemId = decodedItem->valueAt<int>(i);

      if (userId > dims[0]) {
        LOG(WARNING) << "User id out of bound: " << userId << " / " << dims[0];
        userId = 0;
      } else if (itemId > dims[1]) {
        itemId = 0;
        LOG(WARNING) << "Item id out of bound: " << itemId << " / " << dims[1];
      }

      Eigen::Map<Eigen::VectorXf> qiVec(qi_ + itemId * dims[2], dims[2]);
      Eigen::Map<Eigen::VectorXf> puVec(pu_ + userId * dims[2], dims[2]);

      float prediction = bu_[userId] + bi_[itemId] + puVec.dot(qiVec);
      outputValues[i] = prediction;
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .returnType("REAL")
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .build()};
  }

  float* getTensor() const override {
    return weights_;
  }

  std::string getFuncName() {
    return getName();
  };

  static std::string getName() {
    return "svd";
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
    std::vector<double> coefficientVector = getCoefficientVector(getName());
    float cost = coefficientVector[0] * inputDims[0] * inputDims[1];

    return CostEstimate(cost, inputDims[0], inputDims[1]);
  }

 private:
  float* weights_;
  float* bu_;
  float* bi_;
  float* pu_;
  float* qi_;
  std::string weightsFile_;
};