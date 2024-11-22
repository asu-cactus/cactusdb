#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/functions/lib/RowsTranslationUtil.h"
#include "velox/functions/lib/LambdaFunctionUtil.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

// Implementation of embedding layer where the embedding is stored as a 2-D
// array: numEmbedding*embeddingDims, lookup takes a int vector as indices

class CosineSimilarity : public MLFunction {
 public:
  CosineSimilarity(int dim) {
    dims.push_back(dim);
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

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray =
        decodedRightArray->base()->as<ArrayVector>()->elements();
    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input1Matrix(input1Values, numInput, dims[0]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input2Matrix(input2Values, numInput, dims[0]);

    std::vector<float> resultVector;

    rows.applyToSelected([&](vector_size_t i) {
      // Map the input values into Eigen vectors
      Eigen::Map<Eigen::VectorXf> vec1(input1Values + i * dims[0], dims[0]);
      Eigen::Map<Eigen::VectorXf> vec2(input2Values + i * dims[0], dims[0]);

      // Compute cosine similarity
      float dotProduct = vec1.dot(vec2);
      float norm1 = vec1.norm();
      float norm2 = vec2.norm();
      float cosineSim = dotProduct / (norm1 * norm2 + 1e-8);

      // Store the result
      // resultVector.push_back(1.0);
      resultVector.push_back(cosineSim);
    });

    VectorMaker maker{context.pool()};
    output = maker.flatVector<float>(resultVector, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .returnType("REAL")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  // float* getTensor() const override {
  //   return weights_;
  // }

  static std::string getName() {
    return "cosine_similarity";
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

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::vector<int> dims;
  // float* weights_;
  // float* bias_;
  // float eps_;
  // std::string weightsFile_;
  // std::string biasFile_;
};