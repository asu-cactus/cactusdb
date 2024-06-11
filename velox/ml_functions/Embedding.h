#pragma once
#include <fmt/format.h>
#include <iostream>
#include "functions.h"
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
    std::memcpy(weights_, weights, numEmbeddings * embeddingDims * sizeof(float));
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

      BaseVector* input = args[0].get();
      exec::LocalDecodedVector inputHolder(context, *input, rows);
      auto decodedInputArray = inputHolder.get();
      auto arrayVector = decodedInputArray->base()->as<ArrayVector>();
      auto indicesVector = arrayVector->elements();
      int* indicesValues = indicesVector->values()->asMutable<int>();

      int numInputs = rows.size();


      // You can also use sizeof(*arrayVector->rawSizes()) to compute the size of
      // a single entry in BufferPtr
      std::vector<std::vector<float>> result;
      for (int i = 0; i < numInputs; i++) {
        int numSubIndices = arrayVector->sizeAt(i);
        int indicesOffset = arrayVector->offsetAt(i);
        std::vector<float> retrievedEmbedding;
        for (int j = 0; j < numSubIndices; j++) {
          // Support of variadic indexes
          int embedIndex = indicesValues[indicesOffset + j];
          std::vector<float> embedVector(
              weights_ + embedIndex * dims[1],
              weights_ + embedIndex * dims[1] + dims[1]);
          retrievedEmbedding.insert(
              retrievedEmbedding.end(), embedVector.begin(), embedVector.end());
        }
        result.push_back(retrievedEmbedding);
      }

      VectorMaker maker{context.pool()};
      output = maker.arrayVector<float>(result, REAL());
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