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
    weights_ = weights;
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
    auto indicesVector = args[0]->as<ArrayVector>()->elements();
    int* indicesValues = indicesVector->values()->asMutable<int>();
    int numIndices = indicesVector->size();

    std::cout << "[DEBUG] rows size: " << rows.size() << " dims[0]: " << dims[0]
              << " indices size: " << indicesVector->size() << " " << std::endl;

    std::vector<std::vector<float>> result(
        numIndices, std::vector<float>(dims[1]));

    for (int i = 0; i < numIndices; i++) {
      int embedIndex = indicesValues[i];

      for (int j = 0; j < dims[1]; j++) {
        result[i][j] = weights_[embedIndex * dims[1] + j];
      }
    }

    VectorMaker maker{context.pool()};
    std::cout << "[DEBUG] Finish at here 2" << std::endl;
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