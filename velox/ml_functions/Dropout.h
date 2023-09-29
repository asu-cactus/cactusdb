#include <iostream>
#include "functions.h"
#include <Eigen/Dense>
#include <cmath>
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

class Dropout : public MLFunction {
 public:
  Dropout(float p) {
    p_ = p;
    // std::random_device device;
    // std::mt19937 gen(device());
    // std::bernoulli_distribution coin_flip(0.5);
    // bool outcome = coin_flip(gen);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution bernoulli(p_);

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto inputFeatures = args[0]->as<ArrayVector>()->elements();
    float* inputValues = inputFeatures->values()->asMutable<float>();

    int inputSize = inputFeatures->size();
    int numInput = args[0]->size();
    int numFeatures = inputSize / numInput;

    // float* result[inputSize];
    std::vector<std::vector<float>> result(
        numInput, std::vector<float>(numFeatures));

    for (int i = 0; i < numInput; i++) {
      for (int j = 0; j < numFeatures; j++) {
        bool outcome = bernoulli(gen);
        if (outcome) {
          result[i][j] = 0;
        } else {
          result[i][j] = inputValues[i*numFeatures + j];
        }
      }
      
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

  static std::string getName() {
    return "dropout";
  };

  float* getTensor() const override {
    // FIXME
    return nullptr;
  }

  void setWeight(float p) {
    p_ = p;
  }


 private:
  float p_;
  std::mt19937 gen_;
  std::bernoulli_distribution bernoulli_;
};