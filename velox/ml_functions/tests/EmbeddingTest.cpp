// TODO: Resolve dependencies
#include <folly/init/Init.h>
#include <torch/torch.h>
#include <random>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/Dropout.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

// Utility function to generate random float/int values

class RandomGenerator {
 public:
  RandomGenerator(float lb, float ub, int randomSeed = 0) {
    gen_ = std::mt19937(randomSeed);
    distR_ = std::uniform_real_distribution<float>(lb, ub);
    distI_ = std::uniform_int_distribution<int>((int)lb, (int)ub);
  }

  void setFloatRange(float lb, float ub) {
    distR_ = std::uniform_real_distribution<float>(lb, ub);
  }

  void setIntRange(int lb, int ub) {
    distI_ = std::uniform_int_distribution<int>(lb, ub);
  }

  float genRandomFloatValue() {
    return distR_(gen_);
  }

  int genRandomIntValue() {
    return distI_(gen_);
  }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<float> distR_;
  std::uniform_int_distribution<int> distI_;
};

class EmbeddingTest : public HiveConnectorTestBase {
 public:
  EmbeddingTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    SetUp();
  }

  ~EmbeddingTest() {}

  void run();
  void testEmbedding();
  void testBatchNorm1D();
  void testDropout();

  void TestBody() override {}

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TearDown() {
    HiveConnectorTestBase::TearDown();
  }

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::addDefaultLeafMemoryPool();
  VectorMaker maker{pool_.get()};
};

// Test Embedding Layer
void EmbeddingTest::testEmbedding() {
  int numEmbeddings = 6000;
  int embeddingDims = 128;
  int embeddingSize = numEmbeddings * embeddingDims;
  int numSamples = 120;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  randomGenerator.setIntRange(0, numEmbeddings);

  // Initialize the weight matrix
  auto weights = maker.flatVector<float>(embeddingSize);
  for (int i = 0; i < embeddingSize; i++) {
    weights->set(i, randomGenerator.genRandomFloatValue());
  }

  // Initialize the feature vector
  std::vector<std::vector<int>> indicesVectors;
  for (int i = 0; i < numSamples; i++) {
    std::vector<int> inputVector;
    inputVector.push_back(randomGenerator.genRandomIntValue());
    indicesVectors.push_back(inputVector);
  }

  auto indicesArrayVector = maker.arrayVector<int>(indicesVectors, INTEGER());
  auto inputRowVector = maker.rowVector({"x"}, {indicesArrayVector});

  std::cout << "[INFO] Generated Indices:"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          weights->values()->asMutable<float>(), numEmbeddings, embeddingDims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"embedding(x)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results:" << results->toString() << std::endl;
  std::cout << "[INFO] Results:" << results->toString(0, results->size())
            << std::endl;
};

// Test BatchNorm1D Layer
void EmbeddingTest::testBatchNorm1D() {
  int numSamples = 2;
  int numDims = 5;
  int featureSize = numSamples * numDims;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);

  // Initialize the weight matrix
  auto weights = maker.flatVector<float>(numDims);
  for (int i = 0; i < numDims; i++) {
    weights->set(i, randomGenerator.genRandomFloatValue());
  }

  // Initialize the bias matrix
  auto bias = maker.flatVector<float>(numDims);
  for (int i = 0; i < numDims; i++) {
    bias->set(i, randomGenerator.genRandomFloatValue());
  }

  // Initialize the feature vector
  std::vector<std::vector<float>> inputVectors;
  for (int i = 0; i < numSamples; i++) {
    std::vector<float> inputVector;
    for (int j = 0; j < numDims; j++) {
      inputVector.push_back(randomGenerator.genRandomFloatValue());
    }
    inputVectors.push_back(inputVector);
  }

  auto indicesArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {indicesArrayVector});

  std::cout << "[INFO] Generated Indices:"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "batch_norm_1d",
      BatchNorm1D::signatures(),
      std::make_unique<BatchNorm1D>(
          weights->values()->asMutable<float>(),
          bias->values()->asMutable<float>(),
          numDims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"batch_norm_1d(x)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results:" << results->toString() << std::endl;
  std::cout << "[INFO] Results:" << results->toString(0, results->size())
            << std::endl;
};

// Test Dropout Layer
void EmbeddingTest::testDropout() {
  int numSamples = 2;
  int numDims = 5;
  int featureSize = numSamples * numDims;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);

  float p = 0.5;

  // Initialize the feature vector
  std::vector<std::vector<float>> inputVectors;
  for (int i = 0; i < numSamples; i++) {
    std::vector<float> inputVector;
    for (int j = 0; j < numDims; j++) {
      inputVector.push_back(randomGenerator.genRandomFloatValue());
    }
    inputVectors.push_back(inputVector);
  }

  auto indicesArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {indicesArrayVector});

  std::cout << "[INFO] Generated Indices:"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "dropout", Dropout::signatures(), std::make_unique<Dropout>(p));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"dropout(x)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results:" << results->toString() << std::endl;
  std::cout << "[INFO] Results:" << results->toString(0, results->size())
            << std::endl;
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  EmbeddingTest demo;
  // demo.testEmbedding();
  // demo.testBatchNorm1D();
  demo.testDropout();
}