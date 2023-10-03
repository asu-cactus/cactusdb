// TODO: Resolve dependencies
#include <folly/init/Init.h>
#include <torch/torch.h>
#include <random>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/Dropout.h"
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

  std::vector<std::vector<float>> genFloat2dVector(int numRow, int numCol) {
    // Initialize the input1 feature vector
    std::vector<std::vector<float>> float2dVector;

    for (int i = 0; i < numRow; i++) {
      std::vector<float> floatVector;
      for (int j = 0; j < numCol; j++) {
        floatVector.push_back(genRandomFloatValue());
      }
      float2dVector.push_back(floatVector);
    }
    return float2dVector;
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
  void testConcat1();
  void testConcat2();
  // void testConcat3();
  void testCosineSimilarity();

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
  std::cout << "[INFO] Test of Embedding." << std::endl;
  int numEmbeddings = 5;
  int embeddingDims = 2;
  int embeddingSize = numEmbeddings * embeddingDims;
  int numSamples = 5;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  randomGenerator.setIntRange(0, numEmbeddings - 1);

  std::vector<std::vector<float>> weights =
      randomGenerator.genFloat2dVector(numEmbeddings, embeddingDims);
  auto weightsVector = maker.arrayVector<float>(weights, REAL());

  // Initialize the indices vector
  std::vector<std::vector<int>> indicesVectors;
  for (int i = 0; i < numSamples; i++) {
    std::vector<int> inputVector;
    int numI = i % 3;
    if (numI == 0) {
      numI = 1;
    }
    for (int j = 0; j < numI; j++) {
      inputVector.push_back(randomGenerator.genRandomIntValue());
    }
    indicesVectors.push_back(inputVector);
  }

  auto indicesArrayVector = maker.arrayVector<int>(indicesVectors, INTEGER());
  auto inputRowVector = maker.rowVector({"x"}, {indicesArrayVector});

  std::cout << "[INFO] Generated Embedding:\n"
            << weightsVector->toString(0, weightsVector->size()) << std::endl;

  std::cout << "[INFO] Generated Indices:\n"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          weightsVector->elements()->values()->asMutable<float>(),
          numEmbeddings,
          embeddingDims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"embedding(x)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results: \n" << results->toString() << std::endl;
  std::cout << "[INFO] Results: \n"
            << results->toString(0, results->size()) << std::endl;
};

// Test BatchNorm1D Layer
void EmbeddingTest::testBatchNorm1D() {
  std::cout << "[INFO] Test of BatchNorm1D." << std::endl;
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

  std::cout << "[INFO] Results: \n" << results->toString() << std::endl;
  std::cout << "[INFO] Results: \n"
            << results->toString(0, results->size()) << std::endl;
};

// Test Dropout Layer
void EmbeddingTest::testDropout() {
  std::cout << "[INFO] Test of Dropout." << std::endl;
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

  std::cout << "[INFO] Results: \n" << results->toString() << std::endl;
  std::cout << "[INFO] Results: \n"
            << results->toString(0, results->size()) << std::endl;
};

// Test Dropout Layer
void EmbeddingTest::testConcat1() {
  std::cout << "[INFO] Test of Concat1." << std::endl;
  int numSamples = 2;
  int numDims1 = 5;
  int numDims2 = 3;
  int featureSize1 = numSamples * numDims1;
  int featureSize2 = numSamples * numDims2;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1);

  // Initialize the input1 feature vector
  std::vector<std::vector<float>> inputVectors1 =
      randomGenerator.genFloat2dVector(numSamples, numDims1);
  std::vector<std::vector<float>> inputVectors2 =
      randomGenerator.genFloat2dVector(numSamples, numDims2);

  auto indicesArrayVector1 = maker.arrayVector<float>(inputVectors1, REAL());
  auto inputRowVector1 = maker.rowVector({"x1"}, {indicesArrayVector1});

  auto indicesArrayVector2 = maker.arrayVector<float>(inputVectors2, REAL());
  auto inputRowVector2 = maker.rowVector({"x2"}, {indicesArrayVector2});

  auto inputRowVector =
      maker.rowVector({"x1", "x2"}, {indicesArrayVector1, indicesArrayVector2});

  // Print input
  std::cout << "[INFO] input1: \n"
            << inputRowVector1->toString(0, inputRowVector1->size())
            << std::endl;
  std::cout << "[INFO] input2: \n"
            << inputRowVector2->toString(0, inputRowVector2->size())
            << std::endl;

  exec::registerVectorFunction(
      "concat",
      Concat::signatures(),
      std::make_unique<Concat>(numDims1, numDims2));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"concat(x1, x2)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results: \n" << results->toString() << std::endl;
  std::cout << "[INFO] Results: \n"
            << results->toString(0, results->size()) << std::endl;
};

void EmbeddingTest::testConcat2() {
  std::cout << "[INFO] Test of Concat2." << std::endl;
  int numSamples = 2;
  int input1Dims = 5;
  int input2Dims = 4;

  int input1NN = 3;
  int input2NN = 2;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1);

  // Initialize the input1 feature vector
  std::vector<std::vector<float>> inputVectors1 =
      randomGenerator.genFloat2dVector(numSamples, input1Dims);
  std::vector<std::vector<float>> inputVectors2 =
      randomGenerator.genFloat2dVector(numSamples, input2Dims);

  auto indicesArrayVector1 = maker.arrayVector<float>(inputVectors1, REAL());
  auto inputRowVector1 = maker.rowVector({"in1"}, {indicesArrayVector1});
  auto indicesArrayVector2 = maker.arrayVector<float>(inputVectors2, REAL());
  auto inputRowVector2 = maker.rowVector({"in2"}, {indicesArrayVector2});

  auto weights1 = maker.flatVector<float>(input1Dims * input1NN);
  for (int i = 0; i < input1Dims * input1NN; i++) {
    weights1->set(i, randomGenerator.genRandomFloatValue());
  }

  auto weights2 = maker.flatVector<float>(input2Dims * input2NN);
  for (int i = 0; i < input2Dims * input2NN; i++) {
    weights2->set(i, randomGenerator.genRandomFloatValue());
  }

  // Print input
  std::cout << "[INFO] input1: \n"
            << inputRowVector1->toString(0, inputRowVector1->size())
            << std::endl;
  std::cout << "[INFO] input2: \n"
            << inputRowVector2->toString(0, inputRowVector2->size())
            << std::endl;

  exec::registerVectorFunction(
      "mat_mul1",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          weights1->values()->asMutable<float>(), input1Dims, input1NN));

  exec::registerVectorFunction(
      "mat_mul2",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          weights2->values()->asMutable<float>(), input2Dims, input2NN));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto myPlan1 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({inputRowVector1})
                     .project({"mat_mul1(in1) as o1"})
                     .planNode();

  auto myPlan2 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({inputRowVector2})
                     .project({"mat_mul2(in2) as o2"})
                     .planNode();

  auto results1 =
      exec::test::AssertQueryBuilder(myPlan1).copyResults(pool_.get());

  auto results2 =
      exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());

  auto t3 = maker.rowVector(
      {"o1", "o2"}, {results1->childAt(0), results2->childAt(0)});

  std::cout << "[INFO] t3: \n" << t3->toString(0, t3->size()) << std::endl;

  exec::registerVectorFunction(
      "concat",
      Concat::signatures(),
      std::make_unique<Concat>(input1NN, input2NN));

  auto myPlan3 = exec::test::PlanBuilder(pool_.get())
                     .values({t3})
                     .project({"concat(o1, o2)"})
                     .planNode();
  auto results3 =
      exec::test::AssertQueryBuilder(myPlan3).copyResults(pool_.get());
  std::cout << "[INFO] Results3: \n"
            << results3->toString(0, results3->size()) << std::endl;
};

void EmbeddingTest::testCosineSimilarity() {
  std::cout << "[INFO] Test of CosineSimilarity." << std::endl;
  int numSamples = 2;
  int input1Dims = 5;
  int input2Dims = 5;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1);

  // Initialize the input1 feature vector
  std::vector<std::vector<float>> inputVectors1 =
      randomGenerator.genFloat2dVector(numSamples, input1Dims);
  std::vector<std::vector<float>> inputVectors2 =
      randomGenerator.genFloat2dVector(numSamples, input2Dims);

  auto indicesArrayVector1 = maker.arrayVector<float>(inputVectors1, REAL());
  auto inputRowVector1 = maker.rowVector({"in1"}, {indicesArrayVector1});
  auto indicesArrayVector2 = maker.arrayVector<float>(inputVectors2, REAL());
  auto inputRowVector2 = maker.rowVector({"in2"}, {indicesArrayVector2});

  auto inputRowVector = maker.rowVector(
      {"in1", "in2"}, {indicesArrayVector1, indicesArrayVector2});

  // Print input
  std::cout << "[INFO] input: \n"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(input1Dims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"cosine_similarity(in1, in2)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results \n"
            << results->toString(0, results->size()) << std::endl;
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  EmbeddingTest demo;
  demo.testEmbedding();
  demo.testBatchNorm1D();
  demo.testDropout();
  demo.testConcat1();
  demo.testConcat2();
  demo.testCosineSimilarity();
}