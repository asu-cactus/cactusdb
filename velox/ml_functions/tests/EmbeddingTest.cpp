// TODO: Resolve dependencies
#define EIGEN_USE_BLAS

#include <folly/init/Init.h>
#include <torch/torch.h>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/BatchNorm.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/CosineSimilarity.h"
#include "velox/ml_functions/DotProduct.h"
#include "velox/ml_functions/Dropout.h"
#include "velox/ml_functions/Embedding.h"
#include "velox/ml_functions/Encoder.h"
#include "velox/ml_functions/HuggingFaceServerless.h"
#include "velox/ml_functions/PositionEncoding.h"
#include "velox/ml_functions/SequencePooling.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/optimizer/Helper.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

// Utility function to generate random float/int values

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
  void testConcat3();
  void testCosineSimilarity();
  void testDotProduct();
  void testPositionEncoding();
  void testSequencePooling();
  void testEmbedding_MatMul();
  void testHuggingFace();
  void testHuggingFaceTokenizer();

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

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};
  VectorMaker maker{pool_.get()};
};

// Test Embedding Layer
void EmbeddingTest::testEmbedding() {
  std::cout << "[INFO] Test of Embedding." << std::endl;
  int numEmbeddings = 5;
  int embeddingDims = 2;
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

  std::cout << "[INFO] Generated Input: \n"
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
  int numSamples = 1000;
  int input1Dims = 5000;
  int input2Dims = 2000;

  int input1NN = 2000;
  int input2NN = 3000;

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
  // std::cout << "[INFO] input1: \n"
  //           << inputRowVector1->toString(0, inputRowVector1->size())
  //           << std::endl;
  // std::cout << "[INFO] input2: \n"
  //           << inputRowVector2->toString(0, inputRowVector2->size())
  //           << std::endl;

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
  exec::registerVectorFunction(
      "concat",
      Concat::signatures(),
      std::make_unique<Concat>(input1NN, input2NN));

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
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

  auto myPlan3 = exec::test::PlanBuilder(pool_.get())
                     .values({t3})
                     .project({"concat(o1, o2)"})
                     .planNode();

  auto results3 =
      exec::test::AssertQueryBuilder(myPlan3).copyResults(pool_.get());

  // std::cout << "[INFO] t3: \n" << t3->toString(0, t3->size()) << std::endl;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Concat3 (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  // std::cout << "[INFO] Results3: \n"
  //           << results3->toString(0, results3->size()) << std::endl;
};

void EmbeddingTest::testConcat3() {
  std::cout << "[INFO] Test of Concat3." << std::endl;
  int numSamples = 1000;
  int input1Dims = 5000;
  int input2Dims = 2000;

  int input1NN = 2000;
  int input2NN = 3000;

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
  // std::cout << "[INFO] input1: \n"
  //           << inputRowVector1->toString(0, inputRowVector1->size())
  //           << std::endl;
  // std::cout << "[INFO] input2: \n"
  //           << inputRowVector2->toString(0, inputRowVector2->size())
  //           << std::endl;

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

  exec::registerVectorFunction(
      "concat",
      Concat::signatures(),
      std::make_unique<Concat>(input1NN, input2NN));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto myPlan1 = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({inputRowVector1})
                     .project({"mat_mul1(in1) as o1"})
                     .rowNumber({}, std::nullopt, true)
                     .planNode();

  auto myPlan2 =
      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
          .values({inputRowVector2})
          .project({"mat_mul2(in2) as o2"})
          .rowNumber({}, std::nullopt, true)
          .mergeJoin({"row_number"}, {"row_number"}, myPlan1, "", {"o1", "o2"})
          .project({"concat(o1,o2)"})
          .planNode();

  auto results3 =
      exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Concat3 (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;
  // std::cout << "[INFO] Results3: \n"
  //           << results3->toString(0, results3->size()) << std::endl;
};

void EmbeddingTest::testCosineSimilarity() {
  std::cout << "[INFO] Test of CosineSimilarity." << std::endl;
  int numSamples = 5;
  int input1Dims = 20;
  int input2Dims = 20;

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
  auto indicesArrayVector3 = maker.flatVector<int>({0, 1, 2, 3, 4});

  auto inputRowVector = maker.rowVector(
      {"in1", "in2", "id"},
      {indicesArrayVector1, indicesArrayVector2, indicesArrayVector3});

  auto inputRowVectorBatches =
      optimization::splitRowVectorIntoBatches(inputRowVector, 2);
  std::cout << "[INFO] Number of Batches: " << inputRowVectorBatches.size()
            << std::endl;

  // Print input
  std::cout << "[INFO] input: \n" << inputRowVector->toString() << std::endl;

  exec::registerVectorFunction(
      "cosine_similarity",
      CosineSimilarity::signatures(),
      std::make_unique<CosineSimilarity>(input1Dims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values(inputRowVectorBatches)
                    // .values({inputRowVector})
                    .filter("id > 0")
                    .project({"cosine_similarity(in1, in2)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results \n"
            << results->toString(0, results->size()) << std::endl;
};

void EmbeddingTest::testDotProduct() {
  std::cout << "[INFO] Test of DotProduct." << std::endl;
  int numSamples = 4;
  int inputDims = 5;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1);

  // Initialize the input1 feature vector
  std::vector<std::vector<float>> inputVectors1 =
      randomGenerator.genFloat2dVector(numSamples, inputDims);
  std::vector<std::vector<float>> inputVectors2 =
      randomGenerator.genFloat2dVector(numSamples, inputDims);

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
      "dot_product",
      DotProduct::signatures(),
      std::make_unique<DotProduct>(inputDims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"dot_product(in1, in2)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results \n"
            << results->toString(0, results->size()) << std::endl;
};

void EmbeddingTest::testPositionEncoding() {
  std::cout << "[INFO] Test of PositionEncoding." << std::endl;
  int numSamples = 4;
  int inputDims = 6;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1);

  // Initialize the input1 feature vector
  std::vector<std::vector<float>> inputVectors =
      randomGenerator.genFloat2dVector(numSamples, inputDims);

  auto indicesArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  auto inputRowVector = maker.rowVector({"in1"}, {indicesArrayVector});

  // Print input
  std::cout << "[INFO] input: \n"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "position_encoding",
      PositionEncoding::signatures(),
      std::make_unique<PositionEncoding>(inputDims));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"position_encoding(in1)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Results \n"
            << results->toString(0, results->size()) << std::endl;
};

void EmbeddingTest::testHuggingFace() {
  std::cout << "[INFO] Test of HuggingFace Serverless Inference." << std::endl;
  std::vector<std::string> sentences{};
  // Add positive sentences
  sentences.push_back("I really like the new design of your website!");
  sentences.push_back("Hulu has a great UI.");
  sentences.push_back(
      "The final episode was surprising with a fantastic twist at the end.");

  // Add neutral sentences
  sentences.push_back("I'm not sure if I like the new design.");
  sentences.push_back("Disliking horror movies is not uncommon.");
  sentences.push_back("Sometimes I find the show interesting.");

  // Add negative sentences
  sentences.push_back("The new design is awful!");
  sentences.push_back("I dislike horror movies.");
  sentences.push_back(
      "Having to wait two months for the next series to come out is frustrating.");
  auto sentenceFlatVector = maker.flatVector<std::string>(sentences);
  auto inputRowVector = maker.rowVector({"in1"}, {sentenceFlatVector});

  // Print input
  std::cout << "[INFO] input: \n"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  std::string textEmbeddingExtractionAPI =
      "https://api-inference.huggingface.co/pipeline/feature-extraction/cardiffnlp/twitter-roberta-base-sentiment-latest";

  exec::registerVectorFunction(
      "hf_embedding_extractor",
      HuggingFaceServerless::signatures(),
      std::make_unique<HuggingFaceServerless>(
          textEmbeddingExtractionAPI,
          HuggingFaceTaskType::TEXT_FEATURE_EXTRACTION));

  std::string textEmbeddingExtractionMiniLMAPI =
      "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2";

  exec::registerVectorFunction(
      "hf_minilm_embedding_extractor",
      HuggingFaceServerless::signatures(),
      std::make_unique<HuggingFaceServerless>(
          textEmbeddingExtractionMiniLMAPI,
          HuggingFaceTaskType::TEXT_FEATURE_EXTRACTION));

  std::string textClassificationAPI =
      "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest";
  exec::registerVectorFunction(
      "hf_sentiment_classifier",
      HuggingFaceServerless::signatures(),
      std::make_unique<HuggingFaceServerless>(
          textClassificationAPI, HuggingFaceTaskType::TEXT_CLASSIFICATION));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"in1", "hf_sentiment_classifier(in1)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Sentiment Classification Results \n\n"
            << results->toString(0, results->size()) << "\n\n"
            << std::endl;

  auto myPlan1 = exec::test::PlanBuilder(pool_.get())
                     .values({inputRowVector})
                     .project({"in1", "hf_embedding_extractor(in1)"})
                     .planNode();

  auto results1 =
      exec::test::AssertQueryBuilder(myPlan1).copyResults(pool_.get());

  std::cout << "[INFO] Embedding Extraction Results: \n\n\n"
            << results1->toString(0, results1->size()) << std::endl;

  auto myPlan2 = exec::test::PlanBuilder(pool_.get())
                     .values({inputRowVector})
                     .project({"in1", "hf_minilm_embedding_extractor(in1)"})
                     .planNode();

  auto results2 =
      exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());

  std::cout << "[INFO] MiniLM L6 Embedding Extraction Results: \n\n\n"
            << results2->toString(0, results2->size()) << std::endl;
};

// Test Embedding Layer
void EmbeddingTest::testSequencePooling() {
  std::cout << "[INFO] Test of SequencePooling." << std::endl;
  int numEmbeddings = std::atoi(std::getenv("numEmbeddings"));
  int embeddingDims = std::atoi(std::getenv("embeddingDims"));
  int numSamples = std::atoi(std::getenv("numSamples"));
  int maxSampleLen = std::atoi(std::getenv("maxInputLen"));

  std::cout << numEmbeddings << " " << embeddingDims << " " << numSamples << " "
            << maxSampleLen;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  randomGenerator.setIntRange(0, numEmbeddings - 1);

  std::vector<std::vector<float>> weights =
      randomGenerator.genFloat2dVector(numEmbeddings, embeddingDims);
  auto weightsVector = maker.arrayVector<float>(weights, REAL());

  // Initialize the indices vector
  std::vector<std::vector<int>> indicesVectors;
  for (int i = 0; i < numSamples; i++) {
    std::vector<int> inputVector;
    int numI = std::max(1, i % maxSampleLen);

    for (int j = 0; j < numI; j++) {
      inputVector.push_back(randomGenerator.genRandomIntValue());
    }
    indicesVectors.push_back(inputVector);
  }

  auto indicesArrayVector = maker.arrayVector<int>(indicesVectors, INTEGER());
  auto inputRowVector = maker.rowVector({"x"}, {indicesArrayVector});

  // std::cout << "[INFO] Generated Embedding:\n"
  //           << weightsVector->toString(0, weightsVector->size()) <<
  //           std::endl;

  // std::cout << "[INFO] Generated Indices:\n"
  //           << inputRowVector->toString(0, inputRowVector->size()) <<
  //           std::endl;

  exec::registerVectorFunction(
      "embedding",
      Embedding::signatures(),
      std::make_unique<Embedding>(
          weightsVector->elements()->values()->asMutable<float>(),
          numEmbeddings,
          embeddingDims));

  exec::registerVectorFunction(
      "sequence_pooling",
      SequencePooling::signatures(),
      std::make_unique<SequencePooling>(std::string("MEAN"), embeddingDims));

  auto myPlan1 = exec::test::PlanBuilder(pool_.get())
                     .values({inputRowVector})
                     .project({"sequence_pooling(embedding(x))"})
                     .planNode();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto result1 =
      exec::test::AssertQueryBuilder(myPlan1).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for SequencePooling (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;

  // std::cout << "[INFO] Result1: \n"
  //           << result1->toString(0, result1->size()) << std::endl;

  // auto myPlan2 = exec::test::PlanBuilder(pool_.get())
  //                    .values({result1})
  //                    .project({"sequence_pooling(o1)"})
  //                    .planNode();

  // auto result2 =
  //     exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());

  // std::cout << "[INFO] Result2: \n"
  //           << result2->toString(0, result2->size()) << std::endl;
};

// Test Embedding Layer
// 6.75
void EmbeddingTest::testEmbedding_MatMul() {
  std::cout << "[INFO] Test of Embedding Mat Mul." << std::endl;
  int numEmbeddings = std::atoi(std::getenv("numEmbeddings"));
  int embeddingDims = std::atoi(std::getenv("embeddingDims"));
  int numSamples = std::atoi(std::getenv("numSamples"));

  int maxSampleLen = std::atoi(std::getenv("maxInputLen"));
  std::cout << numEmbeddings << " " << embeddingDims << " " << numSamples << " "
            << maxSampleLen;

  RandomGenerator randomGenerator = RandomGenerator(-1, 1, 0);
  randomGenerator.setIntRange(0, numEmbeddings - 1);

  std::vector<float> weights =
      randomGenerator.genFloat1dVector(numEmbeddings * embeddingDims);

  auto weightsVector = maker.flatVector<float>(weights, REAL());

  // Initialize the indices vector
  std::vector<std::vector<int>> indicesVectors;
  for (int i = 0; i < numSamples; i++) {
    std::vector<int> inputVector;
    int numI = std::max(1, i % maxSampleLen);

    for (int j = 0; j < numI; j++) {
      inputVector.push_back(randomGenerator.genRandomIntValue());
    }
    indicesVectors.push_back(inputVector);
  }

  auto indicesArrayVector = maker.arrayVector<int>(indicesVectors, INTEGER());
  auto inputRowVector = maker.rowVector({"x"}, {indicesArrayVector});

  // std::cout << "[INFO] Generated Embedding:\n"
  //           << weightsVector->toString(0, weightsVector->size()) <<
  //           std::endl;

  // std::cout << "[INFO] Generated Indices:\n"
  //           << inputRowVector->toString(0, inputRowVector->size()) <<
  //           std::endl;

  exec::registerVectorFunction(
      "multi_hot_norm_encoder",
      MultiHotNormalizedEncoder::signatures(),
      std::make_unique<MultiHotNormalizedEncoder>(numEmbeddings));

  exec::registerVectorFunction(
      "mat_mul",
      MatrixMultiply::signatures(),
      std::make_unique<MatrixMultiply>(
          weightsVector->values()->asMutable<float>(),
          numEmbeddings,
          embeddingDims));

  auto myPlan1 = exec::test::PlanBuilder(pool_.get())
                     .values({inputRowVector})
                     .project({"mat_mul(multi_hot_norm_encoder(x))"})
                     .planNode();

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();
  auto result1 =
      exec::test::AssertQueryBuilder(myPlan1).copyResults(pool_.get());
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time for Embedding mat mul (sec) = "
            << (std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count()) /
          1000000.0
            << std::endl;

  // std::cout << "[INFO] Result1: \n"
  //           << result1->toString(0, result1->size()) << std::endl;

  // steps : normalised 1 hot encoding E.g [0 1 0 1 0 0] sould be divided by 2
  // since there are 2 1s multiply by embedding matrix matrix [ n * vector_dims]
  // auto myPlan2 = exec::test::PlanBuilder(pool_.get())
  //                    .values({result1})
  //                    .project({"sequence_pooling(o1)"})
  //                    .planNode();

  // auto result2 =
  //     exec::test::AssertQueryBuilder(myPlan2).copyResults(pool_.get());

  // std::cout << "[INFO] Result2: \n"
  //           << result2->toString(0, result2->size()) << std::endl;
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  EmbeddingTest demo;
  // demo.testEmbedding();
  // demo.testBatchNorm1D();
  // demo.testDropout();
  // demo.testConcat1();
  // demo.testConcat2();
  // demo.testConcat3();
  // demo.testCosineSimilarity();
  // demo.testEmbedding_MatMul();
  // demo.testSequencePooling();
  // demo.testDotProduct();
  // demo.testPositionEncoding();
  demo.testHuggingFace();
}