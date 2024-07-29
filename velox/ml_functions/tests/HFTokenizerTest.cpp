// TODO: Resolve dependencies
#define EIGEN_USE_BLAS

#include <folly/init/Init.h>
#include <torch/torch.h>
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/HuggingFaceTokenizer.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

// Utility function to generate random float/int values

class HFTokenizerTest : public HiveConnectorTestBase {
 public:
  HFTokenizerTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

    SetUp();
  }

  ~HFTokenizerTest() {}

  void run();
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

void HFTokenizerTest::testHuggingFaceTokenizer() {
  std::cout << "[INFO] Test of HuggingFace Tokenizer." << std::endl;
  std::vector<std::string> sentences{};
  // Add positive sentences
  sentences.push_back("I really like the new design of your website!");
  sentences.push_back("Hulu has a great UI.");
  sentences.push_back(
      "The final episode was surprising with a fantastic twist at the end.");
  sentences.push_back("I'm not sure if I like the new design.");
  sentences.push_back("Disliking horror movies is not uncommon.");
  sentences.push_back("Sometimes I find the show interesting.");
  sentences.push_back("The new design is awful!");
  sentences.push_back("I dislike horror movies.");
  sentences.push_back("Having to wait two months for the next series to come out is frustrating.");
  auto sentenceFlatVector = maker.flatVector<std::string>(sentences);
  auto inputRowVector = maker.rowVector({"in1"}, {sentenceFlatVector});

  // Print input
  std::cout << "[INFO] input: \n"
            << inputRowVector->toString(0, inputRowVector->size()) << std::endl;

  exec::registerVectorFunction(
      "hf_tokenizer",
      HuggingFaceTokenizer::signatures(),
      std::make_unique<HuggingFaceTokenizer>("/home/velox/resources/model/tokenizer/roberta.json"));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                    .values({inputRowVector})
                    .project({"in1", "hf_tokenizer(in1)"})
                    .planNode();

  auto results =
      exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());

  std::cout << "[INFO] Tokenized results: \n\n\n"
            << results->toString(0, results->size()) << std::endl;
};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});
  HFTokenizerTest demo;
  demo.testHuggingFaceTokenizer();
}