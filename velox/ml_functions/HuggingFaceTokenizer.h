#pragma once
#include <fmt/format.h>
#include <tokenizers_cpp.h>
#include <iostream>
#include "functions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using tokenizers::Tokenizer;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

class HuggingFaceTokenizer : public MLFunction {
 public:
  HuggingFaceTokenizer(std::string pathToTokenizer) {
    pathToTokenizer_ = pathToTokenizer;
    auto blob = LoadBytesFromFile(pathToTokenizer_);
    tokenizer_ = Tokenizer::FromBlobJSON(blob);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    // Read string input
    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();
    int numInputs = rows.size();

    std::vector<std::vector<int>> result;
    for (int i = 0; i < numInputs; i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::vector<int> ids = tokenizer_->Encode(val);

      result.push_back(ids);
    }
    VectorMaker maker{context.pool()};
    output = maker.arrayVector<int>(result, INTEGER());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("array(INTEGER)")
                .build()};
  }

  static std::string getName() {
    return "hf_tokenizer";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::string pathToTokenizer_;
  std::unique_ptr<Tokenizer> tokenizer_;
};
