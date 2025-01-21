#pragma once
#include <cpr/cpr.h>
#include <json/json.h>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "functions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

enum HuggingFaceTaskType {
  TEXT_CLASSIFICATION,
  IMAGE_CLASSIFICATION,
  REGRESSION,
  TEXT_FEATURE_EXTRACTION
};

class HuggingFaceServerless : public MLFunction {
 public:
  HuggingFaceServerless(std::string apiURL, HuggingFaceTaskType taskType) {
    apiURL_ = apiURL;
    taskType_ = taskType;
    apiToken_ = getEnvVar("HF_TOKEN");
    if (apiToken_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] HuggingFace token is not set, please set HF_TOKEN"));
    }
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
    numFailures_ = 0;
  }

  ~HuggingFaceServerless() {
    std::string filename = "huggingfaceServerless.log";
    std::ofstream file(filename, std::ios::app);
    if (!file) {
      std::cerr << "Unable to open file: " << filename << std::endl;
      return;
    }
    // Get the current time
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    // Write the timestamp to the file
    file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " ";

    // Write the uint64_t values to the file
    file << "[HuggingFaceServerless] # Input:" << inputTokenNumber_
         << " # Output: " << outputTokenNumber_
         << " # NumFailure: " << numFailures_ << std::endl;
    file.close();
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

    cpr::Header headers{{"Authorization", fmt::format("Bearer {}", apiToken_)}};

    std::vector<std::vector<float>> result(numInputs);

    // Limit of number of inputs can be sent to serverless API at once
    // HuggingFace itself suggest maximum number is 10K, sometime the API
    // itself is busy and cannot work. Try reduce the limit or deploy a
    // dedicated endpoint
    const int HF_SERVERLESS_INPUT_LIMIT = 5000;

    // HuggingFace inputs are formatted as follows:
    // "inputs": ["Sentence 1", "Sentence 2"],
    std::string strInputs = "[";
    int accuInputCount = 0;
    int insertedDataIdx = 0;
    for (int i = 0; i < numInputs; i++) {
      std::string valString =
          std::string(decodedStringInput->valueAt<StringView>(i));
      const_cast<uint64_t&>(inputTokenNumber_) =
          inputTokenNumber_ + countWords(valString);

      strInputs += "\"" + valString + "\"";
      accuInputCount += 1;

      if (i != (numInputs - 1) && accuInputCount != HF_SERVERLESS_INPUT_LIMIT) {
        strInputs += ",";
      } else {
        // need to post the inputs to huggingface serverless api and get results
        strInputs += "]";
        auto huggingFaceInputs = "{\"inputs\": " + strInputs + "}";

        cpr::Response response = cpr::Post(
            cpr::Url{apiURL_},
            cpr::Body{huggingFaceInputs},
            cpr::Header{headers});

        if (response.status_code == 200) {
          // The response text can be parsed as json objects,
          // it should be a list of result for each input
          auto jsonObj = nlohmann::json::parse(response.text);
          int processedEmbeddingCount = 0;
          for (const auto& innerVector : jsonObj) {
            // iterate response for each sample
            if (taskType_ == HuggingFaceTaskType::TEXT_CLASSIFICATION) {
              std::vector<float> floatVector(3);
              for (const auto& value : innerVector) {
                int dataIdx = 0;
                // TODO: Different model comes with different return value name
                // need more work here to handle such things
                if (value["label"] == "positive") {
                  dataIdx = 0;
                } else if (value["label"] == "neutral") {
                  dataIdx = 1;
                } else if (value["label"] == "negative") {
                  dataIdx = 2;
                }
                floatVector[dataIdx] = value["score"];
              }
              result[insertedDataIdx++] = floatVector;
            } else if (
                taskType_ == HuggingFaceTaskType::TEXT_FEATURE_EXTRACTION) {
              if (processedEmbeddingCount == accuInputCount) {
                break;
              }
              // Need a case-by-case handling for different model
              if (apiURL_.find("all-MiniLM") != std::string::npos) {
                auto returnedEmbedding = innerVector;
                std::vector<float> embeddingVector;
                for (const auto& val : returnedEmbedding) {
                  embeddingVector.push_back(val);
                }
                const_cast<uint64_t&>(outputTokenNumber_) =
                    outputTokenNumber_ + embeddingVector.size();
                processedEmbeddingCount += 1;
                result[insertedDataIdx++] = embeddingVector;
                if (processedEmbeddingCount == accuInputCount) {
                  break;
                }
              } else {
                auto returnedEmbedding = innerVector[0];
                // FIXME sometimes it returns unfixed number of embeeding, need
                // further investigation
                for (const auto& value : returnedEmbedding) {
                  // std::cout << "[DEBUG]: " << value << std::endl;
                  std::vector<float> embeddingVector;
                  for (const auto& val : value) {
                    embeddingVector.push_back(val);
                  }
                  processedEmbeddingCount += 1;
                  result[insertedDataIdx++] = embeddingVector;
                  if (processedEmbeddingCount == accuInputCount) {
                    break;
                  }
                }
              }
              // std::cout << "[DEBUG]: i: " << i << " innerVector.size: " <<
              // innerVector.size() << std::endl; std::cout << "[DEBUG]: i: " <<
              // i << " innerVector[0].size: " << innerVector[0].size() <<
              // std::endl; auto returnedEmbedding = innerVector[0];
            } else {
              throw std::runtime_error(fmt::format(
                  "Current HuggingFace Task Type {} is not supported",
                  taskType_));
            }
          }
        } else {
          // Handle error cases
          std::cerr << "Error in fetchting the results: "
                    << response.error.message << std::endl;
          for (int l = 0; i < accuInputCount; i++) {
            result[insertedDataIdx++] = {0.0};
          }
        }

        // reset
        strInputs = "[";
        accuInputCount = 0;
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("array(REAL)")
                .build()};
  }
  static std::string getName() {
    return "huggingface";
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::string apiURL_;
  std::string apiToken_;
  HuggingFaceTaskType taskType_;
  uint64_t inputTokenNumber_;
  uint64_t outputTokenNumber_;
  uint64_t numFailures_;
};