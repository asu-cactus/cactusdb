/**
 * @file
 * @brief Implementation of a Hugging Face serverless API integration for machine learning tasks.
 * @copyright Copyright (c) 2025 ASU Cactus Lab.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cpr/cpr.h>
#include <json/json.h>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "BaseFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/UtilFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

/**
 * @enum HuggingFaceTaskType
 * @brief Enumeration of supported Hugging Face task types.
 */
enum HuggingFaceTaskType {
  TEXT_CLASSIFICATION,      ///< Text classification task.
  IMAGE_CLASSIFICATION,     ///< Image classification task.
  REGRESSION,               ///< Regression task.
  TEXT_FEATURE_EXTRACTION   ///< Text feature extraction task.
};

/**
 * @class HuggingFaceServerless
 * @brief Implements a machine learning function that interacts with Hugging Face's serverless API.
 */
class HuggingFaceServerless : public MLFunction {
 public:
  /**
   * @brief Constructor for HuggingFaceServerless.
   * @param apiURL The URL of the Hugging Face serverless API.
   * @param taskType The type of task to perform (e.g., text classification, feature extraction).
   */
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

  /**
   * @brief Destructor for HuggingFaceServerless.
   * Logs input/output token counts and failure statistics to a file.
   */
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

  /**
   * @brief Applies the Hugging Face serverless API function to the input data.
   * @param rows Selectivity vector indicating which rows to process.
   * @param args Vector of input arguments.
   * @param type Type of the output vector.
   * @param context Evaluation context.
   * @param output Output vector to store the results.
   */
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
    // HuggingFace itself suggests a maximum number of 10K, but the API
    // may be busy and fail. Try reducing the limit or deploy a dedicated endpoint.
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
        // Need to post the inputs to Hugging Face serverless API and get results
        strInputs += "]";
        auto huggingFaceInputs = "{\"inputs\": " + strInputs + "}";

        cpr::Response response = cpr::Post(
            cpr::Url{apiURL_},
            cpr::Body{huggingFaceInputs},
            cpr::Header{headers});

        if (response.status_code == 200) {
          // The response text can be parsed as JSON objects,
          // it should be a list of results for each input
          auto jsonObj = nlohmann::json::parse(response.text);
          int processedEmbeddingCount = 0;
          for (const auto& innerVector : jsonObj) {
            // Iterate response for each sample
            if (taskType_ == HuggingFaceTaskType::TEXT_CLASSIFICATION) {
              std::vector<float> floatVector(3);
              for (const auto& value : innerVector) {
                int dataIdx = 0;
                // TODO: Different models come with different return value names
                // Need more work here to handle such cases
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
              // Need case-by-case handling for different models
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
                // FIXME: Sometimes it returns an unfixed number of embeddings,
                // need further investigation
                for (const auto& value : returnedEmbedding) {
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
            } else {
              throw std::runtime_error(fmt::format(
                  "Current HuggingFace Task Type {} is not supported",
                  taskType_));
            }
          }
        } else {
          // Handle error cases
          std::cerr << "Error in fetching the results: "
                    << response.error.message << std::endl;
          for (int l = 0; i < accuInputCount; i++) {
            result[insertedDataIdx++] = {0.0};
          }
        }

        // Reset
        strInputs = "[";
        accuInputCount = 0;
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(result, REAL());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("array(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "huggingface";
  }

  /**
   * @brief Returns the tensor associated with the function.
   * @return Pointer to the tensor.
   */
  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  /**
   * @brief Estimates the cost of the function.
   * @param inputDims Dimensions of the input.
   * @return Cost estimate.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::string apiURL_; ///< URL of the Hugging Face serverless API.
  std::string apiToken_; ///< API token for Hugging Face.
  HuggingFaceTaskType taskType_; ///< Type of task to perform.
  uint64_t inputTokenNumber_; ///< Number of input tokens processed.
  uint64_t outputTokenNumber_; ///< Number of output tokens generated.
  uint64_t numFailures_; ///< Number of API failures encountered.
};
