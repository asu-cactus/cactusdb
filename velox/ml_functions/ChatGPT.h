#pragma once
#include <cpr/cpr.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include "functions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

#define MAX_ALLOWED_CHATGPT_TRY 30

std::string getEnvVar(std::string const& key) {
  char const* val = getenv(key.c_str());
  return val == NULL ? std::string() : std::string(val);
}

// Function to count the number of words in a string
int countWords(const std::string& str) {
  std::istringstream iss(str);
  int wordCount = 0;
  std::string word;
  while (iss >> word) {
    ++wordCount;
  }
  return wordCount;
}

// Function to count the number of punctuation marks in a string
int countPunctuation(const std::string& str) {
  int punctuationCount = 0;
  for (char ch : str) {
    if (std::ispunct(ch)) {
      ++punctuationCount;
    }
  }
  return punctuationCount;
}

void sendRequestViaCpr(
    const std::string& url,
    const cpr::Header& headers,
    const std::string& payload,
    cpr::Response& response,
    int& numFailures) {
  response = cpr::Post(cpr::Url{url}, cpr::Header{headers}, cpr::Body{payload});
  int failureCount = 0;
  while (response.status_code != 200) {
    response =
        cpr::Post(cpr::Url{url}, cpr::Header{headers}, cpr::Body{payload});
    if (++failureCount > MAX_ALLOWED_CHATGPT_TRY) {
      throw std::runtime_error(fmt::format(
          "[ERROR] Failed to send request to OpenAI API. Reached maximum number of retries: {}, error message: {}",
          MAX_ALLOWED_CHATGPT_TRY,
          payload));
    }
  }
  numFailures = failureCount;
}

void sendRequestViaCprBatch(
    const std::string& url,
    const cpr::Header& headers,
    const std::vector<std::string>& payloadVector,
    std::vector<cpr::Response>& responseVector,
    const int& startIdx,
    std::vector<int>& numFailureVector) {
  int index = startIdx;
  for (std::string payload : payloadVector) {
    int failureCount = 0;
    auto response =
        cpr::Post(cpr::Url{url}, cpr::Header{headers}, cpr::Body{payload});
    while (response.status_code != 200) {
      response =
          cpr::Post(cpr::Url{url}, cpr::Header{headers}, cpr::Body{payload});
      if (++failureCount > MAX_ALLOWED_CHATGPT_TRY) {
        throw std::runtime_error(fmt::format(
            "[ERROR] Failed to send request to OpenAI API. Reached maximum number of retries: {}, error message: {}",
            MAX_ALLOWED_CHATGPT_TRY,
            payload));
      }
    }
    responseVector[index] = response;
    numFailureVector[index] = failureCount;
    index++;
  }
}

class ChatGPT : public MLFunction {
 public:
  ChatGPT() {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    numThreads_ = getEnvVar("NUM_THREADS") == ""
        ? 8
        : std::stoi(getEnvVar("NUM_THREADS"));
    url_ = "https://api.openai.com/v1/chat/completions";
    model_ = "gpt-3.5-turbo";
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
    numFailures_ = 0;
  }

  ChatGPT(std::string url, std::string model) {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    numThreads_ = getEnvVar("NUM_THREADS") == ""
        ? 8
        : std::stoi(getEnvVar("NUM_THREADS"));
    url_ = url;
    model_ = model;
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
    numFailures_ = 0;
  }

  ~ChatGPT() {
    std::string filename = "chatgpt.log";
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
    file << "# Input:" << inputTokenNumber_
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
    std::string promptPrefix = "";
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();

    int numInput = rows.size();
    int numSelected = rows.countSelected();
    LOG(INFO) << "[INFO ChatGPT:] countSelected: " << rows.countSelected() << " numInput: " << numInput << std::endl;
    

    if (args.size() == 2) {
      exec::LocalDecodedVector decodedStringHolder(context, *args[1], rows);
      auto decodedStringInput = decodedStringHolder.get();
      StringView val = decodedStringInput->valueAt<StringView>(0);
      promptPrefix = std::string(val);
    }

    std::vector<std::string> results;

    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + apiKey_}};

    // Thread vector
    std::vector<std::thread> threads;
    std::vector<cpr::Response> responses(numInput);
    std::vector<int> numFailureVector(numInput);

    int numInputsPerThread = int(std::ceil(float(numSelected) / numThreads_));
    int processedInputCount = 0;
    std::vector<std::string> payloadsBatchVector;
    int processedIndex = 0;

    // Version 1
    // This approach is more efficient by sending requests in batches and leveraging
    // multiple threads to send requests concurrently, it requires additional 
    // isValid check to skip the rows that are not selected. Note: at the end of this
    // approach, it is required to invoke context.moveOrCopyResult to copy the results
    // back to the output vector since we only compute the results for selected ones
    for (int i = 0; i < numInput; i++) {
      // if the row is not selected, skip
      if (!rows.isValid(i)) {
        continue;
      }
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::string valString = promptPrefix + std::string(val);
      nlohmann::json messageArrays = nlohmann::json::array();
      // Add message
      messageArrays.push_back({{"role", "user"}, {"content", valString}});

      nlohmann::json payload = {
          {"model", model_}, {"messages", messageArrays}, {"max_tokens", 150}};

      payloadsBatchVector.push_back(payload.dump());
      processedInputCount++;

      if (processedInputCount == numInputsPerThread || i == numInput - 1) {
        threads.emplace_back(
            sendRequestViaCprBatch,
            url_,
            headers,
            payloadsBatchVector,
            std::ref(responses),
            processedIndex - processedInputCount + 1,
            std::ref(numFailureVector));
        processedInputCount = 0;
        payloadsBatchVector.clear();
      }
      processedIndex++;
    }

    for (auto& thread : threads) {
      thread.join();
    }

    for (int i = 0; i < numSelected; i++) {
      if (responses[i].status_code == 200) {
        // parse the returned value
        nlohmann::json response_json = nlohmann::json::parse(responses[i].text);
        std::string generated_message =
            response_json["choices"][0]["message"]["content"];
        results.push_back(generated_message);
        const_cast<uint64_t&>(inputTokenNumber_) = inputTokenNumber_ +
          response_json["usage"]["prompt_tokens"].get<int>();
        const_cast<uint64_t&>(outputTokenNumber_) = outputTokenNumber_ +
            response_json["usage"]["completion_tokens"].get<int>();
        const_cast<uint64_t&>(numFailures_) =
            numFailures_ + numFailureVector[i];
        if (numFailureVector[i] > 0) {
          LOG(WARNING)
              << "[WARNING] Failed to send request to OpenAI API. Number of retries: "
              << numFailureVector[i] << " numFailures_: " << numFailures_
              << std::endl;
        }
        LOG(INFO) << fmt::format(
                         "[INFO] i: {} / {}, results: {}, numFailures: {}",
                         i + 1,
                         numSelected,
                         generated_message,
                         numFailureVector[i])
                  << std::endl;
      } else {
        LOG(ERROR) << "Error: " << responses[i].status_code << " - "
                   << responses[i].text << std::endl;
      }
    }

    VectorMaker maker{context.pool()};
    VectorPtr localResult = maker.flatVector<std::string>(results);

    context.moveOrCopyResult(localResult, rows, output);

    // Version 2: Leveraging applyToSelected function, while this is done
    // sequentially, it is easier to implement and debug but less efficient
    /*
    auto flatResult = output->asFlatVector<StringView>();
    rows.applyToSelected([&](vector_size_t row) {
      StringView val = decodedStringInput->valueAt<StringView>(row);
      std::string valString = promptPrefix + std::string(val);
      nlohmann::json messageArrays = nlohmann::json::array();
      // Add message
      messageArrays.push_back({{"role", "user"}, {"content", valString}});

      nlohmann::json payload = {
          {"model", model_}, {"messages", messageArrays}, {"max_tokens", 150}};
      sendRequestViaCpr(url_, headers, payload.dump(), responses[row], numFailureVector[row]);

      // parse the returned value
      nlohmann::json response_json = nlohmann::json::parse(responses[row].text);
        std::string generated_message =
            response_json["choices"][0]["message"]["content"];
        flatResult->set(row, StringView(generated_message));
        // results.push_back(generated_message);
        const_cast<uint64_t&>(inputTokenNumber_) = inputTokenNumber_ +
          response_json["usage"]["prompt_tokens"].get<int>();
        const_cast<uint64_t&>(outputTokenNumber_) = outputTokenNumber_ +
            response_json["usage"]["completion_tokens"].get<int>();
        const_cast<uint64_t&>(numFailures_) =
            numFailures_ + numFailureVector[row];
      LOG(INFO) << fmt::format(
                         "[INFO] Selected row: {} / {}, results: {}, numFailures: {}",
                         row + 1,
                         numSelected,
                         generated_message,
                         numFailureVector[row])
                  << std::endl;
    });
    */
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .argumentType("VARCHAR")
            .returnType("VARCHAR")
            .build(),
        // supports with prompt prefix
        exec::FunctionSignatureBuilder()
            .argumentType("VARCHAR")
            .argumentType("VARCHAR")
            .returnType("VARCHAR")
            .build()};
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  static std::string getName() {
    return "chatgpt";
  };

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::string apiKey_;
  std::string url_;
  std::string model_;
  uint64_t inputTokenNumber_;
  uint64_t outputTokenNumber_;
  uint64_t numFailures_;
  int numThreads_;
};

class ChatGPTRecommender : public MLFunction {
 public:
  ChatGPTRecommender() {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    numThreads_ = getEnvVar("NUM_THREADS") == ""
        ? 8
        : std::stoi(getEnvVar("NUM_THREADS"));
    url_ = "https://api.openai.com/v1/chat/completions";
    model_ = "gpt-3.5-turbo";
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
    numFailures_ = 0;
  }

  ChatGPTRecommender(std::string url, std::string model) {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    numThreads_ = getEnvVar("NUM_THREADS") == ""
        ? 8
        : std::stoi(getEnvVar("NUM_THREADS"));
    url_ = url;
    model_ = model;
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
    numFailures_ = 0;
  }

  ~ChatGPTRecommender() {
    std::string filename = "chatgpt_recommender.log";
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
    file << "# Input:" << inputTokenNumber_
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
    std::string promptSuffix = "";
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    exec::LocalDecodedVector decodedStringHolder1(context, *args[0], rows);
    auto decodedStringInput1 = decodedStringHolder1.get();

    exec::LocalDecodedVector decodedStringHolder2(context, *args[1], rows);
    auto decodedStringInput2 = decodedStringHolder2.get();

    int numInput = rows.size();
    int numSelected = rows.countSelected();
    LOG(INFO) << "[INFO ChatGPTRecommender:] countSelected: " << rows.countSelected() << " numInput: " << numInput << std::endl;

    if (args.size() == 3) {
      exec::LocalDecodedVector decodedStringHolder3(context, *args[2], rows);
      auto decodedStringInput3 = decodedStringHolder3.get();
      StringView val = decodedStringInput3->valueAt<StringView>(0);
      promptSuffix = std::string(val);
    }

    std::vector<std::string> results;

    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Authorization", "Bearer " + apiKey_}};

    // Thread vector
    std::vector<std::thread> threads;
    std::vector<cpr::Response> responses(numInput);
    std::vector<int> numFailureVector(numInput);

    int numInputsPerThread = int(std::ceil(float(numSelected) / numThreads_));
    int processedInputCount = 0;
    std::vector<std::string> payloadsBatchVector;
    int processedIndex = 0;

    for (int i = 0; i < numInput; i++) {
      // if the row is not selected, skip
      if (!rows.isValid(i)) {
        continue;
      }
      StringView val1 = decodedStringInput1->valueAt<StringView>(i);
      StringView val2 = decodedStringInput2->valueAt<StringView>(i);
      std::string valString =
          "Summarized user statistics data (preference): " + std::string(val1) +
          ". \n Summarized user movie metadata:  " + std::string(val2) + ".\n" +
          promptSuffix;
      nlohmann::json messageArrays = nlohmann::json::array();
      // Add message
      messageArrays.push_back({{"role", "user"}, {"content", valString}});

      nlohmann::json payload = {
          {"model", model_}, {"messages", messageArrays}, {"max_tokens", 500}};

      payloadsBatchVector.push_back(payload.dump());
      processedInputCount++;

      if (processedInputCount == numInputsPerThread || i == numInput - 1) {
        threads.emplace_back(
            sendRequestViaCprBatch,
            url_,
            headers,
            payloadsBatchVector,
            std::ref(responses),
            processedIndex - processedInputCount + 1,
            std::ref(numFailureVector));
        processedInputCount = 0;
        payloadsBatchVector.clear();
      }
      processedIndex ++;
    }

    for (auto& thread : threads) {
      thread.join();
    }

    for (int i = 0; i < numSelected; i++) {
      if (responses[i].status_code == 200) {
        // parse the returned value
        nlohmann::json response_json = nlohmann::json::parse(responses[i].text);
        std::string generated_message =
            response_json["choices"][0]["message"]["content"];
        results.push_back(generated_message);
        const_cast<uint64_t&>(inputTokenNumber_) = inputTokenNumber_ +
          response_json["usage"]["prompt_tokens"].get<int>();
        const_cast<uint64_t&>(outputTokenNumber_) = outputTokenNumber_ +
            response_json["usage"]["completion_tokens"].get<int>();
        const_cast<uint64_t&>(numFailures_) =
            numFailures_ + numFailureVector[i];
        if (numFailureVector[i] > 0) {
          LOG(WARNING)
              << "[WARNING] Failed to send request to OpenAI API. Number of retries: "
              << numFailureVector[i] << " numFailures_: " << numFailures_
              << std::endl;
        }
        LOG(INFO) << fmt::format(
                         "[INFO] i: {} / {}, results: {}, numFailures: {}",
                         i + 1,
                         numSelected,
                         generated_message,
                         numFailureVector[i])
                  << std::endl;
      } else {
        LOG(ERROR) << "Error: " << responses[i].status_code << " - "
                   << responses[i].text << std::endl;
      }
    }

    VectorMaker maker{context.pool()};
    VectorPtr localResult = maker.flatVector<std::string>(results);

    context.moveOrCopyResult(localResult, rows, output);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {
        exec::FunctionSignatureBuilder()
            .argumentType("VARCHAR")
            .argumentType("VARCHAR")
            .returnType("VARCHAR")
            .build(),
        // supports with prompt suffix
        exec::FunctionSignatureBuilder()
            .argumentType("VARCHAR")
            .argumentType("VARCHAR")
            .argumentType("VARCHAR")
            .returnType("VARCHAR")
            .build()};
  }

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  static std::string getName() {
    return "chatgpt_recommender";
  };

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  std::string apiKey_;
  std::string url_;
  std::string model_;
  uint64_t inputTokenNumber_;
  uint64_t outputTokenNumber_;
  uint64_t numFailures_;
  int numThreads_;
};