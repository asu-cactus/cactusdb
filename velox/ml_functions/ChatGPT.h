#pragma once
#include <fmt/format.h>
#include "functions.h"
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <ctime>
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

class ChatGPT : public MLFunction {
 public:
  ChatGPT() {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    url_ = "https://api.openai.com/v1/chat/completions";
    model_ = "gpt-3.5-turbo";
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
  }

  ChatGPT(std::string url, std::string model) {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    url_ = url;
    model_ = model;
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
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
    file << "# Input:" <<  inputTokenNumber_ << " # Output: " << outputTokenNumber_ << std::endl;
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

    for (int i = 0; i < numInput; i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::string valString = promptPrefix + std::string(val);
      nlohmann::json messageArrays = nlohmann::json::array();
      // Add message
      messageArrays.push_back({{"role", "user"}, {"content", valString}});

      const_cast<uint64_t&>(inputTokenNumber_) = inputTokenNumber_ + countWords(valString) + countPunctuation(valString);

      nlohmann::json payload = {{"model", model_}, {"messages", messageArrays}, {"max_tokens", 150}};

      cpr::Response response = cpr::Post(
          cpr::Url{url_}, cpr::Header{headers}, cpr::Body{payload.dump()});
      int failureCount = 0;
      // retry
      while (response.status_code != 200) {
        response = cpr::Post(
          cpr::Url{url_}, cpr::Header{headers}, cpr::Body{payload.dump()});
        if (failureCount++ > 10) {
          break;
        }
      }
      if (response.status_code == 200) {
        // parse the returned value
        nlohmann::json response_json = nlohmann::json::parse(response.text);
        std::string generated_message =
            response_json["choices"][0]["message"]["content"];
        results.push_back(generated_message);
        const_cast<uint64_t&>(outputTokenNumber_) = outputTokenNumber_ + countWords(generated_message) + countPunctuation(generated_message);
        LOG(INFO) << fmt::format("[INFO] i: {} / {}, results: {}", i+1, numInput, generated_message) << std::endl;
      } else {
        std::cout << "Error: " << response.status_code << " - " << response.text
                  << std::endl;
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<std::string>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
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
};

class ChatGPTRecommender : public MLFunction {
 public:
  ChatGPTRecommender() {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    url_ = "https://api.openai.com/v1/chat/completions";
    model_ = "gpt-3.5-turbo";
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
  }

  ChatGPTRecommender(std::string url, std::string model) {
    apiKey_ = getEnvVar("OPENAI_API_KEY");
    if (apiKey_ == "") {
      throw std::runtime_error(fmt::format(
          "[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY"));
    }
    url_ = url;
    model_ = model;
    inputTokenNumber_ = 0;
    outputTokenNumber_ = 0;
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
    file << "# Input:" <<  inputTokenNumber_ << " # Output: " << outputTokenNumber_ << std::endl;
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

    for (int i = 0; i < numInput; i++) {
      StringView val1 = decodedStringInput1->valueAt<StringView>(i);
      StringView val2 = decodedStringInput2->valueAt<StringView>(i);
      std::string valString = "Summarized user statistics data (preference): " + std::string(val1) + ". \n Summarized user movie metadata:  " + std::string(val2) + ".\n" + promptSuffix;
      nlohmann::json messageArrays = nlohmann::json::array();
      // Add message
      messageArrays.push_back({{"role", "user"}, {"content", valString}});

      const_cast<uint64_t&>(inputTokenNumber_) = inputTokenNumber_ + countWords(valString) + countPunctuation(valString);

      nlohmann::json payload = {{"model", model_}, {"messages", messageArrays}, {"max_tokens", 500}};

      cpr::Response response = cpr::Post(
          cpr::Url{url_}, cpr::Header{headers}, cpr::Body{payload.dump()});
      
      int failureCount = 0;
      // retry
      while (response.status_code != 200) {
        response = cpr::Post(
          cpr::Url{url_}, cpr::Header{headers}, cpr::Body{payload.dump()});
        if (failureCount++ > 10) {
          break;
        }
      }
      
      if (response.status_code == 200) {
        // parse the returned value
        nlohmann::json response_json = nlohmann::json::parse(response.text);
        std::string generated_message =
            response_json["choices"][0]["message"]["content"];
        results.push_back(generated_message);
        const_cast<uint64_t&>(outputTokenNumber_) = outputTokenNumber_ + countWords(generated_message) + countPunctuation(generated_message);
        // std::cout << fmt::format("[INFO] i: {} / {}, results: {}", i+1, numInput, generated_message) << std::endl;
        // std::cout << fmt::format("[DEBUG] payload: {}", payload.dump()) << std::endl;
      } else {
        std::cout << "Error: " << response.status_code << " - " << response.text
                  << std::endl;
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<std::string>(results);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
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
};