/*
 * Copyright (c) 2025 ASU Cactus Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @class ChatGPT
 * @brief A class that interacts with OpenAI's ChatGPT API for text generation, inheriting from MLFunction.
 *
 * This class provides functionality to send text prompts to the ChatGPT API and retrieve generated responses.
 * It supports multi-threaded requests for improved performance.
 */
class ChatGPT : public MLFunction {
public:
    /**
     * @brief Default constructor.
     *
     * Initializes the ChatGPT API with the default URL and model ("gpt-3.5-turbo").
     * Requires the `OPENAI_API_KEY` environment variable to be set.
     */
    ChatGPT() {
        apiKey_ = getEnvVar("OPENAI_API_KEY");
        if (apiKey_ == "") {
            throw std::runtime_error("[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY");
        }
        numThreads_ = getEnvVar("NUM_THREADS") == "" ? 8 : std::stoi(getEnvVar("NUM_THREADS"));
        url_ = "https://api.openai.com/v1/chat/completions";
        model_ = "gpt-3.5-turbo";
        inputTokenNumber_ = 0;
        outputTokenNumber_ = 0;
        numFailures_ = 0;
    }

    /**
     * @brief Constructor with custom URL and model.
     *
     * Initializes the ChatGPT API with a custom URL and model.
     * Requires the `OPENAI_API_KEY` environment variable to be set.
     *
     * @param url The URL of the OpenAI API endpoint.
     * @param model The model to use for text generation (e.g., "gpt-3.5-turbo").
     */
    ChatGPT(std::string url, std::string model) {
        apiKey_ = getEnvVar("OPENAI_API_KEY");
        if (apiKey_ == "") {
            throw std::runtime_error("[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY");
        }
        numThreads_ = getEnvVar("NUM_THREADS") == "" ? 8 : std::stoi(getEnvVar("NUM_THREADS"));
        url_ = url;
        model_ = model;
        inputTokenNumber_ = 0;
        outputTokenNumber_ = 0;
        numFailures_ = 0;
    }

    /**
     * @brief Destructor.
     *
     * Logs the total number of input tokens, output tokens, and API failures to a file.
     */
    ~ChatGPT() {
        std::string filename = "chatgpt.log";
        std::ofstream file(filename, std::ios::app);
        if (!file) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " ";
        file << "[ChatGPT] # Input:" << inputTokenNumber_
             << " # Output: " << outputTokenNumber_
             << " # NumFailure: " << numFailures_ << std::endl;
        file.close();
    }

    /**
     * @brief Applies the ChatGPT function to the input array.
     *
     * This method processes the input array, sends text prompts to the ChatGPT API,
     * and stores the generated responses in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input text prompts).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the generated responses will be stored.
     */
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
    LOG(INFO) << "[INFO ChatGPT:] countSelected: " << rows.countSelected()
              << " numInput: " << numInput << std::endl;

    if (args.size() == 2) {
      exec::LocalDecodedVector decodedStringHolder2(context, *args[1], rows);
      auto decodedStringInput2 = decodedStringHolder2.get();
      StringView val = decodedStringInput2->valueAt<StringView>(0);
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
    // This approach is more efficient by sending requests in batches and
    // leveraging multiple threads to send requests concurrently, it requires
    // additional isValid check to skip the rows that are not selected. Note: at
    // the end of this approach, it is required to invoke
    // context.moveOrCopyResult to copy the results back to the output vector
    // since we only compute the results for selected ones
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
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {
            exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("VARCHAR")
                .build(),
            exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .argumentType("VARCHAR")
                .returnType("VARCHAR")
                .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A null pointer (no tensor is associated with this function).
     */
    float* getTensor() const override {
        return nullptr;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string ("chatgpt").
     */
    static std::string getName() {
        return "chatgpt";
    }

    /**
     * @brief Estimates the computational cost of applying the ChatGPT function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

private:
    std::string apiKey_; ///< OpenAI API key.
    std::string url_;    ///< URL of the OpenAI API endpoint.
    std::string model_;  ///< Model to use for text generation.
    uint64_t inputTokenNumber_; ///< Total number of input tokens processed.
    uint64_t outputTokenNumber_; ///< Total number of output tokens generated.
    uint64_t numFailures_; ///< Total number of API failures.
    int numThreads_; ///< Number of threads for multi-threaded requests.
};
/**
 * @class ChatGPTRecommender
 * @brief A class that interacts with OpenAI's ChatGPT API for recommendation tasks, inheriting from MLFunction.
 *
 * This class provides functionality to send user statistics and movie metadata to the ChatGPT API
 * and retrieve personalized recommendations. It supports multi-threaded requests for improved performance.
 */
class ChatGPTRecommender : public MLFunction {
public:
    /**
     * @brief Default constructor.
     *
     * Initializes the ChatGPT API with the default URL and model ("gpt-3.5-turbo").
     * Requires the `OPENAI_API_KEY` environment variable to be set.
     */
    ChatGPTRecommender() {
        apiKey_ = getEnvVar("OPENAI_API_KEY");
        if (apiKey_ == "") {
            throw std::runtime_error("[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY");
        }
        numThreads_ = getEnvVar("NUM_THREADS") == "" ? 8 : std::stoi(getEnvVar("NUM_THREADS"));
        url_ = "https://api.openai.com/v1/chat/completions";
        model_ = "gpt-3.5-turbo";
        inputTokenNumber_ = 0;
        outputTokenNumber_ = 0;
        numFailures_ = 0;
    }

    /**
     * @brief Constructor with custom URL and model.
     *
     * Initializes the ChatGPT API with a custom URL and model.
     * Requires the `OPENAI_API_KEY` environment variable to be set.
     *
     * @param url The URL of the OpenAI API endpoint.
     * @param model The model to use for text generation (e.g., "gpt-3.5-turbo").
     */
    ChatGPTRecommender(std::string url, std::string model) {
        apiKey_ = getEnvVar("OPENAI_API_KEY");
        if (apiKey_ == "") {
            throw std::runtime_error("[ERROR] OpenAI API key is not set, please set OPENAI_API_KEY");
        }
        numThreads_ = getEnvVar("NUM_THREADS") == "" ? 8 : std::stoi(getEnvVar("NUM_THREADS"));
        url_ = url;
        model_ = model;
        inputTokenNumber_ = 0;
        outputTokenNumber_ = 0;
        numFailures_ = 0;
    }

    /**
     * @brief Destructor.
     *
     * Logs the total number of input tokens, output tokens, and API failures to a file.
     */
    ~ChatGPTRecommender() {
        std::string filename = "chatgpt.log";
        std::ofstream file(filename, std::ios::app);
        if (!file) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " ";
        file << "[ChatGPT Recommender] # Input:" << inputTokenNumber_
             << " # Output: " << outputTokenNumber_
             << " # NumFailure: " << numFailures_ << std::endl;
        file.close();
    }

    /**
     * @brief Applies the ChatGPTRecommender function to the input array.
     *
     * This method processes the input array, sends user statistics and movie metadata to the ChatGPT API,
     * and stores the personalized recommendations in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., user statistics and movie metadata).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the recommendations will be stored.
     */
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
    LOG(INFO) << "[INFO ChatGPTRecommender:] countSelected: "
              << rows.countSelected() << " numInput: " << numInput << std::endl;

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
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {
            exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .argumentType("VARCHAR")
                .returnType("VARCHAR")
                .build(),
            exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .argumentType("VARCHAR")
                .argumentType("VARCHAR")
                .returnType("VARCHAR")
                .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A null pointer (no tensor is associated with this function).
     */
    float* getTensor() const override {
        return nullptr;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string ("chatgpt_recommender").
     */
    static std::string getName() {
        return "chatgpt_recommender";
    }

    /**
     * @brief Estimates the computational cost of applying the ChatGPTRecommender function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

private:
    std::string apiKey_; ///< OpenAI API key.
    std::string url_;    ///< URL of the OpenAI API endpoint.
    std::string model_;  ///< Model to use for text generation.
    uint64_t inputTokenNumber_; ///< Total number of input tokens processed.
    uint64_t outputTokenNumber_; ///< Total number of output tokens generated.
    uint64_t numFailures_; ///< Total number of API failures.
    int numThreads_; ///< Number of threads for multi-threaded requests.
};
