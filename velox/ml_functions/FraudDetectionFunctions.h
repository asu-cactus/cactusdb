/**
 * @file
 * @brief Implementation of various utility functions for machine learning.
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

#include <time.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <locale>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "BaseFunction.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

/**
 * @class IsWeekday
 * @brief Implements a function to check if a given timestamp corresponds to a weekday.
 */
class IsWeekday : public MLFunction {
 public:
  /**
   * @brief Applies the function to check if the timestamp is a weekday.
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

    std::vector<int> results;

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto inputTimes = decodedArray->base()->as<FlatVector<int64_t>>();

    const int secondsInADay = 86400;
    for (int i = 0; i < rows.size(); i++) {
      int64_t timestamp = inputTimes->valueAt(i);

      std::time_t time = static_cast<std::time_t>(timestamp);
      std::tm* time_info = std::localtime(&time);
      int dayOfWeek = time_info->tm_wday;

      // Return true if the day is Saturday (6) or Sunday (0)
      if (dayOfWeek == 0 || dayOfWeek == 6) {
        results.push_back(0);
      } else {
        results.push_back(1);
      }
    }

    VectorMaker maker{context.pool()};
    auto localResult = maker.flatVector<int>(results);
    context.moveOrCopyResult(localResult, rows, output);
    output = maker.flatVector<int>(results, INTEGER());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .returnType("INTEGER")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "is_weekday";
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
};

/**
 * @class GetAge
 * @brief Implements a function to calculate the age based on the birth year.
 */
class GetAge : public MLFunction {
 public:
  /**
   * @brief Applies the function to calculate the age.
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

    std::vector<int> results;

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto birthYears = decodedArray->base()->as<FlatVector<int>>();

    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::tm* localTime = std::localtime(&currentTime);
    int currentYear = 1900 + localTime->tm_year;

    for (int i = 0; i < rows.size(); i++) {
      int birthYear = birthYears->valueAt(i);
      results.push_back(currentYear - birthYear);
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(results);
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .returnType("INTEGER")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "get_age";
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
};

/**
 * @class GetTransactionFeatures
 * @brief Implements a function to extract features from transaction data.
 */
class GetTransactionFeatures : public MLFunction {
 public:
  /**
   * @brief Applies the function to extract transaction features.
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

    int secondsInADay = 86400;
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();
    BaseVector* base2 = args[2].get();
    BaseVector* base3 = args[3].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();

    exec::LocalDecodedVector thirdHolder(context, *base2, rows);
    auto decodedArray2 = thirdHolder.get();

    exec::LocalDecodedVector fourthHolder(context, *base3, rows);
    auto decodedArray3 = fourthHolder.get();

    for (int i = 0; i < rows.size(); i++) {
      float totalOrder = (static_cast<float>(decodedArray0->valueAt<int64_t>(i))) / 79.0;
      float tAmount = (decodedArray1->valueAt<float>(i)) / 16048.0;
      float timeDiff = (static_cast<float>(decodedArray2->valueAt<int64_t>(i))) / 729.0;
      int64_t tTimestamp = decodedArray3->valueAt<int64_t>(i);

      // Calculate day of week
      std::time_t time = static_cast<std::time_t>(tTimestamp);
      std::tm* time_info = std::localtime(&time);
      float dayOfWeek = (static_cast<float>(time_info->tm_wday)) / 6.0;

      // Calculate the number of days since Unix epoch
      float daysSinceEpoch =
          (static_cast<float>(tTimestamp / secondsInADay)) / 15338.0;

      std::vector<float> vec;
      vec.push_back(totalOrder);
      vec.push_back(tAmount);
      vec.push_back(timeDiff);
      vec.push_back(dayOfWeek);
      vec.push_back(daysSinceEpoch);

      results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .argumentType("REAL")
                .argumentType("BIGINT")
                .argumentType("BIGINT")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "get_transaction_features";
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
};

/**
 * @class GetCustomerFeatures
 * @brief Implements a function to extract features from customer data.
 */
class GetCustomerFeatures : public MLFunction {
 public:
  /**
   * @brief Applies the function to extract customer features.
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

    int secondsInADay = 86400;
    std::vector<std::vector<float>> results;

    BaseVector* base0 = args[0].get();
    BaseVector* base1 = args[1].get();
    BaseVector* base2 = args[2].get();
    BaseVector* base3 = args[3].get();

    exec::LocalDecodedVector firstHolder(context, *base0, rows);
    auto decodedArray0 = firstHolder.get();

    exec::LocalDecodedVector secondHolder(context, *base1, rows);
    auto decodedArray1 = secondHolder.get();

    exec::LocalDecodedVector thirdHolder(context, *base2, rows);
    auto decodedArray2 = thirdHolder.get();

    exec::LocalDecodedVector fourthHolder(context, *base3, rows);
    auto decodedArray3 = fourthHolder.get();

    for (int i = 0; i < rows.size(); i++) {
      float cAddressNum =
          (static_cast<float>(decodedArray0->valueAt<int>(i))) / 35352.0;
      float cCustFlag = static_cast<float>(decodedArray1->valueAt<int>(i));
      float cBirthCountry =
          (static_cast<float>(decodedArray2->valueAt<int>(i))) / 211.0;
      float cAge = (static_cast<float>(decodedArray3->valueAt<int>(i))) / 94.0;

      std::vector<float> vec;
      vec.push_back(cAddressNum);
      vec.push_back(cCustFlag);
      vec.push_back(cBirthCountry);
      vec.push_back(cAge);

      results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .argumentType("INTEGER")
                .returnType("ARRAY(REAL)")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "get_customer_features";
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
};

/**
 * @class TimeDiffInDays
 * @brief Implements a function to calculate the difference in days between two timestamps.
 */
class TimeDiffInDays : public MLFunction {
 public:
  /**
   * @brief Applies the function to calculate the time difference in days.
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

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector decodedInput1(context, *args[0], rows);
    exec::LocalDecodedVector decodedInput2(context, *args[1], rows);

    std::vector<int64_t> results;
    int secondsInADay = 86400;

    for (int i = 0; i < rows.size(); i++) {
      if (!rows.isValid(i)) {
        continue;
      }
      int64_t timestamp1 = decodedInput1->valueAt<int64_t>(i);
      int64_t timestamp2 = decodedInput2->valueAt<int64_t>(i);

      int64_t differenceInSeconds = std::abs(timestamp1 - timestamp2);
      int64_t differenceInDays = differenceInSeconds / secondsInADay;
      results.push_back(differenceInDays);
    }

    VectorMaker maker{context.pool()};
    auto localResult = maker.flatVector<int64_t>(results);
    context.moveOrCopyResult(localResult, rows, output);
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("BIGINT")
                .argumentType("BIGINT")
                .returnType("BIGINT")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "time_diff_in_days";
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
};

/**
 * @class DateToTimestamp
 * @brief Implements a function to convert a date string to a timestamp.
 */
class DateToTimestamp : public MLFunction {
 public:
  /**
   * @brief Constructor for DateToTimestamp.
   * @param dateFormat_ The format of the date string.
   */
  DateToTimestamp(const char* dateFormat_) {
    dateFormat = dateFormat_;
  }

  /**
   * @brief Applies the function to convert a date string to a timestamp.
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

    exec::LocalDecodedVector decodedStringHolder(context, *args[0], rows);
    auto decodedStringInput = decodedStringHolder.get();

    std::vector<int64_t> results;
    struct std::tm t = {};

    for (int i = 0; i < rows.size(); i++) {
      StringView val = decodedStringInput->valueAt<StringView>(i);
      std::string inputStr = std::string(val);

      std::istringstream ss(inputStr);
      ss >> std::get_time(&t, dateFormat);

      // Check if parsing was successful
      if (ss.fail()) {
        std::cerr << "Failed to parse date string " << inputStr << std::endl;
        results.push_back(0);
        continue;
      }

      // Convert tm struct to time_t (timestamp)
      time_t tt = mktime(&t);
      // Cast time_t to int64_t
      int64_t timestamp = static_cast<int64_t>(tt);
      results.push_back(timestamp);
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int64_t>(results, BIGINT());
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("BIGINT")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "date_to_timestamp";
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
  const char* dateFormat; ///< The format of the date string.
};

/**
 * @class GetBinaryClass
 * @brief Implements a function to determine the binary class based on probabilities.
 */
class GetBinaryClass : public MLFunction {
 public:
  /**
   * @brief Applies the function to determine the binary class.
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

    std::vector<int> results;

    BaseVector* baseVec = args[0].get();
    exec::LocalDecodedVector vecHolder(context, *baseVec, rows);
    auto decodedArray = vecHolder.get();
    auto inputProbs = decodedArray->base()->as<ArrayVector>();
    auto inputProbsValues = inputProbs->elements()->asFlatVector<float>();

    for (int i = 0; i < rows.size(); i++) {
      int32_t offset = inputProbs->offsetAt(i);
      float prob_0 = inputProbsValues->valueAt(offset);
      float prob_1 = inputProbsValues->valueAt(offset + 1);
      if (std::isnan(prob_0) || std::isnan(prob_1)) {
        results.push_back(0);
      } else {
        int predicted_class = (prob_0 > prob_1) ? 0 : 1;
        results.push_back(predicted_class);
      }
    }

    VectorMaker maker{context.pool()};
    output = maker.flatVector<int>(results);
  }

  /**
   * @brief Returns the function signatures.
   * @return Vector of function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("ARRAY(REAL)")
                .returnType("INTEGER")
                .build()};
  }

  /**
   * @brief Returns the name of the function.
   * @return Function name.
   */
  static std::string getName() {
    return "get_binary_class";
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
};

/**
 * @brief Reads a file containing country mappings and returns an unordered_map.
 * @return An unordered_map mapping country names to their corresponding integer values.
 */
std::unordered_map<std::string, int> getCountryMap() {
  std::unordered_map<std::string, int> countryMap;

  // Open the txt file
  std::string filePath = "/home/velox/resources/data/country_mapping.txt";
  std::ifstream file(filePath.c_str());
  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file!" << std::endl;
    exit(1);
  }

  std::string line;
  // Read the file line by line
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string key;
    std::string value_str;

    // Get the key before the comma
    std::getline(ss, key, ',');

    // Get the value after the comma
    std::getline(ss, value_str);

    // Convert the string value to an integer
    int value = std::stoi(value_str);

    // Insert into the unordered_map
    countryMap[key] = value;
  }

  // Close the file
  file.close();

  return countryMap;
}
