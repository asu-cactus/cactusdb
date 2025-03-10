/**
 * @file
 * @brief This file contains the implementation of XGBoost-based machine learning functions in Velox.
 * @copyright Copyright (c) 2025 ASU Cactus Lab.
 * @license Licensed under the Apache License, Version 2.0 (the "License");
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

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <xgboost/c_api.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include "BaseFunction.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml {

class XGBoost; ///< Forward declaration of the XGBoost class.

typedef std::shared_ptr<XGBoost> XGBoostPtr; ///< Alias for a shared pointer to an XGBoost object.

/**
 * @class XGBoost
 * @brief A class for managing XGBoost models and making predictions.
 */
class XGBoost {
 public:
  /**
   * @brief Construct a new XGBoost object.
   * @param pathToJSON The path to the JSON file containing the XGBoost model.
   */
  XGBoost(std::string pathToJSON) {
    XGBoosterCreate(NULL, 0, &booster);
    XGBoosterSetParam(booster, "seed", "0");
    XGBoosterLoadModel(booster, pathToJSON.c_str());
  }

  /**
   * @brief Make predictions using the XGBoost model.
   * @param input The input vector containing the features.
   * @param resultVector The output vector to store the predictions.
   * @param numInputs The number of input rows.
   * @param numFeatures The number of features per input row.
   */
  inline void predict(
      VectorPtr& input,
      std::vector<float>& resultVector,
      int numInputs,
      int numFeatures) {
    auto inputFeatures = input->as<ArrayVector>()->elements();
    float* inputValues = inputFeatures->values()->asMutable<float>();
    DMatrixHandle dtest;
    XGDMatrixCreateFromMat(inputValues, numInputs, numFeatures, NAN, &dtest);
    unsigned long numOutputs;
    float const* outData = NULL;
    XGBoosterPredictFromDMatrix(booster, dtest, 0, 0, &numOutputs, &outData);
    assert(numOutputs == numInputs);
    memcpy(resultVector.data(), outData, numOutputs * sizeof(float));
    XGDMatrixFree(dtest);
    XGBoosterFree(booster);
  }

  BoosterHandle booster; ///< Handle to the XGBoost booster.
};

/**
 * @class XGBoostPrediction
 * @brief A machine learning function for making predictions using XGBoost models.
 */
class XGBoostPrediction : public MLFunction {
 public:
  /**
   * @brief Construct a new XGBoostPrediction object.
   * @param forestPath The path to the XGBoost model file.
   * @param numFeatures The number of features in the input data.
   */
  XGBoostPrediction(std::string forestPath, int numFeatures) {
    this->forest = std::make_shared<XGBoost>(forestPath);
    this->numFeatures = numFeatures;
    this->forestPath = forestPath;
  }

  /**
   * @brief Apply the function to make predictions.
   * @param rows The selectivity vector indicating which rows to process.
   * @param args The input arguments to the function.
   * @param type The type of the output vector.
   * @param context The evaluation context.
   * @param output The output vector where the results will be stored.
   */
  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int numInputs = rows.size();

    std::vector<float> resultVector(numInputs);

    this->forest->predict(args[0], resultVector, numInputs, this->numFeatures);

    VectorMaker maker{context.pool()};

    output = maker.flatVector<float>(resultVector, REAL());
  }

  /**
   * @brief Get the function signatures.
   * @return A vector of shared pointers to function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("REAL")
                .build()};
  }

  /**
   * @brief Get the tensor data.
   * @return A pointer to the tensor data.
   * @note This implementation may lead to a memory leak.
   */
  float* getTensor() const override {
    return new float[0]; // will this lead to memory leak?
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the name of the function.
   */
  static std::string getName() {
    return "xgboost_predict";
  }

  /**
   * @brief Get the number of features.
   * @return The number of features in the input data.
   */
  int getNumFeatures() {
    return numFeatures;
  }

  /**
   * @brief Get the path to the XGBoost model file.
   * @return A reference to the path string.
   */
  std::string& getForestPath() {
    return this->forestPath;
  }

 private:
  XGBoostPtr forest; ///< Shared pointer to the XGBoost model.

  int numFeatures; ///< The number of features in the input data.

  std::string forestPath; ///< The path to the XGBoost model file.
};

} // namespace ml
