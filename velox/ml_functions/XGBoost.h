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

class XGBoost;

typedef std::shared_ptr<XGBoost> XGBoostPtr;

class XGBoost {
 public:
  XGBoost(std::string pathToJSON) {
    XGBoosterCreate(NULL, 0, &booster);
    XGBoosterSetParam(booster, "seed", "0");
    XGBoosterLoadModel(booster, pathToJSON.c_str());
  }

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

  // handle to booster
  BoosterHandle booster;
};

class XGBoostPrediction : public MLFunction {
 public:
  XGBoostPrediction(std::string forestPath, int numFeatures) {
    this->forest = std::make_shared<XGBoost>(forestPath);

    this->numFeatures = numFeatures;

    this->forestPath = forestPath;
  }

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

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .returnType("REAL")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  float* getTensor() const override {
    return new float[0]; // will this lead to memory leak?
  }

  static std::string getName() {
    return "xgboost_predict";
  }

  int getNumFeatures() {
    return numFeatures;
  }

  std::string& getForestPath() {
    return this->forestPath;
  }

 private:
  XGBoostPtr forest;

  int numFeatures;

  std::string forestPath;
};

} // namespace ml
