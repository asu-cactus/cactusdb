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
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "velox/ml_functions/BaseFunction.h"

using namespace facebook::velox;

// Implementation of TupleCounter to count the number of tuples at the
// intermediate states

class TupleCounter : public MLFunction {
 public:
  TupleCounter(CataLog* cataLog) {
    cataLog_ = cataLog;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    // Timer timer;
    // timer.tic();
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    output->clearNulls(rows);
    output = args[0];

    if (args.size() != 2) {
      LOG(FATAL)
          << "[ERROR TupleCounter:] args.size() != 2, an intermedaite state name is required";
    }

    exec::LocalDecodedVector decodedStringHolder(context, *args[1], rows);
    auto decodedStringInput = decodedStringHolder.get();
    StringView val = decodedStringInput->valueAt<StringView>(0);
    std::string intermediateStateName = std::string(val);

    auto numSelectedRows = rows.countSelected();
    int accumulatedCount =
        cataLog_->getIntermediateStateTupleCounter(intermediateStateName);
    cataLog_->setIntermediateStateTupleCounter(
        intermediateStateName, accumulatedCount + numSelectedRows);
    // std::cout << "[INFO] tupleCounter execution time: " << timer.toc() << "
    // ms"
    //           << std::endl;
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("VARCHAR")
                .returnType("array(REAL)")
                .build()};
  }

  // TODO: add get and set for bias or we have a better way to store the two
  // parameters in a single file
  float* getTensor() const override {
    return weights_;
  }

  float* getWeight() {
    return weights_;
  }

  float* getBias() {
    return bias_;
  }

  static std::string getName() {
    return "tuple_counter";
  };

  std::string getWeightsFile() {
    return weightsFile_;
  }

  void setWeights(float* weights) {
    weights_ = weights;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // Computes the operation cost using a legacy static cost model.
    // This method is retained for compatibility and reference purposes.
    // NOTE: Query optimization uses an ML-based cost model
    // (optimizer/query2vec), and this function is not invoked in that process.

    // TODO: Implement a static cost estimation method for the specified kernel.
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  float* weights_;
  float* bias_;
  // float eps_;
  std::string weightsFile_;
  // std::string biasFile_;
  CataLog* cataLog_;
};
