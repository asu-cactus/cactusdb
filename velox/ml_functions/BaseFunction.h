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
#include "velox/cost_model/CostEstimate.h"
#include "velox/cost_model/UdfCostCoefficient.h"
#include "velox/exec/Task.h"
#include "velox/expression/Expr.h"
#include "velox/expression/VectorFunction.h"

using namespace facebook::velox;

class MLFunction : public exec::VectorFunction {
 public:
  virtual ~MLFunction() = default;

  virtual float* getTensor() const = 0;

  virtual std::vector<int> getDims() {
    return dims;
  }

  virtual std::string getFuncName() {
    return "";
  }

  virtual int getNumDims() {
    return dims.size();
  }

  virtual CostEstimate getCost(std::vector<int> inputDims) {
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 protected:
  std::vector<int> dims;
  double getWeightedCost(std::string name, float cost) {
    std::vector<double> coefficient =
        UdfCostCoefficient::getInstance().getCoefficient(name);
    // FIXME
    return 0;
    // return coefficient[0] * cost;
  }
  std::vector<double> getCoefficientVector(std::string name) {
    return UdfCostCoefficient::getInstance().getCoefficient(name);
  }
};