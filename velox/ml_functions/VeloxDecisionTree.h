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
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include "BaseFunction.h"
#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/type/OpaqueCustomTypes.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml {

class TreeType : public OpaqueType {
  TreeType() : OpaqueType(std::type_index(typeid(ml::Tree))) {}

 public:
  static const std::shared_ptr<const TreeType>& get() {
    static const std::shared_ptr<const TreeType> instance{

        new TreeType()

    };

    return instance;
  }

  std::string toString() const override {
    return name();
  }

  const char* name() const override {
    return "tree_type";
  }
};

struct TreeT {
  using type = std::shared_ptr<Tree>;

  static constexpr const char* typeName = "tree_type";
};
using TheTree = CustomType<TreeT>;

class TreeTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    return TreeType::get();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_UNSUPPORTED();
  }
};

class AlwaysFailingTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    VELOX_UNSUPPORTED();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_UNSUPPORTED();
  }
};

class VeloxTreeConstruction : public exec::VectorFunction {
 public:
  VeloxTreeConstruction() {}

  void apply(

      const SelectivityVector& rows,

      std::vector<VectorPtr>& args,

      const TypePtr& type,

      exec::EvalCtx& context,

      VectorPtr& output) const override {
    auto flatInput = args[0]->as<SimpleVector<StringView>>();

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto flatResult = output->asFlatVector<std::shared_ptr<void>>();

    rows.applyToSelected([&](auto row) {
      flatResult->set(
          row, std::make_shared<Tree>(row, flatInput->valueAt(row)));
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("tree_type")
                .build()};
  }

  static std::string getName() {
    return "velox_tree_construct";
  }
};

class VeloxTreePrediction : public MLFunction {
 public:
  VeloxTreePrediction(int numFeatures) {
    this->numFeatures = numFeatures;
    dims.push_back(numFeatures);
  }

  float* getTensor() const override {
    return new float[0]; // will this lead to memory leak?
  }

  void apply(

      const SelectivityVector& rows,

      std::vector<VectorPtr>& args,

      const TypePtr& type,

      exec::EvalCtx& context,

      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    BaseVector* left = args[0].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);

    auto decodedLeftArray = leftHolder.get();

    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();

    auto flatInput = args[1]->as<SimpleVector<std::shared_ptr<void>>>();

    auto flatResult = output->asFlatVector<float>();

    rows.applyToSelected([&](auto row) {
      flatResult->set(

          row,
          std::static_pointer_cast<Tree>(flatInput->valueAt(row))
              ->predictSingle(input1Values, row * numFeatures)

      );
    });
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("tree_type")
                .returnType("REAL")
                .build()};
  }

  static std::string getName() {
    return "velox_tree_predict";
  }

  std::string getFuncName() {
    return getName();
  };

  CostEstimate getCost(std::vector<int> inputDims) {
    // Compute the operation cost using a static cost model. Note: This is
    // currently not utilized for query optimization as we rely on an ML-based
    // model (optimizer/query2vec).
    // TODO: Implement a static cost estimation method for the specified kernel.
    return CostEstimate(1, inputDims[0], dims[1]);
  }

  int numFeatures;
};

template <typename T>
struct VeloxTreePredictionSimpleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void call(
      out_type<float>& result,
      const arg_type<Array<float>>& a,
      const arg_type<TheTree>& b) {
    result = 0.0;
  }
};

} // namespace ml
