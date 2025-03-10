/**
 * @file
 * @brief This file contains the implementation of custom types and functions for tree-based machine learning models in Velox.
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

/**
 * @class TreeType
 * @brief A custom opaque type for representing tree structures.
 * 
 * This class inherits from `OpaqueType` and is used to define a custom type for trees.
 */
class TreeType : public OpaqueType {
  TreeType() : OpaqueType(std::type_index(typeid(ml::Tree))) {}

 public:
  /**
   * @brief Get a shared instance of TreeType.
   * @return A shared pointer to a constant TreeType instance.
   */
  static const std::shared_ptr<const TreeType>& get() {
    static const std::shared_ptr<const TreeType> instance{
        new TreeType()
    };
    return instance;
  }

  /**
   * @brief Convert the type to a string representation.
   * @return A string representing the type.
   */
  std::string toString() const override {
    return name();
  }

  /**
   * @brief Get the name of the type.
   * @return A C-string representing the name of the type.
   */
  const char* name() const override {
    return "tree_type";
  }
};

/**
 * @struct TreeT
 * @brief A struct representing the custom type for trees.
 */
struct TreeT {
  using type = std::shared_ptr<Tree>; ///< The underlying type for the custom type.

  static constexpr const char* typeName = "tree_type"; ///< The name of the custom type.
};

using TheTree = CustomType<TreeT>; ///< Alias for the custom tree type.

/**
 * @class TreeTypeFactories
 * @brief Factory class for creating instances of TreeType.
 */
class TreeTypeFactories : public CustomTypeFactories {
 public:
  /**
   * @brief Get the TreeType instance.
   * @return A shared pointer to the TreeType instance.
   */
  TypePtr getType() const override {
    return TreeType::get();
  }

  /**
   * @brief Get the cast operator for the type.
   * @return A shared pointer to the cast operator.
   * @throws VeloxException if the operation is unsupported.
   */
  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_UNSUPPORTED();
  }
};

/**
 * @class AlwaysFailingTypeFactories
 * @brief A factory class that always fails to create instances.
 */
class AlwaysFailingTypeFactories : public CustomTypeFactories {
 public:
  /**
   * @brief Get the type instance.
   * @throws VeloxException if the operation is unsupported.
   */
  TypePtr getType() const override {
    VELOX_UNSUPPORTED();
  }

  /**
   * @brief Get the cast operator for the type.
   * @throws VeloxException if the operation is unsupported.
   */
  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_UNSUPPORTED();
  }
};

/**
 * @class VeloxTreeConstruction
 * @brief A vector function for constructing tree structures.
 */
class VeloxTreeConstruction : public exec::VectorFunction {
 public:
  VeloxTreeConstruction() {}

  /**
   * @brief Apply the function to construct trees.
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
    auto flatInput = args[0]->as<SimpleVector<StringView>>();

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    auto flatResult = output->asFlatVector<std::shared_ptr<void>>();

    rows.applyToSelected([&](auto row) {
      flatResult->set(
          row, std::make_shared<Tree>(row, flatInput->valueAt(row)));
    });
  }

  /**
   * @brief Get the function signatures.
   * @return A vector of shared pointers to function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("tree_type")
                .build()};
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the name of the function.
   */
  static std::string getName() {
    return "velox_tree_construct";
  }
};

/**
 * @class VeloxTreePrediction
 * @brief A machine learning function for making predictions using tree models.
 */
class VeloxTreePrediction : public MLFunction {
 public:
  /**
   * @brief Construct a new VeloxTreePrediction object.
   * @param numFeatures The number of features in the input data.
   */
  VeloxTreePrediction(int numFeatures) {
    this->numFeatures = numFeatures;
    dims.push_back(numFeatures);
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

  /**
   * @brief Get the function signatures.
   * @return A vector of shared pointers to function signatures.
   */
  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("tree_type")
                .returnType("REAL")
                .build()};
  }

  /**
   * @brief Get the name of the function.
   * @return A string representing the name of the function.
   */
  static std::string getName() {
    return "velox_tree_predict";
  }

  /**
   * @brief Get the function name.
   * @return A string representing the function name.
   */
  std::string getFuncName() {
    return getName();
  };

  /**
   * @brief Get the cost estimate for the function.
   * @param inputDims The dimensions of the input data.
   * @return A CostEstimate object representing the cost.
   */
  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO
    return CostEstimate(1, inputDims[0], dims[1]);
  }

  int numFeatures; ///< The number of features in the input data.
};

/**
 * @struct VeloxTreePredictionSimpleFunction
 * @brief A simple function for making predictions using tree models.
 * @tparam T The template parameter for the function.
 */
template <typename T>
struct VeloxTreePredictionSimpleFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  /**
   * @brief Call the function to make predictions.
   * @param result The output result.
   * @param a The input array of features.
   * @param b The input tree model.
   */
  void call(
      out_type<float>& result,
      const arg_type<Array<float>>& a,
      const arg_type<TheTree>& b) {
    result = 0.0;
  }
};

} // namespace ml
