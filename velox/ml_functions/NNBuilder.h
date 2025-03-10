/**
 * @file
 * @brief Implementation of a neural network builder for constructing and registering neural network layers.
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
#include "velox/ml_functions/functions.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

/**
 * @enum Activation
 * @brief Enumeration of activation functions supported by the neural network builder.
 */
enum Activation { RELU, ///< Rectified Linear Unit (ReLU) activation.
                 SOFTMAX, ///< Softmax activation.
                 NONE ///< No activation.
};

/**
 * @class NNBuilder
 * @brief Implements a neural network builder for constructing and registering neural network layers.
 */
class NNBuilder {
 public:
  /**
   * @brief Default constructor for NNBuilder.
   */
  NNBuilder() {
    function_count = 0;
    compute_string = "{}";
  }

  /**
   * @brief Constructor for NNBuilder with weights and bias files.
   * @param weightsFile Path to the file containing weights.
   * @param biasFile Path to the file containing biases.
   */
  NNBuilder(std::string weightsFile, std::string biasFile) : NNBuilder() {
    weightsFile_ = weightsFile;
    biasFile_ = biasFile;
  }

  /**
   * @brief Destructor for NNBuilder.
   */
  ~NNBuilder() {}

  /**
   * @brief Builds and returns the computation string representing the neural network.
   * @return The computation string.
   */
  std::string build() {
    return compute_string;
  }

  /**
   * @brief Adds a dense layer to the neural network.
   * @param units Number of units in the dense layer.
   * @param input_size Size of the input to the dense layer.
   * @param weights Pointer to the weights array.
   * @param bias Pointer to the bias array.
   * @param ac Activation function to apply after the dense layer.
   * @return Reference to the NNBuilder object for method chaining.
   */
  NNBuilder& denseLayer(
      int units,
      int input_size,
      float* weights,
      float* bias,
      Activation ac) {
    std::string mat_mul_name =
        MatrixMultiply::getName() + std::to_string(function_count++);
    std::string mat_add_name =
        MatrixVectorAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        mat_mul_name,
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weights, input_size, units));

    exec::registerVectorFunction(
        mat_add_name,
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(bias, units));

    if (ac == Activation::RELU) {
      act_name = Relu::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
          act_name, Relu::signatures(), std::make_unique<Relu>());
    } else {
      act_name = Softmax::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
          act_name, Softmax::signatures(), std::make_unique<Softmax>());
    }

    compute_string = fmt::format(
        "{}({}({}({})))", act_name, mat_add_name, mat_mul_name, compute_string);
    return *this;
  }

  /**
   * @brief Adds a dense layer to the neural network using weights and biases from files.
   * @param units Number of units in the dense layer.
   * @param input_size Size of the input to the dense layer.
   * @param ac Activation function to apply after the dense layer.
   * @return Reference to the NNBuilder object for method chaining.
   */
  NNBuilder& denseLayer(int units, int input_size, Activation ac) {
    std::string mat_mul_name =
        MatrixMultiply::getName() + std::to_string(function_count++);
    std::string mat_add_name =
        MatrixAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        mat_mul_name,
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_, input_size, units));

    exec::registerVectorFunction(
        mat_add_name,
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(biasFile_, units));

    if (ac == Activation::RELU) {
      act_name = Relu::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
          act_name, Relu::signatures(), std::make_unique<Relu>());
    } else {
      act_name = Softmax::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
          act_name, Softmax::signatures(), std::make_unique<Softmax>());
    }

    compute_string = fmt::format(
        "{}({}({}({})))", act_name, mat_add_name, mat_mul_name, compute_string);
    return *this;
  }

  /**
   * @brief Adds a convolutional layer to the neural network.
   * @param num_filters Number of filters in the convolutional layer.
   * @param dims Dimensions of the filters.
   * @param weights Pointer to the weights array.
   * @param bias Pointer to the bias array.
   * @param ac Activation function to apply after the convolutional layer.
   * @return Reference to the NNBuilder object for method chaining.
   */
  NNBuilder& convLayer(
      int num_filters,
      int* dims,
      float* weights,
      float* bias,
      Activation ac) {
    std::string conv_name =
        Convolute::getName() + std::to_string(function_count++);
    std::string scal_add_name =
        VectorScalarAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        conv_name,
        Convolute::signatures(),
        std::make_unique<Convolute>(weights, dims));

    exec::registerVectorFunction(
        scal_add_name,
        VectorScalarAddition::signatures(),
        std::make_unique<VectorScalarAddition>(bias, num_filters));

    if (ac == Activation::RELU) {
      act_name = Relu::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
          act_name, Relu::signatures(), std::make_unique<Relu>());
    } else if (ac == Activation::SOFTMAX) {
      act_name = Softmax::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
          act_name, Softmax::signatures(), std::make_unique<Softmax>());
    }
    if (act_name == "")
      compute_string =
          fmt::format("{}({}({}))", scal_add_name, conv_name, compute_string);
    else
      compute_string = fmt::format(
          "{}({}({}({})))", act_name, scal_add_name, conv_name, compute_string);

    return *this;
  }

  /**
   * @brief Adds a max pooling layer to the neural network.
   * @param side Size of the pooling window.
   * @param height Height of the input feature map.
   * @param width Width of the input feature map.
   * @return Reference to the NNBuilder object for method chaining.
   */
  NNBuilder& maxPoolLayer(int side, int height, int width) {
    std::string max_pool_name =
        MaxPool::getName() + std::to_string(function_count++);
    exec::registerVectorFunction(
        max_pool_name,
        MaxPool::signatures(),
        std::make_unique<MaxPool>(side, height, width));
    compute_string = fmt::format("{}({})", max_pool_name, compute_string);
    return *this;
  }

 private:
  int function_count; ///< Counter for generating unique function names.
  std::string compute_string; ///< String representing the computation graph.
  std::string weightsFile_; ///< Path to the file containing weights.
  std::string biasFile_; ///< Path to the file containing biases.
};
