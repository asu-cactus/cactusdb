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
 * @class BatchNorm1D
 * @brief A class that implements 1D batch normalization, inheriting from MLFunction.
 *
 * This class provides functionality to apply 1D batch normalization to an input array.
 * Batch normalization normalizes the input by subtracting the mean and dividing by the standard deviation,
 * then scales and shifts the result using learned weights and biases.
 */
class BatchNorm1D : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the batch normalization operation with weights, biases, and dimensions.
     *
     * @param weights A pointer to the weight matrix for scaling.
     * @param bias A pointer to the bias vector for shifting.
     * @param numDims The number of dimensions (features) in the input.
     * @param eps A small value added to the variance to avoid division by zero (default: 1e-05).
     */
    BatchNorm1D(float* weights, float* bias, int numDims, float eps = 1e-05) {
        weights_ = new float[numDims];
        bias_ = new float[numDims];
        std::memcpy(weights_, weights, numDims * sizeof(float));
        std::memcpy(bias_, bias, numDims * sizeof(float));
        eps_ = eps;
        dims.push_back(numDims);
    }

    /**
     * @brief Applies 1D batch normalization to the input array.
     *
     * This method processes the input array, applies batch normalization, and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the input array).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the result will be stored.
     */
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);
    output->clearNulls(rows);
    auto arrayOutput = output->as<ArrayVector>();
    auto sizes = arrayOutput->mutableSizes(rows.end());
    auto rawSizes = sizes->asMutable<int32_t>();
    auto offsets = arrayOutput->mutableOffsets(rows.end());
    auto rawOffsets = offsets->asMutable<int32_t>();

    // Initialize sizes and offsets to zero.
    std::fill(rawSizes, rawSizes + rows.end(), 0);
    std::fill(rawOffsets, rawOffsets + rows.end(), 0);

    auto elementsOutput = arrayOutput->elements();
    auto elementsPool = context.pool();

    exec::DecodedArgs decodedArgs(rows, args, context);
    auto decodedInput = decodedArgs.at(0);
    auto numRows = rows.size();

    auto inputArray = decodedInput->base()->as<ArrayVector>();
    auto inputElements = inputArray->elements();
    float* inputValues = inputElements->values()->asMutable<float>();
    auto inputOffsets = inputArray->rawOffsets();
    auto inputSizes = inputArray->rawSizes();

    std::map<vector_size_t, vector_size_t> rowMap;
    std::unordered_set<vector_size_t> uniqueRawIndexeSet;
    std::vector<vector_size_t> uniqueRawIndexeVector;
    vector_size_t numUniqueRows = 0;
    int numCols = dims[0];
    rows.applyToSelected([&](vector_size_t row) {
      auto mappedIndexInRowData = decodedInput->index(row);
      if (uniqueRawIndexeSet.find(mappedIndexInRowData) ==
          uniqueRawIndexeSet.end()) {
        // add it
        rowMap[row] = numUniqueRows;
        uniqueRawIndexeSet.insert(mappedIndexInRowData);
        uniqueRawIndexeVector.push_back(mappedIndexInRowData);
        ++numUniqueRows;
      } else {
        // already added
        rowMap[row] = rowMap[mappedIndexInRowData];
      }
    });

    int numInputMatrixRows = numUniqueRows;
    Eigen::MatrixXf inputMatrix(numInputMatrixRows, numCols);
    int rowIndex = 0;
    for (auto rawIndex : uniqueRawIndexeVector) {
      Eigen::Map<const Eigen::VectorXf> rowVector(
          inputValues + inputOffsets[rawIndex], numCols);
      inputMatrix.row(rowIndex++) = rowVector;
    }

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMatrix(numInputMatrixRows, numCols);
    for (int i = 0; i < numCols; i++) {
      Eigen::VectorXf colData = inputMatrix.col(i);
      float colMean = colData.mean();
      float colVariance =
          (colData.array() - colMean).square().sum() / (numInputMatrixRows - 1);

      resultMatrix.col(i) =
          (colData.array() - colMean) / sqrt(colVariance + eps_) * weights_[i] +
          bias_[i];
    }

    auto baseOffset = elementsOutput->size();
    elementsOutput->resize(baseOffset + rows.end() * numCols);
    float* outputValues = elementsOutput->values()->asMutable<float>();
    vector_size_t outputOffset = 0;
    rows.applyToSelected([&](vector_size_t row) {
      if (rowMap.find(row) == rowMap.end()) {
        throw std::runtime_error(
            "Mapped index not found for the result matrix.");
      }
      auto mappedIndexInResultMatrix = rowMap[row];
      rawOffsets[row] = outputOffset;
      rawSizes[row] = numCols;
      std::memcpy(
          outputValues + outputOffset,
          resultMatrix.row(mappedIndexInResultMatrix).data(),
          numCols * sizeof(float));
    });

    arrayOutput->setElements(elementsOutput);
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .argumentType("array(REAL)")
                    .returnType("array(REAL)")
                    .build()};
    }

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to the weight matrix.
     */
    float* getTensor() const override {
        return weights_;
    }

    /**
     * @brief Returns the weights of the batch normalization.
     *
     * @return A pointer to the weight matrix.
     */
    float* getWeight() {
        return weights_;
    }

    /**
     * @brief Returns the biases of the batch normalization.
     *
     * @return A pointer to the bias vector.
     */
    float* getBias() {
        return bias_;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    static std::string getName() {
        return "batch_norm_1d";
    }

    /**
     * @brief Returns the path to the weights file.
     *
     * @return The path to the weights file as a string.
     */
    std::string getWeightsFile() {
        return weightsFile_;
    }

    /**
     * @brief Sets the weights for the batch normalization.
     *
     * @param weights A pointer to the new weight matrix.
     */
    void setWeights(float* weights) {
        weights_ = weights;
    }

    /**
     * @brief Estimates the computational cost of applying batch normalization.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

private:
    float* weights_; ///< Pointer to the weight matrix for scaling.
    float* bias_;    ///< Pointer to the bias vector for shifting.
    float eps_;      ///< Small value added to the variance to avoid division by zero.
    std::string weightsFile_; ///< Path to the weights file.
    std::string biasFile_;    ///< Path to the biases file.
};
