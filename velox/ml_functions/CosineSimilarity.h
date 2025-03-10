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
 * @class CosineSimilarity
 * @brief A class that computes the cosine similarity between two input arrays, inheriting from MLFunction.
 *
 * This class provides functionality to calculate the cosine similarity between two arrays of real numbers (floats).
 * Cosine similarity measures the cosine of the angle between two vectors, providing a value between -1 and 1.
 */
class CosineSimilarity : public MLFunction {
public:
    /**
     * @brief Constructor that initializes the cosine similarity computation with the dimension of the input arrays.
     *
     * @param dim The dimension (number of features) of the input arrays.
     */
    CosineSimilarity(int dim) {
        dims.push_back(dim);
    }

    /**
     * @brief Applies the cosine similarity computation to the input arrays.
     *
     * This method processes the input arrays, computes the cosine similarity between corresponding vectors,
     * and stores the result in the output vector.
     *
     * @param rows A SelectivityVector specifying the rows to process.
     * @param args A vector of input arguments (e.g., the two input arrays).
     * @param type The type of the output vector.
     * @param context The execution context.
     * @param output The output vector where the cosine similarity results will be stored.
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

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto leftInputOffset =
        decodedLeftArray->base()->as<ArrayVector>()->rawOffsets();
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto rightInputOffset =
        decodedRightArray->base()->as<ArrayVector>()->rawOffsets();
    auto baseRightArray =
        decodedRightArray->base()->as<ArrayVector>()->elements();
    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    std::vector<float> resultVector(numInput);

    rows.applyToSelected([&](vector_size_t i) {
      // Map the input values into Eigen vectors
      auto leftIndexInRaw = decodedLeftArray->index(i);
      auto rightIndexInRaw = decodedRightArray->index(i);
      Eigen::Map<Eigen::VectorXf> vec1(
          input1Values + leftInputOffset[leftIndexInRaw], dims[0]);
      Eigen::Map<Eigen::VectorXf> vec2(
          input2Values + rightInputOffset[rightIndexInRaw], dims[0]);

      // Compute cosine similarity
      float dotProduct = vec1.dot(vec2);
      float norm1 = vec1.norm();
      float norm2 = vec2.norm();
      float cosineSim = dotProduct / (norm1 * norm2 + 1e-8);

      // Store the result
      resultVector[i] = cosineSim;
    });

    VectorMaker maker{context.pool()};
    output = maker.flatVector<float>(resultVector, REAL());
    }

    /**
     * @brief Returns the function signatures supported by this class.
     *
     * @return A vector of shared pointers to FunctionSignature objects.
     */
    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                    .argumentType("array(REAL)")
                    .argumentType("array(REAL)")
                    .returnType("REAL")
                    .build()};
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string ("cosine_similarity").
     */
    static std::string getName() {
        return "cosine_similarity";
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
     * @brief Estimates the computational cost of applying the cosine similarity computation.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

private:
    std::vector<int> dims; ///< Dimensions of the input arrays.
};
