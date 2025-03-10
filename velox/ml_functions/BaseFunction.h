/**
 * @class MLFunction
 * @brief A base class for machine learning functions, inheriting from Velox's VectorFunction.
 *
 * This class provides a common interface for machine learning functions, including methods for
 * retrieving tensors, dimensions, and cost estimates. It also includes utility methods for
 * calculating weighted costs and retrieving cost coefficients.
 */
class MLFunction : public exec::VectorFunction {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~MLFunction() = default;

    /**
     * @brief Returns the tensor associated with this function.
     *
     * @return A pointer to the tensor data.
     */
    virtual float* getTensor() const = 0;

    /**
     * @brief Returns the dimensions of the function.
     *
     * @return A vector containing the dimensions.
     */
    virtual std::vector<int> getDims() {
        return dims;
    }

    /**
     * @brief Returns the name of the function.
     *
     * @return The name of the function as a string.
     */
    virtual std::string getFuncName() {
        return "";
    }

    /**
     * @brief Returns the number of dimensions of the function.
     *
     * @return The number of dimensions.
     */
    virtual int getNumDims() {
        return dims.size();
    }

    /**
     * @brief Estimates the computational cost of applying the function.
     *
     * @param inputDims A vector containing the dimensions of the input.
     * @return A CostEstimate object representing the estimated cost.
     */
    virtual CostEstimate getCost(std::vector<int> inputDims) {
        return CostEstimate(0, inputDims[0], inputDims[1]);
    }

protected:
    std::vector<int> dims; ///< Dimensions of the function.

    /**
     * @brief Calculates the weighted cost of the function.
     *
     * @param name The name of the function.
     * @param cost The base cost of the function.
     * @return The weighted cost as a double.
     */
    double getWeightedCost(std::string name, float cost) {
        std::vector<double> coefficient =
            UdfCostCoefficient::getInstance().getCoefficient(name);
        // FIXME: Implement weighted cost calculation.
        return 0;
    }

    /**
     * @brief Retrieves the cost coefficients for the function.
     *
     * @param name The name of the function.
     * @return A vector of cost coefficients.
     */
    std::vector<double> getCoefficientVector(std::string name) {
        return UdfCostCoefficient::getInstance().getCoefficient(name);
    }
};
