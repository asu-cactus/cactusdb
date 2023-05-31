//#define EIGEN_USE_BLAS
#include "velox/expression/VectorFunction.h"
#include <Eigen/Dense>
#include <cblas.h>
#include <chrono>
#include "velox/exec/Task.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;


// TODO: Refactor
class MLFunction : public exec::VectorFunction {
    public:
        virtual ~MLFunction() = default;
        
        std::vector<int> dims;
        virtual float* getTensor() const = 0;
        
        virtual std::vector<int> getDims() {
            return dims;
        }

        virtual int getNumDims(){
            return dims.size();
        }
};

class MatrixMultiply: public MLFunction {
public:
    MatrixMultiply(float* weights, int num_rows, int num_cols) {
        weights_ = weights; 
        dims.push_back(num_rows);
        dims.push_back(num_cols);
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
        
        BaseVector::ensureWritable(rows, type, context.pool(), output);
        
        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
        int input_size = input_elements->size();

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values, input_size/dims[0], dims[0]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(weights_, dims[0], dims[1]); 
        
       
        std::cout << "Matrix shapes" << std::endl;
        std::cout << "Matrix shape: " << m1.rows() << " x " << m1.cols() << std::endl;
        std::cout << "Matrix shape: " << m2.rows() << " x " << m2.cols() << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 * m2;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
        std::cout << m << std::endl;

        std::vector<std::vector<float>> result(m.rows(), std::vector<float>(m.cols()));
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                result[i][j] = m(i, j);
            }
        }
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(result, REAL());
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("array(REAL)")
                     .argumentType("array(REAL)")
                     .build()};
    }

    float* getTensor() const override {
        return weights_;
    }

private:
    float* weights_;
    
};

class MatrixAddition: public MLFunction {
public:
    MatrixAddition(float* weights, int num_rows, int num_cols) {
        weights_ = weights;
        dims.push_back(num_rows);
        dims.push_back(num_cols);
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        BaseVector::ensureWritable(rows, type, context.pool(), output);

        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
        int input_size = input_elements->size();

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values, dims[0], input_size/dims[0]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(weights_, dims[0], dims[1]);
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 + m2;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        
        std::cout << "Time difference (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
        std::cout << m << std::endl;

        int result_size = m.size();
        float* data = m.data();
        
        std::vector<std::vector<float>> result(m.rows(), std::vector<float>(m.cols()));
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                result[i][j] = m(i, j);
            }
        }
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(result, REAL());
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("array(REAL)")
                     .argumentType("array(REAL)")
                     .build()};
    }


    float* getTensor() const override {
        return weights_;
    }

private:
    float* weights_;

};


class Relu: public MLFunction {
public:
    Relu() {}

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        BaseVector::ensureWritable(rows, type, context.pool(), output);

        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
        int input_size = input_elements->size();
        // considering all arrays have same size
        int num_rows = args[0]->size();
        int num_cols = input_size / num_rows;

        std::vector<std::vector<float>> result(num_rows, std::vector<float>(num_cols));
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                result[i][j] = std::max(0.0f, input_values[i*num_cols + j]);
            }
        }

        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(result, REAL());
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("array(REAL)")
                     .argumentType("array(REAL)")
                     .build()};
    }

    // getters for metadata to be used by optimiser
    float* getTensor() const override {
        return new float[0];
    }
};


class TorchDenseLayer: public MLFunction {
public:
    TorchDenseLayer(int inputSize, int outputSize) {
        dims.push_back(inputSize);
        dims.push_back(outputSize);
    }
    // TODO:  create a constructor that takes in weights. Currently linear layer takes care of init weights
    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        torch::nn::Linear dense(dims[0], dims[1]);
        torch::nn::ReLU relu;

        float* values = args[0]->values()->asMutable<float>();

        torch::Tensor input = torch::from_blob(values, {dims[0]});

        torch::Tensor layerOutput = dense->forward(input);

        torch::Tensor reluOutput = relu->forward(layerOutput);
        // Print the output
        std::cout << "Output:\n" << reluOutput << std::endl;

        float* data = reluOutput.data_ptr<float>();

        auto result = BaseVector::create<FlatVector<float>>(type, dims[1], context.pool());
        for (auto i = 0; i < dims[1]; ++i) {
            result->set(i, data[i]);
        }
        output = result;
        std::cout<< "\n" << output->size();
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("REAL")
                     .argumentType("REAL")
                     .build()};
    }

    // getters for metadata to be used by optimiser
    float* getTensor() const override {
        return new float[0];
    }
};






