#pragma once
#include "velox/expression/VectorFunction.h"
#include "velox/vector/DictionaryVector.h"
#include <Eigen/Dense>
#include <cblas.h>
#include <chrono>
#include "velox/exec/Task.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;


/*
    TODO
    1. conv2d - done
    2. max pooling - done
    3. flatten - not required
    4. batch normalization 
    5. padding
    6. concatenate
    7. embedding
    8. transformer -> exiting libraries, encoder, decoder, how to decompoose it into atomic linear algebra
    // focus on weight 
    9. GRU -> not interesting 

*/

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

     MatrixMultiply(std::string weightsFile, int num_rows, int num_cols) {
        weightsFile_ = weightsFile; 
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

        // std::cout << "input_size:" << "," << input_size << std::endl;
        // std::cout << "input_values:" << "," << input_values[0] << "," << input_values[1] << std::endl;
        
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values, input_size/dims[0], dims[0]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(weights_, dims[0], dims[1]); 
        
        
        // std::cout << "Matrix shapes Matmul" << std::endl;
        // std::cout << "Matrix shape: " << m1.rows() << " x " << m1.cols() << std::endl;
        // std::cout << "Matrix shape: " << m2.rows() << " x " << m2.cols() << std::endl;

        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 * m2;
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time for Matrix multiply (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

        // std::vector<std::vector<float>> result(m.rows(), std::vector<float>(m.cols()));
        // for (int i = 0; i < m.rows(); ++i) {
        //     for (int j = 0; j < m.cols(); ++j) {
        //         result[i][j] = m(i, j);
        //     }
        // }
        std::vector<std::vector<float>> result;
        for (int i = 0; i < m.rows(); i++) {
            std::vector<float> row(
            m.row(i).data(),
            m.row(i).data() + m.cols());
            result.push_back(row);
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

    static std::string getName() {
        return "mat_mul";
    };

    std::string getWeightsFile() {
        return weightsFile_;
    }

    void setWeights(float* weights){
        weights_ = weights;
    }


private:
    float* weights_;
    std::string weightsFile_;
    
};

class MatrixMultiply_b: public MLFunction {
public:
    MatrixMultiply_b(int num_rows, int num_cols, int num_samples, float* weights, int blocks) {
        dims.push_back(num_rows);
        dims.push_back(num_cols);
        dims.push_back(num_samples);
        dims.push_back(blocks);
        weights_ = weights;
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
        
        BaseVector::ensureWritable(rows, type, context.pool(), output);
        VectorMaker maker{context.pool()};

        BaseVector* left = args[0].get();
        BaseVector* right = args[1].get();

        exec::LocalDecodedVector leftHolder(context, *left, rows);
        auto decodedLeftArray = leftHolder.get();
        auto baseLeftArray =
            decodedLeftArray->base()->as<ArrayVector>()->elements();

        exec::LocalDecodedVector rightHolder(context, *right, rows);
        auto decodedRightArray = rightHolder.get();
        auto baseRightArray = rightHolder->base()->as<ArrayVector>()->elements();

        float* input_values_v = baseLeftArray->values()->asMutable<float>();
        float* input_values_w = baseRightArray->values()->asMutable<float>();
 
        // auto varrayVector = std::make_shared<ArrayVector<float>>();
        // const int elements_v_per_row = 1500000; //6000*250
        // const int elements_w_per_row = 125000; // 250*500
        
        // std::vector<std::vector<float>> result(1, std::vector<float>(dims[1]*dims[2])); //6000*500
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values_v, dims[2], dims[0]);//3*2
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(input_values_w, dims[0], dims[1]); //2*5
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 * m2;//3*5

        // for (int i = 0; i < m.rows(); ++i) {
        //         for (int j = 0; j < m.cols(); ++j) {
        //             result[0][i * dims[1] + j] = m(i, j);
        //     }
        // }
        // m = m.reshaped(1, m.size());
        // std::cout << "shape: " << m.rows() << "," <<m.cols() << std::endl;
        std::vector<std::vector<float>> result;
        for (int i = 0; i < m.rows(); i++) {
            std::vector<float> row(
            m.row(i).data(),
            m.row(i).data() + m.cols());
            result.push_back(row);
        }
        output = maker.arrayVector<float>(result, REAL());
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("array(REAL)")
                     .argumentType("array(REAL)")
                     .argumentType("array(REAL)")
                     .build()};
    }

    float* getTensor() const override {
        return weights_;
    }

    static std::string getName() {
        return "mat_mul_b";
    };


private:
    float* weights_;
    
};

class MatrixMultiply_h: public MLFunction {
public:
    MatrixMultiply_h(int num_rows, int num_cols, int block_size, float* weights) {
        dims.push_back(num_rows);
        dims.push_back(num_cols);
        dims.push_back(block_size);
        weights_ = weights;
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {
        
        BaseVector::ensureWritable(rows, type, context.pool(), output);
        VectorMaker maker{context.pool()};
 
        // auto input_elements_w = args[1]->as<ArrayVector>()->elements();
        // float* input_values_w = input_elements_w->values()->asMutable<float>();
        // float* input_values_w = weights_;
        auto v1 = args[0]->as<DictionaryVector<ComplexType>>();
        // auto ss3 = args[0]->as<ArrayVector>()->elements();
        auto v2 = v1->valueVector();
        auto v3 = v2->as<LazyVector>();
        auto v4 = v3->loadedVector();
        auto input_elements_v = v4->as<ArrayVector>()->elements();
        float* input_values_v = input_elements_v->values()->asMutable<float>();

        auto w1 = args[1]->as<DictionaryVector<ComplexType>>();
        auto w2 = w1->valueVector();
        auto w3 = w2->as<LazyVector>();
        auto w4 = w3->loadedVector();
        auto input_elements_w = w4->as<ArrayVector>()->elements();
        float* input_values_w = input_elements_w->values()->asMutable<float>();
        // auto varrayVector = std::make_shared<ArrayVector<float>>();
        // const int elements_v_per_row = 1500000; //6000*250
        // const int elements_w_per_row = 125000; // 250*500
        int current_block_size = (input_elements_w->size()< (dims[0] * dims[2])) ? input_elements_w->size()/dims[0] : dims[2];
        std::vector<std::vector<float>> result; //6000*500
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values_v, input_elements_v->size()/dims[0], dims[0]);//3*2
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(input_values_w, dims[0], current_block_size); //2*5
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 * m2;//3*5
        
        for (int i = 0; i < m.rows(); i++) {
            std::vector<float> row(
            m.row(i).data(),
            m.row(i).data() + m.cols());
            result.push_back(row);
        }

        // for (int i = 0; i < m.rows(); ++i) {
        //         for (int j = 0; j < m.cols(); ++j) {
        //             result[0][i * dims[1] + j] = m(i, j);
        //     }
        // }
        // m = m.reshaped(1, m.size());
        // std::cout << "shape: " << m.rows() << "," <<m.cols() << std::endl;
        // for (int i = 0; i < m.rows(); i++) {
        //     std::vector<float> row(
        //     m.row(i).data(),
        //     m.row(i).data() + m.cols());
        //     result.push_back(row);
        // }
        output = maker.arrayVector<float>(result, REAL());
    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
        return {exec::FunctionSignatureBuilder()
                     .returnType("array(REAL)")
                     .argumentType("array(REAL)")
                     .argumentType("array(REAL)")
                     .build()};
    }

    float* getTensor() const override {
        return weights_;
    }

    static std::string getName() {
        return "mat_mul_h";
    };


private:
    float* weights_;
    
};

// there is no need to pass any parameter here since dimensions can be figured out from the input 
// can the optimiser figure out the dimensions from the context?
class MatrixAddition: public MLFunction {
public:
    MatrixAddition(float* weights, int num_cols) {
        weights_ = weights;
        dims.push_back(num_cols);
    }

    MatrixAddition(std::string weightsFile, int num_cols) {
        weightsFile_ = weightsFile;
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

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values, rows.size(), dims[0]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(weights_, rows.size(), dims[0]);
        
        // std::cout << "Matrix shapes MatAdd" << std::endl;
        // std::cout << "Matrix shape: " << m1.rows() << " x " << m1.cols() << std::endl;
        // std::cout << "Matrix shape: " << m2.rows() << " x " << m2.cols() << std::endl;


        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 + m2;
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        
        // std::cout << "Time difference for Mat Add(sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
        //std::cout << m << std::endl;

        int result_size = m.size();
        float* data = m.data();
        
        // std::vector<std::vector<float>> result(rows.size(), std::vector<float>(dims[0]));
        // for (int i = 0; i < rows.size(); ++i) {
        //     for (int j = 0; j < dims[0]; ++j) {
        //         result[i][j] = m(i,j);
        //     }
        // }
        std::vector<std::vector<float>> result;
        for (int i = 0; i < m.rows(); i++) {
            std::vector<float> row(
            m.row(i).data(),
            m.row(i).data() + m.cols());
            result.push_back(row);
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

    static std::string getName() {
        return "mat_add";
    };

    std::string getWeightsFile() {
        return weightsFile_;
    }

    void setWeights(float* weights){
        weights_ = weights;
    }

private:
    float* weights_;
    std::string weightsFile_;

};


// TODO: add future support to implement matrix addition in one class
// matrix addition, matrix addition brodcast by row/col
class MatrixVectorAddition: public MLFunction {
public:
    MatrixVectorAddition(float* weights, int num_cols) {
        weights_ = weights;
        dims.push_back(num_cols);
    }

    MatrixVectorAddition(std::string weightsFile, int num_cols) {
        weightsFile_ = weightsFile;
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

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values, rows.size(), dims[0]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(weights_, 1, dims[0]);

        m1.rowwise() += m2.row(0);

        std::vector<std::vector<float>> result;
        for (int i = 0; i < m1.rows(); i++) {
            std::vector<float> row(
            m1.row(i).data(),
            m1.row(i).data() + m1.cols());
            result.push_back(row);
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

    static std::string getName() {
        return "mat_vector_add";
    };

    std::string getWeightsFile() {
        return weightsFile_;
    }

    void setWeights(float* weights){
        weights_ = weights;
    }

private:
    float* weights_;
    std::string weightsFile_;

};


class Relu: public MLFunction {
public:
    Relu() {}

    float static relu_function(float x) {
        return (x > 0.0f) ? x : 0.0f;
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
        // considering all arrays have same size
        int num_rows = args[0]->size();
        int num_cols = input_size / num_rows;
        
        std::vector<std::vector<float>> result;
        for (int i = 0; i < num_rows; i++) {
            std::vector<float> rowResult(num_cols);
            std::transform(input_values + i*num_cols, input_values + (i+1)*num_cols,
            rowResult.data(), relu_function);
            result.push_back(rowResult);
        }

        // std::vector<std::vector<float>> result(num_rows, std::vector<float>(num_cols));
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // for (int i = 0; i < num_rows; ++i) {
        //     for (int j = 0; j < num_cols; ++j) {
        //         result[i][j] = std::max(0.0f, input_values[i*num_cols + j]);
        //     }
        // }
        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // std::cout << "Time difference for RELU(sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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

    static std::string getName() {
        return "relu";
    };
};

class Softmax: public MLFunction {
public:
    Softmax() {}

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

        int num_rows = args[0]->size();
        int num_cols = input_size / num_rows;
        
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(input_values, num_rows, num_cols);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::ArrayXXf exp = m.array().exp();
        Eigen::ArrayXXf sum = exp.rowwise().sum();
        for (int i = 0; i < exp.rows(); i++) {
            exp.row(i) /= sum(i);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
     //    std::cout << "Time difference for Softmax(sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
        std::vector<std::vector<float>> result(num_rows, std::vector<float>(num_cols));
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                result[i][j] = exp(i,j);
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

    static std::string getName() {
        return "softmax";
    };
};

class TorchDNN: public MLFunction {
public:
    TorchDNN(float** weights, float** bias, std::vector<int> dimensions) {
        this->weights = weights;
        this->bias = bias;
        dims = dimensions;
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        torch::nn::Linear dense1(dims[0], dims[1]);
        torch::nn::Linear dense2(dims[1],dims[2]);
        torch::nn::ReLU relu;

        torch::Tensor weightTensor1 = torch::from_blob(weights[0], {dims[0], dims[1]}).t();
        torch::Tensor weightTensor2 = torch::from_blob(weights[1], {dims[1], dims[2]}).t();
        torch::Tensor bias1 = torch::from_blob(bias[0], {dims[1]});
        torch::Tensor bias2 = torch::from_blob(bias[1], {dims[2]});
        
        dense1->weight.set_data(weightTensor1);
        dense2->weight.set_data(weightTensor2);
        dense1->bias.set_data(bias1);
        dense2->bias.set_data(bias2);
        
        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
        int input_size = input_elements->size();
        
        torch::Tensor input = torch::from_blob(input_values, {rows.size(), dims[0]});

        torch::Tensor layer1_output = dense1->forward(input);
        torch::Tensor reluOutput = relu->forward(layer1_output);
        torch::Tensor layer2_output = dense2->forward(reluOutput);
        torch::Tensor softmax_output = torch::nn::functional::softmax(layer2_output, 1);
        float* data = softmax_output.data_ptr<float>();

        std::vector<std::vector<float>> results;
        for (int i = 0; i < rows.size(); ++i) {
            // std::vector<float> result;
            std::vector<float> result(data + i*dims[2], data+ (i+1)*dims[2]);
            // for (int j = 0; j < dims[2]; ++j) {
            //     result.push_back(data[i*dims[2] + j]);
            // }
            results.push_back(result);
        }
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
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
    
    // Getter method for weights
    float** getWeights() const {
        return weights;
    }

    // Getter method for bias
    float** getBias() const {
        return bias;
    }

    private:
        float** weights;
        float** bias;
};

class TorchDNN_Multi : public MLFunction {
public:
    TorchDNN_Multi(std::vector<float*> weights, std::vector<float*> bias, std::vector<int> dimensions) {
        this->weights = weights;
        this->bias = bias;
        dims = dimensions;
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        std::vector<torch::nn::Linear> dense_layers;
        std::vector<torch::Tensor> weights_tensors;
        std::vector<torch::Tensor> bias_tensors;
        std::vector<torch::nn::ReLU> relus;

        // Create layers
        for (int i = 0; i < dims.size() - 1; ++i) {
            dense_layers.push_back(torch::nn::Linear(dims[i], dims[i+1]));
            weights_tensors.push_back(torch::from_blob(weights[i], {dims[i], dims[i+1]}).t());
            bias_tensors.push_back(torch::from_blob(bias[i], {dims[i+1]}));
            relus.push_back(torch::nn::ReLU());
        }

        // Set weights and biases
        for (int i = 0; i < dense_layers.size(); ++i) {
            dense_layers[i]->weight.set_data(weights_tensors[i]);
            dense_layers[i]->bias.set_data(bias_tensors[i]);
        }
        
        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
        torch::Tensor input = torch::from_blob(input_values, {rows.size(), dims[0]});

        torch::Tensor output_tensor = input;
        for (int i = 0; i < dense_layers.size(); ++i) {
            output_tensor = dense_layers[i]->forward(output_tensor);
            output_tensor = relus[i]->forward(output_tensor);
        }

        // Softmax output
        output_tensor = torch::nn::functional::softmax(output_tensor, 1);
        float* data = output_tensor.data_ptr<float>();

        // Prepare results
        std::vector<std::vector<float>> results;
        for (int i = 0; i < rows.size(); ++i) {
            std::vector<float> result(data + i*dims.back(), data+ (i+1)*dims.back());
            results.push_back(result);
        }
        
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
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
    
    // Getter method for weights
    const std::vector<float*>& getWeights() const {
        return weights;
    }

    // Getter method for bias
    const std::vector<float*>& getBias() const {
        return bias;
    }

private:
    std::vector<float*> weights;
    std::vector<float*> bias;
};

class Convolute: public MLFunction {
public:
    Convolute(float* weights, int* dims_) {
        weights_ = weights; 
        for(int i=0; i < 6; i++)
            dims.push_back(dims_[i]);
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
       
        int input_height =  dims[4];
        int input_width = dims[5];
        int input_channel_size = input_height * input_width;
        int input_size = input_channel_size * dims[3];

        int filter_channel_size = dims[1] * dims[2];
        int filter_size = filter_channel_size * dims[3];

        int output_height = input_height - dims[1] + 1;
        int output_width = input_width - dims[2] + 1;
        
        std::vector<std::vector<float>> results(rows.size(), std::vector<float>(output_height * output_width * dims[0]));
     
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        
        for (int s = 0; s < rows.size(); s++) {
            // for each channel
            for(int c = 0; c < dims[3]; c++) {
                Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(input_values + s * input_size + c * input_channel_size, input_height, input_width);
                // for every filter 
                for(int f=0; f < dims[0]; f++){
                    int filter_offset = f * output_height * output_width;
                    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> kernel(weights_ + f * filter_size + c * filter_channel_size, dims[1], dims[2]);
                    for (int i = 0; i < output_height; ++i){
                        int offset = filter_offset + i*output_width;
                        for (int j = 0; j < output_width; ++j) {
                            results[s][offset + j] += (input.block(i, j, dims[1], dims[2]).cwiseProduct(kernel)).sum();
                        }
                    }
                }   
            }
        }

        // #pragma omp parallel for
        // for (int s = 0; s < rows.size(); s++) {
        //     for (int f = 0; f < dims[0]; f++) {
        //         // Pre-calculate filter offset
        //         int filter_offset = f * output_height * output_width;
        //         for (int c = 0; c < dims[3]; c++) {
        //             // Map input and filter data
        //             Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(input_values + s * input_size + c * input_channel_size, input_height, input_width);
        //             Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> kernel(weights_ + f * filter_size + c * filter_channel_size, dims[1], dims[2]);

        //             // Convolution operation
        //             for (int i = 0; i < output_height; ++i) {
        //                 int output_row_offset = i * output_width;
        //                 for (int j = 0; j < output_width; ++j) {
        //                     // Compute dot product using Eigen operations
        //                     results[s][filter_offset + output_row_offset + j] += (input.block(i, j, dims[1], dims[2]).cwiseProduct(kernel)).sum();
        //                 }
        //             }
        //         }
        //     }
        // }

       
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time for conv2d (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
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

    static std::string getName() {
        return "conv2d";
    };


private:
    float* weights_;
    
};

class TorchConvolute: public MLFunction {
public:
    TorchConvolute(float* weights, int* dims_) {
        weights_ = weights; 
        for(int i=0; i < 6; i++)
            dims.push_back(dims_[i]);
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();     
        BaseVector::ensureWritable(rows, type, context.pool(), output);
        
        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
       
        int input_height =  dims[4];
        int input_width = dims[5];

        int output_height = input_height - dims[1] + 1;
        int output_width = input_width - dims[2] + 1;
        
        std::vector<std::vector<float>> results(rows.size(), std::vector<float>(output_height * output_width * dims[0]));
       
        torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(dims[3], dims[0], {dims[1], dims[2]}));
        // torch::Tensor conv_weights = torch::tensor(weights_).view({dims[3], dims[0], dims[1], dims[2]});

        // conv_layer->weight = torch::nn::parameter::Parameter (conv_weights);
        torch::Tensor input_data = torch::from_blob(input_values, {rows.size(), dims[3], input_height, input_width});

       
        torch::Tensor output_data = conv_layer(input_data);
        
        float* data = output_data.data_ptr<float>();
        
        int row_size = output_height * output_width * dims[0];
       
        for (int i = 0; i < rows.size(); ++i) {
            std::vector<float> result;
            for (int j = 0; j < row_size; ++j) {
                result.push_back(data[i*row_size + j]);
            }
            results.push_back(result);
        }

        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time for conv2d (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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

    static std::string getName() {
        return "torchconv2d";
    };


private:
    float* weights_;
    
};

class TorchCNN: public MLFunction {
public:
    TorchCNN(float* weights, float* bias, int* dims_) {
        weights_ = weights;
        bias_ = bias; 
        for(int i=0; i < 7; i++)
            dims.push_back(dims_[i]);
    }

    void apply(
        const SelectivityVector& rows,
        std::vector<VectorPtr>& args,
        const TypePtr& type,
        exec::EvalCtx& context,
        VectorPtr& output) const override {

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();     
        BaseVector::ensureWritable(rows, type, context.pool(), output);
        
        auto input_elements = args[0]->as<ArrayVector>()->elements();
        float* input_values = input_elements->values()->asMutable<float>();
       
        int input_height =  dims[4];
        int input_width = dims[5];

        int output_height = input_height - dims[1] + 1;
        int output_width = input_width - dims[2] + 1;

        int input_size = input_elements->size();
        // std::cout << "input_size:" << "," << input_size << std::endl;
        // std::cout << "input_values:" << "," << input_values[0] << "," << input_values[1] << "," << input_values[2080] << std::endl;
        // std::cout << "row size" << "," << rows.size() << std::endl;

        std::vector<std::vector<float>> results(rows.size(), std::vector<float>(output_height * output_width * dims[0]));
       
        torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(dims[0], dims[3], {dims[1], dims[2]}).bias(false));
        // torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(dims[3], dims[0], {dims[1], dims[2]}));
        torch::Tensor conv_weights = torch::from_blob(weights_, {dims[3], dims[0], dims[1], dims[2]}).to(torch::kFloat);

        auto parameters = conv_layer->named_parameters();

        // Find and set the weight parameter
        for (auto& named_param : parameters) {
            if (named_param.key() == "weight") {
                named_param.value().data() = conv_weights;
                break;
            }
        }
        torch::Tensor input_data = torch::from_blob(input_values, {rows.size(), dims[3], input_height, input_width}).to(torch::kFloat);

       
        torch::Tensor output_data = conv_layer->forward(input_data);

        // Convert bias values to a tensor
        torch::Tensor bias_tensor = torch::from_blob(bias_, {dims[0]}); 
        if (conv_layer->bias.defined()) {
            output_data += bias_tensor;
        }

        // output_data = torch::relu(output_data);

        // output_data = torch::max_pool2d(output_data, {dims[6], dims[6]});
        
        float* data = output_data.data_ptr<float>();
        
        int row_size = output_height * output_width * dims[0];
       
        for (int i = 0; i < rows.size(); ++i) {
            std::vector<float> result;
            for (int j = 0; j < row_size; ++j) {
                result.push_back(data[i*row_size + j]);
            }
            results.push_back(result);
        }

        // for (auto entry: results) {
        //     for (int i =0; i < 1000; i++){
        //         std::cout << entry[i] << std::endl;
        //     }
        // }

        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time for conv2d (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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

    // Getter method for weights
    float* getWeights() const {
        return weights_;
    }

    // Getter method for bias
    float* getBias() const {
        return bias_;
    }
    static std::string getName() {
        return "torchcnn";
    };


private:
    float* weights_;
    float* bias_;
};



class VectorScalarAddition: public MLFunction {

public:
    VectorScalarAddition(float* weights, int size) {
        weights_ = weights;
        dims.push_back(size);
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
        int num_cols = input_elements->size() / rows.size();
        
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(input_values, rows.size(), num_cols);
        // for each filter add bias
        for(int i=0, step = num_cols/dims[0]; i < dims[0]; i++){
            input.block(0, i*step, rows.size(), step).array() += weights_[i];
        }

        std::vector<std::vector<float>> results(input.rows(), std::vector<float>(input.cols()));
        for (int i = 0; i < input.rows(); ++i) {
            for (int j = 0; j < input.cols(); ++j) {
                results[i][j] = input(i, j);
            }
        }
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
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

    static std::string getName() {
        return "vec_scal_add";
    };

private:
    float* weights_;
};

class MaxPool: public MLFunction {
public:
    MaxPool(int side, int rows, int cols) {
        dims.push_back(side);
        dims.push_back(rows);
        dims.push_back(cols);
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
        int num_cols = input_elements->size() / rows.size();
        int num_channels = num_cols / (dims[1] * dims[2]);
        int side = dims[0];
        int output_size = (dims[1] * dims [2]) / (side * side);
        int output_rows = dims[1] / side;
        int output_cols = dims[2] / side;
        // this can be done by using one big matrix but padding will not be possible then
        // this doesn't support padding yet but this makes it possible to add it later
        std::vector<std::vector<float>> results(rows.size(), std::vector<float>(num_cols/(side*side)));
        // for each sample
        for(int s=0; s < rows.size(); s++) {
            for(int c=0; c < num_channels; c++){
                Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> input(input_values + s * num_cols +  c * dims[1] * dims[2], dims[1], dims[2]);
                for(int i=0; i < output_rows; i++){
                    for(int j=0; j < output_cols; j++){
                        results[s][c*output_size + i*output_cols + j] = input.block(i * side, j * side, side, side).maxCoeff();
                    }
                }
            }
        }

        // for (const auto& inner_vector : results) {
        //     // Iterate over each element in the inner vector
        //     for (const auto& element : inner_vector) {
        //         std::cout << element << std::endl;
        //     }
        // }

        // for(int i=0; i < 64; i++){

        //     for(int j=0; j < 144; j++){
        //         if(j % 12 == 0)
        //             std::cout << std::endl;
        //         std::cout << results[0][i*144 + j];
        //     }
            
        // }


        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
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

    static std::string getName() {
        return "max_pool";
    };
};
