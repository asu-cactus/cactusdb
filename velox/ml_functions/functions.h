#include "velox/expression/VectorFunction.h"
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

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m1(input_values, input_size/dims[0], dims[0]);
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m2(weights_, dims[0], dims[1]); 
        
        
        // std::cout << "Matrix shapes Matmul" << std::endl;
        // std::cout << "Matrix shape: " << m1.rows() << " x " << m1.cols() << std::endl;
        // std::cout << "Matrix shape: " << m2.rows() << " x " << m2.cols() << std::endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 * m2;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << "Time for Matrix multiply (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;

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
        
        std::cout << "Matrix shapes MatAdd" << std::endl;
        std::cout << "Matrix shape: " << m1.rows() << " x " << m1.cols() << std::endl;
        std::cout << "Matrix shape: " << m2.rows() << " x " << m2.cols() << std::endl;


        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m  =  m1 + m2;
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        
        std::cout << "Time difference for Mat Add(sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
        //std::cout << m << std::endl;

        int result_size = m.size();
        float* data = m.data();
        
        std::vector<std::vector<float>> result(rows.size(), std::vector<float>(dims[0]));
        for (int i = 0; i < rows.size(); ++i) {
            for (int j = 0; j < dims[0]; ++j) {
                result[i][j] = m(i,j);
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
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                result[i][j] = std::max(0.0f, input_values[i*num_cols + j]);
            }
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference for RELU(sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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
        std::cout << rows.size() << std::endl;
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
            std::vector<float> result;
            for (int j = 0; j < dims[2]; ++j) {
                result.push_back(data[i*dims[2] + j]);
            }
            results.push_back(result);
        }
        VectorMaker maker{context.pool()};
        output = maker.arrayVector<float>(results, REAL());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto b = std::chrono::time_point_cast<std::chrono::microseconds>(begin).time_since_epoch().count() / 1000000.0;
        auto e = std::chrono::time_point_cast<std::chrono::microseconds>(end).time_since_epoch().count() / 1000000.0;
        std::cout << "Begin-end" << b << " " << e << std::endl;
        std::cout << "Time difference = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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

    private:
        float** weights;
        float** bias;
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
        // for each input sample
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
        std::cout << "Time for conv2d (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
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