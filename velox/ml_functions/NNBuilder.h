#pragma once
#include <folly/init/Init.h>
#include "velox/connectors/tpch/TpchConnector.h"
#include "velox/connectors/tpch/TpchConnectorSplit.h"
#include "velox/core/Expressions.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/expression/VectorFunction.h"
#include <chrono>
#include <torch/torch.h>
#include "velox/exec/Task.h"
#include "velox/ml_functions/functions.h"
#include <fmt/format.h>


using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;



class NNBuilder {
 public:

  enum Activation {
    RELU,    
    SOFTMAX,
    NONE   
  };

  NNBuilder() {
    function_count = 0;
    compute_string = "{}";
  }

  NNBuilder(std::string weightsFile, std::string biasFile) : NNBuilder() {
    weightsFile_ = weightsFile;
    biasFile_ = biasFile;
  }

  ~NNBuilder() {}

  std::string build(){
    return compute_string;
  }

  NNBuilder& denseLayer(int units, int input_size, float* weights, float* bias, Activation ac){

    std::string mat_mul_name = MatrixMultiply::getName() + std::to_string(function_count++);
    std::string mat_add_name = MatrixVectorAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        mat_mul_name,
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weights, input_size, units)
    );

    exec::registerVectorFunction(
        mat_add_name,
        MatrixVectorAddition::signatures(),
        std::make_unique<MatrixVectorAddition>(bias, units)
    );  

    if(ac == Activation::RELU){
      act_name = Relu::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
        act_name,
        Relu::signatures(),
        std::make_unique<Relu>()
     );
    }
    else{
      act_name = Softmax::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
        act_name,
        Softmax::signatures(),
        std::make_unique<Softmax>()
     );
    }

    compute_string = fmt::format("{}({}({}({})))", act_name, mat_add_name, mat_mul_name, compute_string);
    return *this;
  }


  NNBuilder& denseLayer(int units, int input_size, Activation ac){

    std::string mat_mul_name = MatrixMultiply::getName() + std::to_string(function_count++);
    std::string mat_add_name = MatrixAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        mat_mul_name,
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weightsFile_, input_size, units)
    );

    exec::registerVectorFunction(
        mat_add_name,
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(biasFile_, units)
    );  

    if(ac == Activation::RELU){
      act_name = Relu::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
        act_name,
        Relu::signatures(),
        std::make_unique<Relu>()
     );
    }
    else{
      act_name = Softmax::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
        act_name,
        Softmax::signatures(),
        std::make_unique<Softmax>()
     );
    }

    compute_string = fmt::format("{}({}({}({})))", act_name, mat_add_name, mat_mul_name, compute_string);
    return *this;
  }



  NNBuilder& convLayer(int num_filters, int* dims, float* weights, float* bias, Activation ac){

    std::string conv_name = Convolute::getName() + std::to_string(function_count++);
    std::string scal_add_name = VectorScalarAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        conv_name,
        Convolute::signatures(),
        std::make_unique<Convolute>(weights, dims)
    );


    exec::registerVectorFunction(
        scal_add_name,
        VectorScalarAddition::signatures(),
        std::make_unique<VectorScalarAddition>(bias, num_filters)
    );

    if(ac == Activation::RELU){
      act_name = Relu::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
        act_name,
        Relu::signatures(),
        std::make_unique<Relu>()
     );
    } else if (ac == Activation::SOFTMAX){
      act_name = Softmax::getName() + std::to_string(function_count++);
      exec::registerVectorFunction(
        act_name,
        Softmax::signatures(),
        std::make_unique<Softmax>()
     );
    } 
    if(act_name == "")
      compute_string = fmt::format("{}({}({}))", scal_add_name, conv_name, compute_string);
    else
      compute_string = fmt::format("{}({}({}({})))", act_name, scal_add_name, conv_name, compute_string);

    return *this;
  }

  NNBuilder& maxPoolLayer(int side, int height, int width) {
    std::string max_pool_name = MaxPool::getName() + std::to_string(function_count++);
    exec::registerVectorFunction(
        max_pool_name,
        MaxPool::signatures(),
        std::make_unique<MaxPool>(side, height, width)
    );
    compute_string = fmt::format("{}({})", max_pool_name, compute_string);
    return *this;
  }

  private:
    int function_count;
    std::string compute_string;
    // for now there is just one file
    // we may change it to a list of file names
    // in case the weights for each layer is stored in a separate file
    std::string weightsFile_;
    std::string biasFile_;
};


