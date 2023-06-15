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
#include <Eigen/Dense>
#include <cblas.h>
#include <chrono>
#include <torch/torch.h>
#include "velox/exec/Task.h"
#include "velox/ml_functions/functions.h"
#include <fmt/format.h>


using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;



class DNNBuilder {
 public:

  enum Activation {
    RELU,    
    SOFTMAX   
  };

  DNNBuilder() {
    function_count = 0;
    compute_string = "{}";

  }

  ~DNNBuilder() {}

  std::string build(){
    return compute_string;
  }

  DNNBuilder& denseLayer(int units, int input_size, float* weights, float* bias, Activation ac){

    std::string mat_mul_name = MatrixMultiply::getName() + std::to_string(function_count++);
    std::string mat_add_name = MatrixAddition::getName() + std::to_string(function_count++);
    std::string act_name = "";

    exec::registerVectorFunction(
        mat_mul_name,
        MatrixMultiply::signatures(),
        std::make_unique<MatrixMultiply>(weights, input_size, units)
    );

    exec::registerVectorFunction(
        mat_add_name,
        MatrixAddition::signatures(),
        std::make_unique<MatrixAddition>(bias, units)
    );  

    if(ac == RELU){
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

  private:
    int function_count;
    std::string compute_string;
};