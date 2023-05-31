/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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
//#define EIGEN_USE_BLAS
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


using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

// TODO: Refactor
class MLFunctionsTest {
 public:

  MLFunctionsTest() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();

    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();

    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();

  }

  ~MLFunctionsTest() {
  }

  /// Run the demo.
  void run();
  void test_mat_mul();
  void test_mat_add();
  void test_relu();
  void test_dense_layer();
  void test_torch_dense_layer();

  std::shared_ptr<memory::MemoryPool> pool_ = memory::addDefaultLeafMemoryPool();
  VectorMaker maker{pool_.get()};
  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};
  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

};


void MLFunctionsTest::test_mat_mul() { 
//Eigen::setNbThreads(48); 
  int output_size = 5;
  int input_size = 10;
  int num_features = 5;
  int size = output_size*input_size;
  
  auto weights = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 
  
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_features; i++){
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"mat_mul(x)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;

}

void MLFunctionsTest::test_mat_add() {
//Eigen::setNbThreads(48); 
  int num_rows = 5;
  int num_cols = 10;
  int size = num_rows*num_cols;
  
  auto weights = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 


  std::vector<std::vector<float>> inputVectors;
  for(int i=0; i < num_rows; i++){
    std::vector<float> inputVector;
    for(int j=0; j < num_cols; j++){
      inputVector.push_back(i*j);
    }
    inputVectors.push_back(inputVector);
  }
  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  
  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});


  // step1: Register
  exec::registerVectorFunction(
  "mat_add",
  MatrixAddition::signatures(),
  std::make_unique<MatrixAddition>(weights->values()->asMutable<float>() , num_rows, num_cols));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"mat_add(x)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
}

void MLFunctionsTest::test_relu(){
 
  int num_rows = 10;
  int num_cols = 10; 

  std::vector<std::vector<float>> inputVectors;
  for(int i=0; i < num_rows; i++){
    std::vector<float> inputVector;
    for(int j=0; j < num_cols; j++){
      inputVector.push_back(i*j);
    }
    inputVectors.push_back(inputVector);
  }

  auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {inputArrayVector});
  

  // step1: Register
  exec::registerVectorFunction(
  "relu",
  Relu::signatures(),
  std::make_unique<Relu>());

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"relu(x)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;
  
}

void MLFunctionsTest::test_dense_layer() {

  int output_size = 5;
  int input_size = 10;
  int num_features = 3;
  int size = output_size*input_size;
  
  auto weights = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  weights->set(i, i*10);
  } 

  auto bias = maker.flatVector<float>(size);
  for(int i=0; i < size; i++){
	  bias->set(i, i % output_size);
  } 
  
  std::vector<std::vector<float>> featureVectors;
  for(int i=0; i < num_features; i++){
    std::vector<float> featureVector;
    for(int j=0; j < input_size; j++){
      featureVector.push_back(i*j);
    }
    featureVectors.push_back(featureVector);
  }
  auto featureArrayVector = maker.arrayVector<float>(featureVectors, REAL());
  auto inputRowVector = maker.rowVector({"x"}, {featureArrayVector});

  exec::registerVectorFunction(
    "mat_mul",
    MatrixMultiply::signatures(),
    std::make_unique<MatrixMultiply>(weights->values()->asMutable<float>(), input_size, output_size)
  );

  exec::registerVectorFunction(
    "mat_add",
    MatrixAddition::signatures(),
    std::make_unique<MatrixAddition>(bias->values()->asMutable<float>() , num_features, output_size)
  );  

  exec::registerVectorFunction(
    "relu",
    Relu::signatures(),
    std::make_unique<Relu>()
  );

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"relu(mat_add(mat_mul(x)))"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  std::cout << results->toString(0, results->size()) << std::endl;

}

void MLFunctionsTest::test_torch_dense_layer(){

  int input_size = 10;
  int output_size = 5;
  
  auto col1 = maker.flatVector<float>(input_size); 
  auto col2 = maker.flatVector<float>(input_size); 
  
  for(int i=0; i < input_size; i++){
  	col1->set(i, i*5);
	  col2->set(i, i*2);
  } 
  auto inputRowVector = maker.rowVector({"col1", "col2"}, {col1, col2});

  // step1: Register
  exec::registerVectorFunction(
  "dense_layer",
  TorchDenseLayer::signatures(),
  std::make_unique<TorchDenseLayer>(input_size, output_size));

  auto myPlan = exec::test::PlanBuilder(pool_.get())
                  .values({inputRowVector})
                  .project({"dense_layer(col1)"})
		              .planNode();

  auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
  std::cout << "Results:" << results->toString() << std::endl;
  auto vec = results->childAt(0)->asFlatVector<float>()->values()->asMutable<float>();
}



void MLFunctionsTest::run() {
  //test_mat_mul();
  //test_mat_add();
  //test_relu();
  //test_dense_layer();
  test_torch_dense_layer();
}


int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  MLFunctionsTest demo;
  demo.run();
}