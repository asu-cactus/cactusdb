#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/ml_functions/functions.h"
#include "velox/vector/tests/utils/VectorTestBase.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/type/OpaqueCustomTypes.h"

using namespace std;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::memory;

namespace ml{

class TreeType : public OpaqueType {

    TreeType() : OpaqueType(std::type_index(typeid(ml::Tree))) {
    
        std::cout << "typeid(ml::Tree)=" << typeid(ml::Tree).name() << std::endl;
    
    }

public:

    static const std::shared_ptr<const TreeType>& get() {

	static const std::shared_ptr<const TreeType> instance{
        
	    new TreeType()
	
	};

        return instance;

    }

    std::string toString() const override {
    
        return name();
    
    }

    const char * name() const override {
    
        return "tree_type";
    
    }


    
};


struct TreeT {
  using type = std::shared_ptr<Tree>;

  static constexpr const char* typeName = "tree_type";
  
};
using TheTree = CustomType<TreeT>;


class TreeTypeFactories : public CustomTypeFactories {

public:
  
  TypePtr getType() const override {

        return TreeType::get();

  }

  exec::CastOperatorPtr getCastOperator() const override {

    VELOX_UNSUPPORTED();

  }

};

class AlwaysFailingTypeFactories : public CustomTypeFactories {
 public:
  TypePtr getType() const override {
    VELOX_UNSUPPORTED();
  }

  exec::CastOperatorPtr getCastOperator() const override {
    VELOX_UNSUPPORTED();
  }
};

class VeloxTreeConstruction : public exec::VectorFunction {

public:

    VeloxTreeConstruction() {}

    void apply (

      const SelectivityVector& rows,

      std::vector<VectorPtr>& args,

      const TypePtr& type,

      exec::EvalCtx& context,

      VectorPtr& output) const override {


	   auto flatInput = args[0]->as<SimpleVector<StringView>>();
    
           BaseVector::ensureWritable(rows, type, context.pool(), output);

	   auto flatResult = output->asFlatVector<std::shared_ptr<void>>();

	   rows.applyToSelected([&] (auto row) {
	
	       flatResult->set(row, std::make_shared<Tree>(row, flatInput->valueAt(row)));		   

               std:cout << "tree-" << row << " is built" << std::endl;

           });

    }

    static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
           return {exec::FunctionSignatureBuilder()
                .argumentType("VARCHAR")
                .returnType("tree_type")
                .build()};
  }

  static std::string getName() {
    return "velox_tree_construct";
  }

};

class VeloxTreePrediction : public exec::VectorFunction {

public:

  VeloxTreePrediction(int numFeatures) {
  
      this->numFeatures = numFeatures;
  
  }

  void apply(

      const SelectivityVector& rows,

      std::vector<VectorPtr>& args,

      const TypePtr& type,

      exec::EvalCtx& context,

      VectorPtr& output) const override {

    BaseVector::ensureWritable(rows, type, context.pool(), output);

    BaseVector* left = args[0].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);

    auto decodedLeftArray = leftHolder.get();
    
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();


    auto flatInput = args[1]->as<SimpleVector<std::shared_ptr<void>>>();


    auto flatResult = output->asFlatVector<float>();

    rows.applyToSelected([&](auto row) {

      flatResult->set(

          row, std::static_pointer_cast<Tree>(flatInput->valueAt(row))->predictSingle(input1Values, row * numFeatures)
	  
      );
    });

  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("tree_type")
                .returnType("REAL")
                .build()};
  }

  static std::string getName() {
    return "velox_tree_predict";
  }

  int numFeatures;

};

template <typename T>
struct VeloxTreePredictionSimpleFunction {

  VELOX_DEFINE_FUNCTION_TYPES(T);

  void call(
      out_type<float>& result,
      const arg_type<Array<float>>& a,
      const arg_type<TheTree>& b) {
          result = 0.0;
      }
};

}
