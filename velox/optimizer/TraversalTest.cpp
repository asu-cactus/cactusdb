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
#include <folly/init/Init.h>
#include <random>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include "velox/common/file/FileSystems.h"
#include "velox/core/PlanNode.h"
#include "velox/core/Expressions.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/DecisionForest.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/core/ITypedExpr.h"

using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;


class TraversalTest : public HiveConnectorTestBase{

public:

    TraversalTest() {

	// Register Presto scalar functions.    
        functions::prestosql::registerAllScalarFunctions();

        // Register Presto aggregate functions.
        aggregate::prestosql::registerAllAggregateFunctions();

        // Register type resolver with DuckDB SQL parser.
        parse::registerTypeResolver();

    }

    ~TraversalTest() {

	TearDown();

    }
    
    void SetUp() override {
    }

    void TearDown() override {
        HiveConnectorTestBase::TearDown();
    }

    void TestBody() override {
    }

    void registerFunctions() {
    
        std::cout <<"To register function for TreePrediction" << std::endl;

        exec::registerVectorFunction(
            "decision_tree_predict",
            TreePrediction::signatures(),
            std::make_unique<TreePrediction>(0, "resources/model/fraud_xgboost_10_8/0.txt", 28, false));

        std::cout << "To register type for Tree" << std::endl;

        registerCustomType(
            "tree_type", std::make_unique<TreeTypeFactories>());


        std::cout << "To register function for VeloxTreePrediction" << std::endl;

        exec::registerVectorFunction(
            "velox_decision_tree_predict",
            VeloxTreePrediction::signatures(),
            std::make_unique<VeloxTreePrediction>(28));

        std::cout << "To register function for VeloxTreeConstruction" << std::endl;

        exec::registerVectorFunction(
            "velox_decision_tree_construct",
            VeloxTreeConstruction::signatures(),
            std::make_unique<VeloxTreeConstruction>());

        std::cout << "To register function for ForestPrediction" << std::endl;

        exec::registerVectorFunction(
            "decision_forest_predict",
            TreePrediction::signatures(),
            std::make_unique<ForestPrediction>("resources/model/fraud_xgboost_10_8", 28, true));

    }

    ArrayVectorPtr parseCSVFile(VectorMaker & maker, std::string filePath, int numRows, int numCols) {
    
        int size = numRows * numCols;

        std::cout << "Loading tensor of size " << size << " from " << filePath << std::endl;

        std::ifstream file(filePath.c_str());

        std::vector<std::vector<float>> inputArrayVector;

    
        int index = 0;
    
        std::string line;
    
        while (numRows--) { // Read a line from the file

            std::vector<float> curRow(numCols);
	
            std::getline(file, line);

            std::istringstream iss(line); // Create an input string stream from the line

            std::string numberStr;

	    int colIndex = 0;

            while (std::getline(iss, numberStr, ',')) { // Read each number separated by comma
						    //
                float number = std::stof(numberStr);    // Convert the string to float

	        if (colIndex < numCols)					    

                     curRow[colIndex] = number;

	        colIndex ++;

            }

	    inputArrayVector.push_back(curRow);
        }

        file.close();

        ArrayVectorPtr tensor = maker.arrayVector<float>(inputArrayVector);
    
        return tensor;
    
    }


    RowVectorPtr loadData (std::string & path, int numRows, int numCols) {
    
	 ArrayVectorPtr inputArrayVector = parseCSVFile(maker, path, numRows, numCols);

	 std::vector<int32_t> indexVector;

         for (int i = 0; i < numRows; i++) {

            indexVector.push_back(i);

         }

	 auto inputIndexVector = maker.flatVector<int32_t>(indexVector);

         return  maker.rowVector({"row_id", "x"}, {inputIndexVector, inputArrayVector});
    
    }

    void testTraverseDecisionForestUDFPlan() {
    
        registerFunctions();

        int numRows = 56962;

        int numCols = 28;

	std::string dataFilePath = "resources/data/creditcard_test.csv";

        auto inputRowVector = loadData(dataFilePath, numRows, numCols);

        auto myPlan = exec::test::PlanBuilder(pool_.get())
             .values({inputRowVector})
             .project({"decision_forest_predict(x)"})
             .planNode();

	traversePlan(myPlan);
    
    }

    void testTraverseDecisionForestRelationPlan() {

	registerFunctions();

        int numRows = 56962;

	int numCols = 28;

        std::string dataFilePath = "resources/data/creditcard_test.csv";

	auto inputRowVector = loadData(dataFilePath, numRows, numCols);

	std::vector<std::string> pathVectors;

  std::string forestFolderPath = "resources/model/fraud_xgboost_10_8";

        Forest::vectorizeForestFolder(forestFolderPath, pathVectors);

        int numTrees = pathVectors.size();

        auto model = maker.flatVector<StringView> (pathVectors.size());

        for (int i = 0; i < numTrees; i++) {

             model->set(i, StringView(pathVectors[i].c_str()));

        }

        auto treeIndexVector = maker.flatVector<int16_t>(numTrees);

        for (int i = 0; i < numTrees; i++) {

            treeIndexVector->set(i, i);

        }

        auto treeRowVector = maker.rowVector({"tree_id", "tree_path"}, {treeIndexVector, model});

        auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

        PlanNodeId p0;

        auto myPlanFragment = exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                  .tableScan(asRowType(inputRowVector->type()))
                  .capturePlanNodeId(p0)
                  .nestedLoopJoin(
                      exec::test::PlanBuilder(planNodeIdGenerator, pool_.get())
                          .values({treeRowVector})
                          .project({"tree_id as tree_id", "velox_decision_tree_construct(tree_path) as tree"})
                          .planNode(), {"row_id", "x", "tree_id", "tree"})
                   .project({"row_id as row_id", "tree_id as tree_id", "velox_decision_tree_predict(x, tree) as prediction"})
                   .aggregation({"row_id"},
                                {"sum(prediction) as sum"},
                                {},
                                core::AggregationNode::Step::kPartial,
                                false)
                   .project({"row_id as row_id", "if (sum > 0.0, 1.0, 0.0)"})
                   .planFragment();

	 std::shared_ptr<const core::PlanNode> root = myPlanFragment.planNode;
	 traversePlan(root);


    }

    void handleValuesNode (std::shared_ptr<const core::PlanNode> & curNode) {

	 std::cout << "To handle a ValuesNode instance" << std::endl;
    
         std::shared_ptr<const ValuesNode> myValuesNode = std::dynamic_pointer_cast<const ValuesNode> (curNode);

	 for (auto value : myValuesNode->values()) {
	 
	     std::cout << value->retainedSize() << ";" << value->estimateFlatSize() << std::endl;
	 
	 }
    
    }


    void handleDecisionForestPredict(std::string callName) {
    
	 std::cout << "UDF for Decision Forest Prediction: " << callName << std::endl;

	 core::QueryConfig config({});

         std::shared_ptr<VectorFunction> myUDF = getVectorFunction(callName, {ARRAY(REAL())}, {}, config);

         if (myUDF) {

              std::shared_ptr<ForestPrediction> myMLUDF = std::dynamic_pointer_cast<ForestPrediction>(myUDF);

              if (myMLUDF) {

		  std::cout << "Num Input Features: " << myMLUDF->getNumFeatures() << std::endl;

              } else {
	      
	          std::cout << "Could not obtain the ForestPrediction instance" << std::endl;
	      
	      }

          } else {

              std::cout << "Could not obtain the VectorFunction instance" << std::endl;

          }
    
    }

    void handleDeisionTreeCrossProduct(std::string callName) {
    
         std::cout << "UDF for Decision Forest Prediction: " << callName << std::endl;

         core::QueryConfig config({});

         std::shared_ptr<VectorFunction> myUDF = getVectorFunction(callName, {ARRAY(REAL()), OPAQUE<ml::Tree>()}, {}, config);

         if (myUDF) {

             std::shared_ptr<MLFunction> myMLUDF = std::dynamic_pointer_cast<MLFunction>(myUDF);

             if (myMLUDF) {

                    std::cout << "Meet an ML function:" << std::endl;

                    std::cout << "Num Dimensions:" << myMLUDF->getNumDims() << std::endl;

             }

         } else {
             
	     std::cout << "Could not obtain the VectorFunction instance" << std::endl;

         } 
    
    }


    void handleExpressions(const std::vector<core::TypedExprPtr> & expressions) {

	 std::cout << "There are " << expressions.size() << " expressions." << std::endl;
    
         for (auto expression : expressions) {
	 
              std::cout << expression->toString() << std::endl;

	      if (auto call = std::dynamic_pointer_cast<const core::CallTypedExpr>(expression)) {

    		  std::string callName = call->name();

		  std::cout << "Analyzing user supplied function: " << callName << std::endl;

	          return vectorFunctionFactories().withRLock([&](auto& functionMap) {
                      auto it = functionMap.find(callName);
		   
		      if (it != functionMap.end()) {
		      
		           //extract signatures
			   std::vector<FunctionSignaturePtr> signatures = it->second.signatures;

			   for (auto signature : signatures) {
			   
			       //return type
			       TypeSignature returnType = signature->returnType();

			       std::cout << returnType.toString() << std::endl;
			       
			       //argument types
			       std::vector<TypeSignature> arguments = signature->argumentTypes();

			       for (auto argument : arguments) {
			       
				   std::cout << argument.toString() << std::endl;
			       
			       }
			   
			   }

			   /*We can use the following statement to extract function factory to 
			    *reconstruct an instance from the corresponding VectorFunction type:
			    *VectorFunctionFactory functionFac = it->second.factory;
			    *However, it will be better to use the following function:
			    *std::shared_ptr<VectorFunction> getVectorFunction(
    				const std::string& name,
    				const std::vector<TypePtr>& inputTypes,
    				const std::vector<VectorPtr>& constantInputs,
    				const core::QueryConfig& config)
			    */
			   
			   if (callName.find("decision_forest_predict")!=std::string::npos) {
			   
                                 handleDecisionForestPredict(callName);				   
			   
			   } else if (callName.find("velox_decision_tree_predict")!=std::string::npos) {
			   
                                 handleDeisionTreeCrossProduct(callName);				   
			   
			   } 
			   

                           //TODO
			   //Invoke the factory to generate a class

			   //extract metadata
			   VectorFunctionMetadata metadata = it->second.metadata;

			   std::cout << metadata.supportsFlattening << std::endl;
		      
		      }    
				  
		});
	      
	      }
	 
	 }
    
    }


    void handleExpression(const core::TypedExprPtr expression) {

	 if (!expression) return;

         std::cout << expression->type()->toString() << std::endl;

         std::cout << expression->toString() << std::endl;

         std::vector<TypedExprPtr> expressions = expression->inputs();

         handleExpressions(expressions);

    }


    void handleFilterNode (std::shared_ptr<const core::PlanNode> & curNode) {

	 std::cout << "To handle a FilterNode instance" << std::endl;
    
         std::shared_ptr<const FilterNode> myFilterNode = std::dynamic_pointer_cast<const FilterNode> (curNode);
    
	 const TypedExprPtr & filterExpr = myFilterNode->filter();
         
	 handleExpression(filterExpr);

    }

    void handleProjectNode (std::shared_ptr<const core::PlanNode> & curNode) {

         std::cout << "To handle a ProjectNode instance" << std::endl;

         std::shared_ptr<const ProjectNode> myProjectNode = std::dynamic_pointer_cast<const ProjectNode> (curNode);

	    const std::vector<TypedExprPtr> & projections = myProjectNode->projections();

         handleExpressions(projections);

    }

    void handleTableScanNode (std::shared_ptr<const core::PlanNode> & curNode) {

	 std::cout << "To handle a TableScanNode instance" << std::endl;

	 std::shared_ptr<const TableScanNode> myTableScanNode = std::dynamic_pointer_cast<const TableScanNode> (curNode);

	 std::shared_ptr<connector::ConnectorTableHandle> tableHandle = myTableScanNode->tableHandle();

	 std::cout << tableHandle->toString() << std::endl;

    }

    void handleFieldAccessTypedExpressions(const std::vector<FieldAccessTypedExprPtr> & expressions) {
    
	 for (auto expression : expressions) {
	 
		 std::cout << expression->name() << std::endl;

		 std::cout << expression->toString() << std::endl;
	 
	 }
    
    }


    void handleAggregationNode (std::shared_ptr<const core::PlanNode> & curNode) {

         std::cout << "To handle an AggregationNode instance" << std::endl;

         std::shared_ptr<const AggregationNode> myAggregationNode = std::dynamic_pointer_cast<const AggregationNode> (curNode);

         std::vector<FieldAccessTypedExprPtr> groupingKeys = myAggregationNode->groupingKeys();

	 std::cout << "There are " << groupingKeys.size() << " grouping keys" << std::endl;

	 handleFieldAccessTypedExpressions(groupingKeys);

	 std::vector<std::string> aggregateNames = myAggregationNode->aggregateNames();

	 for (auto aggregateName : aggregateNames) {
	 
	      std::cout << aggregateName << std::endl;

	 }
    }

    void handleAbstractJoinNode (std::shared_ptr<const core::PlanNode> & curNode) {

	 std::cout << "To handle an AbstractJoinNode instance" << std::endl;

	 std::shared_ptr<const AbstractJoinNode> myJoinNode = std::dynamic_pointer_cast<const AbstractJoinNode> (curNode);

	 const std::vector<FieldAccessTypedExprPtr>& leftKeys = myJoinNode->leftKeys();

	 handleFieldAccessTypedExpressions(leftKeys);

	 const std::vector<FieldAccessTypedExprPtr>& rightKeys = myJoinNode->leftKeys();

	 handleFieldAccessTypedExpressions(rightKeys);

	 const TypedExprPtr filter = myJoinNode->filter();

         handleExpression (filter);

    }

    void handleHashJoinNode (std::shared_ptr<const core::PlanNode> & curNode) {

         std::cout << "To handle a HashJoinNode instance" << std::endl;

	 handleAbstractJoinNode(curNode);

    }

    void handleMergeJoinNode (std::shared_ptr<const core::PlanNode> & curNode) {


         std::cout << "To handle a MergeJoinNode instance" << std::endl;
         
         handleAbstractJoinNode(curNode);

    }

    void handleNestedLoopJoinNode (std::shared_ptr<const core::PlanNode> & curNode) {

	  std::cout << "To handle a NestedLoopJoinNode instance" << std::endl;

	  std::shared_ptr<const NestedLoopJoinNode> myNestedLoopJoinNode = std::dynamic_pointer_cast<const NestedLoopJoinNode> (curNode);

	  TypedExprPtr joinCondition = myNestedLoopJoinNode->joinCondition();

	  handleExpression(joinCondition);

    }

    enum PlanNodeType {
            Values,
        ArrowStream,
        Filter,
        Project,
            TableScan,
            Aggregation,
        TableWrite,
            TableWriteMerge,
        Exchange,
            MergeExchange,
            LocalMerge,
            LocalPartition,
        PartitionedOutput,
        HashJoin,
        MergeJoin,
            NestedLoopJoin,
            OrderBy,
        TopN,
        Limit,
        Unnest,
        EnforceSingleRow,
        AssignUniqueId,
        Window,
        RowNumber,
        MarkDistinct,
        TopNRowNumber
    };

    PlanNodeType hashPlanNode(std::string_view name) {
        if (name == "Values") return Values;
	if (name == "ArrowStream") return ArrowStream;
        if (name == "Filter") return Filter;
        if (name == "Project") return Project;
        if (name == "TableScan") return TableScan;
        if (name == "Aggregation") return Aggregation;
        if (name == "TableWrite") return TableWrite;
        if (name == "TableWriteMerge") return TableWriteMerge;
        if (name == "Exchange") return Exchange;
	if (name == "MergeExchange") return  MergeExchange;
        if (name == "LocalMerge") return LocalMerge;
	if (name == "LocalPartition") return LocalPartition;
	if (name == "PartitionedOutput") return PartitionedOutput;
	if (name == "HashJoin") return HashJoin;
	if (name == "MergeJoin") return MergeJoin;
	if (name == "NestedLoopJoin") return NestedLoopJoin;
	if (name == "OrderBy") return OrderBy;
	if (name == "TopN") return TopN;
	if (name == "Limit") return Limit;
	if (name == "Unnest") return Unnest;
	if (name == "EnforceSingleRow") return EnforceSingleRow;
	if (name == "AssignUniqueId") return AssignUniqueId;
	if (name == "Window") return Window;
	if (name == "RowNumber") return RowNumber;
	if (name == "MarkDistinct") return MarkDistinct;
	if (name == "TopNRowNumber") return TopNRowNumber;
    }

    inline void traversePlan(std::shared_ptr<const core::PlanNode> & root) {
    
	if (root) {	
	
	   std::shared_ptr<const core::PlanNode> curNode = root;

	   if (!curNode) return;

	   std::cout << curNode->id() << ":" << curNode->outputType()->toString() << std::endl;
        
	   switch (hashPlanNode(curNode->name())) {
	
	        case (Values):

                    handleValuesNode(curNode);

	            break;

	        case (ArrowStream):

		    break;

                case (Filter):

                    handleFilterNode(curNode);

		    break;

                case (Project):

		    handleProjectNode(curNode);

		    break;

	        case (TableScan):	

		    handleTableScanNode(curNode);

	            break;

	        case (Aggregation):

                    handleAggregationNode(curNode);      

	            break;

	        case (TableWrite):

	            break;

	        case (TableWriteMerge):

	            break;

	        case (Exchange):

	            break;

	        case (MergeExchange):

	            break;

	        case (LocalMerge):

	            break;

	        case (LocalPartition):

	            break;

	        case (PartitionedOutput):

	            break;

	        case (HashJoin):

		    handleHashJoinNode(curNode);

	            break;

	        case (MergeJoin):

		    handleMergeJoinNode(curNode);

	            break;

                case (NestedLoopJoin):

		    handleNestedLoopJoinNode(curNode);

		    break;

	        case (OrderBy):

	            break;

	        case (TopN):

	            break;

	        case (Limit):

	            break;

	        case (Unnest):

	            break;

	        case (EnforceSingleRow):

	            break;

	        case (AssignUniqueId):	

		    break;

                case (Window):

		    break;

	        case (RowNumber):

		    break;

                case (MarkDistinct):

		    break;

	        case (TopNRowNumber):

		    break;

	        default:

		    break; 
	   }

	   std::vector<std::shared_ptr<const PlanNode>> sources = curNode->sources();

	   if (sources.size() == 0) {
	   
	       return;
	   
	   } else {
	   
	       for (auto source : sources) {
	       
	           traversePlan(source);
	       
	       }
	   
	   }
	
	} 

    }


private:

    std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

    VectorMaker maker{pool_.get()}; 

};

int main(int argc, char** argv) {
  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  TraversalTest demo;

  std::cout << "================= UDF-Centric Decision Forest ===================" << std::endl << std::endl;

  demo.testTraverseDecisionForestUDFPlan();

  std::cout << "================= Relation-Centric Decision Forest ===================" << std::endl << std::endl;

  demo.testTraverseDecisionForestRelationPlan();
}

