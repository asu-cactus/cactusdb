#include <folly/init/Init.h>
//#include <folly/init/Init.h>
#include <torch/torch.h>
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
#include <vector>
#include <cctype>
#include <climits>
#include <unordered_map>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <json/json.h>
#include "velox/type/Type.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/XGBoost.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/NNBuilder.h"
#include <fstream>
#include <sstream>
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/expression/VectorFunction.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include <ctime>
#include <iomanip>
#include <time.h>
#include <chrono>
#include <locale>
#include <regex>
#include <algorithm>
#include <H5Cpp.h>
#include <Eigen/Dense>

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

/*
 * The structure to describe the context of a neural network model architecture
 */
struct NNModelContext {

    int inputFeatures;

    int numLayers;

    int hiddenLayerNeurons;

    int outputLayerNeurons;

};

/*
 * The structure to describe the context of a decision tree model architecture
 */

struct DTModelContext {

    int inputFeatures;
    
    int treeDepth;

};


/*
 * The structure to describe the push down status of a feature
 */
struct FeatureStatus {
    int isFeature;

    int isPushed;

    int vectorSize;

    int featureStartPos;

};


class VectorAddition : public MLFunction {
 public:
  VectorAddition(int inputDims) {
    inputDims_ = inputDims;
  }

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
    auto baseLeftArray =
        decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = rightHolder->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input1Matrix(input1Values, numInput, inputDims_);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input2Matrix(input2Values, numInput, inputDims_);

    std::vector<std::vector<float>> results;

    for (int i = 0; i < numInput; i++) {
      //Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor> vSum = input1Matrix.row(i) + input2Matrix.row(i);
      Eigen::VectorXf vSum = input1Matrix.row(i) + input2Matrix.row(i);
      std::vector<float> curVec(vSum.data(), vSum.data() + vSum.size());
      //std::vector<float> std_vector(vSum.data(), vSum.data() + vSum.size());
      results.push_back(curVec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("array(REAL)")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "vector_addition";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

 private:
  int inputDims_;
};


class RewriteFactorized : HiveConnectorTestBase {
    public:
        RewriteFactorized() {
            // Register Presto scalar functions.
            functions::prestosql::registerAllScalarFunctions();
            // Register Presto aggregate functions.
            aggregate::prestosql::registerAllAggregateFunctions();
            // Register type resolver with DuckDB SQL parser.
            parse::registerTypeResolver();
            // HiveConnectorTestBase::SetUp();
            //parquet::registerParquetReaderFactory();

            auto hiveConnector =
                connector::getConnectorFactory(
                    connector::hive::HiveConnectorFactory::kHiveConnectorName)
                    ->newConnector(kHiveConnectorId, std::make_shared<core::MemConfig>());
            connector::registerConnector(hiveConnector);

            // SetUp();

        }

        ~RewriteFactorized() {}

	void SetUp() override {
            // TODO: not used for now
            // HiveConnectorTestBase::SetUp();
            // parquet::registerParquetReaderFactory();
        }

        void TearDown() override {
            HiveConnectorTestBase::TearDown();
        }

        void TestBody() override {
        }


        static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {

            while (!task->isFinished()) {

                usleep(1000); // 0.01 second.

            }
        }

        std::shared_ptr<folly::Executor> executor_{
            std::make_shared<folly::CPUThreadPoolExecutor>(
                std::thread::hardware_concurrency())};

        std::shared_ptr<core::QueryCtx> queryCtx_{
            std::make_shared<core::QueryCtx>(executor_.get())};

        std::shared_ptr<memory::MemoryPool> pool_{memory::MemoryManager::getInstance()->addLeafPool()};

        VectorMaker maker{pool_.get()};

        std::unordered_map<std::string, RowVectorPtr> tableName2RowVector;
        std::unordered_map<std::string, std::vector<std::vector<float>>> operatorParam2Weights;
        std::unordered_map<std::string, std::vector<std::string>> tabel2Columns;
        std::unordered_map<std::string, FeatureStatus> allFeatureStatus;
        std::vector<std::string> modelOperators;
        std::map<std::string, int> factorizationPlans;
        std::map<std::string, int> featureStartPos;
        int totalFeatures = 0;



        // Function to check if a string represents a valid integer
        bool isInteger(const std::string& s) {

            if (s.empty() || (s.size() > 1 && s[0] == '0')) {

                return false; // prevent leading zeros for non-zero integers

            }

            for (char c : s) {

                if (!std::isdigit(c)) {

                    return false;
                }
            }

            return true;

        }



        void findFactorizationPlans(std::string filePath) {
            // Open the file
            std::ifstream inputFile(filePath);
            if (!inputFile.is_open()) {
                std::cerr << "Failed to open the file: " << filePath << std::endl;
                //return 1;
            }

            // Map to store key-value pairs
            std::map<std::string, int> myMap;

            // Read the file line by line
            std::string line;
            while (std::getline(inputFile, line)) {
                // Find the separator " = " to split the key and value
                size_t separatorPos = line.find(" = ");
                if (separatorPos != std::string::npos) {
                    // Extract the key and value
                    std::string key = line.substr(0, separatorPos);
                    int value = std::stoi(line.substr(separatorPos + 3)); // Convert value to int

                    // Store in the map
                    myMap[key] = value;
                }
            }

            // Close the file
            inputFile.close();

            factorizationPlans = myMap;
        }



        std::vector<std::string> extractOperatorsInReverse(const std::string& input) {
            std::vector<std::string> operators;
    std::regex operatorPattern(R"(([a-zA-Z_]+\d+)\()"); // Match patterns like mat_mul1(, mat_add2(, etc.
    std::smatch match;

    std::string::const_iterator searchStart(input.cbegin());
    while (std::regex_search(searchStart, input.cend(), match, operatorPattern)) {
        operators.push_back(match[1]); // Extract the operator name
        searchStart = match.suffix().first; // Move the search start position
    }

    // Reverse the order of operators to match actual execution order
    std::reverse(operators.begin(), operators.end());
    return operators;
        }



        std::vector<std::vector<float>> loadHDF5Array(const std::string& filename, const std::string& datasetName, int doPrint) {
            H5::H5File file(filename, H5F_ACC_RDONLY);
            H5::DataSet dataset = file.openDataSet(datasetName);
            H5::DataSpace dataspace = dataset.getSpace();

            // Get the number of dimensions
            int rank = dataspace.getSimpleExtentNdims();
            // std::cout << "Rank: " << rank << std::endl;

            // Allocate space for the dimensions
            std::vector<hsize_t> dims(rank);

            // Get the dataset dimensions
            dataspace.getSimpleExtentDims(dims.data(), nullptr);

            size_t rows;
            size_t cols;

            if (rank == 1) {
                rows = dims[0];
                cols = 1;
            }
            else if (rank == 2) {
                rows = dims[0];
                cols = dims[1];
            } else {
                throw std::runtime_error("Unsupported rank: " + std::to_string(rank));
            }

            // Read data into a 1D vector
            std::vector<float> flatData(rows * cols);
            dataset.read(flatData.data(), H5::PredType::NATIVE_FLOAT);

            // Convert to 2D vector
            std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[i][j] = flatData[i * cols + j];
                    if (doPrint == 1)
                        std::cout << result[i][j] << ", ";
                }
                if (doPrint == 1)
                    std::cout << std::endl;
            }

            // Close the dataset and file
            dataset.close();
            file.close();

            return result;
        }




        void findWeights(std::string modelPath) {
            // read the parameter weights from file
            std::vector<std::vector<float>> w1 = loadHDF5Array(modelPath, "fc1.weight", 0);
            std::vector<std::vector<float>> b1 = loadHDF5Array(modelPath, "fc1.bias", 0);
            std::vector<std::vector<float>> w2 = loadHDF5Array(modelPath, "fc2.weight", 0);
            std::vector<std::vector<float>> b2 = loadHDF5Array(modelPath, "fc2.bias", 0);
            std::vector<std::vector<float>> w3 = loadHDF5Array(modelPath, "fc3.weight", 0);
            std::vector<std::vector<float>> b3 = loadHDF5Array(modelPath, "fc3.bias", 0);

            // store the weights in map with same name as operator
            operatorParam2Weights["mat_mul1"] = w1;
            operatorParam2Weights["mat_add1"] = b1;
            operatorParam2Weights["mat_mul2"] = w2;
            operatorParam2Weights["mat_add2"] = b2;
            operatorParam2Weights["mat_mul3"] = w3;
            operatorParam2Weights["mat_add3"] = b3;

            std::cout << "Shape of mat_mul1 weight: " << w1.size() << ", " << w1[0].size() << std::endl;
        }



        std::vector<std::vector<float>> extractSubweight(const std::vector<std::vector<float>>& matrix, int start, int n) {
            std::vector<std::vector<float>> result;

            std::cout << "Extracting subweight" << std::endl;
            std::cout << start << ", " << n << std::endl;

            // Ensure that the range [start, start + n) is within bounds
            //int end = std::min(start + n, matrix.size());
            int end = start + n;
            for (int i = start; i < end; ++i) {
                result.push_back(matrix[i]);  // Copy rows within the range
            }

            return result;
        }



        RowVectorPtr getTableFromCSVFile(VectorMaker & maker, std::string csvFilePath, std::string tableName, std::string joinKey) {

            std::ifstream file(csvFilePath.c_str());
            if (file.fail()) {

                std::cerr << "Error in reading data file:" << csvFilePath << std::endl;
                exit(1);

            }
            
            std::string line;

            std::cout << tableName << std::endl;

            std::vector<int> joinVec;
            std::vector<std::vector<float>> featuresVec;

            // Ignore the first line (header)
           if (std::getline(file, line)) {
               std::cout << "Ignoring header: " << line << std::endl;
           }

            int count  = 0;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                std::string numberStr;
                std::vector<float> features;
                int colIndex = 0;

                while (std::getline(iss, numberStr, ',')) {
                    if (numberStr.size() >= 2 && numberStr.front() == '"' && numberStr.back() == '"') {
                        numberStr = numberStr.substr(1, numberStr.size() - 2);
                    }

                    if (colIndex == 0) {
                        joinVec.push_back(std::stoi(numberStr));
                    }
                    else {
                        features.push_back(std::stof(numberStr));
                    }
                    colIndex += 1;
                }
                featuresVec.push_back(features);
                count += 1;
            }
            file.close();

            std::string featureCol = "f_" + tableName;
            auto joinVector = maker.flatVector<int>(joinVec);
            auto featuresVector = maker.arrayVector<float>(featuresVec, REAL());
            auto tableRowVector = maker.rowVector(
                {joinKey, featureCol}, {joinVector, featuresVector}
                );

            std::vector<std::string> cols = {joinKey, featureCol};
            tabel2Columns[tableName] = cols;
            totalFeatures += featuresVec[0].size();

            FeatureStatus fs1;
            fs1.isFeature = 0;
            allFeatureStatus[joinKey] = fs1;

            FeatureStatus fs2;
            fs2.isFeature = 1;
            fs2.isPushed = 0;
            fs2.vectorSize = featuresVec[0].size();
            allFeatureStatus[featureCol] = fs2;

            return tableRowVector;
        }




int sampleQuery() {

      return 29;

}

int sampleModel() {

      return 0;

}


void sampleNNModelArch(int numInputFeatures, NNModelContext & nn) {

      nn.inputFeatures = numInputFeatures;
      nn.numLayers = 3;
      nn.hiddenLayerNeurons = 16;
      nn.outputLayerNeurons = 2; 

}

void sampleDTModelArch (int numInputFeatures, DTModelContext & dt) {

      dt.inputFeatures = numInputFeatures;
      dt.treeDepth = 8;

}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}



void registerNNFunction(std::string op_name, std::vector<std::vector<float>> weightMatrice, int dim1, int dim2) {

    if (op_name.find("mat_mul") != std::string::npos) {
        auto nnWeightVector = maker.arrayVector<float>(weightMatrice, REAL());
        exec::registerVectorFunction(
            op_name,
            MatrixMultiply::signatures(),
            std::make_unique<MatrixMultiply>(
                std::move(nnWeightVector->elements()->values()->asMutable<float>()), dim1, dim2)
            );
        std::cout << "Registered a mat_mul function of name " << op_name << " with dimension " << dim1 << ", " << dim2 << endl;
    }

    else if (op_name.find("mat_add") != std::string::npos) {
        auto nnWeightVector = maker.arrayVector<float>(weightMatrice, REAL());
        exec::registerVectorFunction(
            op_name,
            MatrixVectorAddition::signatures(),
            std::make_unique<MatrixVectorAddition>(
                std::move(nnWeightVector->elements()->values()->asMutable<float>()), dim1)
            );
        std::cout << "Registered a mat_add function of name " << op_name << " with dimension " << dim1 << endl;
    }

    else if (op_name.find("relu") != std::string::npos) {
        exec::registerVectorFunction(
            op_name, Relu::signatures(), std::make_unique<Relu>(),
            {},
            true);
        std::cout << "Registered a relu function of name " << op_name << endl;
    }

    else if (op_name.find("softmax") != std::string::npos) {
        exec::registerVectorFunction(
            op_name, Softmax::signatures(), std::make_unique<Softmax>());
        std::cout << "Registered a softmax function of name " << op_name << endl;
    }

    else if (op_name.find("vector_addition") != std::string::npos) {
        exec::registerVectorFunction(
            op_name,
            VectorAddition::signatures(),
            std::make_unique<VectorAddition>(dim1)
            );
        std::cout << "Registered a vector_addition function of name " << op_name << " with dimension " << dim1 << endl;
    }

    
}





float * genWeight(int dim1, int dim2) {

   int total_size = dim1 * dim2;

   //generate weight matrix
   float * weight = new float[total_size];
   for (int i = 0; i < total_size; i++) {
       if (i % 2 == 0) {
           weight[i] = 1.0;
       } else {
           weight[i] = 0.0;
       }
   }
   return weight;
}

bool addModelInferenceToQueryPlanAfterFactorize(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {

        std::string modelProjString = "";
        std::vector<std::vector<float>> emptyMatrix;
        for (int i = modelOperators.size() - 1; i > 0; i--) {
            std::string opName = modelOperators[i];
            if (opName.find("mat_mul") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                int dim2 = param[0].size();
                registerNNFunction(opName, param, dim1, dim2);
            }
            else if (opName.find("mat_add") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                registerNNFunction(opName, param, dim1, -1);
            }
            else {
                registerNNFunction(opName, emptyMatrix, -1, -1);
            }
            modelProjString += opName + "(";
        }
        modelProjString += "features";

        for (int i = modelOperators.size() - 1; i > 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";

        std::cout << "Inference Part: " << modelProjString << endl;
        planBuilder.project({modelProjString});

      return true;

}

bool addModelInferenceToQueryPlan(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {

        std::string modelProjString = "";
        std::vector<std::vector<float>> emptyMatrix;
        for (int i = modelOperators.size() - 1; i >= 0; i--) {
            std::string opName = modelOperators[i];
            if (opName.find("mat_mul") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                int dim2 = param[0].size();
                registerNNFunction(opName, param, dim1, dim2);
            }
            else if (opName.find("mat_add") != std::string::npos) {
                std::vector<std::vector<float>> param = operatorParam2Weights[opName];
                int dim1 = param.size();
                registerNNFunction(opName, param, dim1, -1);
            }
            else {
                registerNNFunction(opName, emptyMatrix, -1, -1);
            }
            modelProjString += opName + "(";
        }
        modelProjString += "features";

        for (int i = modelOperators.size() - 1; i >= 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";

        std::cout << "Inference Part: " << modelProjString << endl;
        planBuilder.project({modelProjString});

      return true;

}


bool rewriteWithFactorization(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, std::string joinOrderStr) {
    
    // Create a JSON reader and root object
    Json::CharReaderBuilder readerBuilder;
    Json::Value root; // Root will hold the parsed JSON array
    std::string errors;

    // Parse the JSON string
    std::istringstream stream(joinOrderStr);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errors)) {
        std::cerr << "Error parsing JSON: " << errors << "\n";
        return 1;
    }
    //JSON Reading successful"

    std::unordered_map<std::string, PlanBuilder> sources; //with filters and projections pushed down;

    std::string outName;
    std::string leftName;
    std::string rightName;

    std::string firstOpName = modelOperators[0]; // name of operator of split layer
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName]; // weight of the split layer
    int fCurrentTotal = 0;
    int numCols = firstWeight.size(); // number of columns in split layer
    int numNeurons = firstWeight[0].size(); // number of neurons in split layer
    int k = 0;
    int addIdx = 0;

    std::vector<std::vector<float>> emptyMatrix;
    bool isAdditionRegistered = false;

    // Iterate through the array
    for (const auto& item : root) {
        std::cout << "ID: " << item["ID"].asString() << "\n";
        std::cout << "Left: " << item["Left"].asString() << "\n";
        std::cout << "Right: " << item["Right"].asString() << "\n";
        std::cout << "Pred: " << item["Pred"].asString() << "\n";
        std::cout << "ProbeKeys: " << item["ProbeKeys"].asString() << "\n";
        std::cout << "BuildKeys: " << item["BuildKeys"].asString() << "\n";

        std::string joinId = item["ID"].asString();
        std::string leftTable = item["Left"].asString();
        std::string rightTable = item["Right"].asString();
        std::string probKeys = item["ProbeKeys"].asString();
        std::string buildKeys = item["BuildKeys"].asString();
        int NumDimLeft = item["NumDimLeft"].asInt();
        int NumDimRight = item["NumDimRight"].asInt();

        std::string leftFactorizationKey = leftTable + "--->" + joinId;
        std::string rightFactorizationKey = rightTable + "--->" + joinId;

        std::cout << "left factorization key: " << leftFactorizationKey << std::endl;
        std::cout << "right factorization key: " << leftFactorizationKey << std::endl;

        bool isLeftTableNotLeaf = isInteger(leftTable);

        if (isLeftTableNotLeaf == false) {
            // left table is leaf

            featureStartPos[leftTable] = fCurrentTotal;
            fCurrentTotal += NumDimLeft;

            std::string fName = tabel2Columns[leftTable][1];

            if (factorizationPlans[leftFactorizationKey] == 1) {
               // Doing factorization of left edge
                std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, featureStartPos[leftTable], NumDimLeft);
                std::string newOpName = firstOpName + "_" + std::to_string(k);
                registerNNFunction(newOpName, subWeight, NumDimLeft, numNeurons);
                std::string fNewName = "factorized_" + std::to_string(k);
                std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                tabel2Columns[leftTable][1] = fNewName;
                k += 1;

                FeatureStatus fs;
                fs.isFeature = 1;
                fs.isPushed = 1;
                fs.vectorSize = numNeurons;
                allFeatureStatus[fNewName] = fs;

                auto leftPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector[leftTable]})
            .project({tabel2Columns[leftTable][0], fNewNameFull});

            sources[leftTable] = leftPlan;
            }
            else {
                // Not Doing factorization of left edge
                FeatureStatus fs = allFeatureStatus[fName];
                fs.featureStartPos = featureStartPos[leftTable];
                allFeatureStatus[fName] = fs;

                auto leftPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector[leftTable]})
            .project({tabel2Columns[leftTable]});

            sources[leftTable] = leftPlan;
            }

        }

        bool isRightTableNotLeaf = isInteger(rightTable);
        if (isRightTableNotLeaf == false) {
            // right table is leaf

            featureStartPos[rightTable] = fCurrentTotal;
            fCurrentTotal += NumDimRight;

            std::string fName = tabel2Columns[rightTable][1];

            if (factorizationPlans[rightFactorizationKey] == 1) {
                // Doing factorization of right edge
                std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, featureStartPos[rightTable], NumDimRight);
                std::string newOpName = firstOpName + "_" + std::to_string(k);
                registerNNFunction(newOpName, subWeight, NumDimRight, numNeurons);
                std::string fNewName = "factorized_" + std::to_string(k);
                std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                tabel2Columns[rightTable][1] = fNewName;
                k += 1;

                FeatureStatus fs;
                fs.isFeature = 1;
                fs.isPushed = 1;
                fs.vectorSize = numNeurons;
                allFeatureStatus[fNewName] = fs;

                auto rightPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector[rightTable]})
            .project({tabel2Columns[rightTable][0], fNewNameFull});

            sources[rightTable] = rightPlan;
            }
            else {
                // Not Doing factorization of right edge
                FeatureStatus fs = allFeatureStatus[fName];
                fs.featureStartPos = featureStartPos[rightTable];
                allFeatureStatus[fName] = fs;

                auto rightPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector[rightTable]})
            .project({tabel2Columns[rightTable]});

            sources[rightTable] = rightPlan;
            }

        }

    featureStartPos[joinId] = featureStartPos[leftTable];


    PlanBuilder left, right, out;
    
    if (sources.count(leftTable) > 0) {

        left = sources[leftTable];

    }

    if (sources.count(rightTable) > 0) {

        right = sources[rightTable];

    }

    //compose join features
    std::vector<std::string> leftProj;
    std::vector<std::string> joinCols;
    int totalPushed = 0;

    if (factorizationPlans[leftFactorizationKey] == 1) {
        // Factorizing all left features that were not pushed earlier
        int newFactorizedCount = 0;
        for (int i = 0; i < tabel2Columns[leftTable].size(); i++) {
                std::string fName = tabel2Columns[leftTable][i];
                if (allFeatureStatus.count(fName) <= 0) {
                   std::cout << "Feature status not found for feature " << fName << std::endl;
                   continue;
                }
                FeatureStatus fs = allFeatureStatus[fName];
                if (fs.isFeature == 1 && fs.isPushed == 0) {

                   // Feature not pushed earlier, pushing now
                    std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, fs.featureStartPos, fs.vectorSize);
                    std::string newOpName = firstOpName + "_" + std::to_string(k);
                    registerNNFunction(newOpName, subWeight, fs.vectorSize, numNeurons);
                    std::string fNewName = "factorized_" + std::to_string(k);
                    std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                    k += 1;

                    FeatureStatus fs;
                    fs.isFeature = true;
                    fs.isPushed = 1;
                    fs.vectorSize = numNeurons;
                    allFeatureStatus[fNewName] = fs;
                    leftProj.push_back(fNewNameFull);
                    joinCols.push_back(fNewName);
                    newFactorizedCount += 1;
                    totalPushed += 1;
                }
                else {
                    // Feature not a feature or pushed earlier
                    leftProj.push_back(fName);
                    joinCols.push_back(fName);
                    if (fs.isFeature == 1) {
                        // Feature pushed earlier
                        totalPushed += 1;
                    }
                }
            }

            if (newFactorizedCount > 0) {
                left = left.project({leftProj});
            }
    }
    else {
        // Just adding all left columns without factorization
        for (int i = 0; i < tabel2Columns[leftTable].size(); i++) {
            std::string fName = tabel2Columns[leftTable][i];
            leftProj.push_back(fName);
            joinCols.push_back(fName);
        }
    }


    std::vector<std::string> rightProj;

    if (factorizationPlans[rightFactorizationKey] == 1) {
        // Factorizing all right features that were not pushed earlier
        int newFactorizedCount = 0;
        for (int i = 0; i < tabel2Columns[rightTable].size(); i++) {
                std::string fName = tabel2Columns[rightTable][i];
                if (allFeatureStatus.count(fName) <= 0) {
                   std::cout << "Feature status not found for feature " << fName << std::endl;
                   continue;
                }
                FeatureStatus fs = allFeatureStatus[fName];
                if (fs.isFeature == 1 && fs.isPushed == 0) {

                   // Feature not pushed earlier, pushing now
                    std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, fs.featureStartPos, fs.vectorSize);
                    std::string newOpName = firstOpName + "_" + std::to_string(k);
                    registerNNFunction(newOpName, subWeight, fs.vectorSize, numNeurons);
                    std::string fNewName = "factorized_" + std::to_string(k);
                    std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                    k += 1;

                    FeatureStatus fs;
                    fs.isFeature = true;
                    fs.isPushed = 1;
                    fs.vectorSize = numNeurons;
                    allFeatureStatus[fNewName] = fs;
                    rightProj.push_back(fNewNameFull);
                    joinCols.push_back(fNewName);
                    newFactorizedCount += 1;
                    totalPushed += 1;
                }
                else {
                    // Feature not a feature or pushed earlier
                    rightProj.push_back(fName);
                    joinCols.push_back(fName);
                    if (fs.isFeature == 1) {
                        // Feature pushed earlier
                        totalPushed += 1;
                    }
                }
            }

            if (newFactorizedCount > 0) {
                right = right.project({rightProj});
            }
    }
    else {
        // Just adding all right columns without factorization
        for (int i = 0; i < tabel2Columns[rightTable].size(); i++) {
            std::string fName = tabel2Columns[rightTable][i];
            rightProj.push_back(fName);
            joinCols.push_back(fName);
        }
    }

    tabel2Columns[joinId] = joinCols;
    // Writing join
    out = left.hashJoin(
        {item["ProbeKeys"].asString()},
        {item["BuildKeys"].asString()},
        right.planNode(),
        "",
            {joinCols}
    );

    if (totalPushed > 1) {
       // some features were pushed before join, they need to be aggregated
        std::vector<std::string> joinColsNew;
        std::vector<std::string> joinProj;

        bool isFirstVec = true;
        std::string projString = "";
        for (int i = 0; i < joinCols.size(); i++) {
            std::string fName = joinCols[i];
            FeatureStatus fs = allFeatureStatus[fName];
            if (fs.isFeature == 0 || fs.isPushed == 0) {
                joinColsNew.push_back(fName);
                joinProj.push_back(fName);
                continue;
            }

            if (isFirstVec == true) {
                projString = fName;
                isFirstVec = false;
                continue;
            }

            std::string newOpName = "vector_addition";
            std::string opName = newOpName + "(" + projString + ", " + fName + ")";
            projString = opName;
            if (isAdditionRegistered == false) {
                registerNNFunction(newOpName, emptyMatrix, numNeurons, -1);
                isAdditionRegistered = true;
            }
        }

        std::string fNewName = "added_vec_" + std::to_string(addIdx);
        addIdx += 1;
        projString += " AS " + fNewName;
        joinColsNew.push_back(fNewName);
        joinProj.push_back(projString);

        FeatureStatus fs;
        fs.isFeature = true;
        fs.isPushed = 1;
        fs.vectorSize = numNeurons;
        allFeatureStatus[fNewName] = fs;

        out = out.project({joinProj});
        tabel2Columns[joinId] = joinColsNew;
    }
    // Finished Writing join

    outName = joinId;
    sources[outName] = out;

    }

    planBuilder = sources[outName];

    std::cout << "After the last join" << std::endl;
    std::vector<std::string> joinProj;
    std::vector<std::string> joinCols;
    int newPushedCount = 0;

    // iterate over the final join projection output to compute not pushed feature
    for (int i = 0; i < tabel2Columns[outName].size(); i++) {
            std::string fName = tabel2Columns[outName][i];
            FeatureStatus fs = allFeatureStatus[fName];
            if (fs.isFeature == 1) {
                if (fs.isPushed == 0) {

                    // found a feature in the final join output which were not pushed
                    std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, fs.featureStartPos, fs.vectorSize);
                    std::string newOpName = firstOpName + "_" + std::to_string(k);
                    std::cout << fName << ", " << fs.featureStartPos << ", " << fs.vectorSize << ", " << subWeight.size() << ", " << subWeight[0].size() << std::endl;
                    registerNNFunction(newOpName, subWeight, fs.vectorSize, numNeurons);
                    std::string fNewName = "factorized_" + std::to_string(k);
                    std::string fNewNameFull = newOpName + "(" + fName + ") AS " + fNewName;
                    k += 1;

                    FeatureStatus fs;
                    fs.isFeature = true;
                    fs.isPushed = 1;
                    fs.vectorSize = numNeurons;
                    allFeatureStatus[fNewName] = fs;
                    joinProj.push_back(fNewNameFull);
                    joinCols.push_back(fNewName);
                    newPushedCount += 1;
                }
                else {
                    joinProj.push_back(fName);
                    joinCols.push_back(fName);
                }
            }
        }

    if (newPushedCount > 0) {
       // final join output projection changed
        planBuilder = planBuilder.project({joinProj});
    }

    // perform aggregation of the final join output
    std::string projString = joinCols[0];
    for (int i = 1; i < joinCols.size(); i++) {
        std::string newOpName = "vector_addition";
        std::string opName = newOpName + "(" + projString + ", " + joinCols[i] + ")";
        projString = opName;
        if (isAdditionRegistered == false) {
            registerNNFunction(newOpName, emptyMatrix, numNeurons, -1);
            isAdditionRegistered = true;
        }
    }
    projString += " AS features";

    planBuilder = planBuilder.project({projString});
    std::cout << "Plan: " << planBuilder.planNode()->toString(true, true) << std::endl;
    return true;

}



bool writeWithoutFactorization(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, std::string joinOrderStr) {

    // Create a JSON reader and root object
    Json::CharReaderBuilder readerBuilder;
    Json::Value root; // Root will hold the parsed JSON array
    std::string errors;

    // Parse the JSON string
    std::istringstream stream(joinOrderStr);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errors)) {
        std::cerr << "Error parsing JSON: " << errors << "\n";
        return 1;
    }

    std::unordered_map<std::string, PlanBuilder> sources; //with filters and projections pushed down;

    std::string outName;
    std::string leftName;
    std::string rightName;
    std::vector<std::string> projections;

    std::string firstOpName = modelOperators[0];
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName];
    int fCurrentTotal = 0;
    int numCols = firstWeight.size();
    int numNeurons = firstWeight[0].size();
    int k = 0;

    // Iterate through the array
    for (const auto& item : root) {
        std::cout << "ID: " << item["ID"].asString() << "\n";
        std::cout << "Left: " << item["Left"].asString() << "\n";
        std::cout << "Right: " << item["Right"].asString() << "\n";
        std::cout << "Pred: " << item["Pred"].asString() << "\n";
        std::cout << "ProbeKeys: " << item["ProbeKeys"].asString() << "\n";
        std::cout << "BuildKeys: " << item["BuildKeys"].asString() << "\n";

        std::string joinId = item["ID"].asString();
        std::string leftTable = item["Left"].asString();
        std::string rightTable = item["Right"].asString();
        std::string probKeys = item["ProbeKeys"].asString();
        std::string buildKeys = item["BuildKeys"].asString();
        int NumDimLeft = item["NumDimLeft"].asInt();
        int NumDimRight = item["NumDimRight"].asInt();

        bool isLeftTableNotLeaf = isInteger(leftTable);
        if (isLeftTableNotLeaf == false) {
            // left table is leaf
            fCurrentTotal += NumDimLeft;

            auto leftPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector[leftTable]})
            .project({tabel2Columns[leftTable]});

            sources[leftTable] = leftPlan;

        }

        bool isRightTableNotLeaf = isInteger(rightTable);
        if (isRightTableNotLeaf == false) {
            // right table is leaf
            fCurrentTotal += NumDimRight;

            auto rightPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector[rightTable]})
            .project({tabel2Columns[rightTable]});

            sources[rightTable] = rightPlan;

        }


    PlanBuilder left, right, out;

    //retrieve the corresponding PlanBuilder
    if (sources.count(leftTable) > 0) {

        left = sources[leftTable];

    }

    if (sources.count(rightTable) > 0) {

        right = sources[rightTable];

    }

    //compose join
    projections.clear();
    // Access Projection if it exists
    if (item.isMember("Projection")) {
        for (const auto& proj : item["Projection"]) {
		     std::cout << proj << std::endl;
             projections.push_back(proj.asString());
	    }
    }

    std::cout << "Writing join" << std::endl;
    out = left.hashJoin(
        {item["ProbeKeys"].asString()},
        {item["BuildKeys"].asString()},
        right.planNode(),
        "",
            {projections}
    );
    std::cout << "Finished Writing join" << std::endl;

    outName = joinId;
    sources[outName] = out;

    }
    planBuilder = sources[outName];

    // After the final join
    std::string projString="";
    bool isFirst = true;
    for (std::string proj : projections) {
       FeatureStatus fs = allFeatureStatus[proj];
       if (fs.isFeature == 1) {
           if (isFirst) {
               projString += proj;
	           isFirst = false;
           } else {
               projString += "," + proj;
           }
       }
    }
    projString = "concat(" + projString + ") as features";

    planBuilder = planBuilder.project({projString});
    std::cout << "Plan: " << planBuilder.planNode()->toString(true, true) << std::endl;
    return true;

}



bool createAndExecuteQuery(std::string queryJsonStr, bool withFactorization) {
      PlanBuilder planBuilder{pool_.get()};
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
      if (withFactorization) {
        // Rewriting with factorization
        rewriteWithFactorization(planBuilder, planNodeIdGenerator, queryJsonStr);

        // Step 2. Add model inference to query plan
        addModelInferenceToQueryPlanAfterFactorize(planBuilder, planNodeIdGenerator);
      }
      else {
          // Writing without factorization
          writeWithoutFactorization(planBuilder, planNodeIdGenerator, queryJsonStr);

          // Step 2. Add model inference to query plan
          addModelInferenceToQueryPlan(planBuilder, planNodeIdGenerator);
      }

      //std::cout << "Plan: " << planBuilder.planNode()->toString(true, true) << std::endl;
      auto myPlan = planBuilder.planNode();
      std::cout << myPlan->toString(true, true) << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
      std::cout << "Results Size: " << results->size() << std::endl;
      std::cout << "Results:" << results->toString(0, 5) << std::endl;
      //std::cout << results->toString(0, results->size()) << std::endl;
      return true;

}


};

int main(int argc, char** argv) {

      setlocale(LC_TIME, "C");

      folly::init(&argc, &argv, false);
      memory::MemoryManager::initialize({});

      RewriteFactorized rewriteObj;

      // Load table1
      std::cout << "Reading Table 1 CSV file" << std::endl;
      RowVectorPtr table1Vec = rewriteObj.getTableFromCSVFile(rewriteObj.maker, "resources/data/synthetic_dataset_4_3/table_1.csv", "table_1", "join_key1");

      std::cout << "Reading Table 2 CSV file" << std::endl;
      // Load table2
      RowVectorPtr table2Vec = rewriteObj.getTableFromCSVFile(rewriteObj.maker, "resources/data/synthetic_dataset_4_3/table_2.csv", "table_2", "join_key2");

      // Load table3
      std::cout << "Reading Table 3 CSV file" << std::endl;
      RowVectorPtr table3Vec = rewriteObj.getTableFromCSVFile(rewriteObj.maker, "resources/data/synthetic_dataset_4_3/table_3.csv", "table_3", "join_key3");

      // Load table4
      std::cout << "Reading Table 4 CSV file" << std::endl;
      RowVectorPtr table4Vec = rewriteObj.getTableFromCSVFile(rewriteObj.maker, "resources/data/synthetic_dataset_4_3/table_4.csv", "table_4", "join_key4");


      rewriteObj.tableName2RowVector["table_1"] = table1Vec;
      rewriteObj.tableName2RowVector["table_2"] = table2Vec;
      rewriteObj.tableName2RowVector["table_3"] = table3Vec;
      rewriteObj.tableName2RowVector["table_4"] = table4Vec;


      // retrieve the weights and set to map
      std::cout << "Reading model parameters" << std::endl;
      rewriteObj.findWeights("resources/model/dummy.h5");

      // retrieve the model operators from model expression IR
      std::string modelInput = "softmax3(mat_add3(mat_mul3(relu2(mat_add2(mat_mul2(relu1(mat_add1(mat_mul1(features)))))))))";
      std::cout << "Extracting model operators" << std::endl;
      std::vector<std::string> operators = rewriteObj.extractOperatorsInReverse(modelInput);
      rewriteObj.modelOperators = operators;

      std::string jsonString = R"([
          {"ID":"0","Left":"table_1","Right":"table_2","Pred":"table_1.join_key1 = table_2.join_key2","ProbeKeys":"join_key1","BuildKeys":"join_key2","Projection":["join_key1","f_table_1","join_key2","f_table_2"],"NumTuplesLeft":1000,"NumDimLeft":20,"NumTuplesRight":300,"NumDimRight":39,"NumTuplesOutput":300,"NumDimOutput":59},
          {"ID":"1","Left":"0","Right":"table_3","Pred":"table_2.join_key2 = table_3.join_key3","ProbeKeys":"join_key2","BuildKeys":"join_key3","Projection":["join_key1","f_table_1","join_key2","f_table_2","join_key3","f_table_3"],"NumTuplesLeft":300,"NumDimLeft":59,"NumTuplesRight":12000,"NumDimRight":31,"NumTuplesOutput":12000,"NumDimOutput":90}
      ])";

      std::cout << "Reading factorization Plan" << std::endl;
      rewriteObj.findFactorizationPlans("resources/plans/factorization_plan4_3.txt");

      std::cout << "Performing rewriting" << std::endl;
      bool ret = rewriteObj.createAndExecuteQuery(jsonString, true);

      return 1;
}

