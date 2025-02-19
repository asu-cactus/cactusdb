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
#include <malloc.h>

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
    dims.push_back(inputDims);
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    /*auto input_elements1 = args[0]->as<ArrayVector>()->elements();
    float* input1Values = input_elements1->values()->asMutable<float>();

    auto input_elements2 = args[1]->as<ArrayVector>()->elements();
    float* input2Values = input_elements2->values()->asMutable<float>();*/

    BaseVector* left = args[0].get();
    BaseVector* right = args[1].get();

    exec::LocalDecodedVector leftHolder(context, *left, rows);
    auto decodedLeftArray = leftHolder.get();
    auto baseLeftArray = decodedLeftArray->base()->as<ArrayVector>()->elements();

    exec::LocalDecodedVector rightHolder(context, *right, rows);
    auto decodedRightArray = rightHolder.get();
    auto baseRightArray = decodedRightArray->base()->as<ArrayVector>()->elements();

    float* input1Values = baseLeftArray->values()->asMutable<float>();
    float* input2Values = baseRightArray->values()->asMutable<float>();

    int numInput = rows.size();

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input1Matrix(input1Values, numInput, dims[0]);
    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        input2Matrix(input2Values, numInput, dims[0]);

    std::vector<std::vector<float>> results;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> sumMat  =  input1Matrix + input2Matrix;
    for (int i = 0; i < numInput; i++) {
        std::vector<float> curVec(
            sumMat.row(i).data(),
            sumMat.row(i).data() + sumMat.cols());
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

};




class GetFeatureVec : public MLFunction {
 public:

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& type,
      exec::EvalCtx& context,
      VectorPtr& output) const override {
    BaseVector::ensureWritable(rows, type, context.pool(), output);

    int64_t vecSizeLarge = 0;
    if (args.size() == 2) {
        // an optional parameter can be passed to enable the GPU for mat_mul
        vecSizeLarge = args[1]->as<ConstantVector<int64_t>>()->valueAt(0);
    }
    int vecSize = static_cast<int>(vecSizeLarge);

    std::vector<std::vector<float>> results;

    for (int i = 0; i < rows.size(); i++) {
        std::vector<float> vec;

        for (int j = 0; j < vecSize; j++) {
            if (j % 2 == 0)
                vec.push_back(1.0);
            else
                vec.push_back(0.0);
        }
        results.push_back(vec);
    }

    VectorMaker maker{context.pool()};
    output = maker.arrayVector<float>(results, REAL());
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    return {exec::FunctionSignatureBuilder()
                .argumentType("array(REAL)")
                .argumentType("BIGINT")
                .returnType("array(REAL)")
                .build()};
  }

  static std::string getName() {
    return "get_feature_vec";
  };

  float* getTensor() const override {
    // TODO: need to implement
    return nullptr;
  }

  CostEstimate getCost(std::vector<int> inputDims) {
    // TODO: need to implement
    return CostEstimate(0, inputDims[0], inputDims[1]);
  }

};



class Job2WayJoin : HiveConnectorTestBase {
 public:
  Job2WayJoin() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();
    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();
    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~Job2WayJoin() {}

  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

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

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};

  std::unordered_map<std::string, RowVectorPtr> tableName2RowVector;
  std::unordered_map<std::string, std::vector<std::vector<float>>> operatorParam2Weights;
  std::unordered_map<std::string, std::vector<std::string>> tabel2Columns;
  std::vector<std::string> modelOperators;

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


  int getStringIndex(const std::vector<std::string>& strVec, const std::string& target) {
    auto it = std::find(strVec.begin(), strVec.end(), target);

    if (it != strVec.end()) {
        // Calculate the index
        int index = std::distance(strVec.begin(), it);
        return index;
    } else {
        return -1;
    }

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



  void findWeights(const std::string& modelPath) {
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
            std::cout << "Shape of mat_add1 weight: " << b1.size() << std::endl;
            std::cout << "Shape of mat_mul2 weight: " << w2.size() << ", " << w2[0].size() << std::endl;
            std::cout << "Shape of mat_add2 weight: " << b2.size() << std::endl;
            std::cout << "Shape of mat_mul3 weight: " << w3.size() << ", " << w3[0].size() << std::endl;
            std::cout << "Shape of mat_add3 weight: " << b3.size() << std::endl;
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


  RowVectorPtr getTableFromCSVFile(
      VectorMaker& maker,
      std::string csvFilePath,
      std::string tableName,
      int k) {
    std::ifstream file(csvFilePath.c_str());
    if (file.fail()) {
      std::cerr << "Error in reading data file:" << csvFilePath << std::endl;
      exit(1);
    }

    std::cout << tableName << std::endl;

    std::unordered_map<std::string, std::vector<string>> colName2colHeader;

    std::unordered_map<std::string, std::vector<int>> colName2colType;

    colName2colType["aka_name"] = {0, 0, 1, 1, 1, 1, 1, 1};
    colName2colType["aka_title"] = {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1};
    colName2colType["cast_info"] = {0, 0, 0, 0, 1, 0, 0};
    colName2colType["char_name"] = {0, 1, 1, 0, 1, 1, 1};
    colName2colType["comp_cast_type"] = {0, 1};
    colName2colType["company_name"] = {0, 1, 1, 0, 1, 1, 1};
    colName2colType["company_type"] = {0, 1};
    colName2colType["complete_cast"] = {0, 0, 0, 0};
    colName2colType["info_type"] = {0, 1};
    colName2colType["keyword"] = {0, 1, 1};
    colName2colType["kind_type"] = {0, 1};
    colName2colType["link_type"] = {0, 1};
    colName2colType["movie_companies"] = {0, 0, 0, 0, 1};
    colName2colType["movie_info_idx"] = {0, 0, 0, 1, 1};
    colName2colType["movie_keyword"] = {0, 0, 0};
    colName2colType["movie_link"] = {0, 0, 0, 0};
    colName2colType["name"] = {0, 1, 1, 0, 1, 1, 1, 1, 1};
    colName2colType["role_type"] = {0, 1};
    colName2colType["title"] = {0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1};
    colName2colType["movie_info"] = {0, 0, 0, 1, 1};
    colName2colType["person_info"] = {0, 0, 0, 1, 1};

    std::vector<std::string> aka_name_columns = {
        "id",
        "person_id",
        "name",
        "imdb_index",
        "name_pcode_cf",
        "name_pcode_ndf",
        "surname_pcode",
        "md5sum",
        "an_features"};

    colName2colHeader["aka_name"] = aka_name_columns;

    std::vector<std::string> aka_title_columns = {
        "id",
        "movie_id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "note",
        "md5sum",
        "at_features"};

    colName2colHeader["aka_title"] = aka_title_columns;

    std::vector<std::string> cast_info_columns = {
        "id",
        "person_id",
        "movie_id",
        "person_role_id",
        "note",
        "nr_order",
        "role_id",
        "ci_features"};

    colName2colHeader["cast_info"] = cast_info_columns;

    std::vector<std::string> char_name_columns = {
        "id",
        "name",
        "imdb_index",
        "imdb_id",
        "name_pcode_nf",
        "surname_pcode",
        "md5sum",
        "chn_features"};

    colName2colHeader["char_name"] = char_name_columns;

    std::vector<std::string> comp_cast_type_columns = {
        "id", "kind", "cct_features"};

    colName2colHeader["comp_cast_type"] = comp_cast_type_columns;

    std::vector<std::string> company_name_columns = {
        "id",
        "name",
        "country_code",
        "imdb_id",
        "name_pcode_nf",
        "name_pcode_sf",
        "md5sum",
        "cn_features"};

    colName2colHeader["company_name"] = company_name_columns;

    std::vector<std::string> company_type_columns = {
        "id", "kind", "ct_features"};

    colName2colHeader["company_type"] = company_type_columns;

    std::vector<std::string> complete_cast_columns = {
        "id", "movie_id", "subject_id", "status_id", "cc_features"};

    colName2colHeader["complete_cast"] = complete_cast_columns;

    std::vector<std::string> info_type_columns = {"id", "info", "it_features"};

    colName2colHeader["info_type"] = info_type_columns;

    std::vector<std::string> keyword_columns = {
        "id", "keyword", "phonetic_code", "k_features"};

    colName2colHeader["keyword"] = keyword_columns;

    std::vector<std::string> kind_type_columns = {"id", "kind", "kt_features"};

    colName2colHeader["kind_type"] = kind_type_columns;

    std::vector<std::string> link_type_columns = {"id", "link", "lt_features"};

    colName2colHeader["link_type"] = link_type_columns;

    std::vector<std::string> movie_companies_columns = {
        "id",
        "movie_id",
        "company_id",
        "company_type_id",
        "note",
        "mc_features"};

    colName2colHeader["movie_companies"] = movie_companies_columns;

    std::vector<std::string> movie_info_idx_columns = {
        "id", "movie_id", "info_type_id", "info", "note", "mii_features"};

    colName2colHeader["movie_info_idx"] = movie_info_idx_columns;

    std::vector<std::string> movie_keyword_columns = {
        "id", "movie_id", "keyword_id", "mk_features"};

    colName2colHeader["movie_keyword"] = movie_keyword_columns;

    std::vector<std::string> movie_link_columns = {
        "id", "movie_id", "linked_movie_id", "link_type_id", "ml_features"};

    colName2colHeader["movie_link"] = movie_link_columns;

    std::vector<std::string> name_columns = {
        "id",
        "name",
        "imdb_index",
        "imdb_id",
        "gender",
        "name_pcode_cf",
        "name_pcode_nf",
        "surname_pcode",
        "md5sum",
        "n_features"};

    colName2colHeader["name"] = name_columns;

    std::vector<std::string> role_type_columns = {"id", "role", "rt_features"};

    colName2colHeader["role_type"] = role_type_columns;

    std::vector<std::string> title_columns = {
        "id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "imdb_id",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "series_years",
        "md5sum",
        "t_features"};

    colName2colHeader["title"] = title_columns;

    std::vector<std::string> movie_info_columns = {
        "id", "movie_id", "info_type_id", "info", "note", "mi_features"};

    colName2colHeader["movie_info"] = movie_info_columns;

    std::vector<std::string> person_info_columns = {
        "id", "person_id", "info_type_id", "info", "note", "pi_features"};

    colName2colHeader["person_info"] = person_info_columns;

    std::string line;

    std::vector<std::vector<int>> intCols;

    std::vector<std::vector<std::string>> stringCols;

    std::vector<int> colTypeIndex = colName2colType[tableName];

    std::vector<int> colIndexInType;

    int colIndex = 0;

    std::string cell;

    int numRows = 0;

    while (std::getline(file, line)) {
      // std::cout << line << std::endl;

      // analyze the first line
      std::stringstream iss(line);

      bool fragmentFlag = false;

      std::string fragmentedStr;

      colIndex = 0;

      // The JOB tables only have two types of columns: integer and string
      while (std::getline(iss, cell, ',')) {
        if ((fragmentFlag == false) && (cell.size() == 1) && (cell[0] == '"')) {
          fragmentFlag = true;

          fragmentedStr = ",";

          continue;

        } else if (
            (fragmentFlag == true) && (cell.size() == 1) && (cell[0] == '"')) {
          fragmentFlag = false;

          cell = fragmentedStr;

          fragmentedStr = "";

        } else if (
            (fragmentFlag == false) && (cell[0] == '"') &&
            ((cell[cell.size() - 1] != '"') ||
             ((cell[cell.size() - 1] == '"') &&
              (cell[cell.size() - 2] == '\\')))) {
          fragmentFlag = true;

          fragmentedStr = cell;

          continue;

        } else if (
            (fragmentFlag == true) && (cell[0] != '"') &&
            (cell[cell.size() - 1] == '"') && (cell[cell.size() - 2] != '\\')) {
          fragmentFlag = false;

          fragmentedStr += cell;

          cell = fragmentedStr;

          fragmentedStr = "";

        } else if (fragmentFlag == true) {
          fragmentedStr += cell;

          continue;
        }

        // std::cout << colIndex << ":" << cell << std::endl;

        if (!fragmentFlag) {
          if (!colTypeIndex[colIndex]) {
            // this is an integer column

            if (numRows == 0) {
              if (cell == "")

                intCols.push_back(std::vector<int>{INT_MIN});

              else

                intCols.push_back(std::vector<int>{stoi(cell)});

              colIndexInType.push_back(intCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[colIndex];

              if (cell == "")

                intCols[vecIndex].push_back(INT_MIN);

              else

                intCols[vecIndex].push_back(stoi(cell));
            }

          } else {
            // this is a string column

            if (numRows == 0) {
              stringCols.push_back(std::vector<std::string>{cell});

              colIndexInType.push_back(stringCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[colIndex];

              stringCols[vecIndex].push_back(cell);
            }
          }

          colIndex++;
        }
      }

      if (colIndex < colTypeIndex.size()) {
        // std::cout << "colIndex:" << colIndex << std::endl;
        // std::cout << "colTypeIndex.size():"<< colTypeIndex.size() <<
        // std::endl;

        for (int i = colIndex; i < colTypeIndex.size(); i++) {
          if (!colTypeIndex[i]) {
            if (numRows == 0) {
              intCols.push_back(std::vector<int>{INT_MIN});

              colIndexInType.push_back(intCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[i];

              intCols[vecIndex].push_back(INT_MIN);
            }

          } else {
            if (numRows == 0) {
              stringCols.push_back(std::vector<std::string>{""});

              colIndexInType.push_back(stringCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[i];

              stringCols[vecIndex].push_back("");
            }
          }
        }
      }

      colIndex = colTypeIndex.size();

      /*if (numRows == 0) {

          for (int i = 0; i < colIndex; i++) {

               std::cout << colTypeIndex[i] << ":" << colIndexInType[i] <<
      std::endl;

          }

      }*/

      numRows++;
    }

    std::vector<VectorPtr> vecs;

    std::cout << "Building RowVector for this table with " << colIndex
              << " columns and " << numRows << " rows." << std::endl;

    for (int i = 0; i < colIndex; i++) {
      int type = colTypeIndex[i];

      int vecIndex = colIndexInType[i];

      // std::cout << i << ":" << type << ":" << vecIndex << std::endl;

      if (!type) {
        auto vec = maker.flatVector<int>(intCols[vecIndex]);

        vecs.push_back(vec);

      } else {
        auto vec = maker.flatVector<std::string>(stringCols[vecIndex]);

        vecs.push_back(vec);
      }
    }

    // to create the last column, which is a feature vector of length k

    if (k < 0)
      k = 8;

    std::vector<std::vector<float>> inputVectors;

    for (int i = 0; i < numRows; i++) {
      std::vector<float> inputVector;

      for (int j = 0; j < k; j++) {
        if (j % 2 == 0)

          inputVector.push_back(1.0);

        else

          inputVector.push_back(0.0);
      }

      inputVectors.push_back(inputVector);
    }

    auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

    vecs.push_back(inputArrayVector);

    RowVectorPtr myRowVector =
        maker.rowVector(colName2colHeader[tableName], vecs);

    return myRowVector;
  }

  int sampleQuery() {
    return 29;
  }

  int sampleModel() {
    return 0;
  }

  void sampleNNModelArch(int numInputFeatures, NNModelContext& nn) {
    nn.inputFeatures = numInputFeatures;
    nn.numLayers = 3;
    nn.hiddenLayerNeurons = 16;
    nn.outputLayerNeurons = 2;
  }

  void sampleDTModelArch(int numInputFeatures, DTModelContext& dt) {
    dt.inputFeatures = numInputFeatures;
    dt.treeDepth = 8;
  }

  bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
      return false;
    str.replace(start_pos, from.length(), to);
    return true;
  }


  void registerNNFunction(const std::string& op_name, const std::vector<std::vector<float>>& weightMatrice, int dim1, int dim2) {

    if (op_name.find("mat_mul") != std::string::npos) {
        auto nnWeightVector = maker.arrayVector<float>(weightMatrice, REAL());
        exec::registerVectorFunction(
            op_name,
            MatrixMultiply::signatures(),
            std::make_unique<MatrixMultiply>(
                nnWeightVector->elements()->values()->asMutable<float>(), dim1, dim2)
            );
        std::cout << "Registered a mat_mul function of name " << op_name << " with dimension " << dim1 << ", " << dim2 << endl;
    }

    else if (op_name.find("mat_add") != std::string::npos) {
        auto nnWeightVector = maker.arrayVector<float>(weightMatrice, REAL());
        exec::registerVectorFunction(
            op_name,
            MatrixVectorAddition::signatures(),
            std::make_unique<MatrixVectorAddition>(
                nnWeightVector->elements()->values()->asMutable<float>(), dim1)
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




    std::unordered_map<std::string, PlanBuilder> getAllTableSources(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
      std::unordered_map<std::string, PlanBuilder>
        sources; // with filters and projections pushed down;

    auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_name"]})
                    .project({"person_id as an_person_id", "name as an_name", "imdb_index as an_imdb_index", "an_features"});
    tabel2Columns["an"] = {"an_person_id", "an_name", "an_imdb_index", "an_features"};
    sources["an"] = an_a;

    auto at_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_title"]})
                    .project({"id as at_id", "movie_id as at_movie_id", "title as at_title", "imdb_index as at_imdb_index", "kind_id as at_kind_id", "at_features"});
    tabel2Columns["at"] = {"at_id", "at_movie_id", "at_title", "at_imdb_index", "at_kind_id", "at_features"};
    sources["at"] = at_a;

    auto ci_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cast_info"]})
            .project(
                {"movie_id as ci_movie_id",
                 "person_id as ci_person_id",
                 "role_id as ci_role_id",
                 "person_role_id as ci_person_role_id",
                 "ci_features"})
            .limit(0, 3624434, false);
    tabel2Columns["ci"] = {"ci_movie_id",
                 "ci_person_id",
                 "ci_role_id",
                 "ci_person_role_id",
                 "ci_features"};
    sources["ci"] = ci_a;

    auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["char_name"]})
                     .project({"id as chn_id", "name as chn_name", "imdb_index as chn_imdb_index", "imdb_id as chn_imdb_id", "chn_features"});
    tabel2Columns["chn"] = {"chn_id", "chn_name", "chn_imdb_index", "chn_imdb_id", "chn_features"};
    sources["chn"] = chn_a;

    auto cc_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["complete_cast"]})
            .project({"id as cc_id", "movie_id as cc_movie_id", "subject_id as cc_subject_id", "status_id as cc_status_id", "cc_features"});
    tabel2Columns["cc"] = {"cc_id", "cc_movie_id", "cc_subject_id", "cc_status_id", "cc_features"};
    sources["cc"] = cc_a;

    auto cct_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .project({"id as cct_id", "kind as cct_kind", "cct_features"});
    tabel2Columns["cct"] = {"cct_id", "cct_kind", "cct_features"};
    sources["cct"] = cct_a;


    auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_name"]})
                    .project({"id as cn_id", "name as cn_name", "country_code as cn_country_code", "imdb_id as cn_imdb_id", "cn_features"});
    tabel2Columns["cn"] = {"cn_id", "cn_name", "cn_country_code", "cn_imdb_id", "cn_features"};
    sources["cn"] = cn_a;

    auto ct_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_type"]})
                    .project({"id as ct_id", "kind as ct_kind", "ct_features"});
    tabel2Columns["ct"] = {"ct_id", "ct_kind", "ct_features"};
    sources["ct"] = ct_a;

    auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["info_type"]})
                    .project({"id as it_id", "info as it_info", "it_features"});
    tabel2Columns["it"] = {"it_id", "it_info", "it_features"};
    sources["it"] = it_a;

    auto kt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["kind_type"]})
                    .project({"id as kt_id", "kind as kt_kind", "kt_features"});
    tabel2Columns["kt"] = {"kt_id", "kt_kind", "kt_features"};
    sources["kt"] = kt_a;

    auto lt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["link_type"]})
                    .project({"id as lt_id", "link as lt_link", "lt_features"});
    tabel2Columns["lt"] = {"lt_id", "lt_link", "lt_features"};
    sources["lt"] = lt_a;

    auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["keyword"]})
                   .project({"id as k_id", "keyword as k_keyword", "k_features"});
    tabel2Columns["k"] = {"k_id", "k_keyword", "k_features"};
    sources["k"] = k_a;

    auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_companies"]})
                    .project({"id as mc_id", "movie_id as mc_movie_id", "company_id as mc_company_id", "company_type_id as mc_company_type_id", "mc_features"});
    tabel2Columns["mc"] = {"mc_id", "mc_movie_id", "mc_company_id", "mc_company_type_id", "mc_features"};
    sources["mc"] = mc_a;

    auto mi_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info"]})
            .project({"id as mi_id", "movie_id as mi_movie_id", "info_type_id as mi_info_type_id", "info as mi_info", "mi_features"});
    tabel2Columns["mi"] = {"mi_id", "mi_movie_id", "mi_info_type_id", "mi_info", "mi_features"};
    sources["mi"] = mi_a;

    auto mii_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info_idx"]})
            .project({"id as mii_id", "movie_id as mii_movie_id", "info_type_id as mii_info_type_id", "info as mii_info", "mii_features"});
    tabel2Columns["mii"] = {"mii_id", "mii_movie_id", "mii_info_type_id", "mii_info", "mii_features"};
    sources["mii"] = mii_a;

    auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_keyword"]})
                    .project({"id as mk_id", "movie_id as mk_movie_id", "keyword_id as mk_keyword_id", "mk_features"});
    tabel2Columns["mk"] = {"mk_id", "mk_movie_id", "mk_keyword_id", "mk_features"};
    sources["mk"] = mk_a;

    auto ml_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_link"]})
                    .project({"id as ml_id", "movie_id as ml_movie_id", "linked_movie_id as ml_linked_movie_id", "link_type_id as ml_link_type_id", "ml_features"});
    tabel2Columns["ml"] = {"ml_id", "ml_movie_id", "ml_linked_movie_id", "ml_link_type_id", "ml_features"};
    sources["ml"] = ml_a;

    auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["name"]})
                   .project({"id as n_id", "name as n_name", "imdb_index as n_imdb_index", "imdb_id as n_imdb_id", "n_features"});
    tabel2Columns["n"] = {"n_id", "n_name", "n_imdb_index", "n_imdb_id", "n_features"};
    sources["n"] = n_a;

    auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["person_info"]})
                    .project(
                        {"id as pi_id",
                         "person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
    tabel2Columns["pi"] = {"pi_id", "pi_person_id", "pi_info_type_id", "pi_features"};
    sources["pi"] = pi_a;

    auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["role_type"]})
                    .project({"id as rt_id", "role as rt_role", "rt_features"});
    tabel2Columns["rt"] = {"rt_id", "rt_role", "rt_features"};
    sources["rt"] = rt_a;

    auto t_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["title"]})
            .project({"id as t_id", "title as t_title", "imdb_index as t_imdb_index", "kind_id as t_kind_id", "imdb_id as t_imdb_id", "t_features"});
    tabel2Columns["t"] = {"t_id", "t_title", "t_imdb_index", "t_kind_id", "t_imdb_id", "t_features"};
    sources["t"] = t_a;

    return sources;
  }



bool addModelInferenceToQueryPlanAfterFactorize(PlanBuilder & planBuilder, std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, const std::string& colFeature) {

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
        modelProjString += colFeature;

        for (int i = modelOperators.size() - 1; i > 0; i--) {
            modelProjString += ")";
        }
        modelProjString += " AS output";

        std::cout << "Inference Part: " << modelProjString << endl;
        planBuilder.project({modelProjString});

      return true;
}



/*
param pushDown: 0 -> no push, 1 -> all push, 2 -> left push, 3 -> right push
*/
bool createAndExecuteQuery(std::string leftTable, std::string rightTable, std::string probKey, std::string buildKey, int dimLeft, int dimRight, int pushDown) {
    std::string firstOpName = modelOperators[0]; // name of operator of split layer
    std::vector<std::vector<float>> firstWeight = operatorParam2Weights[firstOpName]; // weight of the split layer
    int numCols = firstWeight.size(); // number of columns in split layer
    int numNeurons = firstWeight[0].size(); // number of neurons in split layer

    //registering vector addition operator for later use as aggregation
    std::vector<std::vector<float>> emptyMatrix;
    registerNNFunction("vector_addition", emptyMatrix, numNeurons, -1);

    PlanBuilder planBuilder{pool_.get()};
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

    // Fetch plan builders for each plan
    std::unordered_map<std::string, PlanBuilder> sources = getAllTableSources(planNodeIdGenerator);

    // fetch plan builders corresponding to left and right table
    PlanBuilder leftPlan = sources[leftTable];
    PlanBuilder rightPlan = sources[rightTable];

    // form the left and right feature columns
    std::string lFeatureName = leftTable + "_features";
    std::string rFeatureName = rightTable + "_features";

    std::vector<std::string> projections;
    projections.push_back(probKey);
    projections.push_back(buildKey);

    std::string fNewName; //new name after applying operator on feature
    std::string fNewNameFull; // fNewName with its full projection details

    // checking if push left feature
    if (pushDown == 1 || pushDown == 2) {
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, 0, dimLeft);
        std::string newOpName = firstOpName + "_left";
        registerNNFunction(newOpName, subWeight, dimLeft, numNeurons);
        fNewName = "factorized_left";
        fNewNameFull = newOpName + "(get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ")) AS " + fNewName;
    }
    else {
        fNewName = "mapped_left";
        fNewNameFull = "get_feature_vec(" + lFeatureName + ", " + std::to_string(dimLeft) + ") AS " + fNewName;
    }
    projections.push_back(fNewName);
    leftPlan= leftPlan.project({probKey, fNewNameFull});

    // checking if push right feature
    if (pushDown == 1 || pushDown == 3) {
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, dimLeft, dimRight);
        std::string newOpName = firstOpName + "_right";
        registerNNFunction(newOpName, subWeight, dimRight, numNeurons);
        fNewName = "factorized_right";
        fNewNameFull = newOpName + "(get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ")) AS " + fNewName;
    }
    else {
        fNewName = "mapped_right";
        fNewNameFull = "get_feature_vec(" + rFeatureName + ", " + std::to_string(dimRight) + ") AS " + fNewName;
    }
    projections.push_back(fNewName);
    rightPlan= rightPlan.project({buildKey, fNewNameFull});

    // Perform the join
    PlanBuilder out = leftPlan.hashJoin(
        {probKey},
        {buildKey},
        rightPlan.planNode(),
        "",
        {projections}
    );

    // Apply operators after join
    std::string projString;

    if (pushDown == 0) {
        // no features pushed, so concatenate them and apply first operator in the model
        registerNNFunction(firstOpName, firstWeight, dimLeft + dimRight, numNeurons);
        projString = firstOpName + "(concat(mapped_left, mapped_right)) AS features";
    }
    else if (pushDown == 1) {
        // all features pushed, so just perform aggregation
        projString = "vector_addition(factorized_left, factorized_right) AS features";
    }
    else if (pushDown == 2) {
        // only left features pushed, so apply first operator on right feature and perform aggregation
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, dimLeft, dimRight);
        std::string newOpName = firstOpName + "_right";
        registerNNFunction(newOpName, subWeight, dimRight, numNeurons);
        std::string fNewNameFull = newOpName + "(mapped_right)";

        projString = "vector_addition(factorized_left, " + fNewNameFull + ") AS features";
    }

    else {
        // only right features pushed, so apply first operator on left feature and perform aggregation
        std::vector<std::vector<float>> subWeight = extractSubweight(firstWeight, 0, dimLeft);
        std::string newOpName = firstOpName + "_left";
        registerNNFunction(newOpName, subWeight, dimLeft, numNeurons);
        std::string fNewNameFull = newOpName + "(mapped_left)";

        projString = "vector_addition(" + fNewNameFull + ", factorized_right) AS features";
    }

    planBuilder = out.project({projString});
    addModelInferenceToQueryPlanAfterFactorize(planBuilder, planNodeIdGenerator, "features");

      auto myPlan = planBuilder.planNode();
      std::cout << myPlan->toString(true, true) << std::endl;
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
      auto results = exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0 << std::endl;
      std::cout << "Results Size: " << results->size() << std::endl;
      std::cout << "Results:" << results->toString(0, 5) << std::endl;
      return true;
  }

};


int main(int argc, char** argv) {
  setlocale(LC_TIME, "C");
  malloc_trim(0);

  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  Job2WayJoin bench;
  std::cout
      << "[WARNING] the data path is hardcoded and needs to be modified accordingly."
      << std::endl;

  // Load Aka Name table
  RowVectorPtr akaNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/aka_name.csv", "aka_name", 8);

  // Load Aka Title table
  RowVectorPtr akaTitleVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/aka_title.csv", "aka_title", 8);

  // Load Cast Info table
  RowVectorPtr castInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/cast_info.csv", "cast_info", 8);

  // Load Char Name table
  RowVectorPtr charNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/char_name.csv", "char_name", 8);

  // Load Comp Cast Type table
  RowVectorPtr compCastTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/comp_cast_type.csv", "comp_cast_type", 8);

  // Load Company Name table
  RowVectorPtr companyNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/company_name.csv", "company_name", 8);

  // Load Company Type table
  RowVectorPtr companyTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/company_type.csv", "company_type", 8);

  // Load Complete Cast table
  RowVectorPtr completeCastVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/complete_cast.csv", "complete_cast", 8);

  // Load Info Type table
  RowVectorPtr infoTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/info_type.csv", "info_type", 8);

  // Load Keyword table
  RowVectorPtr keywordVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/keyword.csv", "keyword", 8);

  // Load Kind Type table
  RowVectorPtr kindTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/kind_type.csv", "kind_type", 8);

  // Load Link Type table
  RowVectorPtr linkTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/link_type.csv", "link_type", 8);

  // Load Movie Companies table
  RowVectorPtr movieCompaniesVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_companies.csv", "movie_companies", 8);

  // Load Movie Info Index table
  RowVectorPtr movieInfoIdxVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_info_idx.csv", "movie_info_idx", 8);

  // Load Movie Keyword table
  RowVectorPtr movieKeywordVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_keyword.csv", "movie_keyword", 8);

  // Load Movie Link table
  RowVectorPtr movieLinkVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_link.csv", "movie_link", 8);

  // Load Name table
  RowVectorPtr nameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/name.csv", "name", 8);

  // Load Role Type table
  RowVectorPtr roleTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/role_type.csv", "role_type", 8);

  // Load Title table
  RowVectorPtr titleVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/title.csv", "title", 8);

  // Load Movie Info table
  RowVectorPtr movieInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_info.csv", "movie_info", 8);

  // Load Person Info table
  RowVectorPtr personInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb//person_info.csv", "person_info", 8);

  bench.tableName2RowVector["aka_name"] = akaNameVec;

  bench.tableName2RowVector["aka_title"] = akaTitleVec;

  bench.tableName2RowVector["cast_info"] = castInfoVec;

  bench.tableName2RowVector["char_name"] = charNameVec;

  bench.tableName2RowVector["comp_cast_type"] = compCastTypeVec;

  bench.tableName2RowVector["company_name"] = companyNameVec;

  bench.tableName2RowVector["company_type"] = companyTypeVec;

  bench.tableName2RowVector["complete_cast"] = completeCastVec;

  bench.tableName2RowVector["info_type"] = infoTypeVec;

  bench.tableName2RowVector["keyword"] = keywordVec;

  bench.tableName2RowVector["kind_type"] = kindTypeVec;

  bench.tableName2RowVector["link_type"] = linkTypeVec;

  bench.tableName2RowVector["movie_companies"] = movieCompaniesVec;

  bench.tableName2RowVector["movie_info_idx"] = movieInfoIdxVec;

  bench.tableName2RowVector["movie_keyword"] = movieKeywordVec;

  bench.tableName2RowVector["movie_link"] = movieLinkVec;

  bench.tableName2RowVector["name"] = nameVec;

  bench.tableName2RowVector["role_type"] = roleTypeVec;

  bench.tableName2RowVector["title"] = titleVec;

  bench.tableName2RowVector["movie_info"] = movieInfoVec;

  bench.tableName2RowVector["person_info"] = personInfoVec;
  

  exec::registerVectorFunction(
          "get_feature_vec",
          GetFeatureVec::signatures(),
          std::make_unique<GetFeatureVec>());
  std::cout << "Completed registering function for get_feature_vec" << std::endl;

  // retrieve the weights and set to map
  std::cout << "Reading model parameters" << std::endl;
  bench.findWeights("resources/model/job_any_64.h5");

  // retrieve the model operators from model expression IR
  std::string modelInput = "softmax3(mat_add3(mat_mul3(relu2(mat_add2(mat_mul2(relu1(mat_add1(mat_mul1(features)))))))))";
  std::cout << "Extracting model operators" << std::endl;
  std::vector<std::string> operators = bench.extractOperatorsInReverse(modelInput);
  bench.modelOperators = operators;

  std::cout << "Performing Join" << std::endl;
  bool ret = bench.createAndExecuteQuery("ci", "rt", "ci_role_id", "rt_id", 50, 50, 3);

  return ret;
}

