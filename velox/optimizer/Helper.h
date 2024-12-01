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
#pragma once
#include <iostream>
#include <regex>
#include <vector>
#include "CataLog.h"
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox::exec;

namespace optimization {
// Structure to represent the file structure with paths and schema
struct FileStructure {
  std::vector<std::shared_ptr<TempFilePath>> paths;
  RowTypePtr schema;
};

// Test class inheriting from HiveConnectorTestBase
class MyFileTest : public HiveConnectorTestBase {
 public:
  MyFileTest() {
    // SetUp();
  }
  ~MyFileTest() {}

  void SetUp() {
    HiveConnectorTestBase::SetUp();
  }

  void TestBody() override {}
};
/**
 * @brief A function to create a block index based on parts and flag
 *
 * @param parts The number of blocks
 * @param flag The flag to denote the left side or right side of multiplication.
 * 0 denotes the left side, usually values; 1 denotes the right side, usually
 * weights.
 *
 * @return indexs of blocks
 */
std::vector<std::vector<float>> create_block_index(int parts, int flag) {
  std::vector<std::vector<float>> indexs;
  if (flag == 0) {
    std::vector<float> indexs_row;
    std::vector<float> indexs_col;
    for (int i = 0; i < parts; i++) {
      indexs_row.push_back(0);
    }
    for (int i = 0; i < parts; i++) {
      indexs_col.push_back(i);
    }
    indexs.push_back(indexs_row);
    indexs.push_back(indexs_col);
  } else {
    std::vector<float> indexs_row;
    std::vector<float> indexs_col;
    for (int i = 0; i < parts; i++) {
      indexs_row.push_back(i);
    }
    for (int i = 0; i < parts; i++) {
      indexs_col.push_back(0);
    }
    indexs.push_back(indexs_row);
    indexs.push_back(indexs_col);
  }
  return indexs;
}

// Function to create input block based on total_size, values, and block_numbers
std::vector<std::vector<float>> create_input_block(
    int total_size,
    std::vector<std::vector<float>>& values,
    int block_numbers) {
  std::vector<std::vector<float>> valuesArray;
  std::vector<float> flattened;
  for (const auto& row : values) {
    flattened.insert(flattened.end(), row.begin(), row.end());
  }
  // TODO: needs to handle the case: that can not be exactly blocked
  auto block_size = total_size / block_numbers;
  int num_cols = total_size / values.size();
  int cols_per_block = block_size / values.size();
  // std::cout
  //     << fmt::format(
  //            "Total Size: {}, block size: {}, value size: cols per block:
  //            {}", total_size, block_size, cols_per_block)
  //     << std::endl;
  for (int i = 0; i < block_numbers; i++) {
    std::vector<float> valuesArraySingleBlock;
    for (int j = 0; j < block_size; j++) {
      // get the data's x and y in original block
      int data_x = j / cols_per_block;
      int data_y = i * cols_per_block + j % cols_per_block;
      valuesArraySingleBlock.push_back(flattened[data_x * num_cols + data_y]);
    }
    valuesArray.push_back(valuesArraySingleBlock);
  }
  return valuesArray;
}

// Function to create weight block based on total_size, values, and
// block_numbers
std::vector<std::vector<float>>
create_weight_block(int total_size, float* values, int block_numbers) {
  std::vector<std::vector<float>> valuesArray;
  auto block_size = total_size / block_numbers;
  for (int i = 0; i < block_numbers; i++) {
    std::vector<float> valuesArraySingleBlock;
    for (int j = 0; j < block_size; j++) {
      valuesArraySingleBlock.push_back(values[i * block_size + j]);
    }
    valuesArray.push_back(valuesArraySingleBlock);
  }
  return valuesArray;
}

std::vector<std::vector<float>>
create_blocks(int row, int col, float* values, int block_size) {
  int num_blocks =
      static_cast<int>(std::ceil(static_cast<float>(col) / block_size));
  std::vector<std::vector<float>> blocks(
      num_blocks); // Initialize vector of blocks
  int current_col = 0; // Current column index in the values array
  for (int i = 0; i < num_blocks; ++i) {
    int current_block_size = (i == num_blocks - 1)
        ? col - current_col
        : block_size; // Adjust block size for the last block

    // Create a new block of size row x current_block_size
    std::vector<float> block(row * current_block_size);

    // Fill the block with values
    for (int r = 0; r < row; ++r) {
      for (int c = 0; c < current_block_size; ++c) {
        block[r * current_block_size + c] = values[r * col + current_col + c];
      }
    }

    blocks[i] = std::move(block); // Store the block in the vector of blocks
    current_col += current_block_size; // Move to the next column
  }

  return blocks;
}

FileStructure save_blocks_to_files(
    std::vector<std::vector<float>> valuesArray,
    std::string name) {
  optimization::MyFileTest myFile;
  optimization::FileStructure myFileStructure;
  std::vector<std::shared_ptr<TempFilePath>> paths;
  auto pool_ = memory::MemoryManager::getInstance()->addLeafPool();
  VectorMaker maker{pool_.get()};
  RowVectorPtr input;
  // Create indexs for blocks
  int parts = valuesArray.size();
  int flag = 0;
  auto indexs = create_block_index(parts, flag);
  // Use maker to create rowVector for "w", "w_row", and "w_col"
  for (int i = 0; i < parts; i++) {
    input = maker.rowVector(
        {name + "_wb", name + "_wb_row", name + "_wb_col"},
        {maker.arrayVector<float>({valuesArray[i]}, REAL()),
         maker.flatVector({indexs[0][i]}),
         maker.flatVector({indexs[1][i]})});

    auto file = TempFilePath::create();
    // Store blocks to file
    myFile.writeToFile(file->path, {input});
    // Store file object to paths
    paths.push_back(file);
  }
  myFileStructure.paths = paths;
  // Store schema
  myFileStructure.schema = asRowType(input->type());
  return myFileStructure;
}

// Function to convert block data to files and return FileStructure
FileStructure block_to_files(
    std::vector<std::vector<float>> valuesArray,
    int parts,
    int flag) {
  optimization::MyFileTest myFile;
  optimization::FileStructure myFileStructure;
  std::vector<std::shared_ptr<TempFilePath>> paths;
  auto pool_{memory::MemoryManager::getInstance()->addLeafPool()};
  VectorMaker maker{pool_.get()};
  RowVectorPtr input;
  if (flag == 0) {
    // Create indexs for blocks
    auto indexs = create_block_index(parts, flag);
    // Use maker to create rowVector for "v", "v_row", and "v_col"
    for (int i = 0; i < parts; i++) {
      input = maker.rowVector(
          {"v", "v_row", "v_col"},
          {maker.arrayVector<float>({valuesArray[i]}, REAL()),
           maker.flatVector({indexs[0][i]}),
           maker.flatVector({indexs[1][i]})});

      auto file = TempFilePath::create();
      // Store blocks to file
      myFile.writeToFile(file->path, {input});
      // Store file object to paths
      paths.push_back(file);
    }
    myFileStructure.paths = paths;
    // Store schema
    myFileStructure.schema = asRowType(input->type());

    return myFileStructure;
  } else {
    // Create indexs for blocks
    auto indexs = create_block_index(parts, flag);
    // Use maker to create rowVector for "w", "w_row", and "w_col"
    for (int i = 0; i < parts; i++) {
      input = maker.rowVector(
          {"w", "w_row", "w_col"},
          {maker.arrayVector<float>({valuesArray[i]}, REAL()),
           maker.flatVector({indexs[0][i]}),
           maker.flatVector({indexs[1][i]})});

      auto file = TempFilePath::create();
      // Store blocks to file
      myFile.writeToFile(file->path, {input});
      // Store file object to paths
      paths.push_back(file);
    }
    myFileStructure.paths = paths;
    // Store schema
    myFileStructure.schema = asRowType(input->type());
    return myFileStructure;
  }
}

void replaceSourceWithIdInSerializedPlan(
    folly::dynamic& serializedPlan,
    folly::dynamic& serializedNewSource,
    std::string nodeId) {
  if (!serializedPlan.count("sources")) {
    return;
  }
  for (auto& source : serializedPlan["sources"]) {
    if (source["id"].asString() == nodeId) {
      source = serializedNewSource;
      return;
    } else {
      optimization::replaceSourceWithIdInSerializedPlan(
          source, serializedNewSource, nodeId);
    }
  }
}

std::vector<int> extractUDFDimension(std::string udfName) {
  core::QueryConfig config({});
  std::shared_ptr<VectorFunction> myUDF =
      getVectorFunction(udfName, {ARRAY(REAL())}, {}, config);
  if (udfName.find("mat_mul") != std::string::npos) {
    if (udfName.find("_h") != std::string::npos) {
      // block based MatrixMultiply
      myUDF = getVectorFunction(
          udfName, {ARRAY(REAL()), ARRAY(REAL())}, {}, config);
      std::shared_ptr<MatrixMultiply_h> myMulUDF =
          std::dynamic_pointer_cast<MatrixMultiply_h>(myUDF);
      std::vector<int> dims = myMulUDF->getDims();
      return {dims[2] /*block size */, dims[1]};
    } else {
      // non-block based MatrixMultiply
      std::shared_ptr<MatrixMultiply> myMulUDF =
          std::dynamic_pointer_cast<MatrixMultiply>(myUDF);
      return myMulUDF->getDims();
    }
  } else if (udfName.find("mat_vector_add") != std::string::npos) {
    std::shared_ptr<MatrixVectorAddition> myAddUDF =
        std::dynamic_pointer_cast<MatrixVectorAddition>(myUDF);
    return myAddUDF->getDims();
  } else {
    return {};
  }
}

void augmentFunctionExpression(folly::dynamic& serializedPlan) {
  if (serializedPlan.count("functionName")) {
    std::string functionName = serializedPlan["functionName"].asString();
    try {
      std::vector<int> dims = extractUDFDimension(functionName);
      if (dims.size() > 0) {
        folly::dynamic jsonArray = folly::dynamic::array;
        for (int value : dims) {
          jsonArray.push_back(value);
        }
        serializedPlan["dims"] = jsonArray;
      }
    } catch (const std::exception& e) {
      std::cout << "Error: extractUDFDimension: " << functionName << " "
                << e.what() << std::endl;
    }
    if (serializedPlan.count("inputs")) {
      for (auto& input : serializedPlan["inputs"]) {
        augmentFunctionExpression(input);
      }
    }
  }
}

void augmentTableScanNode(folly::dynamic& serializedPlan, CataLog& cataLog) {
  if (serializedPlan["name"].asString() == "TableScanNode") {
    std::string nodeId = serializedPlan["id"].asString();
    std::shared_ptr<Source> tableSource = cataLog.getSource(nodeId);
    std::shared_ptr<OutputStat> tableStats =
        std::static_pointer_cast<OutputStat>(tableSource->getStats());
    int numRows = tableStats->getRows();
    int numCols = tableStats->getCols();

    // Set number of rows and columns
    folly::dynamic jsonArray = folly::dynamic::array;
    jsonArray.push_back(numRows);
    jsonArray.push_back(numCols);
    serializedPlan["tableStats"] = jsonArray;

    // Set tableName
    std::string tableName = cataLog.getNodeIdRelationName(nodeId);
    serializedPlan["tableName"] = tableName;
  }
}

void augmentSerializedPlan(folly::dynamic& serializedPlan, CataLog& cataLog) {
  if (serializedPlan.count("projections")) {
    for (auto& project : serializedPlan["projections"]) {
      if (project.count("functionName")) {
        augmentFunctionExpression(project);
      }
    }
  }

  augmentTableScanNode(serializedPlan, cataLog);

  if (serializedPlan.count("sources")) {
    for (auto& source : serializedPlan["sources"]) {
      augmentSerializedPlan(source, cataLog);
    }
  }
}

std::string extractExprWithinTarget(
    const std::string& source,
    const std::string& target) {
  size_t pos = source.find(target);
  if (pos == std::string::npos) {
    return ""; // Target function not found in source
  }

  int count = 0;
  size_t start_pos = source.find('(', pos);
  for (size_t i = start_pos + 1; i < source.size(); ++i) {
    if (source[i] == '(') {
      ++count;
    } else if (source[i] == ')') {
      if (count == 0) {
        return source.substr(start_pos + 1, i - start_pos - 1);
      } else {
        --count;
      }
    }
  }
  return ""; // Matching ')' not found
}

// Function to escape special characters in a regex string
std::string escapeRegex(const std::string& str) {
  std::string escapedStr;
  for (char c : str) {
    if (c == '\\' || c == '[' || c == ']' || c == '(' || c == ')' || c == '{' ||
        c == '}' || c == '+' || c == '*' || c == '?' || c == '.' || c == '^' ||
        c == '$' || c == '|') {
      escapedStr += '\\'; // Add escape character
    }
    escapedStr += c;
  }
  return escapedStr;
}
/**
 * @brief Function to replace double quotes with single quotes in a string
 *
 * @param str The input string
 * @return std::string The string with double quotes replaced by single quotes
 */
std::string replaceDoubleQuotes(std::string str) {
  for (char& ch : str) {
    if (ch == '"') {
      ch = '\'';
    }
  }
  return str;
}

// Iterate over all files in a directory and return their paths
std::vector<std::string> getFilePathsFromDir(const std::string& dirPath) {
  std::vector<std::string> filePaths;
  for (auto const& dirEntry : fs::directory_iterator(dirPath)) {
    if (!dirEntry.is_regular_file()) {
      continue;
    }
    // Ignore hidden files.
    if (dirEntry.path().filename().c_str()[0] == '.') {
      continue;
    }
    // auto dataFile = CustomTempFilePath::create(dirEntry.path());
    filePaths.push_back(dirEntry.path());
  }
  return filePaths;
}

// Function to trim spaces from both front and end of a string
std::string trim(const std::string& str) {
  const char* whitespace = " \t\n\r";
  const size_t first = str.find_first_not_of(whitespace);
  if (first == std::string::npos)
    return "";
  const size_t last = str.find_last_not_of(whitespace);
  return str.substr(first, (last - first + 1));
}

// Recursive function to parse the nested DL expressions
// The parsed order is from the outermost expression to the innermost expression
// Example: exp1(exp2(exp3(exp4(input))))
// Return: parsedSingleExpr: {"exp1", "exp2", "exp3", "exp4"}
//         matchedExpr: {"exp1(exp2(exp3(exp4(input))))",
//         "exp2(exp3(exp4(input)))", "exp3(exp4(input))", "exp4(input)"}
void parseDLExpressions(
    const std::string& input,
    std::vector<std::string>& parsedSingleExpr,
    std::vector<std::string>& matchedExpr) {
  size_t openParen = input.find('(');
  if (openParen == std::string::npos) {
    return;
  }

  size_t closeParen = input.rfind(')');
  if (closeParen == std::string::npos) {
    return;
  }

  // Extract the function name
  std::string funcName = input.substr(0, openParen);
  funcName = trim(funcName);
  parsedSingleExpr.push_back(funcName);
  matchedExpr.push_back(trim(input));

  // Extract the argument within the parentheses
  std::string inner = input.substr(openParen + 1, closeParen - openParen - 1);
  inner = trim(inner);

  // Recursively parse the inner expression
  parseDLExpressions(inner, parsedSingleExpr, matchedExpr);
}

// Function to split a string based on a delimiter
std::vector<std::string> splitString(const std::string& str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

// Function to check if a string contains a substring but is not equal to it
bool containsStrButNotEqual(const std::string& str, const std::string& subStr) {
  // Check if the substring is found within the string
  size_t found = str.find(subStr);

  // Ensure that the substring is found and the two strings are not equal
  return (found != std::string::npos) && (str != subStr);
}

/**
 * @brief Function to find the data sources from the expression and return a
 * vector of data sources
 *
 * @param expr The expression string to search for data sources
 * @return std::vector<std::string> A vector of data sources found in the
 * expression
 */
std::vector<std::string> findDataSrcFromExpr(const std::string& expr) {
  std::regex patternToMatchRawSource("ROW\\[\"(.*?)\"\\]");
  std::smatch matches;
  // Object to capture the matched data source
  std::vector<std::string> matchedDataSources;
  // Start position for the search
  std::string::const_iterator searchStart(expr.cbegin());

  // Search out the matched data source and store in matches
  while (std::regex_search(
      searchStart, expr.cend(), matches, patternToMatchRawSource)) {
    // The captured group is in matches[1]
    matchedDataSources.push_back(matches[1].str());
    // Update the search start position
    searchStart = matches.suffix().first;
  }
  return matchedDataSources;
}

/**
 * @brief Function to find the pushdown planNode based on the nodeId
 *
 * @param planNode The current planNode to search for the pushdown planNode
 * @param nodeId The nodeId to search for the pushdown planNode
 * @return std::shared_ptr<const core::PlanNode> The pushdown planNode
 */
std::shared_ptr<const core::PlanNode> findPlanNodeById(
    const std::shared_ptr<const core::PlanNode>& planNode,
    const std::string& nodeId) {
  if (planNode->id() == nodeId) {
    return planNode;
  }

  for (const auto& child : planNode->sources()) {
    auto found = findPlanNodeById(child, nodeId);
    if (found) {
      return found;
    }
  }

  return nullptr;
}

/**
 * @brief Function to find the nodeIds between two nodeId
 *
 * @param planNode The current planNode to search for the nodeIds
 * @param sourceNodeId The source nodeId
 * @param targetNodeId The target nodeId
 * @return std::vector<std::string> The nodeIds between the source and target
 * nodeId
 */

std::vector<std::string> findNodeIdsBetweenIds(
    const std::shared_ptr<const core::PlanNode>& planNode,
    std::string sourceNodeId,
    std::string targetNodeId,
    std::vector<std::string>& nodeIds) {
  auto planNodeId = planNode->id();
  if (planNodeId != sourceNodeId && planNodeId != targetNodeId) {
    nodeIds.push_back(planNodeId);
  }
  if (planNodeId == sourceNodeId) {
    return nodeIds;
  }

  for (const auto& child : planNode->sources()) {
    auto childNodeIds =
        findNodeIdsBetweenIds(child, sourceNodeId, targetNodeId, nodeIds);
    if (!childNodeIds.empty()) {
      return childNodeIds;
    }
  }
  if (nodeIds.size() > 0) {
    nodeIds.pop_back();
  }

  return {};
}

/**
 * @brief Function to add the projection field in the serialized plan
 *
 * @param serializedPlan The serialized plan to add the projection field
 * @param filedToBeAdded The field to be added in the projection
 * @param nodeIds The nodeIds to add the projection field
 */
void addProjectionFiledInSerializedPlan(
    folly::dynamic& serializedPlan,
    folly::dynamic& filedToBeAdded,
    std::vector<std::string> nodeIds) {
  if (!serializedPlan.count("sources")) {
    return;
  }

  std::string currentNodeId = serializedPlan["id"].asString();
  std::string currentNodeName = serializedPlan["name"].asString();
  // Use std::find to search for the string in the vector
  auto it = std::find(nodeIds.begin(), nodeIds.end(), currentNodeId);
  if (it != nodeIds.end()) {
    if (currentNodeName.find("Project") != std::string::npos) {
      // Add the filed to the ProjectNode
      serializedPlan["projections"].push_back(filedToBeAdded);
      serializedPlan["names"].push_back(filedToBeAdded["fieldName"]);
    } else if (currentNodeName.find("Join") != std::string::npos) {
      // Add the filed to the FilterNode
      serializedPlan["outputType"]["cTypes"].push_back(filedToBeAdded["type"]);
      serializedPlan["outputType"]["names"].push_back(
          filedToBeAdded["fieldName"]);
    } else if (currentNodeName.find("Filter") != std::string::npos) {
      // No need to add the filed to the FilterNode
    } else {
      throw std::runtime_error(
          "[Helper] addProjectionFiledInSerializedPlan: Unsupported node type: " +
          currentNodeName);
    }
  }

  for (auto& source : serializedPlan["sources"]) {
    optimization::addProjectionFiledInSerializedPlan(
        source, filedToBeAdded, nodeIds);
  }

  return;
}

// In Velox, when parsing the cast function, parentheses are omitted in the
// exprStr output, making it unusable directly. This function reconstructs the
// correct cast expression by adding the necessary parentheses to match the body
// of the expression. For example:
// Input: eq(cast argmax(ROW["trending_prediction"]) as BIGINT, 1)
// Output: eq(cast(argmax(ROW["trending_prediction"]) as BIGINT), 1)
std::string fix_cast_function_parsing(std::string input) {
  // Use regex to match the pattern "cast" followed by a space and a function
  // call
  std::regex cast_regex(R"(cast\s+(\w+.*?)\s+as\s+(\w+))");

  // Use a lambda function for the replacement to insert parentheses
  std::string result =
      std::regex_replace(input, cast_regex, R"(cast($1 as $2))");
  return result;
}

/**
 * @brief Function to reformat the comparison expression to a standard format,
 * the input following the format:
 * [Operator](cast [Expression] as [DataType], [CompareValue])
 * The output format: [Expression] [Operator] [CompareValue]
 *
 *
 * @param exprStr The input comparison expression string
 * @return std::string The reformatted comparison expression
 */

std::string reformatComparisonExpr(std::string exprStr) {
  std::regex pattern(R"((\w+)\(cast\s+(.*?)\s+as\s+(\w+),\s*(.*?)\s*\))");
  // Match the exprStr string against the pattern
  std::smatch match;
  if (std::regex_search(exprStr, match, pattern)) {
    // Extract the matched groups
    std::string operatorStr = match[1]; // Operator
    std::string expression = match[2]; // Expression
    std::string dataType = match[3]; // Expression
    std::string compareValue = match[4]; // CompareValue

    if (operatorStr == "eq") {
      operatorStr = "=";
    } else if (operatorStr == "neq") {
      operatorStr = "!=";
    } else if (operatorStr == "lt") {
      operatorStr = "<";
    } else if (operatorStr == "lte") {
      operatorStr = "<=";
    } else if (operatorStr == "gt") {
      operatorStr = ">";
    } else if (operatorStr == "gte") {
      operatorStr = ">=";
    } else {
      throw std::runtime_error(
          "Unsupported operator in the expression: " + exprStr);
    }

    if (dataType == "DOUBLE" || dataType == "REAL") {
      compareValue = std::to_string(std::stod(compareValue));
    } 
    
    // Return the reformatted expression
    return expression + " " + operatorStr + " " + compareValue;
  } else {
    throw std::runtime_error("Failed to match the pattern: " + exprStr);
  }

  // If no match, return an empty optional
  return "";
}

std::vector<RowVectorPtr> splitRowVectorIntoBatches(
    RowVectorPtr inputVector,
    size_t batchSize) {
  // Total number of rows in the input RowVector
  size_t totalRows = inputVector->size();

  // Result vector to hold all the batches
  std::vector<RowVectorPtr> batches;

  // Split the input RowVector into batches
  for (size_t start = 0; start < totalRows; start += batchSize) {
    // Calculate the size of the current batch
    size_t currentBatchSize = std::min(batchSize, totalRows - start);

    // Slice the RowVector directly
    RowVectorPtr batch = std::dynamic_pointer_cast<RowVector>(
        inputVector->slice(start, currentBatchSize));

    // Add the batch to the result
    batches.push_back(batch);
  }

  return batches;
}

std::vector<std::shared_ptr<TempFilePath>> splitRowVectorIntoBatchFiles(
    RowVectorPtr inputVector,
    size_t batchSize) {
  optimization::MyFileTest myFile;

  // Split the input RowVector into batches
  auto batches = splitRowVectorIntoBatches(inputVector, batchSize);

  // Result vector to hold all the batch file paths
  std::vector<std::shared_ptr<TempFilePath>> batchFiles;

  // Write each batch to a temporary file
  for (size_t i = 0; i < batches.size(); ++i) {
    auto file = TempFilePath::create();
    myFile.writeToFile(file->path, {batches[i]});
    batchFiles.push_back(file);
  }
  return batchFiles;
}

} // namespace optimization