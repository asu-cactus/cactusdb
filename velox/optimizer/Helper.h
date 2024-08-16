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
#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/vector/tests/utils/VectorMaker.h"

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
  int num_blocks = (col + block_size - 1) /
      block_size; // Calculate the number of blocks needed
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
  if (serializedPlan["sources"].isNull()) {
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
} // namespace optimization