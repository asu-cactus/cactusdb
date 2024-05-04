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
#include <vector>
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
  //            "Total Size: {}, block size: {}, value size: cols per block: {}",
  //            total_size,
  //            block_size,
  //            cols_per_block)
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
  int num_blocks = (col + block_size - 1) / block_size; // Calculate the number of blocks needed
  std::vector<std::vector<float>> blocks(num_blocks); // Initialize vector of blocks
  int current_col = 0; // Current column index in the values array
    for (int i = 0; i < num_blocks; ++i) {
        int current_block_size = (i == num_blocks - 1) ? col - current_col : block_size; // Adjust block size for the last block

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

FileStructure create_blocks_to_files(int row, int col, int block_size) {
  optimization::MyFileTest myFile;
  optimization::FileStructure myFileStructure;
  std::vector<std::shared_ptr<TempFilePath>> paths;
  auto pool_ = memory::MemoryManager::getInstance()->addLeafPool();
  VectorMaker maker{pool_.get()};
  RowVectorPtr input;


  int num_blocks = (col + block_size - 1) / block_size; // Calculate the number of blocks needed

  int parts = num_blocks;
  int flag = 0;
  auto indexs = create_block_index(parts, flag);

  int current_col = 0; // Current column index in the values array
    for (int i = 0; i < num_blocks; ++i) {
        int current_block_size = (i == num_blocks - 1) ? col - current_col : block_size; // Adjust block size for the last block

        // Create a new block of size row x current_block_size
        std::vector<float> block(row * current_block_size);

        // Fill the block with values
        for (size_t r = 0; r < row; ++r) {
            for (int c = 0; c < current_block_size; ++c) {
                
                block[r * current_block_size + c] = 0.000001;
            }
        }
        input = maker.rowVector(
          {"w", "w_row", "w_col"},
          {maker.arrayVector<float>({block}, REAL()),
           maker.flatVector({indexs[0][i]}),
           maker.flatVector({indexs[1][i]})});

        auto file = TempFilePath::create();
      // Store blocks to file
        myFile.writeToFile(file->path, {input});
      // Store file object to paths
        paths.push_back(file);
       // Store the block in the vector of blocks
        current_col += current_block_size; // Move to the next column

    }

    myFileStructure.paths = paths;
    // Store schema
    myFileStructure.schema = asRowType(input->type());
    return myFileStructure;

}

FileStructure save_blocks_to_files(
    std::vector<std::vector<float>> valuesArray) {
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

} // namespace optimization