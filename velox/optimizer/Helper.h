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

    struct FileStructure {
        std::vector<std::shared_ptr<TempFilePath>> paths;
        RowTypePtr schema;
    };

    class MyFileTest : public HiveConnectorTestBase {
        public:
            MyFileTest(){
                // SetUp();
            }
            ~MyFileTest() {
            }

            void SetUp() {
                HiveConnectorTestBase::SetUp();
            }

            void TestBody() override {}

    };

    std::vector<std::vector<float>> create_block_index(int parts, int flag){
        std::vector<std::vector<float>> indexs;
        if (flag == 0){
            std::vector<float> indexs_row;
            std::vector<float> indexs_col;
            for (int i = 0; i < parts; i++){
            indexs_row.push_back(0);
            }
            for (int i = 0; i < parts; i++){
            indexs_col.push_back(i);
            }
            indexs.push_back(indexs_row);
            indexs.push_back(indexs_col);
        }
        else {
            std::vector<float> indexs_row;
            std::vector<float> indexs_col;
            for (int i = 0; i < parts; i++){
            indexs_row.push_back(i);
            }
            for (int i = 0; i < parts; i++){
            indexs_col.push_back(0);
            }
            indexs.push_back(indexs_row);
            indexs.push_back(indexs_col);
        }
        return indexs;
    }

    std::vector<std::vector<float>> create_input_block(int total_size, std::vector<std::vector<float>>& values, int block_numbers){
        std::vector<std::vector<float>> valuesArray;
        std::vector<float> flattened;
        for (const auto& row : values) {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }

        auto block_size = total_size / block_numbers;
        for (int i = 0; i < block_numbers; i++) {
            std::vector<float> valuesArraySingleBlock;
            for (int j = 0; j < block_size; j++) {
                    valuesArraySingleBlock.push_back(flattened[i*block_size+j]);
            }
            valuesArray.push_back(valuesArraySingleBlock);
        }
        return valuesArray;
    }


    std::vector<std::vector<float>> create_weight_block(int total_size, float* values, int block_numbers){
        std::vector<std::vector<float>> valuesArray;
        auto block_size = total_size / block_numbers;
        for (int i = 0; i < block_numbers; i++) {
            std::vector<float> valuesArraySingleBlock;
            for (int j = 0; j < block_size; j++) {
                    valuesArraySingleBlock.push_back(values[i*block_size+j]);
            }
            valuesArray.push_back(valuesArraySingleBlock);
        }
        return valuesArray;
    }

    FileStructure block_to_files(std::vector<std::vector<float>> valuesArray, int parts, int flag){
        optimization::MyFileTest myFile;
        optimization::FileStructure myFileStructure;
        std::vector<std::shared_ptr<TempFilePath>> paths;
        auto pool_ = memory::addDefaultLeafMemoryPool();
        VectorMaker maker{pool_.get()};
        RowVectorPtr input;
        if (flag == 0){
            auto indexs = create_block_index(parts, flag);
            for (int i = 0; i < parts; i++){
            input = maker.rowVector({"v", "v_row", "v_col"}, 
            {maker.arrayVector<float>({valuesArray[i]}, REAL()), maker.flatVector({indexs[0][i]}), maker.flatVector({indexs[1][i]})});
            auto file = TempFilePath::create();
            myFile.writeToFile(file->path, {input});
            paths.push_back(file);
        }
        myFileStructure.paths = paths;
        myFileStructure.schema = asRowType(input->type());
        return myFileStructure;
        }
        else {
            auto indexs = create_block_index(parts, flag);
            for (int i = 0; i < parts; i++){
            input = maker.rowVector({"w", "w_row", "w_col"}, 
            {maker.arrayVector<float>({valuesArray[i]}, REAL()), maker.flatVector({indexs[0][i]}), maker.flatVector({indexs[1][i]})});
            auto file = TempFilePath::create();
            myFile.writeToFile(file->path, {input});
            paths.push_back(file);
        }
        myFileStructure.paths = paths;
        myFileStructure.schema = asRowType(input->type());
        return myFileStructure;
        }

    }

}