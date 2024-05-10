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
#include "velox/expression/VectorFunction.h"
#include <unordered_map>
#include "Helper.h"
#include "CataLog.h"



namespace optimization {
/**
 * @brief Update the CataLog based on the registered vector function.
 * 
 * @param name Name of the registered vector function.
 * @param sharedFunc Shared pointer to the registered vector function.
 * @param cataLog Reference to a CataLog object to store metadata and information.
 */
void updateCataLog(const std::string& name, std::shared_ptr<VectorFunction> sharedFunc, CataLog& cataLog, bool isVerticalPartition = false) {
    // TODO:Finish all branches, here we only complete mat_mul
    // Check if the registered vector function is a TorchDNN, here
    if (name.find("torchDNN") != std::string::npos) {
        // Dynamic cast to TorchDNN
        std::shared_ptr<TorchDNN> TorchUDF = std::dynamic_pointer_cast<TorchDNN>(sharedFunc);
    }
    // Check if the registered vector function is a MatrixMultiply (mat_mul)
    else if (name.find("mat_mul") != std::string::npos) {
        // Dynamic cast to MatrixMultiply
        std::shared_ptr<MatrixMultiply> MulUDF = std::dynamic_pointer_cast<MatrixMultiply>(sharedFunc);
        if (MulUDF) {
            // Retrieve dimensions and weights from MatrixMultiply UDF
            std::vector<int> dims = MulUDF->getDims();
            float* weights = MulUDF->getTensor();
            // Get blocking threshold and default blocks number from cataLog
            int blockingThreshold = cataLog.getBlockingThreshold();
            int blocksSize = cataLog.getDefaultBlocksSize();
            int blocksNum = cataLog.getDefaultBlocksNum();
            // Check if blocking is required based on dimensions
            if (dims[0] > blockingThreshold) {
                std::vector<std::vector<float>> weightBlocks;
                optimization::FileStructure weightsFileStructure;
                // Create weight blocks and convert to files
                std::string nameSuffix = "";
                if (isVerticalPartition) {
                    // in vertical partition approach: inputs are partitioned vertically
                    // while weights are partitioned horizontally
                    weightBlocks = create_weight_block(dims[0]*dims[1], weights, blocksNum);
                    weightsFileStructure = block_to_files(weightBlocks, blocksNum, 1);
                    nameSuffix = "_horizontal";
                } else {
                    // in horizontal partition approach: inputs are partitioned horizontally 
                    // by using Velox's batches, while weights are partitioned vertically
                    weightBlocks = create_blocks(dims[0], dims[1], std::move(weights), blocksSize);
                    weightsFileStructure = save_blocks_to_files(std::move(weightBlocks), name);
                    nameSuffix = "_vertical";
                    weightBlocks.clear();
                    weightBlocks.shrink_to_fit();
                }
                // Add the updated information to cataLog
                cataLog.add(name, weightsFileStructure.schema, weightsFileStructure.paths, 1, nameSuffix);
            }
        }
    }
    else {
        // TODO: need to change it back
        // std::cout << "INFO: " << name << ": No update for catalog." << std::endl;
    }
}
// We encapsulate the original registerFunction, adding a step to update the catalog when registering functions.
bool registerVectorFunction(
    const std::string& name,
    std::vector<FunctionSignaturePtr> signatures,
    std::unique_ptr<VectorFunction> func,
    VectorFunctionMetadata metadata,
    bool overwrite,
    CataLog& cataLog,
    bool isVerticalPartition = false) {
        
    std::shared_ptr<VectorFunction> sharedFunc = std::move(func);
    // Only added this update step 
    updateCataLog(name, sharedFunc, cataLog, isVerticalPartition);
    auto factory = [sharedFunc](
                        const auto& /*name*/,
                        const auto& /*vectorArg*/,
                        const auto& /*config*/) { return sharedFunc; };
    return facebook::velox::exec::registerStatefulVectorFunction(
        name, signatures, factory, metadata, overwrite);
}


}



