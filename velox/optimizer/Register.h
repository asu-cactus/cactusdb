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
void updateCataLog(const std::string& name, std::shared_ptr<VectorFunction> sharedFunc, CataLog& cataLog) {
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
            int blocksNum = cataLog.getDefaultBlocksNum();
            // Check if blocking is required based on dimensions
            if (dims[0] > blockingThreshold) {
                // Create weight blocks and convert to files
                auto weightBlocks = create_weight_block(dims[0]*dims[1], weights, blocksNum);
                auto weights = block_to_files(weightBlocks, blocksNum, 1);
                // Add the updated information to cataLog
                cataLog.add(name, weights.schema, weights.paths, 1);
            }
        }
    }
    else {
        std::cout << "Warning: " << name << ": No update for catalog." << std::endl;
    }
}
// We encapsulate the original registerFunction, adding a step to update the catalog when registering functions.
bool registerVectorFunction(
    const std::string& name,
    std::vector<FunctionSignaturePtr> signatures,
    std::unique_ptr<VectorFunction> func,
    VectorFunctionMetadata metadata,
    bool overwrite,
    CataLog& cataLog) {
        
    std::shared_ptr<VectorFunction> sharedFunc = std::move(func);
    // Only added this update step 
    updateCataLog(name, sharedFunc, cataLog);
    auto factory = [sharedFunc](
                        const auto& /*name*/,
                        const auto& /*vectorArg*/,
                        const auto& /*config*/) { return sharedFunc; };
    return facebook::velox::exec::registerStatefulVectorFunction(
        name, signatures, factory, metadata, overwrite);
}


}



