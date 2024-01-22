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
// #include "folly/Singleton.h"
// #include "folly/Synchronized.h"
// #include "velox/expression/SignatureBinder.h"


namespace optimization {

void updateCataLog(const std::string& name, std::shared_ptr<VectorFunction> sharedFunc, CataLog& cataLog) {
    if (name.find("torchDNN") != std::string::npos) {
            std::shared_ptr<TorchDNN> TorchUDF = std::dynamic_pointer_cast<TorchDNN>(sharedFunc);
        }
    else if (name.find("mat_mul") != std::string::npos) {
        std::shared_ptr<MatrixMultiply> MulUDF = std::dynamic_pointer_cast<MatrixMultiply>(sharedFunc);
        std::vector<int> dims = MulUDF->getDims();
		float* weights = MulUDF->getTensor();
        int blockingThreshold = cataLog.getBlockingThreshold();
        int blocksNum = cataLog.getDefaultBlocksNum();
        if (dims[0] > blockingThreshold) {
            auto weightBlocks = create_weight_block(dims[0]*dims[1], weights, blocksNum);
            auto weights = block_to_files(weightBlocks, blocksNum, 1);
            cataLog.add(name, weights.schema, weights.paths, 1);
        }
    }
    else {
        std::cerr << "Error: No update for mat_mul catalog: "  << std::endl;
    }
}
// Returns true iff an insertion actually happened
bool registerVectorFunction(
    const std::string& name,
    std::vector<FunctionSignaturePtr> signatures,
    std::unique_ptr<VectorFunction> func,
    VectorFunctionMetadata metadata,
    bool overwrite,
    CataLog& cataLog) {
        
  std::shared_ptr<VectorFunction> sharedFunc = std::move(func);
  updateCataLog(name, sharedFunc, cataLog);
  auto factory = [sharedFunc](
                     const auto& /*name*/,
                     const auto& /*vectorArg*/,
                     const auto& /*config*/) { return sharedFunc; };
  return facebook::velox::exec::registerStatefulVectorFunction(
      name, signatures, factory, metadata, overwrite);
}


}



