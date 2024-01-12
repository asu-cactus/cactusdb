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

#include <memory>
#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/vector/tests/utils/VectorMaker.h"

using namespace facebook::velox;
using namespace facebook::velox::memory;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

namespace optimization {

class RewriteAction {

public:

    virtual bool apply( std::shared_ptr<const core::PlanNode> curNode,
		        std::shared_ptr<const core::PlanNode> prevNode,
		        VectorMaker & maker,
		        PlanBuilder & planBuilder,
		        std::shared_ptr<MemoryPool> pool_,
		        std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator ) = 0;

    virtual std::string name() = 0;



};


}
