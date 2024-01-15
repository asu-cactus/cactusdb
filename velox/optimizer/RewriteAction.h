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

    virtual bool apply(std::shared_ptr<const core::PlanNode> curNode,
		        std::shared_ptr<const core::PlanNode> prevNode,
		        VectorMaker & maker,
		        PlanBuilder & planBuilder,
		        std::shared_ptr<memory::MemoryPool> pool_,
		        std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, 
				std::vector<std::string> targets) = 0;

<<<<<<< HEAD
=======

    virtual std::string name() = 0;
>>>>>>> 6286b5dc17740428b22b8396623779d5c45d184f

    virtual std::string name() = 0;
	// rootNode is the root node of the original plan;
	// targetAction is the vector of possible rewritten UDF name (such as torchdnn0)
	// check function will search the whole plan start by rootNode, storing the applicable UDF name that can apply such a rule at targetActions
	virtual bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> &targetActions) = 0;

};


}
