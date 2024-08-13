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
#include "CataLog.h"

using namespace facebook::velox;
using namespace facebook::velox::memory;
using namespace facebook::velox::test;
using namespace facebook::velox::exec::test;

namespace optimization {

class RewriteAction {

public:
	/**
	 * @brief A virtual function to apply a rule for rewriting the logical plan.
	 * 
	 * @param curNode A pointer to the current plan node, usually point to the last node of logical plan.
	 * @param prevNode A pointer to the previous plan node, usually point to the previous node before current node.
	 * @param maker A pointer to the VectorMaker, which is a helper class used to build the data source vector.
	 * @param planBuilder A pointer to the planBuilder, which is a helper class used to build the logical plan.
	 * @param pool_ A pointer to the memory pool, which is used to build the logical plan.
	 * @param planNodeIdGenerator A pointer to the planNodeIdGenerator, which is used to track the ID of the plan Node.
	 * @param targets A vector for multiple strings, representing the target UDF name that can apply this rewritten rule.
	 * @param cataLog Reference to a CataLog object to store metadata and information.
	 * 
	 * @return A boolean value indicating whether the rewrite was successful.
	 * 
	*/
    virtual bool apply(std::shared_ptr<const core::PlanNode> curNode,
		        std::shared_ptr<const core::PlanNode> prevNode,
		        VectorMaker & maker,
		        PlanBuilder & planBuilder,
		        std::shared_ptr<memory::MemoryPool> pool_,
		        std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator, 
				std::vector<std::string> targets,
				CataLog &cataLog) = 0;

	/**
	 * @brief A virtual function to get the name of rewritten rule.
	 * 
	 * @return A string value denoting the name of the rule
	*/
    virtual std::string name() = 0;

	/**
	 * @brief A virtual function to check if this rule can be applied in a logical plan and to store the possible UDF name.
	 * 
	 * @param rootNode A pointer to the logical plan.
	 * @param targetActions A pointer to the vector used to store possible UDF names applicable for this rule.
	 * @param cataLog Reference to a CataLog object to store metadata and information.
	 * 
	 * @return A boolean value indicating whether the check was successful.
	*/
	virtual bool check(std::shared_ptr<const core::PlanNode> rootNode, std::vector<std::string> &targetActions, CataLog &cataLog) = 0;

};


}
