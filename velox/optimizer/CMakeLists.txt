# Copyright (c) 2025 ASU Cactus Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(SOURCES
  OptimizerTest.cpp
)

set(Torch_DIR "$HOME/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(Faiss REQUIRED)

include_directories(/home/velox/third_party/tokenizers-cpp/include)
include_directories(/home/velox/third_party/tokenizers-cpp/src)

add_executable(velox_op ${SOURCES})
target_link_libraries(
  velox_op
  ml_functions
)
#[[
add_executable(traversal_test TraversalTest.cpp)
target_link_libraries(
  traversal_test
  ml_functions
)



add_executable(torch2twolayer_test TorchNN2TwoLayerUDFRewriteActionTest.cpp)
target_link_libraries(
  torch2twolayer_test
  ml_functions
)

add_executable(twolayer2torch_test TwoLayerUDF2TorchNNRewriteActionTest.cpp)
target_link_libraries(
  twolayer2torch_test
  ml_functions
)

add_executable(multilayer2torch_test MultiLayerUDF2TorchNNRewriteActionTest.cpp)
target_link_libraries(
  multilayer2torch_test
  ml_functions
)

add_executable(multilayer2torchcuda_test MultiLayerUDF2TorchNNCUDARewriteActionTest.cpp)
target_link_libraries(
  multilayer2torchcuda_test
  ml_functions
)

add_executable(ml_decomp_pushdown_test MLDecompositionPushdownRewriteActionTest.cpp)
target_link_libraries(
  ml_decomp_pushdown_test
  ml_functions
)


add_executable(decisionforest2rel_test DecisionForestUDF2RelationRewriteActionTest.cpp)
target_link_libraries(
  decisionforest2rel_test
  ml_functions
)

add_executable(mul2joinagg_test Mul2JoinAggRewriteActionTest.cpp)
target_link_libraries(
  mul2joinagg_test
  ml_functions
)

add_executable(mul2joinagghorizontal_test Mul2JoinAggHorizontalRewriteActionTest.cpp)
target_link_libraries(
  mul2joinagghorizontal_test
  ml_functions
)

add_executable(matmuldense2sparse_test MatMulDense2SparseRewriteActionTest.cpp)
target_link_libraries(
  matmuldense2sparse_test
  ml_functions
)

add_executable(conv_test Conv2dActionTest.cpp)
target_link_libraries(
  conv_test
  ml_functions
)
]]
if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()
