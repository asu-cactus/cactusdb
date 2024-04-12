# Copyright (c) Facebook, Inc. and its affiliates.
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

#include_directories(/home/linqi/velox/velox)
set(SOURCES
  OptimizerTest.cpp
)

set(Torch_DIR "$HOME/libtorch/share/cmake/Torch")
# set(Torch_DIR "/home/local/ASUAD/qlin36/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

find_package(jsoncpp REQUIRED)


add_executable(velox_op ${SOURCES})
target_link_libraries(
  velox_op
  velox_type
  velox_vector
  velox_exec
  velox_exec_test_lib
  velox_memory
  velox_vector_test_lib
  pthread
  jsoncpp_lib
  # openblas
  ${TORCH_LIBRARIES}
)

add_executable(traversal_test TraversalTest.cpp)
target_link_libraries(
  traversal_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)



add_executable(torch2twolayer_test TorchNN2TwoLayerUDFRewriteActionTest.cpp)
target_link_libraries(
  torch2twolayer_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(twolayer2torch_test TwoLayerUDF2TorchNNRewriteActionTest.cpp)
target_link_libraries(
  twolayer2torch_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(multilayer2torch_test MultiLayerUDF2TorchNNRewriteActionTest.cpp)
target_link_libraries(
  multilayer2torch_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)


add_executable(decisionforest2rel_test DecisionForestUDF2RelationRewriteActionTest.cpp)
target_link_libraries(
  decisionforest2rel_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(mul2joinagg_test Mul2JoinAggRewriteActionTest.cpp)
target_link_libraries(
  mul2joinagg_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(mul2joinagghorizontal_test Mul2JoinAggHorizontalRewriteActionTest.cpp)
target_link_libraries(
  mul2joinagghorizontal_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(conv_test Conv2dActionTest.cpp)
target_link_libraries(
  conv_test
  velox_aggregates
  velox_type
  velox_vector
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_common_base
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()