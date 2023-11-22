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

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_PREFIX_PATH "$HOME/libtorch")

find_package(Torch REQUIRED)


add_executable(ml_functions_test tests/MLFunctionsTest.cpp)

target_link_libraries(
  ml_functions_test
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
  openblas
  ${TORCH_LIBRARIES}
)

add_executable(embedding_test tests/EmbeddingTest.cpp)
target_link_libraries(
  embedding_test
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
  openblas
  ${TORCH_LIBRARIES}
)

add_executable(two_tower_model_test tests/TwoTowerModelTest.cpp)
target_link_libraries(
  two_tower_model_test
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
  openblas
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(two_tower_model_pipeline_test tests/TowTowerModelPipelineTest.cpp)
target_link_libraries(
  two_tower_model_pipeline_test
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
  openblas
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

# TODO: add or remove this unit test
# add_executable(parquest_loader tests/TestParquetLoader.cpp)
# target_link_libraries(
#   parquest_loader
#   velox_dwio_parquet_reader
#   velox_exec_test_lib
#   velox_exec
#   velox_hive_connector
#   velox_link_libs
#   ${TEST_LINK_LIBS}
#   ${TORCH_LIBRARIES}
# )

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -march=corei7-avx")

