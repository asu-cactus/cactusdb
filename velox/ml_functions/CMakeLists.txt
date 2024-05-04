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
#set(CMAKE_PREFIX_PATH "$HOME/libtorch; $CONDA_PREFIX")
set(CMAKE_PREFIX_PATH "$HOME/libtorch")
set(CMAKE_PREFIX_PATH "$CONDA_PREFIX")
set(cpr_DIR "/home/ubuntu/cpr")

find_package(Torch REQUIRED)
find_package(xgboost REQUIRED)
find_package(cpr REQUIRED)
find_package(jsoncpp REQUIRED)

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
  velox_dwio_parquet_reader
  ${TORCH_LIBRARIES}
)

add_executable(nn_tests tests/NNTest.cpp)
target_link_libraries(
  nn_tests
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
  jsoncpp_lib
  openblas
  ${TORCH_LIBRARIES}
  cpr::cpr
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

add_executable(decision_forest_prediction_test tests/DecisionForestTest.cpp)
target_link_libraries(
  decision_forest_prediction_test
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

add_executable(xgboost_test tests/XGBoostTest.cpp)
target_link_libraries(
  xgboost_test
  xgboost::xgboost
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

add_executable(ml_sql_test tests/MLSQLTest.cpp)
target_link_libraries(
  ml_sql_test
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

add_executable(array_array_unnest_test tests/ArrayofArrayUnnestTest.cpp)
target_link_libraries(
  array_array_unnest_test
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

add_executable(ml_kernel_benchmark tests/MLKernelBenchmarkTest.cpp)
target_link_libraries(
  ml_kernel_benchmark
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
