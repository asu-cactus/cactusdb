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
set(CMAKE_PREFIX_PATH "$HOME/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "$CONDA_PREFIX")
# include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
# set(TORCH_LIBRARIES "$HOME/libtorch")

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

# include tokenizer cpp as a sub directory



include_directories(/home/velox/third_party/tokenizers-cpp/include)
include_directories(/home/velox/third_party/tokenizers-cpp/src)

add_library(tokenizer_cpp STATIC IMPORTED)
add_library(tokenizer_c STATIC IMPORTED)
set_target_properties(tokenizer_cpp PROPERTIES IMPORTED_LOCATION /home/velox/third_party/tokenizers-cpp/example/build/tokenizers/libtokenizers_cpp.a)
set_target_properties(tokenizer_c PROPERTIES IMPORTED_LOCATION /home/velox/third_party/tokenizers-cpp/example/build/tokenizers/libtokenizers_c.a)


# TODO: temporary disable it until we can fix the build of dependency
# add_executable(standalone_hf_tokenizer_test tests/StandaloneHFTokenizerTest.cpp)
# target_link_libraries(standalone_hf_tokenizer_test 
# # FIXME:  for some reason tokenizer cpp needs to be placed before tokenizer c, 
# # needs fix to automatically compile it from 3rd library
# tokenizer_cpp
# tokenizer_c 
# )

# add_executable(hftokenizer_Test tests/HFTokenizerTest.cpp)
# target_link_libraries(
#   hftokenizer_Test
#   velox_aggregates
#   velox_type
#   velox_vector
#   velox_vector_test_lib
#   velox_exec
#   velox_exec_test_lib
#   velox_tpch_connector
#   velox_memory
#   velox_common_base 
#   velox_vector_fuzzer
#   jsoncpp_lib
#   openblas
#   ${TORCH_LIBRARIES}
#   cpr::cpr
#   tokenizer_cpp
#   tokenizer_c
# )


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

add_executable(two_tower_model_pipeline_test tests/TwoTowerModelPipelineTest.cpp)
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

include_directories("/usr/include/hdf5/serial")
include_directories("/home/h5cpp/build/src/h5cpp")

find_package(h5cpp REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)

link_directories( ${HDF5_LIBRARY_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )

add_executable(fraud_detection_test tests/FraudDetectionTest.cpp)
target_link_libraries(
  fraud_detection_test
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
  h5cpp::h5cpp
  hdf5_serial
  ${HDF5_CXX_LIBRARIES}
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

add_executable(duckdb_optimizer_test tests/DuckDBOptimizerTest.cpp)
target_link_libraries(
  duckdb_optimizer_test
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

add_executable(serialize_test tests/SerializeTest.cpp)
target_link_libraries(
  serialize_test
  velox_aggregates
  velox_type
  velox_vector_test_lib
  velox_exec
  velox_exec_test_lib
  velox_tpch_connector
  velox_memory
  velox_vector_fuzzer
  velox_dwio_parquet_reader
  velox_serialization
  velox_hive_connector
  velox_file
  velox_core
  velox_connector
  velox_time
  velox_codegen
  velox_test_util
  velox_arrow_bridge
  velox_common_compression
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

add_executable(chatgpt_test tests/ChatGPTTest.cpp)
target_link_libraries(
  chatgpt_test
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
  jsoncpp_lib
  openblas
  ${TORCH_LIBRARIES}
  cpr::cpr
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
