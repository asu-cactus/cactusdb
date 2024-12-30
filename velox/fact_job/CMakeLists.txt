cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(Torch_DIR "$HOME/libtorch")


find_package(Torch REQUIRED)
find_package(jsoncpp REQUIRED)
add_executable(job_test tests/JobBenchmark.cpp)
target_link_libraries(
  job_test
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
  jsoncpp_lib
)
