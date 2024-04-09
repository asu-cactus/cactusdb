add_executable(mcts_test MCTSTest.cpp)

add_definitions(-DC10_USE_GLOG)

target_link_libraries(
    mcts_test
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
    ${TORCH_LIBRARIES}
)