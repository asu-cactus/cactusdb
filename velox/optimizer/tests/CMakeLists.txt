add_executable(mcts_test MCTSTest.cpp)
add_executable(profile_query_generator ProfileQueryGenerator.cpp)
add_executable(reusable_mcts_test ReusableMCTSTest.cpp)

add_definitions(-DC10_USE_GLOG)

include_directories("/usr/include/hdf5/serial")
include_directories("/home/h5cpp/build/src/h5cpp")

find_package(h5cpp REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
link_directories( ${HDF5_LIBRARY_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )

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
    cpr::cpr
    h5cpp::h5cpp
    hdf5_serial
    ${HDF5_CXX_LIBRARIES}
)

target_link_libraries(
    profile_query_generator
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
    cpr::cpr
    h5cpp::h5cpp
    hdf5_serial
    ${HDF5_CXX_LIBRARIES}
)

target_link_libraries(
    reusable_mcts_test
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
    cpr::cpr
    h5cpp::h5cpp
    hdf5_serial
    ${HDF5_CXX_LIBRARIES}
)