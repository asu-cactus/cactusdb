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


include_directories("/usr/include/hdf5/serial")
include_directories("/home/h5cpp/build/src/h5cpp")

find_package(h5cpp REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)

link_directories( ${HDF5_LIBRARY_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )


add_executable(factorize_test tests/RewriteFactorized.cpp)
target_link_libraries(
  factorize_test
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
  h5cpp::h5cpp
  hdf5_serial
  ${HDF5_CXX_LIBRARIES}
)

