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
# find_package(Torch REQUIRED)
set(Torch_DIR "$HOME/libtorch")

# set(Torch_DIR "/home/local/ASUAD/qlin36/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(nn_test tests/NNTests.cpp)
target_link_libraries(
  nn_test
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