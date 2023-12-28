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
include_directories(/home/linqi/velox/velox)
set(SOURCES
  OptimizerTest.cpp
)

# set (CMAKE_PREFIX_PATH "$HOME/libtorch")
set(Torch_DIR "/home/local/ASUAD/qlin36/libtorch/share/cmake/Torch")
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
# set (CMAKE_CXX_FLAGS "-fopenmp -march=corei7-avx")