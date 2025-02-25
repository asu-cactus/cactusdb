# Copyright (c) 2025 ASU Cactus Lab.
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
set(Torch_DIR "$HOME/libtorch")


find_package(Torch REQUIRED)

include_directories(/home/velox/third_party/tokenizers-cpp/include)
include_directories(/home/velox/third_party/tokenizers-cpp/src)

add_library(tokenizer_cpp STATIC IMPORTED)
add_library(tokenizer_c STATIC IMPORTED)
set_target_properties(tokenizer_cpp PROPERTIES IMPORTED_LOCATION /home/velox/third_party/tokenizers-cpp/example/build/tokenizers/libtokenizers_cpp.a)
set_target_properties(tokenizer_c PROPERTIES IMPORTED_LOCATION /home/velox/third_party/tokenizers-cpp/example/build/tokenizers/libtokenizers_c.a)



add_executable(nn_test tests/NNTests.cpp)
target_link_libraries(
  nn_test
  ml_functions
)
