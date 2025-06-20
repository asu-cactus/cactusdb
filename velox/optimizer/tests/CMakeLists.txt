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
add_definitions(-DC10_USE_GLOG)


include_directories(/usr/include/hdf5/serial)
include_directories(/home/h5cpp/build/src/h5cpp)

find_package(h5cpp REQUIRED)
find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Faiss REQUIRED)

include_directories(/home/velox/third_party/tokenizers-cpp/include)
include_directories(/home/velox/third_party/tokenizers-cpp/src)

add_library(tokenizer_cpp STATIC IMPORTED)
add_library(tokenizer_c STATIC IMPORTED)
set_target_properties(tokenizer_cpp PROPERTIES IMPORTED_LOCATION /home/velox/third_party/tokenizers-cpp/example/build/tokenizers/libtokenizers_cpp.a)
set_target_properties(tokenizer_c PROPERTIES IMPORTED_LOCATION /home/velox/third_party/tokenizers-cpp/example/build/tokenizers/libtokenizers_c.a)

link_directories(${HDF5_LIBRARY_DIRS})
include_directories(${HDF5_INCLUDE_DIRS})

add_executable(ablation_study_test AblationStudyTest.cpp)

target_link_libraries(
  ablation_study_test
  ml_functions
  OpenMP::OpenMP_CXX
)

add_dependencies(ablation_study_test ml_functions)

add_executable(profile_query_generator ProfileQueryGenerator.cpp)
target_link_libraries(
  profile_query_generator
  ml_functions
)

add_executable(model_query_profiler ModelQueryProfiler.cpp)
target_link_libraries(
  model_query_profiler
  ml_functions
)

add_executable(reusable_mcts_test ReusableMCTSTest.cpp)
target_link_libraries(
  reusable_mcts_test
  ml_functions
)

add_executable(factorization_test FactorizeTest.cpp)
target_link_libraries(
  factorization_test
  ml_functions
)
