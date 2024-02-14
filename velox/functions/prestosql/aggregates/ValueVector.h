/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once


#include "velox/exec/Aggregate.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"
#include <iostream>


namespace facebook::velox::aggregate {

// Represents a list of values, including nulls, for an array/map/distinct value
// set in aggregation. Bit-packed null flags are stored separately from the
// non-null values.
class ValueVector {
 public:
  // void appendValueFromCopy()
  void addValue(float* newValues, vector_size_t offset, vector_size_t size) {
    if (storedValue == nullptr){
      storedValue = new float[size];
      for (vector_size_t i = 0; i < size; ++i) {
        storedValue[i] = newValues[offset + i]; // Copy the elements from newValues
      }
      size_ = size; 
    }
    else {
      float* current_ptr = newValues + offset;
      for (vector_size_t i = 0; i < size; ++i) {
        storedValue[i] = storedValue[i] + current_ptr[i];
      }
    }
  }

  void insertValue(float* newValues, vector_size_t offset, vector_size_t size, float* newIndexs) {
    int index = static_cast<int>(newIndexs[0]);
    float* slicedValues = new float[size];
    std::copy(newValues + (offset*size), newValues + (offset*size) + size, slicedValues);

    insertedValue[index] = slicedValues;
    insertedSize[index] = size;
    if (size > block_size) { block_size = size;}
    size_ += size;
  }
  // void extractValues (FlatVector<float>& values, vector_size_t offset) {
  //   vector_size_t index = offset;
  //   for (vector_size_t i = 0; i < size_; ++i) {
  //     values.set(index++, storedValue[i]);
  //   }
  // }
    void extractValues (FlatVector<float>& values, vector_size_t index) {
        values.set(index, storedValue[index]);
    }

    void concatValues (FlatVector<float>& values, vector_size_t offset) {
      for (auto entry: insertedValue) {
        int indexs = entry.first;
        auto sizes = insertedSize[indexs];
        for (int i = 0; i< sizes; i++){
          values.set(offset + indexs * block_size + i, entry.second[i]);
        }
      }
    }

  void extractValues (FlatVector<float>& values, vector_size_t offset , vector_size_t length) {
    // vector_size_t index = offset;
    for (vector_size_t i = 0; i < length; ++i) {
      values.set(i + offset, storedValue[i + offset]);
    }
  }

  void free() {
    delete[] storedValue;
}

  int32_t size() const {
    return size_;
  }

  void print(size_t endPos = -1) {
    size_t end = size_;
    if (endPos != -1) {
      end = endPos;
    }
    std::cout << "[INFO] ValueVector: \n";
    for (size_t i = 0; i < end; i++){
      std::cout << storedValue[i] << ", ";
    }
    std::cout<<std::endl;
  }


 private:
  // An array_agg or related begins with an allocation of 5 words and
  // 4 bytes for header. This is compact for small arrays (up to 5
  // bigints) and efficient if needs to be extended (stores 4 bigints
  // and a next pointer. This could be adaptive, with smaller initial


  // Number of values added, including nulls.
  uint32_t size_{0};
  float* storedValue = nullptr;
  std::map<int, float*> insertedValue;
  std::map<int, vector_size_t> insertedSize;
  vector_size_t block_size{0};
  // Last nulls word. 'size_ % 64' is the null bit for the next element.

};

// Extracts values from the ValueList into provided vector.


} // namespace facebook::velox::aggregate
