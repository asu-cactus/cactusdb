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
#include "velox/exec/ContainerRowSerde.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/AggregateNames.h"
#include "velox/functions/prestosql/aggregates/ValueVector.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

struct ArrayAccumulator {
  ValueVector elements;
};

class SumArrayAggregate : public exec::Aggregate {
 public:
  explicit SumArrayAggregate(TypePtr resultType) : Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(ArrayAccumulator);
  }

  bool isFixedSize() const override {
    return false;
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    for (auto index : indices) {
      new (groups[index] + offset_) ArrayAccumulator();
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<ArrayVector>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);

    auto elements = vector->elements()->as<FlatVector<float>>();
    elements->resize(1024000);//Todo auto

    uint64_t* rawNulls = getRawNulls(vector);
    vector_size_t offset = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto& values = value<ArrayAccumulator>(groups[i])->elements;
      auto arraySize = values.size();
      if (arraySize) {
        clearNull(rawNulls, i);

        // ValueListReader reader(values);
        // for (auto index = 0; index < arraySize; ++index) {
        //   reader.next(*elements, offset + index);
        // }
        values.extractValues(*elements, offset);
        vector->setOffsetAndSize(i, offset, arraySize);
        offset += arraySize;
      } else {
        vector->setNull(i, true);
      }
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    extractValues(groups, numGroups, result);
  }

  void addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /*mayPushdown*/) override {
  decodedElements_.decode(*args[0], rows);
  auto Values = decodedElements_.base()->as<ArrayVector>();
  float* Valuesfloat = Values->elements()->values()->asMutable<float>();
  rows.applyToSelected([&](vector_size_t row) {
    auto group = groups[row];
    auto tracker = trackRowSize(group);
    auto decodedRow = decodedElements_.index(row);
    auto rowOffset =  Values->offsetAt(decodedRow);
    auto rowSize = Values->sizeAt(decodedRow);
 
    auto& oldValues = value<ArrayAccumulator>(group)->elements;
    oldValues.addValue(Valuesfloat, rowOffset, rowSize);
    // auto newValues = decodedElements_.base()->as<ArrayVector>();
    // if (oldValues.size() == 0) {
    //   oldValues.appendValue(decodedElements_, row, allocator_);//index is row old.addvalue(newvalue[index,offsetsize])
    // } else {
    //   auto size = oldValues.size();
    //   const auto newValues = decodedElements_.base()->as<ArrayVector>();

    //   float* input_values_new = newValues->elements()->values()->asMutable<float>();
    //   // auto resultVector = BaseVector::create(args[0].type(), 0, allocator_);
    //   // auto resultVector = args[0]->as<ArrayVector>()->elements();
    //   // auto resultVector = BaseVector::create(newValues->type(), 0, allocator_);


    //   // for (auto index = 0; index < size; ++index) {
    //   //   valueListReader.next(*resultVector, index);
    //   // }
    //   // auto ss = resultVector->values()->asMutable<float>();
    //   // for (vector_size_t i = 0; i < size; ++i) {
    //   //   // auto newv = newValues->as<float>()[i];
    //   //   // resultVector->values()->asMutable<float>()[i] += newv;
    //   // }

    // }
  });
}


  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    auto arrayVector = decodedIntermediate_.base()->as<ArrayVector>();
    auto& elements = arrayVector->elements();
    float* Valuesfloat = arrayVector->elements()->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
      auto group = groups[row];
      auto decodedRow = decodedIntermediate_.index(row);
      auto tracker = trackRowSize(group);
      auto rowOffset =  arrayVector->offsetAt(decodedRow);
      auto rowSize = arrayVector->sizeAt(decodedRow);
      auto& values = value<ArrayAccumulator>(group)->elements;

      values.addValue(Valuesfloat, rowOffset, rowSize);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    auto& values = value<ArrayAccumulator>(group)->elements;

    decodedElements_.decode(*args[0], rows);
    auto tracker = trackRowSize(group);
    auto Values = decodedElements_.base()->as<ArrayVector>();
    float* Valuesfloat = Values->elements()->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
    auto decodedRow = decodedElements_.index(row);
    auto rowOffset =  Values->offsetAt(decodedRow);
    auto rowSize = Values->sizeAt(decodedRow);
 

    values.addValue(Valuesfloat, rowOffset, rowSize);
    });
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedIntermediate_.decode(*args[0], rows);
    auto arrayVector = decodedIntermediate_.base()->as<ArrayVector>();

    auto& values = value<ArrayAccumulator>(group)->elements;
    float* Valuesfloat = arrayVector->elements()->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
      auto decodedRow = decodedIntermediate_.index(row);
      auto rowOffset =  arrayVector->offsetAt(decodedRow);
      auto rowSize = arrayVector->sizeAt(decodedRow);
      values.addValue(Valuesfloat, rowOffset, rowSize);
    });
  }

  void destroy(folly::Range<char**> groups) override {
    for (auto group : groups) {
      value<ArrayAccumulator>(group)->elements.free();
    }
  }

 private:
  vector_size_t countElements(char** groups, int32_t numGroups) const {
    vector_size_t size = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      size += value<ArrayAccumulator>(groups[i])->elements.size();
    }
    return size;
  }

  // Reusable instance of DecodedVector for decoding input vectors.
  DecodedVector decodedElements_;
  DecodedVector decodedIntermediate_;
};

bool registerSumArray(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("array(REAL)")
          .intermediateType("array(double)")
          .argumentType("array(REAL)")
          .build()};
          // .typeVariable("E")
          // .returnType("array(E)")
          // .intermediateType("array(E)")
          // .argumentType("E")
          // .build()};

  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 1, "{} takes at most one argument", name);
        return std::make_unique<SumArrayAggregate>(resultType);
      });
  return true;
}

} // namespace

void registerSumArrayAggregate(const std::string& prefix) {
  registerSumArray(prefix + kSumArray);
}

} // namespace facebook::velox::aggregate::prestosql
