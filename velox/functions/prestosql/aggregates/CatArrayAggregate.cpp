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
#include <iostream>
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

class CatArrayAggregate : public exec::Aggregate {
 public:
  explicit CatArrayAggregate(TypePtr resultType) : Aggregate(resultType) {}

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
    auto originElements = vector->elements();
    auto elementsFloat = originElements->as<FlatVector<float>>();
    elementsFloat->resize(countElements(groups, numGroups));

    uint64_t* rawNulls = getRawNulls(vector);
    vector_size_t offset = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      auto& values = value<ArrayAccumulator>(groups[i])->elements;
      auto arraySize = values.size();
      if (arraySize) {
        clearNull(rawNulls, i);

        vector->setOffsetAndSize(i, offset, arraySize);
        values.extractAndConcatValue(*elementsFloat, offset);
        offset += arraySize;
      } else {
        vector->setNull(i, true);
      }
    }
  }

  // In cases where the block size cannot perfectly match the tensor shape in
  // the second dimension and the final size of the output tensor can only be
  // determined at the final aggregation stage, the intermediate aggregation
  // results are stored in a map structure. The block ID and block's values
  // serve as a key-value pair in the map for each gorup.
  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    // extractValues(groups, numGroups, result);
    auto mapVector = (*result)->as<MapVector>();
    VELOX_CHECK(mapVector);
    mapVector->resize(numGroups);

    auto mapKeys = mapVector->mapKeys();
    auto mapValueArrays = mapVector->mapValues()->as<ArrayVector>();
    auto numKeys = countBlocks(groups, numGroups);
    mapKeys->resize(numKeys);
    mapValueArrays->resize(numKeys);

    auto mapKeysVector = mapKeys->asFlatVector<float>();

    auto mapValuesElements = mapValueArrays->elements();
    mapValuesElements->resize(countElements(groups, numGroups));

    auto* rawNulls = getRawNulls(mapVector);
    vector_size_t keyOffset = 0;
    vector_size_t valueOffset = 0;

    for (int32_t i = 0; i < numGroups; i++) {
      char* group = groups[i];
      auto accumulatorValues = value<ArrayAccumulator>(group)->elements;
      auto numBlocks = accumulatorValues.numBlocks();
      auto arraySize = accumulatorValues.size();
      if (arraySize) {
        mapVector->setOffsetAndSize(i, keyOffset, numBlocks);
        accumulatorValues.extractIntermediate(
            mapKeys, *mapValueArrays, keyOffset, valueOffset);
      } else {
        mapVector->setOffsetAndSize(i, 0, 0);
      }
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedElements_.decode(*args[0], rows);
    auto values = decodedElements_.base()->as<ArrayVector>();
    float* valuesFloat = values->elements()->values()->asMutable<float>();

    decodedIndexes.decode(*args[1], rows);
    auto indexes =
        decodedIndexes.base()->asFlatVector<float>(); // intermediate result
                                                      // changed to dict, TODO
    float* indexesFloat = indexes->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
      auto group = groups[row];
      auto tracker = trackRowSize(group);
      // auto decodedRow = decodedElements_.index(row);
      auto rowOffset = values->offsetAt(row);
      auto rowSize = values->sizeAt(row);

      auto& oldValues = value<ArrayAccumulator>(group)->elements;
      oldValues.insertValue(
          valuesFloat, rowOffset, rowSize, decodedIndexes.valueAt<float>(row));
    });
  }

  // In cases where the block size cannot perfectly match the tensor shape in
  // the second dimension and the final size of the output tensor can only be
  // determined at the final aggregation stage, the intermediate aggregation
  // results are stored in a map structure. The block ID and block's values
  // serve as a key-value pair in the map for each gorup.
  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedMaps_.decode(*args[0], rows);
    auto mapVector = decodedMaps_.base()->template as<MapVector>();

    decodedKeys_.decode(*mapVector->mapKeys());
    decodedValueArrays_.decode(*mapVector->mapValues());

    auto* valueArrays_ = decodedValueArrays_.base()->template as<ArrayVector>();
    decodedValues_.decode(*valueArrays_->elements());

    float* valuesFloat = valueArrays_->elements()->values()->asMutable<float>();

    rows.applyToSelected([&](vector_size_t row) {
      auto group = groups[row];
      auto& accumulatorElements = value<ArrayAccumulator>(group)->elements;
      clearNull(group);
      auto decodedRow = decodedMaps_.index(row);
      auto offset = mapVector->offsetAt(decodedRow);
      auto size = mapVector->sizeAt(decodedRow);

      for (auto i = offset; i < offset + size; i++) {
        auto rowKey = decodedKeys_.valueAt<float>(i);
        auto numValues = valueArrays_->sizeAt(decodedValueArrays_.index(i));
        auto valueOffset = valueArrays_->offsetAt(decodedValueArrays_.index(i));
        accumulatorElements.insertValue(
            valuesFloat, valueOffset, numValues, rowKey);
      }

      auto decodedKeyRow = decodedKeys_.index(row);
      auto keySize = decodedKeys_.valueAt<float>(decodedKeyRow);
    });
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    auto& groupElements = value<ArrayAccumulator>(group)->elements;

    decodedElements_.decode(*args[0], rows);
    auto tracker = trackRowSize(group);
    auto values = decodedElements_.base()->as<ArrayVector>();
    float* valuesFloat = values->elements()->values()->asMutable<float>();

    decodedElements_.decode(*args[1], rows);
    auto indexes = decodedElements_.base()->asFlatVector<float>();
    float* indexesFloat = indexes->values()->asMutable<float>();

    rows.applyToSelected([&](vector_size_t row) {
      auto decodedRow = decodedElements_.index(row);
      auto rowOffset = values->offsetAt(decodedRow);
      auto rowSize = values->sizeAt(decodedRow);

      groupElements.insertValue(
          valuesFloat, row, rowSize, decodedIndexes.valueAt<float>(row));
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
    float* valuesFloat = arrayVector->elements()->values()->asMutable<float>();

    decodedIntermediate_.decode(*args[1], rows);
    auto indexes = decodedIntermediate_.base()->asFlatVector<float>();
    float* indexesFloat = indexes->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
      auto decodedRow = decodedIntermediate_.index(row);
      auto rowOffset = arrayVector->offsetAt(decodedRow);
      auto rowSize = arrayVector->sizeAt(decodedRow);
      values.insertValue(
          valuesFloat, row, rowSize, decodedIndexes.valueAt<float>(row));
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

  vector_size_t countBlocks(char** groups, int32_t numGroups) const {
    vector_size_t size = 0;
    for (int32_t i = 0; i < numGroups; ++i) {
      size += value<ArrayAccumulator>(groups[i])->elements.numBlocks();
    }
    return size;
  }

  // Reusable instance of DecodedVector for decoding input vectors.
  DecodedVector decodedElements_;
  DecodedVector decodedIndexes;
  DecodedVector decodedIntermediate_;

  DecodedVector decodedKeys_;
  DecodedVector decodedValues_;
  DecodedVector decodedMaps_;
  DecodedVector decodedValueArrays_;
};

bool registerCatArray(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .argumentType("array(REAL)")
          .argumentType("REAL")
          .intermediateType("map(REAL,array(REAL))")
          .returnType("array(REAL)")
          .build()};

  exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType,
          const core::QueryConfig&) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(
            argTypes.size(), 2, "{} takes at most one argument", name);
        return std::make_unique<CatArrayAggregate>(resultType);
      });
  return true;
}

} // namespace

void registerCatArrayAggregate(const std::string& prefix) {
  registerCatArray(prefix + kCatArray);
}

} // namespace facebook::velox::aggregate::prestosql
