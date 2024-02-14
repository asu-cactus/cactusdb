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

  /* addRawInput is used for SingleAggregation and PartialAggregation which process 
  *  the input vector and store the values in the 
  */
  void addRawInput(
    char** groups,
    const SelectivityVector& rows,
    const std::vector<VectorPtr>& args,
    bool /*mayPushdown*/) override {
  decodedElements_.decode(*args[0], rows);
  auto Values = decodedElements_.base()->as<ArrayVector>();
  float* Valuesfloat = Values->elements()->values()->asMutable<float>();
  auto numSamples = Values->size();
  auto numElements = Values->elements()->size();
  numSamples_ = numSamples;
  numCols_ = numElements / numSamples;
  // std::cout << fmt::format("[INFO addRawInput] numSample: {}, numCols: {}, rows size: {}\n", numSamples_, numCols_, rows.size());
  rows.applyToSelected([&](vector_size_t row) {
    auto group = groups[row];
    auto tracker = trackRowSize(group);
    auto decodedRow = decodedElements_.index(row);
    auto rowOffset =  Values->offsetAt(decodedRow);
    auto rowSize = Values->sizeAt(decodedRow);
    // Change row size to the whole block size
    rowSize = numSamples_*numCols_;
 
    auto& oldValues = value<ArrayAccumulator>(group)->elements;
    oldValues.addValue(Valuesfloat, rowOffset, rowSize);
  });
}
  // This function is invoked by intermediateAggregation and finalAggregation
  // which takes the inputs from the partialAggregation and add the results into
  // the accumulator
void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedIntermediate_.decode(*args[0], rows);

    auto arrayVector = decodedIntermediate_.base()->as<ArrayVector>();
    auto& elements = arrayVector->elements();
    float* Valuesfloat = arrayVector->elements()->values()->asMutable<float>();
    auto numElements = arrayVector->elements()->size();
    numSamples_ = arrayVector->size();
    numCols_ = numElements/numSamples_;
    // there will be only one group after aggregation if we partition the input
    // features vertically and weight matrix horizontally, so all result values
    // should be stored in group 0.
    vector_size_t groupIdx = 0;
    auto group = groups[groupIdx];
    auto decodedRow = decodedIntermediate_.index(groupIdx);
    auto tracker = trackRowSize(group);
    auto rowOffset =  arrayVector->offsetAt(decodedRow);
    auto rowSize = arrayVector->sizeAt(decodedRow);
    auto& values = value<ArrayAccumulator>(group)->elements;
    rowSize = numSamples_*numCols_;
    values.addValue(Valuesfloat, rowOffset, rowSize);
  }

  // This function is invoked by singleAggregation or finalAggregation to
  // produce the final results from the accumulator, where addRawInput and
  // addIntermediateResults store the aggregated values.
  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<ArrayVector>();
    VELOX_CHECK(vector);
    // Unblocking: set the number of groups equals the sample size
    vector->resize(numSamples_);
    auto elements = vector->elements()->as<FlatVector<float>>();
    // Resize the FlatVector length equals to the whole block size
    elements->resize(numSamples_*numCols_);

    uint64_t* rawNulls = getRawNulls(vector);
      // Only one group eventually and the aggregated results are stored in group 0
      auto& aggregatedValues = value<ArrayAccumulator>(groups[0])->elements;
      vector_size_t offset = 0;
      for (int32_t i = 0; i < numSamples_; i++) {
        if (numCols_) {
          aggregatedValues.extractValues(*elements, offset, numCols_);
          vector->setOffsetAndSize(i, offset, numCols_);
          offset += numCols_;
        } else {
          vector->setNull(i, true);
        }
      }
  }

  // This function is invoked for partialAggregation and intermediateAggregation
  // to produce the partial aggregated results that will be used for the next
  // aggregation.
  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {

    // std::cout << fmt::format("[INFO extractAccumulators] : numSamples: {}, numCols: {} numGroups: {}\n", numSamples_, numCols_, numGroups);
    auto vector = (*result)->as<ArrayVector>();
    vector->resize(numSamples_);
    auto elements = vector->elements()->as<FlatVector<float>>();
    elements->resize(numSamples_*numCols_);
    
    // all aggregated values are stored in group 0
    auto aggregatedValues = value<ArrayAccumulator>(groups[0])->elements;
    vector_size_t offset = 0;

    for (int32_t i = 0; i < numSamples_; i++) {
      if (numCols_) {
        aggregatedValues.extractValues(*elements, offset, numCols_);
        vector->setOffsetAndSize(i, offset, numCols_);
        offset += numCols_;
      } else {
        vector->setNull(i, true);
      }
    }
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
    // std::cout << fmt::format("addSingleGroupRawInput:  Values \n {} \n {}\n", Values->toString(), Values->toString(0, Values->size()));
    float* Valuesfloat = Values->elements()->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
    auto decodedRow = decodedElements_.index(row);
    auto rowOffset =  Values->offsetAt(decodedRow);
    auto rowSize = Values->sizeAt(decodedRow);
 
    rowSize = numSamples_*numCols_;
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
    // std::cout << fmt::format("addSingleGroupIntermediateResults:  Values \n {} \n {}\n", arrayVector->toString(), arrayVector->toString(0, arrayVector->size()));
    auto& values = value<ArrayAccumulator>(group)->elements;
    float* Valuesfloat = arrayVector->elements()->values()->asMutable<float>();
    rows.applyToSelected([&](vector_size_t row) {
      auto decodedRow = decodedIntermediate_.index(row);
      auto rowOffset =  arrayVector->offsetAt(decodedRow);
      auto rowSize = arrayVector->sizeAt(decodedRow);
      rowSize = numSamples_*numCols_;
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
  size_t numSamples_;
  size_t numCols_;
};

bool registerSumArray(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("array(REAL)")
          .intermediateType("array(REAL)")
          .argumentType("array(REAL)")
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
