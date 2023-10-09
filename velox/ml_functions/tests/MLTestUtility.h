#pragma once
#include <random>

// TODO: add namespace

class RandomGenerator {
 public:
  RandomGenerator(float lb, float ub, int randomSeed = 0) {
    gen_ = std::mt19937(randomSeed);
    distR_ = std::uniform_real_distribution<float>(lb, ub);
    distI_ = std::uniform_int_distribution<int>((int)lb, (int)ub);
  }

  void setFloatRange(float lb, float ub) {
    distR_ = std::uniform_real_distribution<float>(lb, ub);
  }

  void setIntRange(int lb, int ub) {
    // NOTE: ub is included when sampling
    distI_ = std::uniform_int_distribution<int>(lb, ub);
  }

  float genRandomFloatValue() {
    return distR_(gen_);
  }

  int genRandomIntValue() {
    return distI_(gen_);
  }

  std::vector<std::vector<int>>
  genLookUpIndices(int numRow, int maxIndexNumber, int maxVariadic = 1) {
    // max variadic is the maximum number of sampled indices for each data
    // point, in two-tower model, the value is 6 for genre
    setIntRange(0, maxIndexNumber);
    std::vector<std::vector<int>> indicesVectors;
    for (int i = 0; i < numRow; i++) {
      std::vector<int> sampledIndices;
      int numSampledIndices = 1;
      if (maxVariadic > 1) {
        numSampledIndices = (i % maxVariadic == 0) ? 1 : i % maxVariadic;
      }
      for (int j = 0; j < numSampledIndices; j++) {
        sampledIndices.push_back(genRandomIntValue());
      }
      indicesVectors.push_back(sampledIndices);
    }
    return indicesVectors;
  }

  std::vector<std::vector<float>> genFloat2dVector(int numRow, int numCol) {
    // Initialize the input1 feature vector
    std::vector<std::vector<float>> float2dVector;

    for (int i = 0; i < numRow; i++) {
      std::vector<float> floatVector;
      for (int j = 0; j < numCol; j++) {
        floatVector.push_back(genRandomFloatValue());
      }
      float2dVector.push_back(floatVector);
    }
    return float2dVector;
  }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<float> distR_;
  std::uniform_int_distribution<int> distI_;
};