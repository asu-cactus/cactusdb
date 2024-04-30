#pragma once
#include <random>

// TODO: add namespace

void countRowsAndColumnsFromCSV(const std::string& filename, int& numRows, int& numCols) {
    std::ifstream file(filename);
    std::string line;

    numRows = 0;
    numCols = 0;

    while (std::getline(file, line)) {
        ++numRows;
        std::stringstream ss(line);
        std::string cell;
        int currentCols = 0;
        while (std::getline(ss, cell, ',')) {
            ++currentCols;
        }
        if (currentCols > numCols) {
            numCols = currentCols;
        }
    }
}


class RandomGenerator {
 public:
  
  RandomGenerator() {
    gen_ = std::mt19937(0);
    distR_ = std::uniform_real_distribution<float>(0, 1);
    distI_ = std::uniform_int_distribution<int>((int)0, (int)10);
  }
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

  std::vector<int> gen1DInt(int numRow, int minIndexNumber, int maxIndexNumber) {
    // max variadic is the maximum number of sampled indices for each data
    // point, in two-tower model, the value is 6 for genre
    setIntRange(minIndexNumber, maxIndexNumber);
    std::vector<int> intVector;
    for (int i = 0; i < numRow; i++) {
      intVector.push_back(genRandomIntValue());
    }
    return intVector;
  }

  std::vector<std::vector<float>> genFloat2dVector(int numRow, int numCol) {
    // Initialize the input1 feature vector
    std::vector<std::vector<float>> float2dVector;

    for (int i = 0; i < numRow; i++) {
      float2dVector.push_back(std::move(genFloat1dVector(numCol)));
    }
    return float2dVector;
  }

  float* genFloat1dArray(int size) {
    // Initialize the input1 feature vector
    float* float1dArray = new float[size];

    for (int i = 0; i < size; i++) {
      float1dArray[i] = genRandomFloatValue();
    }
    return float1dArray;
  }

  std::vector<float> genFloat1dVector(int size) {
    // Initialize the input1 feature vector
    std::vector<float> float1dVector;
    for (int i = 0; i < size; i++) {
        float1dVector.push_back(genRandomFloatValue());
    }
    return float1dVector;
  }

  std::vector<int> genIntRange(int low, int high) {
    std::vector<int> result;
    for (int i = low; i < high; ++i) {
        result.push_back(i);
    }
    return result;
  }

 private:
  std::mt19937 gen_;
  std::uniform_real_distribution<float> distR_;
  std::uniform_int_distribution<int> distI_;
};