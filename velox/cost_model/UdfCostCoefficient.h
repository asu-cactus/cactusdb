/*
 * Copyright (c) 2025 ASU Cactus Lab.
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
#include <glog/logging.h>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

// can also call init(filename) followed by getInstance()
// to initialise the map based on filename param
class UdfCostCoefficient {
 private:
  UdfCostCoefficient() {
    loadCoefficients();
  }

  ~UdfCostCoefficient() {}

  void loadCoefficients() {
    const char* homePath = std::getenv("HOME");
    std::string udfCoefficientPath = "";
    if (homePath) {
      udfCoefficientPath = std::string(homePath) +
          "/velox/resources/data/udf_coefficient_aws.txt";
    } else {
      udfCoefficientPath = "/home/velox/resources/data/udf_coefficient_aws.txt";
    }
    std::ifstream inputFile(udfCoefficientPath);

    if (!inputFile.is_open()) {
      std::cerr << "Error opening the file." << std::endl;
      return;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
      size_t pos = line.find(':');

      if (pos != std::string::npos) {
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        key.erase(0, key.find_first_not_of(" \t\n\r\f\v"));
        key.erase(key.find_last_not_of(" \t\n\r\f\v") + 1);
        value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
        value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);

        coefficientMap[key].push_back(std::stod(value));
      } else {
        std::cerr << "Invalid line format: " << line << std::endl;
      }
    }

    inputFile.close();
    // LOG(INFO) << "[UDFCostCoefficient] Loaded coefficient map" << std::endl;
    // for (const auto& entry : coefficientMap) {
    //     LOG(INFO) << "Key: " << entry.first << ", Value: " << entry.second <<
    //     std::endl;
    // }
  }

  static UdfCostCoefficient instance;

  std::unordered_map<std::string, std::vector<double>> coefficientMap;

 public:
  static UdfCostCoefficient& getInstance() {
    return instance;
  }

  std::vector<double> getCoefficient(const std::string& udf) {
    LOG(INFO) << "[INFO] UdfCostCoefficient, retrieved coefficient for: " << udf
              << std::endl;
    if (coefficientMap.find(udf) != coefficientMap.end()) {
      return coefficientMap[udf];
    } else {
      // TODO: should failed or return a default value
      LOG(ERROR) << "UDF " + udf + " not found";
      return {1};
    }
  }
};

UdfCostCoefficient UdfCostCoefficient::instance;
