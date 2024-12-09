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
#include <map>
#include <string>
#include "velox/cost_model/Source.h"

class CataLog {
 public:
  virtual ~CataLog() = default;

  CataLog() {}

  CataLog(std::string name) : name(name) {}

  // Add or update UDF associate information (schema, file address) based on the
  // flag (weights or values)
  void add(
      const std::string& name,
      RowTypePtr schema,
      std::vector<std::shared_ptr<TempFilePath>> filePath,
      int flag,
      std::string nameSuffix = "") {
    // TODO: better naming convention for the flag variable
    for (auto& path : filePath) {
      preserveTempFilePaths.push_back(path);
    }
    if (flag == 1) {
      std::string key = name + "_weights" + nameSuffix;

      auto fileAddrIt = UDFFileAddrMap.find(key);
      auto schemaIt = UDFSchemaMap.find(key);

      if (fileAddrIt != UDFFileAddrMap.end() &&
          schemaIt != UDFSchemaMap.end()) {
        // Key exists, update values
        fileAddrIt->second = filePath;
        schemaIt->second = schema;
      } else {
        // Key doesn't exist, create new entry
        UDFFileAddrMap[key] = filePath;
        UDFSchemaMap[key] = schema;
      }
    } else if (flag == 0) {
      std::string key = name + "_values" + nameSuffix;

      auto fileAddrIt = UDFFileAddrMap.find(key);
      auto schemaIt = UDFSchemaMap.find(key);

      if (fileAddrIt != UDFFileAddrMap.end() &&
          schemaIt != UDFSchemaMap.end()) {
        // Key exists, update values
        fileAddrIt->second = filePath;
        schemaIt->second = schema;
      } else {
        // Key doesn't exist, create new entry
        UDFFileAddrMap[key] = filePath;
        UDFSchemaMap[key] = schema;
      }
    }
  }

  // Set schema and file address for data source blocks
  void setDataSourceBlocks(
      RowTypePtr schema,
      std::vector<std::shared_ptr<TempFilePath>> filePath) {
    dataSourceBlocksSchemaMap["values"] = schema;
    dataSourceBlocksFileAddrMap["values"] = filePath;
  }

  // Get data source blocks schema based on key
  RowTypePtr getDataSourceBlocksSchema(std::string key) {
    return findSchemaInMap(dataSourceBlocksSchemaMap, key);
  }

  // Get data source blocks file address based on key
  std::vector<std::shared_ptr<TempFilePath>> getDataSourceBlocksFileAddr(
      std::string key) {
    return findFileAddrInMap(dataSourceBlocksFileAddrMap, key);
  }

  void setUDFSchema(std::string key, RowTypePtr schema) {
    UDFSchemaMap[key] = schema;
  }

  // Get UDF schema based on key
  RowTypePtr getUDFSchema(std::string key) {
    RowTypePtr result = findSchemaInMap(UDFSchemaMap, key);
    if (result == 0) {
      throw std::runtime_error(
          fmt::format("Key: {} was not found in UDFSchemaMap.", key));
    }
    return result;
  }

  // Get UDF file address based on key
  std::vector<std::shared_ptr<TempFilePath>> getUDFFileAddr(std::string key) {
    std::vector<std::shared_ptr<TempFilePath>> result =
        findFileAddrInMap(UDFFileAddrMap, key);
    if (result.size() == 0) {
      throw std::runtime_error(
          fmt::format("Key: {} was not found in UDFFileAddrMap.", key));
    }
    return result;
  }

  // Check if UDF file address exists for the given key
  bool checkExistsUDFFileAddr(const std::string& key) {
    return UDFFileAddrMap.find(key) != UDFFileAddrMap.end();
  }

  // Set schema and file address for data source
  void setDataSource(
      RowTypePtr schema,
      std::vector<std::shared_ptr<TempFilePath>> filePath) {
    dataSourceSchemaMap["values"] = schema;
    dataSourceFileAddrMap["values"] = filePath;
  }

  // Set data source statistics based on key
  void setDataSourceStat(std::vector<int> dims) {
    dataSourceStatMap["values"] = dims;
  }

  // Get data source statistics based on key
  std::vector<int> getDataSourceStat(std::string key) {
    std::vector<int> result = findStatInMap(dataSourceStatMap, key);
    if (result.size() == 0) {
      throw std::runtime_error(
          fmt::format("Key: {} was not found in dataSourceStatMap.", key));
    }
    return result;
  }

  // Set file address map for a given PlanNodeId
  void setIdAddressMap(core::PlanNodeId p, std::vector<std::string> filePath) {
    idFileAddrMap[p] = filePath;
    idFileFormatMap[p] = dwio::common::FileFormat::DWRF;
  }

  // Set file address map for a given PlanNodeId
  void setIdAddressMap(
      core::PlanNodeId p,
      std::vector<std::string> filePath,
      dwio::common::FileFormat format) {
    idFileAddrMap[p] = filePath;
    idFileFormatMap[p] = format;
  }

  // Deprecating warning: in the future development, it is expected to pass path
  // as std::vector<std::string>
  void setIdAddressMap(
      core::PlanNodeId p,
      std::vector<std::shared_ptr<TempFilePath>> filePath) {
    std::vector<std::string> filePathStr;
    for (auto& path : filePath) {
      filePathStr.push_back(path->path);
      // preserve it for cleanup
      preserveTempFilePaths.push_back(path);
    }
    idFileAddrMap[p] = filePathStr;
    idFileFormatMap[p] = dwio::common::FileFormat::DWRF;
  }

  // Get file address map for a given PlanNodeId
  std::vector<std::string> getFileAddress(core::PlanNodeId p) {
    return idFileAddrMap[p];
  }

  // Delete file address map entry for a given PlanNodeId
  void deleteIdAddressMap(core::PlanNodeId p) {
    idFileAddrMap.erase(p);
  }

  // Clear file address map entry for a given PlanNodeId
  void clearIdAddressMap() {
    idFileAddrMap.clear();
  }

  // Get the entire file address map for PlanNodeId
  std::map<core::PlanNodeId, std::vector<std::string>> getIdAddressMap() {
    return idFileAddrMap;
  }

  void setIdAddressMap(
      std::map<core::PlanNodeId, std::vector<std::string>> map) {
    idFileAddrMap = map;
  }

  // Set the schema for a given PlanNodeId
  void setFileSchema(core::PlanNodeId p, RowTypePtr schema) {
    fileSchemaMap[p] = schema;
  }

  // Get the schema for a given PlanNodeId
  RowTypePtr getFileSchema(core::PlanNodeId p) {
    auto it = fileSchemaMap.find(p);
    if (it != fileSchemaMap.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  // Get the map of PlanNodeId to schema
  std::map<core::PlanNodeId, RowTypePtr> getFileSchemaMap() {
    return fileSchemaMap;
  }

  // Clear file schema map
  void clearFileSchemaMap() {
    fileSchemaMap.clear();
  }

  // Get the map of PlanNodeId to FileFormat
  std::map<core::PlanNodeId, dwio::common::FileFormat> getIdFileFormatMap() {
    return idFileFormatMap;
  }

  // Set file address map for a given PlanNodeId
  void setIdFileFormat(core::PlanNodeId p, dwio::common::FileFormat format) {
    idFileFormatMap[p] = format;
  }

  // Get file format for a given PlanNodeId
  dwio::common::FileFormat getIdFileFormat(core::PlanNodeId p) {
    auto it = idFileFormatMap.find(p);
    if (it != idFileFormatMap.end()) {
      return it->second;
    } else {
      return dwio::common::FileFormat::DWRF;
    }
  }

  // Set mapping from vector values to PlanNodeId
  void setVectorIdMap(core::PlanNodeId p, std::string values) {
    vectorIdMap[values] = p;
  }

  // Delete mapping from vector values to PlanNodeId
  void deleteVectorIdMap(std::string values) {
    vectorIdMap.erase(values);
  }

  // Clear mapping from vector values to PlanNodeId
  void clearVectorIdMap() {
    vectorIdMap.clear();
  }

  // Get PlanNodeId based on vector values
  core::PlanNodeId getVectorIdMap(const std::string& values) {
    auto it = vectorIdMap.find(values);
    return (it != vectorIdMap.end())
        ? it->second
        : core::PlanNodeId(); // Return default PlanNodeId
  }

  // Set blocking threshold
  void setBlockingThreshold(int newThreshold) {
    blockingThreshold = newThreshold;
  }

  // Get blocking threshold
  int getBlockingThreshold() {
    return blockingThreshold;
  }

  // Set default number of blocks
  void setDefaultBlocksNum(int newBlocksNum) {
    defaultBlocksNum = newBlocksNum;
  }

  // Get default number of blocks
  int getDefaultBlocksNum() {
    return defaultBlocksNum;
  }

  void setDefaultBlocksSize(int newBlocksSize) {
    defaultBlocksSize = newBlocksSize;
  }

  // Get default number of blocks
  int getDefaultBlocksSize() {
    return defaultBlocksSize;
  }

  int getDefaultSplits() {
    return defaultSplits;
  }

  void addNodeIdRelationName(core::PlanNodeId p, std::string relationName) {
    nodeIdRelationNameMap[p] = relationName;
  }

  std::string getNodeIdRelationName(core::PlanNodeId p) {
    auto it = nodeIdRelationNameMap.find(p);
    if (it != nodeIdRelationNameMap.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] PlanNodeId: {} not exist in nodeIdRelationNameMap", p);
    }
  }

  void clearNodeIdRelationNameMap() {
    nodeIdRelationNameMap.clear();
  }

  void addSource(std::shared_ptr<Source> src) {
    sourceMap.insert({src->getName(), src});
  }

  void removeSource(std::string name) {
    sourceMap.erase(name);
  }

  void registerDataSrc(
      std::string name,
      std::vector<std::shared_ptr<TempFilePath>> filePath,
      RowTypePtr schema,
      std::pair<int, int> stats) {
    auto it = registeredDataSrcFiles.find(name);
    if (it != registeredDataSrcFiles.end()) {
      LOG(FATAL) << fmt::format(
          "[ERROR] name: {} already exists in registeredDataSrcFiles", name);
    } else {
      std::vector<std::string> files;
      for (auto& path : filePath) {
        preserveTempFilePaths.push_back(path);
        files.push_back(path->path);
      }
      registeredDataSrcFiles[name] = files;
      registeredDataSrcFormat[name] = dwio::common::FileFormat::DWRF;
      registeredDataSrcSchema[name] = schema;
      registeredDataSrcStats[name] = stats;
    }
  }

  void registerDataSrc(
      std::string name,
      std::vector<std::string> filePath,
      dwio::common::FileFormat format,
      RowTypePtr schema,
      std::pair<int, int> stats) {
    auto it = registeredDataSrcFiles.find(name);
    if (it != registeredDataSrcFiles.end()) {
      LOG(FATAL) << fmt::format(
          "[ERROR] name: {} already exists in registeredDataSrcFiles", name);
    } else {
      registeredDataSrcFiles[name] = filePath;
      registeredDataSrcFormat[name] = format;
      registeredDataSrcSchema[name] = schema;
      registeredDataSrcStats[name] = stats;
    }
  }

  std::vector<std::string> getRegisteredDataSrcFiles(std::string name) {
    auto it = registeredDataSrcFiles.find(name);
    if (it != registeredDataSrcFiles.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] name: {} not exist in registeredDataSrcFiles", name);
      return {};
    }
  }

  dwio::common::FileFormat getRegisteredDataSrcFormat(std::string name) {
    auto it = registeredDataSrcFormat.find(name);
    if (it != registeredDataSrcFormat.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] name: {} not exist in registeredDataSrcFormat", name);
      return dwio::common::FileFormat::DWRF;
    }
  }

  RowTypePtr getRegisteredDataSrcSchema(std::string name) {
    auto it = registeredDataSrcSchema.find(name);
    if (it != registeredDataSrcSchema.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] name: {} not exist in registeredDataSrcSchema", name);
      return nullptr;
    }
  }

  std::pair<int, int> getRegisteredDataSrcStats(std::string name) {
    auto it = registeredDataSrcStats.find(name);
    if (it != registeredDataSrcStats.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] name: {} not exist in registeredDataSrcStats", name);
      return std::make_pair(0, 0);
    }
  }

  void clearRegisteredDataSrc() {
    registeredDataSrcFiles.clear();
    registeredDataSrcFormat.clear();
    registeredDataSrcSchema.clear();
    registeredDataSrcStats.clear();
  }

  std::shared_ptr<Source> getSource(std::string srcName) {
    auto it = sourceMap.find(srcName);
    if (it != sourceMap.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] srcName: {} not exist in sourceMap", srcName);
      return nullptr;
    }
  }

  void clearSourceMap() {
    sourceMap.clear();
  }

  void addNumericalColMinMax(std::string colName, double min, double max) {
    numericalColMinMaxs[colName] = std::make_pair(min, max);
  }

  std::pair<int, int> getNumericalColMinMax(std::string colName) {
    auto it = numericalColMinMaxs.find(colName);
    if (it != numericalColMinMaxs.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] colName: {} not exist in numericalColMinMaxs", colName);
      return std::make_pair(0, 0);
    }
  }

  void clearNumericalColMinMax() {
    numericalColMinMaxs.clear();
  }

  void addCategoricalColVals(
      std::string colName,
      std::vector<std::string> uniqueValues) {
    categoricalColVals[colName] = uniqueValues;
  }

  std::vector<std::string> getCategoricalColVals(std::string colName) {
    auto it = categoricalColVals.find(colName);
    if (it != categoricalColVals.end()) {
      return it->second;
    } else {
      LOG(FATAL) << fmt::format(
          "[ERROR] colName: {} not exist in categoricalColVals", colName);
      return {};
    }
  }

  void clearCategoricalColVals() {
    categoricalColVals.clear();
  }

  template <typename T>
  void processNumericColumn(
      std::string colName,
      facebook::velox::FlatVector<T>* numericVector,
      size_t numRows,
      int numBins,
      std::vector<double>& frequencies,
      std::vector<std::string>& bins) {
    T minValue = std::numeric_limits<T>::max();
    T maxValue = std::numeric_limits<T>::lowest();

    if (numericalColMinMaxs.find(colName) != numericalColMinMaxs.end()) {
      auto [min, max] = numericalColMinMaxs[colName];
      minValue = min;
      maxValue = max;
    } else {
      // Compute min and max for binning
      for (size_t j = 0; j < numRows; ++j) {
        if (!numericVector->isNullAt(j)) {
          T value = numericVector->valueAt(j);
          minValue = std::min(minValue, value);
          maxValue = std::max(maxValue, value);
        }
      }
    }

    if (minValue == maxValue) {
      bins[0] = std::to_string(minValue);
      frequencies[0] = 1.0; // All values fall into one bin
      return;
    }

    double binSize = static_cast<double>(maxValue - minValue) / numBins;
    for (int b = 0; b < numBins; ++b) {
      double binValue = static_cast<double>(minValue) + b * binSize;
      bins[b] = std::to_string(binValue);
    }
    bins[numBins] = std::to_string(maxValue);

    for (size_t j = 0; j < numRows; ++j) {
      if (!numericVector->isNullAt(j)) {
        T value = numericVector->valueAt(j);
        int binIndex = static_cast<int>((value - minValue) / binSize);
        if (binIndex >= 0 && binIndex < numBins) {
          ++frequencies[binIndex];
        }
      }
    }

    // Normalize frequencies
    for (auto& freq : frequencies) {
      freq /= numRows;
    }
  }

  void processCategoricalColumn(
      std::string colName,
      facebook::velox::FlatVector<StringView>* stringVector,
      size_t numRows,
      std::vector<double>& frequencies,
      std::vector<std::string>& bins) {
    std::map<std::string, double> categoryCounts;
    std::set<std::string> uniqueCategories;
    double totalCount = 0;

    for (size_t j = 0; j < numRows; ++j) {
      if (!stringVector->isNullAt(j)) {
        std::string value = stringVector->valueAt(j).str();
        categoryCounts[value]++;
        totalCount++;
        uniqueCategories.insert(value);
      }
    }
    std::vector<std::string> categoricalValsToIterate;
    if (categoricalColVals.find(colName) != categoricalColVals.end()) {
      // Use the unique values from the categoricalColVals map if exists
      categoricalValsToIterate = categoricalColVals[colName];
    } else {
      categoricalValsToIterate = std::vector<std::string>(
          uniqueCategories.begin(), uniqueCategories.end());
    }

    size_t index = 0;
    for (const auto& category : categoricalValsToIterate) {
      if (index >= bins.size()) {
        break;
      }
      bins[index] = category;
      if (categoryCounts.find(category) == categoryCounts.end()) {
        frequencies[index] = 0;
      } else {
        frequencies[index] = categoryCounts[category] / totalCount;
      }
      index += 1;
    }
  }

  void outputHistogramForData(
      const std::shared_ptr<RowVector>& rowVector,
      const std::string& tableName,
      int numBins,
      const std::string& outputFilePath) {
    std::ofstream outFile(outputFilePath, std::ios::app);
    if (!outFile.is_open()) {
      throw std::runtime_error("Failed to open the output file.");
    }

    const auto& children = rowVector->children();
    const auto& type = rowVector->type()->asRow();
    size_t numRows = rowVector->size();
    for (size_t i = 0; i < children.size(); i++) {
      std::string columnType;
      const auto& child = children[i];
      const std::string& columnName = type.nameOf(i);

      // initialize the bins
      std::vector<double> frequencies(numBins, 0);
      std::vector<std::string> bins(numBins + 1); // bins store the bin edges
      if (child->typeKind() == TypeKind::INTEGER) {
        // Handle INTEGER type
        auto numericVector = child->asFlatVector<int32_t>();
        // std::dynamic_pointer_cast<FlatVector<T>>(out)
        processNumericColumn<int32_t>(
            columnName, numericVector, numRows, numBins, frequencies, bins);
        columnType = "Numerical";
      } else if (child->typeKind() == TypeKind::BIGINT) {
        // Handle BIGINT type
        auto numericVector = child->asFlatVector<int64_t>();
        processNumericColumn<int64_t>(
            columnName, numericVector, numRows, numBins, frequencies, bins);
        columnType = "Numerical";
      } else if (child->typeKind() == TypeKind::REAL) {
        // Handle REAL type
        auto numericVector = child->asFlatVector<float>();
        processNumericColumn<float>(
            columnName, numericVector, numRows, numBins, frequencies, bins);
        columnType = "Numerical";
      } else if (child->typeKind() == TypeKind::DOUBLE) {
        // Handle DOUBLE type
        auto numericVector = child->asFlatVector<double>();
        processNumericColumn<double>(
            columnName, numericVector, numRows, numBins, frequencies, bins);
        columnType = "Numerical";
      } else if (child->typeKind() == TypeKind::VARCHAR) {
        // Handle VARCHAR type
        if (columnName.find("title") == std::string::npos) {
          // skip title columns
          auto stringVector = child->asFlatVector<StringView>();
          processCategoricalColumn(
              columnName, stringVector, numRows, frequencies, bins);
          columnType = "Categorical";
        } else {
          columnType = "Varchar";
        }
      } else {
        // other types
        columnType = mapTypeKindToName(child->typeKind());
      }

      // Output format: table_name|column_name|frequencies list|bins list|
      // table_name_table_column|is_categorical
      outFile << tableName << "|" << columnName << "|" << tableName << "."
              << columnName << "|" << columnType << "|[";
      for (size_t j = 0; j < frequencies.size(); ++j) {
        outFile << frequencies[j];
        if (j < frequencies.size() - 1) {
          outFile << ",";
        }
      }
      outFile << "]|[";
      for (size_t j = 0; j < bins.size(); ++j) {
        outFile << "\"" << bins[j] << "\""; // Quoting string bins for safety
        if (j < bins.size() - 1) {
          outFile << ",";
        }
      }
      outFile << "]" << "\n";
    }
    outFile.close();
  }

 private:
  std::string name;
  // Default values
  int defaultBlocksNum = 4;
  int defaultBlocksSize = 256;
  int blockingThreshold = 256;
  int defaultSplits = 392;
  // Maps for storing data
  std::map<std::string, std::vector<int>> dataSourceStatMap;
  std::map<core::PlanNodeId, std::vector<std::string>> idFileAddrMap;
  std::map<core::PlanNodeId, dwio::common::FileFormat> idFileFormatMap;
  std::map<std::string, core::PlanNodeId> vectorIdMap;
  std::map<std::string, RowTypePtr> dataSourceSchemaMap;
  std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>>
      dataSourceFileAddrMap;
  std::map<std::string, RowTypePtr> dataSourceBlocksSchemaMap;
  std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>>
      dataSourceBlocksFileAddrMap;
  std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>>
      UDFFileAddrMap;
  std::map<std::string, RowTypePtr> UDFSchemaMap;
  std::map<core::PlanNodeId, RowTypePtr> fileSchemaMap;
  std::unordered_map<std::string, std::shared_ptr<Source>> sourceMap;
  std::vector<std::shared_ptr<TempFilePath>> preserveTempFilePaths;
  std::unordered_map<std::string, std::vector<std::string>>
      registeredDataSrcFiles;
  std::unordered_map<std::string, dwio::common::FileFormat>
      registeredDataSrcFormat;
  std::unordered_map<std::string, RowTypePtr> registeredDataSrcSchema;
  std::unordered_map<std::string, std::pair<int, int>> registeredDataSrcStats;
  std::unordered_map<core::PlanNodeId, std::string> nodeIdRelationNameMap;
  // vars to store per-column statistics
  std::unordered_map<std::string, std::pair<int, int>> numericalColMinMaxs;
  std::unordered_map<std::string, std::vector<std::string>> categoricalColVals;

  // Helper function to find schema in a map based on key
  RowTypePtr findSchemaInMap(
      const std::map<std::string, RowTypePtr>& schemaMap,
      const std::string& key) {
    auto schemaIt = schemaMap.find(key);
    return (schemaIt != schemaMap.end())
        ? schemaIt->second
        : nullptr; // Return null shared pointer
  }

  // Helper function to find file address in a map based on key
  std::vector<std::shared_ptr<TempFilePath>> findFileAddrInMap(
      const std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>>&
          fileAddrMap,
      const std::string& key) {
    auto addrIt = fileAddrMap.find(key);
    return (addrIt != fileAddrMap.end())
        ? addrIt->second
        : std::vector<std::shared_ptr<TempFilePath>>(); // Return empty vector
  }

  // Helper function to find statistics in a map based on key
  std::vector<int> findStatInMap(
      const std::map<std::string, std::vector<int>>& statMap,
      const std::string& key) {
    auto statIt = statMap.find(key);
    return (statIt != statMap.end())
        ? statIt->second
        : std::vector<int>(); // Return empty vector
  }
};
