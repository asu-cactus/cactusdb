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

        // Add or update UDF associate information (schema, file address) based on the flag (weights or values)
        void add(const std::string& name, RowTypePtr schema, std::vector<std::shared_ptr<TempFilePath>> filePath, int flag, std::string nameSuffix = "") {
            // TODO: better naming convention for the flag variable
            if (flag == 1) {
                std::string key = name + "_weights" + nameSuffix;

                auto fileAddrIt = UDFFileAddrMap.find(key);
                auto schemaIt = UDFSchemaMap.find(key);

                if (fileAddrIt != UDFFileAddrMap.end() && schemaIt != UDFSchemaMap.end()) {
                    // Key exists, update values
                    fileAddrIt->second = filePath;
                    schemaIt->second = schema;
                } else {
                    // Key doesn't exist, create new entry
                    UDFFileAddrMap[key] = filePath;
                    UDFSchemaMap[key] = schema;
                }
            }
            else if (flag == 0) {
                std::string key = name + "_values" + nameSuffix;

                auto fileAddrIt = UDFFileAddrMap.find(key);
                auto schemaIt = UDFSchemaMap.find(key);

                if (fileAddrIt != UDFFileAddrMap.end() && schemaIt != UDFSchemaMap.end()) {
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
        void setDataSourceBlocks(RowTypePtr schema, std::vector<std::shared_ptr<TempFilePath>> filePath) {
            dataSourceBlocksSchemaMap["values"] = schema;
            dataSourceBlocksFileAddrMap["values"] = filePath;
        }

        // Get data source blocks schema based on key
        RowTypePtr getDataSourceBlocksSchema(std::string key) {
            return findSchemaInMap(dataSourceBlocksSchemaMap, key);
        }

        // Get data source blocks file address based on key
        std::vector<std::shared_ptr<TempFilePath>> getDataSourceBlocksFileAddr(std::string key) {
            return findFileAddrInMap(dataSourceBlocksFileAddrMap, key);
        }

        void setUDFSchema(std::string key, RowTypePtr schema) {
            UDFSchemaMap[key] = schema;
        }

        // Get UDF schema based on key
        RowTypePtr getUDFSchema(std::string key) {
            RowTypePtr result = findSchemaInMap(UDFSchemaMap, key);
            if (result == 0) {
                throw std::runtime_error(fmt::format("Key: {} was not found in UDFSchemaMap.", key));
            }         
            return result;
        }

        // Get UDF file address based on key
        std::vector<std::shared_ptr<TempFilePath>> getUDFFileAddr(std::string key) {
            std::vector<std::shared_ptr<TempFilePath>> result =  findFileAddrInMap(UDFFileAddrMap, key);
            if (result.size() == 0) {
                throw std::runtime_error(fmt::format("Key: {} was not found in UDFFileAddrMap.", key));
            }
            return result;
        }

        // Check if UDF file address exists for the given key
        bool checkExistsUDFFileAddr(const std::string& key) {
            return UDFFileAddrMap.find(key) != UDFFileAddrMap.end();
        }

        // Set schema and file address for data source
        void setDataSource(RowTypePtr schema, std::vector<std::shared_ptr<TempFilePath>> filePath) {
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
                throw std::runtime_error(fmt::format("Key: {} was not found in dataSourceStatMap.", key));
            }
            return result;
        }

        // Set file address map for a given PlanNodeId
        void setIdAddressMap(core::PlanNodeId p, std::vector<std::shared_ptr<TempFilePath>> filePath) {
            idFileAddrMap[p] = filePath;
        }

       std::vector<std::shared_ptr<TempFilePath>> getFileAddress(core::PlanNodeId p) {
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
        std::map<core::PlanNodeId, std::vector<std::shared_ptr<TempFilePath>>> getIdAddressMap() {
            return idFileAddrMap;
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
            return (it != vectorIdMap.end()) ? it->second : core::PlanNodeId();  // Return default PlanNodeId
        }

        // Set blocking threshold
        void setBlockingThreshold(int newThreshold){
            blockingThreshold = newThreshold;
        }

        // Get blocking threshold
        int getBlockingThreshold (){
            return blockingThreshold;
        }

        // Set default number of blocks
        void setDefaultBlocksNum(int newBlocksNum) {
            defaultBlocksNum = newBlocksNum;
        }

        // Get default number of blocks
        int getDefaultBlocksNum () {
            return defaultBlocksNum;
        }

        void setDefaultBlocksSize(int newBlocksSize) {
            defaultBlocksSize = newBlocksSize;
        }

        // Get default number of blocks
        int getDefaultBlocksSize () {
            return defaultBlocksSize;
        }

        int getDefaultSplits (){
            return defaultSplits;
        }

        void addSource(std::shared_ptr<Source> src){
            sourceMap.insert({src->getName(), src});
        }

        void removeSource(std::string name){
            sourceMap.erase(name);
        }

        std::shared_ptr<Source> getSource(std::string srcName){
            auto it = sourceMap.find(srcName);
            if (it != sourceMap.end()) {
                return it->second;
            } else {
                LOG(FATAL) << fmt::format("[ERROR] srcName: {} not exist in sourceMap", srcName);
                return nullptr; 
            }
        }

        void clearSourceMap() {
            sourceMap.clear();
        }


    private:
        std::string name;
        // Default values
        int defaultBlocksNum = 4;
        int defaultBlocksSize = 256;
        int blockingThreshold = 1;
        int defaultSplits = 392;
        // Maps for storing data
        std::map<std::string, std::vector<int>> dataSourceStatMap;
        std::map<core::PlanNodeId, std::vector<std::shared_ptr<TempFilePath>>> idFileAddrMap;
        std::map<std::string, core::PlanNodeId> vectorIdMap;
        std::map<std::string, RowTypePtr> dataSourceSchemaMap;
        std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>> dataSourceFileAddrMap;
        std::map<std::string, RowTypePtr> dataSourceBlocksSchemaMap;
        std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>> dataSourceBlocksFileAddrMap;
        std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>> UDFFileAddrMap;
        std::map<std::string, RowTypePtr> UDFSchemaMap;
        std::unordered_map<std::string, std::shared_ptr<Source>> sourceMap;

         // Helper function to find schema in a map based on key
        RowTypePtr findSchemaInMap(const std::map<std::string, RowTypePtr>& schemaMap, const std::string& key) {
            auto schemaIt = schemaMap.find(key);
            return (schemaIt != schemaMap.end()) ? schemaIt->second : nullptr;  // Return null shared pointer
        }

        // Helper function to find file address in a map based on key
        std::vector<std::shared_ptr<TempFilePath>> findFileAddrInMap(const std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>>& fileAddrMap, const std::string& key) {
            auto addrIt = fileAddrMap.find(key);
            return (addrIt != fileAddrMap.end()) ? addrIt->second : std::vector<std::shared_ptr<TempFilePath>>();  // Return empty vector
        }

        // Helper function to find statistics in a map based on key
        std::vector<int> findStatInMap(const std::map<std::string, std::vector<int>>& statMap, const std::string& key) {
            auto statIt = statMap.find(key);
            return (statIt != statMap.end()) ? statIt->second : std::vector<int>();  // Return empty vector
        }
};
