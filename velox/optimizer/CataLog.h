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

class CataLog {
    public:
        void add(const std::string& name, RowTypePtr schema, std::vector<std::shared_ptr<TempFilePath>> filePath, int flag) {
            if (flag == 1) {
                std::string key = name + "_weights";

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
                std::string key = name + "_values";

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

        void setDataSourceBlocks(RowTypePtr schema, std::vector<std::shared_ptr<TempFilePath>> filePath) {
            dataSourceBlocksSchemaMap["values"] = schema;
            dataSourceBlocksFileAddrMap["values"] = filePath;
        }

        RowTypePtr getDataSourceBlocksSchema(std::string key) {
            auto schemaIt = dataSourceBlocksSchemaMap.find(key);
            if (schemaIt != dataSourceBlocksSchemaMap.end()) {
                return schemaIt->second;
            }
            else {
                // Return a null shared pointer to indicate an empty state
                return RowTypePtr();
            }
        }

        std::vector<std::shared_ptr<TempFilePath>> getDataSourceBlocksFileAddr(std::string key) {
            auto addrIt = dataSourceBlocksFileAddrMap.find(key);
            if (addrIt != dataSourceBlocksFileAddrMap.end()) {
                return addrIt->second;
            }
            else {
                // Return a null shared pointer to indicate an empty state
                return {};
            }
        }

        RowTypePtr getUDFSchema(std::string key) {
            auto schemaIt = UDFSchemaMap.find(key);
            if (schemaIt != UDFSchemaMap.end()) {
                return schemaIt->second;
            }
            else {
                // Return a null shared pointer to indicate an empty state
                return RowTypePtr();
            }
        }

        std::vector<std::shared_ptr<TempFilePath>> getUDFFileAddr(std::string key){
            auto addrIt = UDFFileAddrMap.find(key);
            if (addrIt != UDFFileAddrMap.end()) {
                return addrIt->second;
            }
            else {
                // Return a null shared pointer to indicate an empty state
                return {};
            }
        }

        bool checkExistsUDFFileAddr(const std::string& key) {
            auto addrIt = UDFFileAddrMap.find(key);
            return addrIt != UDFFileAddrMap.end();
        }


        void setDataSource(RowTypePtr schema, std::vector<std::shared_ptr<TempFilePath>> filePath) {
            dataSourceSchemaMap["values"] = schema;
            dataSourceFileAddrMap["values"] = filePath;
        }

        void setDataSourceStat(std::vector<int> dims) {
            dataSourceStatMap["values"] = dims;
        }
        std::vector<int> getDataSourceStat(std::string key) {
            auto statIt = dataSourceStatMap.find(key);
            if (statIt != dataSourceStatMap.end()) {
                return statIt->second;
            }
            else {
                return std::vector<int>(); // Return an empty vector
            }
        }

        void setIdAddressMap(core::PlanNodeId p, std::vector<std::shared_ptr<TempFilePath>> filePath) {
            idFileAddrMap[p] = filePath;
        }

        void deleteIdAddressMap(core::PlanNodeId p) {
            idFileAddrMap.erase(p);
        }

         std::map<core::PlanNodeId, std::vector<std::shared_ptr<TempFilePath>>> getIdAddressMap() {
            return idFileAddrMap;
         }



        void setVectorIdMap(core::PlanNodeId p, std::string values) {
            vectorIdMap[values] = p;
        }

        core::PlanNodeId getVectorIdMap(const std::string& values) {
    // Assuming vectorIdMap is a std::unordered_map or std::map
            auto it = vectorIdMap.find(values);
            
            if (it != vectorIdMap.end()) {
                // Key found, return the associated value
                return it->second;
            } else {
                // Key not found, return an appropriate default or indicate the absence
                // Here, using std::nullopt to indicate the absence of a value
                return "";
            }
        }


        void setBlockingThreshold(int newThreshold){
            blockingThreshold = newThreshold;
        }
        int getBlockingThreshold (){
            return blockingThreshold;
        }
        void setDefaultBlocksNum(int newBlocksNum) {
            defaultBlocksNum = newBlocksNum;
        }
        int getDefaultBlocksNum () {
            return defaultBlocksNum;
        }


    private:
        int defaultBlocksNum = 4;
        int blockingThreshold = 2000;
        std::map<std::string, std::vector<int>> dataSourceStatMap;
        std::map<core::PlanNodeId, std::vector<std::shared_ptr<TempFilePath>>> idFileAddrMap;
        std::map<std::string, core::PlanNodeId> vectorIdMap;
        std::map<std::string, RowTypePtr> dataSourceSchemaMap;
        std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>> dataSourceFileAddrMap;
        std::map<std::string, RowTypePtr> dataSourceBlocksSchemaMap;
        std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>> dataSourceBlocksFileAddrMap;
        std::map<std::string, std::vector<std::shared_ptr<TempFilePath>>> UDFFileAddrMap;
        std::map<std::string, RowTypePtr> UDFSchemaMap;
    };
