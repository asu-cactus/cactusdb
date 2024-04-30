#pragma once
#include<unordered_map>


using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

class Catalog {

    public:
        virtual ~Catalog() = default;

        Catalog(std::string name) : name(name) {}
        
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
                return nullptr; 
            }
        }
        

    private:
        std::string name;
        std::unordered_map<std::string, std::shared_ptr<Source>> sourceMap;

};
