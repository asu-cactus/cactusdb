#pragma once
#include "velox/cost_model/Stat.h"

class Source {

    public:

        enum Type {
            NODE,
            FILE,
            VECTOR,
            DATABASE
        };

        Source(std::string name = "", Type type = Type::NODE, std::shared_ptr<Stat> stats = nullptr) : name(name), type(type), stats(stats) {}

        std::string getName(){
            return name;
        }

        std::shared_ptr<Stat> getStats(){
            return stats;
        };

        Type getType(){
            return type;
        }

    private:
        std::string name;
        Type type;
        std::shared_ptr<Stat> stats;     
};
