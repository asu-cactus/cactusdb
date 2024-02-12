#include <iostream>
#include <unordered_map>


// can also call init(filename) followed by getInstance()
// to initialise the map based on filename param
class UdfCostCoefficient {
private:
    UdfCostCoefficient() {
        loadCoefficients();
    }

    ~UdfCostCoefficient() {
    }

    void loadCoefficients() { 
        std::ifstream inputFile("../../../../resources/data/udf_coefficient.txt");

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

                coefficientMap[key] = std::stof(value);
            } else {
                std::cerr << "Invalid line format: " << line << std::endl;
            }
        }

        inputFile.close();

        for (const auto& entry : coefficientMap) {
            std::cout << "Key: " << entry.first << ", Value: " << entry.second << std::endl;
        }

    }

    static UdfCostCoefficient instance;

    std::unordered_map<std::string, float> coefficientMap;

public:
    static UdfCostCoefficient& getInstance() {
        return instance;
    }

    float getCoefficient(const std::string& udf) {
        if (coefficientMap.find(udf) != coefficientMap.end()) {
            return coefficientMap[udf];
        } else {
            return -1;
            throw std::runtime_error("UDF " + udf +  " not found");
        }
    }
};

UdfCostCoefficient UdfCostCoefficient::instance;

