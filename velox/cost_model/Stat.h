#pragma once

class Stat {

    private:
        std::string name;
   
    public:

        Stat(const std::string& n) : name(n) {}
        
        std::string getName() const{
            return name;
        }


};


class OutputStat : public Stat {
    private:
        
        int numRows;
        int numCols;

    public:

        OutputStat(const std::string& n, int rows=0, int cols = 0) :  Stat(n), numRows(rows), numCols(cols) {}

        OutputStat(int rows, int cols) :  Stat(""), numRows(rows), numCols(cols) {}

        int getRows(){
            return numRows;
        }

        int getCols(){
            return numCols;
        }
};