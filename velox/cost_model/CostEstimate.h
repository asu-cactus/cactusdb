#pragma once
struct CostEstimate {
    float cost;
    int outputRows;
    int outputCols;

    CostEstimate(float cost, int rows, int cols) : cost(cost) , outputRows(rows), outputCols(cols) {}
};