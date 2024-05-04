#pragma once
struct CostEstimate {
    float cost;
    int outputRows;
    int outputCols;
    float eigenCost = 0.0;
    float torchCost = 0.0;

    CostEstimate(float cost, int rows, int cols) : cost(cost) , outputRows(rows), outputCols(cols) {}
};