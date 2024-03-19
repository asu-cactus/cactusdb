#pragma once

#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>

template <typename T>
struct CublasType {};

template <>
struct CublasType<float> {
    static const cudaDataType_t type = CUDA_R_32F;
    static const cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
};

template <>
struct CublasType<double> {
    static const cudaDataType_t type = CUDA_R_64F;
    static const cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
};

template <typename T>
void multiplyMatrices(int m, int n, int k,
                      const T* A, int lda, const T* B, int ldb,
                      T* C, int ldc);

