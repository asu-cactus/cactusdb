#include "velox/ml_functions/gpufunctions.h"
#include "velox/experimental/gpu/Common.h"

#define CUBLAS_ERROR(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    printf("Error %s at %s:%d\n", cublasGetStatusString(x), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);} } while(0)
template <typename T>
void multiplyMatrices(int m, int n, int k,
                      const T* A, int lda, const T* B, int ldb,
                      T* C, int ldc) {

    cublasHandle_t handle;
    CUBLAS_ERROR(cublasCreate(&handle));
    // Allocate device memory
    T *d_A, *d_B, *d_C;
    CUDA_CHECK_FATAL(cudaMalloc((void**)&d_A, m * k * sizeof(T)));
    CUDA_CHECK_FATAL(cudaMalloc((void**)&d_B, k * n * sizeof(T)));
    CUDA_CHECK_FATAL(cudaMalloc((void**)&d_C, m * n * sizeof(T)));

    // Copy data from host to device
    CUDA_CHECK_FATAL(cudaMemcpy(d_A, A, m * k * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK_FATAL(cudaMemcpy(d_B, B, k * n * sizeof(T), cudaMemcpyHostToDevice));

    T alpha = 1.0;
    T beta = 0.0;
    // Perform matrix multiplication on GPU
    cublasStatus_t status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                         n, m, k, &alpha,
                                         d_B, CublasType<T>::type, n,
                                         d_A, CublasType<T>::type, k,
                                         &beta, d_C, CublasType<T>::type, n,
                                         CublasType<T>::type, CublasType<T>::algo);

    CUBLAS_ERROR(status);

    // Copy result from device to host
    CUDA_CHECK_FATAL(cudaMemcpy(C, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_LOG(cudaFree(d_A));
    CUDA_CHECK_LOG(cudaFree(d_B));
    CUDA_CHECK_LOG(cudaFree(d_C));

    // Destroy cuBLAS handle
    CUBLAS_ERROR(cublasDestroy(handle));
}

// Explicit instantiation for float and double
template void multiplyMatrices<float>(int, int, int,
                                       const float*, int, const float*, int, float*, int);
template void multiplyMatrices<double>(int, int, int,
                                        const double*, int, const double*, int, double*, int);

