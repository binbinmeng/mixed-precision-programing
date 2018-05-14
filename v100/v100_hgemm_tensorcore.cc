//
// Created by binbin on 18-5-14.
//

//
// Created by binbin on 18-5-14.
//

#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fp16_conversion.h"
#include "../util/util.h"

using namespace std;

int main(int argc, char ** argv){


    int min_m_k_n = 2;
    int max_m_k_n = 4096*8;
    int repeats = 10;
    int verbose = 1;


    cout << "\nrunning cublasHgemm test\n" << endl;

    if(verbose)
        cout << "running with"
             << " min_m_k_n: " << min_m_k_n
             << " max_m_k_n: " << max_m_k_n
             << " repeats: " << repeats
             << endl;

    cublasStatus_t stat;
    cublasHandle_t handle;

    checkCublas(cublasCreate(&handle));
    
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    if(verbose) cout << "allocating device variables" << endl;

    // Allocate 3 arrays on CPU

    float *h_A = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
    float *h_B = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));
    float *h_C = (float *)malloc(max_m_k_n * max_m_k_n * sizeof(float));

    CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
    CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
    CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);


    // Allocate 3 arrays on GPU

    __half *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(__half)));
    checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(__half)));

    for (int i = 0; i < max_m_k_n * max_m_k_n; i++) {
        d_A[i] = approx_float_to_half(h_A[i]);
        d_B[i] = approx_float_to_half(h_B[i]);
        d_C[i] = approx_float_to_half(h_C[i]);
    }

    int lda, ldb, ldc, m, n, k;
    const __half alf = approx_float_to_half(1.0);
    const __half bet = approx_float_to_half(0.0);
    const __half *alpha = &alf;
    const __half *beta = &bet;



    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int size = min_m_k_n; size <= max_m_k_n; size=size*2){

        m=n=k=size;
        lda = m;
        ldb = k;
        ldc = m;

        cudaEventRecord(start, 0);

        for(int rep = 0; rep < repeats; rep++){

             stat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                	m, n, k,
                	alpha,
                	d_A, CUDA_R_16F, m,//MATRIX_M,
                	d_B, CUDA_R_16F, k,//MATRIX_K,
                	beta,
                	d_C, CUDA_R_32F, m,//MATRIX_M,
                	CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);


        }

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;

        if(stat != CUBLAS_STATUS_SUCCESS){
            cerr << "cublasHgemm failed" << endl;
            exit(1);
        }
        assert(!cudaGetLastError());

        cout << "float16; size "

             << size << " average: " << elapsed/repeats << " s "<< endl;

    }

    //Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}