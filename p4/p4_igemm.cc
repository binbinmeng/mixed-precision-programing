
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
//#include "fp16_conversion.h"
#include "../util/util.h"

using namespace std;

int main(int argc, char ** argv){


  int min_m_k_n = 2;
  int max_m_k_n = 4096*8;
  int repeats = 10;
  int verbose = 1;

  cout << "\nrunning cublasSgemm test\n" << endl;
  
  if(verbose) 
    cout << "running with" 
	 << " min_m_k_n: " << min_m_k_n
	 << " max_m_k_n: " << max_m_k_n
	 << " repeats: " << repeats
	 << endl;

  cublasStatus_t stat;
  cublasHandle_t handle;

  checkCublas(cublasCreate(&handle));

  if(verbose) cout << "allocating device variables" << endl;
  
  // Allocate 3 arrays on CPU
  
  int8_t *h_A = (int8_t *)malloc(max_m_k_n * max_m_k_n * sizeof(int8_t));
  int8_t *h_B = (int8_t *)malloc(max_m_k_n * max_m_k_n * sizeof(int8_t));
  int8_t *h_C = (int8_t *)malloc(max_m_k_n * max_m_k_n * sizeof(int8_t));
  
  //CPU_fill_rand(h_A, max_m_k_n, max_m_k_n);
  //CPU_fill_rand(h_B, max_m_k_n, max_m_k_n);
  //CPU_fill_rand(h_C, max_m_k_n, max_m_k_n);

    // Allocate 3 arrays on GPU
    int8_t *d_A, *d_B, *d_C;
    checkCuda(cudaMallocManaged(&d_A, max_m_k_n * max_m_k_n * sizeof(int8_t)));
    checkCuda(cudaMallocManaged(&d_B, max_m_k_n * max_m_k_n * sizeof(int8_t)));
    checkCuda(cudaMallocManaged(&d_C, max_m_k_n * max_m_k_n * sizeof(int8_t)));
    
    checkCuda(cudaMemcpy(d_A,h_A,max_m_k_n * max_m_k_n * sizeof(int8_t),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B,h_B,max_m_k_n * max_m_k_n * sizeof(int8_t),cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_C,h_C,max_m_k_n * max_m_k_n * sizeof(int8_t),cudaMemcpyHostToDevice));
    
    int lda, ldb, ldc, m, n, k;
    const int8_t alf = 1.0f;
    const int8_t bet = 0.0f;
    const int8_t *alpha = &alf;
    const int8_t *beta = &bet;
  

  
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
                            d_A, CUDA_R_8I, lda, 
                            d_B, CUDA_R_8I, ldb, 
                            beta, 
                            d_C, CUDA_R_32I, ldc,
                            CUDA_R_32I, CUBLAS_GEMM_DFALT);
         //stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    elapsed /= 1000.0f;

    if(stat != CUBLAS_STATUS_SUCCESS){
      cerr << "cublasSgemm failed" << endl;
      exit(1);
    }

    assert(!cudaGetLastError());

    cout << "int8_t; size "

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

