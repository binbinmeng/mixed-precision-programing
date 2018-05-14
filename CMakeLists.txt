cmake_minimum_required(VERSION 2.8)
project(mixed-precision)

set(CMAKE_BUILD_TYPE debug)

include(FindCUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -Wall")
LIST(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -lineinfo -use_fast_math -Xptxas -dlcm=cg)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52,arch=compute_60,code=sm_60,arch=compute_61,code=sm_61,arch=compute_70,code=sm_70)

add_subdirectory(p100)
#add_subdirectory(v100)
#add_subdirectory(p4&p40)

