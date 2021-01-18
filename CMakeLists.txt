cmake_minimum_required(VERSION 3.12)
project(matmult)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_STANDARD 14)

add_executable(matmult main.cpp matmult_seq.cpp matrix_helpers.cpp matrix_helpers.h matmult_functions.h matmult_parallel.cpp min_examples.cpp min_examples.h matmult_measuring.cpp matmult_measuring.h)