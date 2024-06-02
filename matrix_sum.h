#ifndef MATRIX_SUM_H
#define MATRIX_SUM_H

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <unistd.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/flow_graph.h>
#include <oneapi/tbb.h>

#define SHOW_VALUES 5 
#define GPU_CUDA_1 1
#define PARA_REDUCE 2
#define GRAPH_BASED 3 
#define GPU_CUDA_2 4

std::string get_label(int option);


// Function to initialize matrices
void initialize_matrices(std::vector<std::vector<float>>& A, 
                         std::vector<std::vector<float>>& B, 
                         std::vector<std::vector<float>>& C, int N);

// Function to sum matrices using TBB and CUDA
void sum_matrices_tbb_cuda(std::vector<std::vector<float>>& A, 
                           std::vector<std::vector<float>>& B, 
                           std::vector<std::vector<float>>& C, 
                           std::vector<std::vector<float>>& result, int N);

// TBB parallelism function using a graph of nodes
void sum_matrices_tbb_graph(std::vector<std::vector<float>>& A, 
                          std::vector<std::vector<float>>& B, 
                          std::vector<std::vector<float>>& C, 
                          std::vector<std::vector<float>>& result, int N);

// TBB using parallel for
void sum_matrices_tbb_parallel(std::vector<std::vector<float>>& A, 
                  std::vector<std::vector<float>>& B, 
                  std::vector<std::vector<float>>& C, 
                  std::vector<std::vector<float>>& result, int N);

#endif // MATRIX_SUM_H
