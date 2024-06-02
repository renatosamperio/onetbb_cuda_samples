// Compilation:
//  reset; /usr/local/cuda/bin/nvcc -c matrix_add_gpu.cu -o matrix_add_gpu.o
//  reset; /usr/local/cuda/bin/nvcc -c matrix_add_gpu.cu -o matrix_sum_gpu.o
//  reset; g++ -std=c++17 -o matrix_sum main.cpp matrix_sum.cpp matrix_sum.o matrix_sum_utils.o -ltbb -lcudart -pthread -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -L/opt/intel/oneapi/tbb/latest/lib -I/opt/intel/oneapi/tbb/latest/include -Wl,-R/opt/intel/oneapi/tbb/latest/lib
//  ./matrix_sum 
// sudo nvidia-smi -pm 1

#include "matrix_sum.h"
#include "human_size.h"
// #include "matrix_sum_utils.h"

#include <oneapi/tbb.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

void initialize_matrix(std::vector<std::vector<float>> &matrix, int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff;
    std::cout << "  Initialised matrix";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    matrix.resize(rows);

    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(cols);
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    int v_size = rows * cols;
    std::cout << " with "<< v_size << " elements [L2="<< human_size(v_size*sizeof(float)) << "/"<< human_size(1048576) << "]"<< std::endl;

}

void print(int num_values, std::vector<std::vector<float>>& result) {

    for (int i =0; i < num_values; i++) {
        std::cout << "\t[ ";
        for (int j =0; j < num_values; j++) {std::cout << result[i][j] << " "; }
        std::cout << "]" << std::endl;
    }
}

void run_program (
    std::vector<std::vector<float>>& A, 
    std::vector<std::vector<float>>& B, 
    std::vector<std::vector<float>>& C, 
    int N, int iterations, int option) {

    // std::cout <<"     Entering run_program" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> diff;
    std::vector<std::vector<float>> result;
    // std::cout <<"     Rezising result vector: "<< C.size() << " " << N*N << std::endl;
    result.resize(N, std::vector<float>(N, 0.0f));
    float accum_times = 0.0;

    for (int i = 0; i<iterations; i++) {
        
        start = std::chrono::high_resolution_clock::now();
        switch(option) {
            case GPU_CUDA_1: 
                sum_matrices_tbb_cuda(A, B, C, result, N);
                break;
            // case GPU_CUDA_2: 
                // sum_matrices_graph_cuda(A, B, C, result, N);
                break;
            case GRAPH_BASED: 
                // std::cout <<"     Calling sum_matrices_tbb_graph" << std::endl;
                sum_matrices_tbb_graph(A, B, C, result, N);
                break;
            case PARA_REDUCE: 
                sum_matrices_tbb_parallel(A, B, C, result, N);
                break;
        }
        
        diff = std::chrono::high_resolution_clock::now() - start;
        double elapsed_time =  diff.count();
        accum_times += elapsed_time;
        // std::cout << "["<< i <<"] acum_times = " << accum_times << " " << elapsed_time << std::endl;
    }

    print(5, result);
    std::cout << "  "<< get_label(option) 
              <<": Time to sum matrices: " 
              << accum_times/iterations << " s\n";
}

int main(int argc, char* argv[]) {

    // Interpreting arguments
    int N = 1024; 
    int iterations = 1; // amount of iterations

    if (argc > 1) {
        // Collecting input parameters
        int input = std::stoi( argv[1] );
        N = input;
        if (argc == 3) {
            input = std::stoi( argv[2] );
            if (input != iterations) {
                iterations = input;
            }
        }
    }
    std::cout << "Using matrixes of size: " << N << " and "<< iterations << " iterations" << std::endl;

    // Size of the matrices
    std::vector<std::vector<float>> A, B, C;
    auto start = std::chrono::high_resolution_clock::now();

    // initialising matrices
    initialize_matrix(A, N, N);
    initialize_matrix(B, N, N);
    initialize_matrix(C, N, N);
    std::chrono::duration<float> diff = std::chrono::high_resolution_clock::now() - start;
    std::cout << "     Time to initialise matrixes: " << diff.count() << " s\n";

    // Start timing to initialise matrixes
    run_program (A, B, C, N, iterations, PARA_REDUCE);
    run_program (A, B, C, N, iterations, GRAPH_BASED);
    run_program (A, B, C, N, iterations, GPU_CUDA_1);

    return 0;
}
