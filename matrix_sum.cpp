#include "matrix_sum.h"

extern "C" void matrix_sum_cuda(const float* A, const float* B, const float* C, float* result, int N, int idx);

std::string get_label(int option) {

    std::string value;
    switch(option) {
        case GPU_CUDA_1: 
            value = "CUDA1";
            break;
        case GRAPH_BASED: 
            value = "GRAPH";
            break;
        case PARA_REDUCE: 
            value = "REDUCE";
            break;
        default:
            value = "";
    }

    return value;
}


void sum_matrices_tbb_cuda(
    std::vector<std::vector<float>>& A, 
    std::vector<std::vector<float>>& B, 
    std::vector<std::vector<float>>& C, 
    std::vector<std::vector<float>>& result, int N) {

    using namespace oneapi::tbb;
    int rows = N;
    int cols = N;
    // int total_sum = 0;

    parallel_for(blocked_range<size_t>(0, rows),
        [&](const blocked_range<size_t>& range) {

            for (size_t i = range.begin(); i < range.end(); ++i) {
                
                float* p_result = result[i].data();
                matrix_sum_cuda(
                    A[i].data(), 
                    B[i].data(), 
                    C[i].data(), 
                    p_result, N, i);
                
                // std::cout << "["<<i<<"] " ;
                // for (int jdx = 0; jdx<5; jdx++) 
                //     std::cout << p_result[jdx]<< " ";
                // std::cout << std::endl;

                // std::vector<float> v(p_result, p_result + N);
                // std::copy(result.at(i).begin(), result.at(i).end(), v.begin());
                std::copy(result[i].begin(), result[i].end(), p_result);
            }
        }
    );

}

void sum_matrices_tbb_graph(std::vector<std::vector<float>>& A, 
                          std::vector<std::vector<float>>& B, 
                          std::vector<std::vector<float>>& C, 
                          std::vector<std::vector<float>>& result, int N) {

    // std::cout << "  "<< get_label(GRAPH_BASED) <<":   Entering function" << std::endl;
    using namespace std;
    using namespace oneapi::tbb;
    using namespace oneapi::tbb::flow;
    size_t available_cores = oneapi::tbb::info::default_concurrency();
    int chunk_size = N / available_cores;

    // std::cout << "  "<< get_label(GRAPH_BASED) <<":   Initialising graph" << std::endl;
    tbb::flow::graph g;
    broadcast_node<continue_msg> start(g);
    std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> nodes;

    // std::cout << "  "<< get_label(GRAPH_BASED) <<":   Setting up nodes loop" << std::endl;
    for (int i = 0; i < available_cores; ++i) {
        nodes.emplace_back(g, [i, N, chunk_size, &A, &B, &C, &result](const tbb::flow::continue_msg&) {
            for (int j = i * chunk_size; j < (i + 1) * chunk_size; ++j) {
                for (int k = 0; k < N; ++k) {
                    result[j][k] = A[j][k] + B[j][k] + C[j][k];
                }
            }
        });
    }

    // std::cout << "  "<< get_label(GRAPH_BASED) <<":   Making start thread layout" << std::endl;
    for (int i = 1; i < available_cores; ++i) {
        tbb::flow::make_edge(start, nodes[i]);
        // tbb::flow::make_edge(nodes[i-1], nodes[i]);
    }

    nodes[0].try_put(tbb::flow::continue_msg());
    // std::cout << "  "<< get_label(GRAPH_BASED) <<":   Waiting for all" << std::endl;
    g.wait_for_all();

}

void sum_matrices_tbb_parallel(
    std::vector<std::vector<float>>& A, 
    std::vector<std::vector<float>>& B, 
    std::vector<std::vector<float>>& C, 
    std::vector<std::vector<float>>& result, int N) {

    using namespace oneapi::tbb;
    int rows = A.size();
    int cols = A[0].size();
    int total_sum = 0;

    parallel_for(blocked_range<size_t>(0, rows),
        [&](const blocked_range<size_t>& range) {

            int chunk_size = (range.end() - range.begin());
            std::vector<int> buffer(range.begin(), range.end()); 
            total_sum += chunk_size;

            // std::cout << "["<< buffer.size() <<", "<< chunk_size <<", "<< total_sum <<"]" << std::endl;

            for (size_t i = range.begin(); i < range.end(); ++i) {
                // std::cout << "  "<<i<<" "<< cols << std::endl;
                for (size_t j = 0; j < cols; ++j) {
                    // std::cout << "  "<<i<<" "<<j<< std::endl;
                    result[i][j] = A[i][j] + B[i][j] + C[i][j];
                }
            }
        }
    );
}

/*
    struct body {
        std::string my_name;
        body(const char *name) : my_name(name) {}
        void operator()(oneapi::tbb::flow::continue_msg) const {
            printf("%s\n", my_name.c_str());
        }
    };

    void sum_matrices_tbb_cpu(std::vector<std::vector<float>>& A, 
                            std::vector<std::vector<float>>& B, 
                            std::vector<std::vector<float>>& C, 
                            std::vector<std::vector<float>>& result, int N) {
        
        using namespace oneapi::tbb;
        using namespace oneapi::tbb::flow;

        int num_threads = task_scheduler_init::default_num_threads();
        int edges = 1; // 2; // depends on 'continue_node' calls

        graph g;
        broadcast_node<continue_msg> start(g);
        // continue_node<continue_msg> a(g, body("A"));
        continue_node<continue_msg> calculate(g, 
            [&](const continue_msg&) {
                parallel_for(blocked_range<size_t>(0, num_elements, num_elements / num_threads),
                    [&](const blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                            result[i] = A[i] + B[i] + C[i];
                        }
                    });
            }
        );

        make_edge(start, calculate);
        // make_edge(calculate, a);

        // Each broadcast noe is waiting for a single 
        //  continue_msg, since they both have only a 
        //  single predecessor, start
        for (int i = 0; i < edges; ++i) {
            start.try_put(continue_msg());
            g.wait_for_all();
        }
    }
*/
