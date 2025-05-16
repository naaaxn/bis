
#include <immintrin.h>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <numeric>


// AVX 邻居采样函数

    
extern "C" {


void avx_th_neighbor_sampling(const int* neighbors, const int* neighbor_counts, int n, int* sampled_neighbors,const int* num_samples) {
    auto sampling_task = [&](int node_idx) {
        int m = neighbor_counts[node_idx];  // 当前节点的邻居数量
        int temp_node_sampled_neighbors=0;
        int node_neighbors=0;
        for (int i = 0; i < node_idx; ++i) {
            node_neighbors += neighbor_counts[i];  // 确定当前节点的邻居在neighbors数组中的起始位置
            temp_node_sampled_neighbors+= num_samples[i]; //确定当前节点在结果数组中的起始位置
            }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, m - 1);
        std::vector<int> indices(m);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        int i = 0;
        while (i < num_samples[node_idx]) {
            int temp_batch_size = std::min(16, num_samples[node_idx] - i);
            alignas(64) int temp_indices[16] = {0}; 
            std::copy(indices.begin()+i, indices.begin() +i + temp_batch_size, temp_indices);

            __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(temp_indices));

            __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors+node_neighbors), sizeof(int));

            if (temp_batch_size == 16) {
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + temp_node_sampled_neighbors + i), gathered_neighbors);
            } else {
                alignas(64) int temp[16];
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp), gathered_neighbors);
                for (int j = 0; j < temp_batch_size; ++j) {
                    sampled_neighbors[i +temp_node_sampled_neighbors+j] = temp[j];
                }
            }

            i += temp_batch_size;
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n);

    for (int node_idx = 0; node_idx < n; ++node_idx) {
        threads.emplace_back(sampling_task, node_idx);
    }

    for (auto& t : threads) {
        t.join();
    }
}

void avx_neighbor_sampling(const int* neighbors, int n, int num_samples, int* sampled_neighbors) {
    if (num_samples > n) {
        num_samples = n;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    std::vector<int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::shuffle(numbers.begin(), numbers.end(), gen);
    
    int i = 0;
    while (i < num_samples) {
        
        int temp_batch_size = std::min(16, num_samples - i);
        
        
        std::shuffle(numbers.begin(), numbers.end(), gen);
        int indices[16] = {0}; 
        std::copy(numbers.begin()+i, numbers.begin() + temp_batch_size+i, indices);

        
        __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices));
        
        __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors), sizeof(int));
        
        if (temp_batch_size == 16) {
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i), gathered_neighbors);
        } else {           
            alignas(32) int temp[16];
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp), gathered_neighbors);
            for (int j = 0; j < temp_batch_size; ++j) {
                sampled_neighbors[i + j] = temp[j];
            }
        }

        i += temp_batch_size;
    }
}


}