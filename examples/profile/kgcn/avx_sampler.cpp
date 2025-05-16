#include <immintrin.h>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <numeric>

extern "C" {

int max_threads=60;

void avx_th_neighbor_sampling(const int* neighbors, const int* neighbor_counts, int n, int* sampled_neighbors, const int* num_samples) {
    auto sampling_task = [&](int start_idx, int end_idx) {
        for (int node_idx = start_idx; node_idx < end_idx; ++node_idx) {
            int m = neighbor_counts[node_idx];  // 当前节点的邻居数量
            int temp_node_sampled_neighbors = 0;
            int node_neighbors = 0;

            for (int i = 0; i < node_idx; ++i) {
                node_neighbors += neighbor_counts[i];  // 确定当前节点的邻居在neighbors数组中的起始位置
                temp_node_sampled_neighbors += num_samples[i]; // 确定当前节点在结果数组中的起始位置
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
                std::copy(indices.begin() + i, indices.begin() + i + temp_batch_size, temp_indices);

                __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(temp_indices));

                __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors + node_neighbors), sizeof(int));

                if (temp_batch_size == 16) {
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + temp_node_sampled_neighbors + i), gathered_neighbors);
                } else {
                    alignas(64) int temp[16];
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp), gathered_neighbors);
                    for (int j = 0; j < temp_batch_size; ++j) {
                        sampled_neighbors[temp_node_sampled_neighbors + i + j] = temp[j];
                    }
                }

                i += temp_batch_size;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(max_threads);

    for (int start_idx = 0; start_idx < n; start_idx += max_threads) {
        int end_idx = std::min(start_idx + max_threads, n);
        
        // 创建线程并传递开始和结束索引
        threads.emplace_back(sampling_task, start_idx, end_idx);
    }

    for (auto& t : threads) {
        t.join();
    }
}

}