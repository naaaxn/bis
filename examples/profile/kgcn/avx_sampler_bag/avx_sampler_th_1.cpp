#include <immintrin.h>
#include <vector>
#include <random>
#include <algorithm>
#include <thread>
#include <numeric>
#include <unordered_set>
#include <cstdlib>  // for rand()
#include <ctime>    // for time()
#include <cstring>
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <omp.h>
#include <cassert>
#include <atomic>
using namespace std;

std::vector<std::vector<int>> map;
int flag=0;
std::vector<int> seed_nodes_mapped;
// AVX 邻居采样函数
struct csr{
    int num_nodes;
    int etypes;
    int src_vtype;
    int dst_vtype;
    int* inptr;
    int* indices;
    int* data;
};

struct heterograph{
    int num_nodes;
    int num_edges;
    int etypes;
    int src_vtype;
    int dst_vtype;
    int* src_node;
    int* dst_node;
    int* data;
};
    
extern "C" {

bool BoolCompareAndSwap(int* addr, int expected, int desired) {
    return __atomic_compare_exchange_n(
        addr,         // 地址
        &expected,    // 期望值的指针
        desired,      // 要设置的新值
        false,        // 是否弱CAS（false 表示强CAS）
        __ATOMIC_SEQ_CST, // 内存顺序（强一致性）
        __ATOMIC_SEQ_CST  // 内存顺序（失败时）
    );
}

void reverseElements(int arr[], int n) {

    // 反转从位置 0 到 n-1 的元素
    for (int i = 0; i < n / 2; ++i) {
        // 交换元素
        int temp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = temp;
    }
}

void construct_map(const int* a, int max_value, int* map){
    for(int i=0;i<max_value;i++){
        map[a[i]]=i;
    }
}

void construct_layer_off_data(const int* layer_sizes, int num_layers, int* layer_off_data) {
    // 初始化 layer_off_data，长度比 layer_sizes 多 1，因为需要存储最后的偏移量
    layer_off_data[0] = 0;
    // 从最外层到最内层，倒序计算
    for (int i = 1; i <= num_layers; ++i) {
        // layer_off_data[i] 是前一层的偏移量 + 前一层的节点数
        layer_off_data[i] = layer_off_data[i - 1] + layer_sizes[num_layers - i];
    }
}

void find_edges_avx512(const int* v_array, int v_size, const int* indptr, const int* indices, const int* data,
                       int node, int* output_edge_ids) {
    int start = indptr[node];       // 获取邻居节点在 indices 中的起始位置
    int end = indptr[node + 1];     // 获取邻居节点在 indices 中的结束位置
    int size = end - start;         // 邻居节点的数量

    // 初始化输出数组，默认值为 -1，表示未找到边
    std::fill(output_edge_ids, output_edge_ids + v_size, -1);

    // 使用 AVX-512 进行并行查找
    const int block_size = 16;

    for (int i = 0; i < v_size; i += block_size) {
        int batch_size = std::min(block_size, v_size - i);  // 处理的目标节点批量大小

        // 加载 v_array 中的批量目标节点到 AVX-512 寄存器
        __m512i target_vec = _mm512_loadu_si512(&v_array[i]);

        // 遍历 indices 中节点 u 的邻居，查找这些目标节点
        for (int j = 0; j < size; j += block_size) {
            int current_batch_size = std::min(block_size, size - j);

            // 加载邻居节点到 AVX-512 寄存器
            __m512i neighbor_vec = _mm512_loadu_si512(&indices[start + j]);

            // 并行比较目标节点和邻居节点
            __mmask16 mask = _mm512_cmpeq_epi32_mask(neighbor_vec, target_vec);

            // 对于匹配的节点，将对应的边 ID 存储到 output_edge_ids 中
            if (mask != 0) {
                // 计算出匹配的每个位置
                for (int k = 0; k < current_batch_size; ++k) {
                    if (mask & (1 << k)) {
                        int v_index = i + k;  // 目标节点在 v_array 中的索引
                        output_edge_ids[v_index] = data[start + j + k];  // 存储边 ID
                    }
                }
            }
        }
    }
}

int prepare_for_sample(
    const int* seed_nodes,    //种子节点
    const int* indptr,        // CSC indptr 数组
    const int* indices,       // CSC indices 数组
    int num_samples,          //采样数量，为下面采样做准备
    int num_nodes,            // 节点总数
    int* src_node,            // 输出：源节点数组（需要足够大）
    int* neighbor_count       // 输出：写入的边数量             
) {
    int temp_count = 0,count=0;
    src_node[0]=0;
    for (int node = 0; node < num_nodes; ++node) {
        int dst=seed_nodes[node];
        int start = indptr[dst];
        int end = indptr[dst + 1];
        int num_edges = end - start;
        neighbor_count[node]=min(num_edges,num_samples);
        temp_count=count;
        src_node[node+1]=src_node[node]+min(num_edges,num_samples);
        count+=min(num_edges,num_samples);
    }
    return count;
}

void fill_array(const std::vector<int>& array,int* goal_array,int size,int start_place){
    int j=0;
     while (j < size) {
            int batch_size = std::min(16, size - j);
            if (batch_size == 16) {
                __m512i neighbors_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(array.data() + j));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(goal_array + j+ start_place), neighbors_vec);
            } else {
                for (int z = 0; z < batch_size; ++z) {
                    goal_array[j+start_place+z] = array[j+z];
                }
            }
            j += batch_size;
        }
    }


void avx512_fill_map_from_goal(const int* goal_array, int* map_array, int goal_size, int map_value) {
    int batch_size = 16;  // 每次处理 16 个元素
    __m512i fill_value = _mm512_set1_epi32(map_value);  // 填充的值，比如设为 1
    
    for (int i = 0; i < goal_size; i += batch_size) {
        int size = std::min(batch_size, goal_size - i);
        
        // 加载 `goal_array` 的一部分到向量寄存器
        __m512i indices = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(goal_array + i));
        
        // 使用 scatter 指令将 fill_value 填充到 `map_array` 的索引处
        if (size == batch_size) {
            _mm512_i32scatter_epi32(map_array, indices, fill_value, 4);
        } else {
            // 对于不满 batch_size 的情况，手动处理
            alignas(32) int temp[16];
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp), indices);
            for (int j = 0; j < size; ++j) {
                map_array[temp[j]] = map_value;
            }
        }
    }
}

void fill_array_from_array(const int* source_array, int* goal_array, int size, int start_place) {
    int j = 0;
    while (j < size) {
        // 选择每次处理的批量大小，最多16个元素，确保不超出数组大小
        int batch_size = std::min(16, size - j);
        
        // 加载源数组的批量数据到向量寄存器中
        // 如果 batch_size 刚好等于 16，则直接将数据写入目标数组
        if (batch_size == 16) {
            __m512i neighbors_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(source_array + j));
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(goal_array + j + start_place), neighbors_vec);
        } else {
            for (int z = 0; z < batch_size; ++z) {
                goal_array[j + start_place + z] = source_array[j+z];
            }
        }
        j += batch_size;
    }
}

void fill_array_from_array_th(const int* source_array, int* goal_array, int size, int start_place, int num_threads) {
    // 确定每个线程要处理的块大小
    int chunk_size = size / num_threads;
    int remainder = size % num_threads;

    // 创建future用于管理线程任务
    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(std::launch::async, [=]() {
            int local_start = i * chunk_size;
            int local_size = (i == num_threads - 1) ? chunk_size + remainder : chunk_size;
            fill_array_from_array(source_array + local_start, goal_array, local_size, local_start + start_place);
        }));
    }

    // 等待所有任务完成
    for (auto& f : futures) {
        f.get();
    }
}


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
            //std::copy(indices.begin()+i, indices.begin() +i + temp_batch_size, temp_indices);
            fill_array_from_array(indices.data()+i,temp_indices,temp_batch_size,0);
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


void avx_th_neighbor_sampling_with_mixed_precision(
    const int* neighbors, const int* neighbor_counts, int n, 
    int* sampled_neighbors, const int num_samples,
    int* edge_index, int* result_index) 
{
    int max_threads=8;
    // 确保线程数不超过最大线程数
    int thread_count = std::min(max_threads, n);

    // 计算每个线程的任务范围
    int nodes_per_thread = n / thread_count;
    int remaining_nodes = n % thread_count;
    std::random_device rd;
    std::mt19937 gen(rd());
    // 将任务分配给线程
    auto sampling_task = [&](int start_node_idx, int end_node_idx) {
        for (int node_idx = start_node_idx; node_idx < end_node_idx; ++node_idx) {
            int node_neighbors = 0;
            int m = neighbor_counts[node_idx];  // 当前节点的邻居数量
            int temp_node_sampled_neighbors = 0;
            std::uniform_int_distribution<> dis(0, m-1);
            std::vector<int> indices(m);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            for (int i = 0; i < node_idx; ++i) {
                node_neighbors += neighbor_counts[i];  // 确定当前节点的邻居在 neighbors 数组中的起始位置
            }
            temp_node_sampled_neighbors = node_idx * num_samples; // 确定当前节点在结果数组中的起始位置
            int i = 0;
            while (i < num_samples) {
                int temp_batch_size = std::min(16, num_samples - i); // 一次处理最多 16 个
                if (temp_batch_size == 16) {
                    __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices.data() + i));
                    __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors+node_neighbors), sizeof(int));
                    __m512i gathered_edge = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(edge_index+node_neighbors), sizeof(int));
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i+temp_node_sampled_neighbors), gathered_neighbors);
                    _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_index + i+temp_node_sampled_neighbors), gathered_edge);
                } 
                 else { // temp_batch_size <= 16
                    // 存储所有的元素逐个存储
                    for (int j = 0; j < temp_batch_size; ++j) {
                        sampled_neighbors[i +temp_node_sampled_neighbors+j] = neighbors[node_neighbors+indices[i+j]];
                        result_index[i +temp_node_sampled_neighbors+j]= edge_index[node_neighbors+indices[i+j]];
                    }
                }
                i += temp_batch_size;
            }
        }
    };

    // 创建线程池，分配节点任务
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    int start_node_idx = 0;
    for (int i = 0; i < thread_count; ++i) {
        int end_node_idx = start_node_idx + nodes_per_thread;
        if (i < remaining_nodes) {
            end_node_idx++;  // 为了平衡剩余节点的工作负载
        }

        threads.emplace_back(sampling_task, start_node_idx, end_node_idx);
        start_node_idx = end_node_idx;
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

void fill_array_with_one(int* array, int value, int count) {
    // 通过 _mm512_set1_epi32 将 'value' 复制到 512 位寄存器中的 16 个元素
    __m512i val_vec = _mm512_set1_epi32(value);
    
    // 计算需要使用 AVX-512 的部分
    int vec_size = 16;
    int full_chunks = count / vec_size;  // 完整的 16 次块
    int remainder = count % vec_size;    // 剩余不足 16 的部分

    // 用 AVX-512 处理完整的 16 个元素块
    for (int i = 0; i < full_chunks * vec_size; i += vec_size) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(array + i), val_vec);
    }

    // 处理不足 16 的剩余部分
    if (remainder > 0) {
        // 使用普通循环逐个填充剩下的元素
        for (int i = full_chunks * vec_size; i < count; ++i) {
            array[i] = value;
        }
    }
}

void avx_th_neighbor_edges_sampling(
    const int* neighbors, const int* neighbor_counts,const int* node_place, int* seed_node,std::vector<int>& map,int map_seed_node,
    int& map_size,int n,
    int* sampled_indptr,
    int* sampled_neighbors, const int num_samples,
    int* edge_index, int* result_index) 
{
    int max_threads=8;
    std::vector<int> temp_sampled_indptr(n,0);
    // 确保线程数不超过最大线程数
    int thread_count = std::min(max_threads, n);
    // 计算每个线程的任务范围
    int nodes_per_thread = n / thread_count;
    int remaining_nodes = n % thread_count;
    std::random_device rd;
    std::mt19937 gen(rd());
    // 将任务分配给线程
    auto sampling_task = [&](int start_node_idx, int end_node_idx) {
        for (int node_idx = start_node_idx; node_idx < end_node_idx; ++node_idx) {
            int node_neighbors = 0;
            int m = neighbor_counts[node_idx];  // 当前节点的邻居数量
            int temp_node_sampled_neighbors = node_place[node_idx];
            std::uniform_int_distribution<> dis(0, m-1);
            std::vector<int> indices(m);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            for (int i = 0; i < node_idx; ++i) {
                node_neighbors += neighbor_counts[i];  // 确定当前节点的邻居在 neighbors 数组中的起始位置
            }
            if(m<=num_samples){
               fill_array_from_array(neighbors+node_neighbors,sampled_neighbors,m,temp_node_sampled_neighbors); 
            }
            else{
               int i = 0;
                while (i < num_samples) {
                    int temp_batch_size = std::min(16, num_samples - i); // 一次处理最多 16 个
                    if (temp_batch_size == 16) {
                        __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices.data() + i));
                        __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors+node_neighbors), sizeof(int));
                        __m512i gathered_edge = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(edge_index+node_neighbors), sizeof(int));
                        _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i+temp_node_sampled_neighbors), gathered_neighbors);
                        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_index + i+temp_node_sampled_neighbors), gathered_edge);
                    } 
                    else { // temp_batch_size <= 16
                        // 存储所有的元素逐个存储
                        for (int j = 0; j < temp_batch_size; ++j) {
                            sampled_neighbors[i +temp_node_sampled_neighbors+j] = neighbors[node_neighbors+indices[i+j]];
                            result_index[i +temp_node_sampled_neighbors+j]= edge_index[node_neighbors+indices[i+j]];
                        }
                    }
                    i += temp_batch_size;
                }
            }
        }
    };

    // 创建线程池，分配节点任务
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    int start_node_idx = 0;
    for (int i = 0; i < thread_count; ++i) {
        int end_node_idx = start_node_idx + nodes_per_thread;
        if (i < remaining_nodes) {
            end_node_idx++;  // 为了平衡剩余节点的工作负载
        }

        threads.emplace_back(sampling_task, start_node_idx, end_node_idx);
        start_node_idx = end_node_idx;
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    int seed_node_size=n;
    int num_threads_col=8;
    int num_cols=seed_node_size;
    std::vector<std::vector<int>> src_nodes_local(num_threads_col); // 每个线程的局部节点集合
    std::vector<int> global_prefix_col(num_threads_col + 1, 0);    // 全局前缀数组

    // 并行处理
    if(map_seed_node==0){
#pragma omp parallel num_threads(num_threads_col)
    {
        int thread_id = omp_get_thread_num();
        num_threads_col = omp_get_num_threads();

        int start_i = thread_id * (num_cols / num_threads_col) +min(thread_id, num_cols % num_threads_col);
        int end_i = (thread_id + 1) * (num_cols / num_threads_col) +min((thread_id + 1), num_cols % num_threads_col);

        assert(thread_id + 1 < num_threads_col || end_i == num_cols);

        // 局部收集新节点
        for (int i = start_i; i < end_i; ++i) {
            int picked_idx = seed_node[i];
            bool spot_claimed=1; 
            if(picked_idx<map_size){
                spot_claimed = BoolCompareAndSwap(&map[picked_idx],-1,i);
            }
            if (spot_claimed) src_nodes_local[thread_id].push_back(picked_idx);
        }

        global_prefix_col[thread_id + 1] = src_nodes_local[thread_id].size();

#pragma omp barrier
#pragma omp master
        {
            global_prefix_col[0] = map_size;
            for (int t = 0; t < num_threads_col; ++t) {
                global_prefix_col[t + 1] += global_prefix_col[t]; // 更新全局前缀数组
            }
        }

#pragma omp barrier
        // 更新全局映射关系
        int mapping_shift = global_prefix_col[thread_id];
        for (size_t i = 0; i < src_nodes_local[thread_id].size(); ++i) {
            map[src_nodes_local[thread_id][i]] = mapping_shift + i;
        }

#pragma omp barrier
        // 更新列索引到全局映射
        for (int i = start_i; i < end_i; ++i) {
            int picked_idx = seed_node[i];
            int mapped_idx = map[picked_idx];
            seed_node[i] = mapped_idx;
        }
    }
    map_size=(global_prefix_col.back());
}
int temp_max_node=0;
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int old_index = seed_node[i];                          // 原始节点索引
        int new_index = map[old_index];             // 映射后的新节点索引
        temp_max_node=max(temp_max_node,new_index);
        temp_sampled_indptr[new_index] = node_place[old_index+1]-node_place[old_index];  // 将原 indptr 的值拷贝到新位置
    }
    // 计算新的累积 indptr 范围
    int temp_temp=0;
    for (int i = 0; i <=temp_max_node; ++i) {
        fill_array_with_one(sampled_indptr+temp_temp,i,temp_sampled_indptr[i]);
        temp_temp+=temp_sampled_indptr[i];
    }
}

void avx_mixed_precision_neighbor_sampling(const int* neighbors, int n, int num_samples, int* sampled_neighbors,int* edge_index,int* result_index) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    int i = 0;
    if (n < 65536) {
    std::vector<unsigned short int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::shuffle(numbers.begin(), numbers.end(), gen);
    // 使用 16 位指令的情况
    while (i < num_samples) {
        int temp_batch_size = std::min(32, num_samples - i); // 一次处理最多 32 个
        if(temp_batch_size==32){
           // 选择随机索引
            // 使用 AVX-512 加载 32 个 16 位整数
            __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(numbers.data()+i));

            // 分别扩展为 32 位整数的低 16 位和高 16 位
            __m512i expanded_indices_low = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(index_vec, 0));
            __m512i expanded_indices_high = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(index_vec, 1));

            // 从邻居数组中采样
            __m512i gathered_neighbors_low = _mm512_i32gather_epi32(expanded_indices_low, reinterpret_cast<const void*>(neighbors), sizeof(int));
            __m512i gathered_neighbors_high = _mm512_i32gather_epi32(expanded_indices_high, reinterpret_cast<const void*>(neighbors), sizeof(int));
    
            //把边的index也采出来：
            __m512i gathered_edge_low = _mm512_i32gather_epi32(expanded_indices_low, reinterpret_cast<const void*>(edge_index), sizeof(int));
            __m512i gathered_edge_high = _mm512_i32gather_epi32(expanded_indices_high, reinterpret_cast<const void*>(edge_index), sizeof(int));
            // 将结果分成两部分存储
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i), gathered_neighbors_low);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i + 16), gathered_neighbors_high);
            // 将结果分成两部分存储
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_index + i), gathered_edge_low);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_index + i + 16), gathered_edge_high);
        }
        else{
            for (int j = 0; j < temp_batch_size; ++j) {
                        sampled_neighbors[i +j] = neighbors[numbers[i+j]];
                        result_index[i+j]= edge_index[numbers[i+j]];
                    }
        }
        i += temp_batch_size;
    }
} else {
    // 使用 32 位指令的情况
    std::vector<int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::shuffle(numbers.begin(), numbers.end(), gen);
    while (i < num_samples) {
        int temp_batch_size = std::min(16, num_samples - i);
        // 使用 32 位指令
        __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(numbers.data()+i));
        __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors), sizeof(int));
        __m512i gathered_edge = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(edge_index), sizeof(int));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i), gathered_neighbors);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_index + i), gathered_edge);
        i += temp_batch_size;
    }
}

}


void avx_negate_sample_neighbors(const int* neighbors, size_t total_neighbors, size_t sample_size, int* sampled_neighbors,int* edge_index,int* result_index) {
    size_t exclude_size = total_neighbors - sample_size;
    
    // 生成要排除的随机索引
    std::vector<size_t> exclude_indices(total_neighbors);
    std::iota(exclude_indices.begin(), exclude_indices.end(), 0); // 填充索引 0 到 total_neighbors-1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(exclude_indices.begin(), exclude_indices.end(), gen);
    std::sort(exclude_indices.begin(), exclude_indices.begin()+exclude_size); // 排序
    auto it = exclude_indices.begin();
    size_t j = 0, k = 0;
    size_t i = 0;
    for (; i < total_neighbors && it != exclude_indices.begin()+exclude_size; ++i) {
        if (*it == i) {
            it++;
            continue;
        }
        result_index[j]=edge_index[i];
        sampled_neighbors[j++]=neighbors[i];
    }
    fill_array_from_array(neighbors + i, sampled_neighbors, sample_size - j, j);
    fill_array_from_array(edge_index + i, result_index, sample_size - j, j);
}


int avx512_filter(const int* input_array, const int* map, int* output_array, int size,const int* data,int* edge_mapping) {
    int output_index = 0;
    // 遍历数组
    for (int i = 0; i < size; i += 16) {
        // 计算当前处理的批次大小
        int batch_size = std::min(16, size - i);
        
        // 加载 16 个 input_array、map 和 data 的元素
        __m512i input_vec = _mm512_loadu_si512(&input_array[i]);
        __m512i map_vec = _mm512_loadu_si512(&map[i]);
        __m512i data_vec = _mm512_loadu_si512(&data[i]);

        // 生成掩码，保留 map 数组中为 1 的位置
        __mmask16 mask = _mm512_test_epi32_mask(map_vec, _mm512_set1_epi32(1));

        // 根据掩码过滤 input_array 中的元素，并存储到 output_array 中
        _mm512_mask_compressstoreu_epi32(&output_array[output_index], mask, input_vec);

        // 根据掩码将 data 数组中对应位置的元素写入 edge_mapping
        _mm512_mask_compressstoreu_epi32(&edge_mapping[output_index], mask, data_vec);

        // 计算存储了多少个元素
        output_index += _mm_popcnt_u32(mask);
    }

    // 处理剩余元素（如果有）
    for (int i = size - (size % 16); i < size; ++i) {
        if (map[i] == 1) {
            output_array[output_index] = input_array[i];
            edge_mapping[output_index] = data[i];
            output_index++;
        }
    }
    return output_index;
}


int avx512_filter_need(const int* input_array, const int* map, int* output_array, int size) {
    int output_index = 0;
    if(size>=16){
       for (int i = 0; i < size; i += 16) {
    // 加载 16 个 input_array 的元素
    __m512i input_vec = _mm512_loadu_si512(&input_array[i]);

    // 创建一个用于存储 map[input_array[i]] 的数组
    alignas(64) int temp[16];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(temp), input_vec);

    // 生成一个掩码来保留 map[input_array[i]] == 0 的位置
    __mmask16 mask = 0;
    for (int j = 0; j < 16; ++j) {
        if (map[temp[j]] == 0) {
            mask |= (1 << j); // 如果 map[input_array[i]] == 0，设置掩码的相应位
        }
    }

    // 根据掩码过滤 input_array 中的元素，并存储到 output_array 中
    _mm512_mask_compressstoreu_epi32(&output_array[output_index], mask, input_vec);

    // 计算存储了多少个元素
    output_index += _mm_popcnt_u32(mask);}
    }
    for (int i = size - (size % 16); i < size; ++i) {
        if (map[input_array[i]] == 0) {
            output_array[output_index] = input_array[i];
            output_index++;
        }
    }
    return output_index;
}


void build_csr_from_edges_with_max_node(
    int* source, 
    int* target,
    size_t num_edges, 
    int max_node_id, 
    std::vector<int>& inptr, 
    std::vector<int>& indices) 
{
    // Step 1: 初始化入度数组, 用于记录源节点的邻居数量
    std::vector<int> in_degrees(max_node_id + 1, 0);
    std::mutex degree_mutex;

    // 多线程统计入度
    auto count_in_degrees = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            int src = source[i];
            if (src <= max_node_id) {
                std::lock_guard<std::mutex> lock(degree_mutex);
                ++in_degrees[src];  // 只统计源节点的入度
            }
        }
    };
    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = (num_edges + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_edges);
        if (start < end) {
            threads.emplace_back(count_in_degrees, start, end);
        }
    }
    for (auto& th : threads) {
        th.join();
    }

    // Step 2: 构造 inptr 数组，inptr[i+1] - inptr[i] 表示以 i 为源节点的邻居数量
    inptr.resize(max_node_id + 2, 0);  // inptr 数组的大小
    std::partial_sum(in_degrees.begin(), in_degrees.end(), inptr.begin() + 1);  // 累加入度

    // Step 3: 填充 indices 数组
    indices.resize(inptr.back());  // 根据 inptr 最后的值确定 indices 大小
    std::vector<int> fill_position(max_node_id + 1, 0);
    std::mutex index_mutex;

    // 填充 indices 数组的多线程函数
    auto fill_indices = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            int src = source[i];
            int tgt = target[i];

            // 只将源节点的邻居填入 indices 中
            if (src <= max_node_id && tgt <= max_node_id) {
                int pos = inptr[src] + fill_position[src];
                {
                    std::lock_guard<std::mutex> lock(index_mutex);
                    indices[pos] = tgt;  // 将目标节点添加到 indices
                    ++fill_position[src];
                }
            }
        }
    };

    threads.clear();
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_edges);
        if (start < end) {
            threads.emplace_back(fill_indices, start, end);
        }
    }
    for (auto& th : threads) {
        th.join();
    }
}

void avx_copy_with_indices(const int* source_array, const int* indices, int* target_array, int num_elements) {
    // 遍历索引数组并使用 AVX-512 指令进行元素复制
    int i = 0;
    for (; i <= num_elements - 16; i += 16) {
        // 从索引数组中获取当前批次的索引
        __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));

        // 从源数组中根据索引加载对应的值
        __m512i values_vec = _mm512_i32gather_epi32(index_vec, source_array, 4);

        // 将加载的值存储到目标数组中
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(target_array + i), values_vec);
    }

    // 处理不足16的部分
    for (; i < num_elements; ++i) {
        target_array[i] = source_array[indices[i]];
    }
}



void neighbor_sampler(
    const int* indptr,        // CSC 形式的 indptr 数组
    const int* indices,       // CSC 形式的 indices 数组
    const int* data,          // 边的信息（可以是边的ID或权重等）
    int indptr_size,          // indptr 数组的大小
    int indices_size,         // indices 数组的大小
    int data_size,            // data 数组的大小
    int batch_size,           // 每批种子节点的大小
    int num_neighbors,        // 每个节点的邻居采样数量
    const int* seed_nodes,    // 初始的种子节点
    int seed_nodes_size,      // 种子节点的大小
    int num_hops,             // 采样的跳数
    int current_nodeflow_index,
    int* node_mapping,        // 输出的 node_mapping 数组
    int* edge_mapping,        // 输出的 edge_mapping 数组
    int* layer_offsets,       // 输出的层偏移量数组
    int* block_offsets,       // 输出的块偏移量数组
    int* indptr_out,            // 输出的源节点数组
    int* indices_out,             // 输出的目标节点数组
    int* data_out             // 输出的边数组
) {
    //auto time_start = chrono::high_resolution_clock::now();
    int max_node_id = indptr_size - 1;
    int map[max_node_id + 1];  // 用来记录哪些节点已经访问过
    memset(map, 0, (max_node_id + 1) * sizeof(int));
    int src_node[max_node_id + 1];         // 输出的源节点数组
    int dst_node[max_node_id + 1];             // 输出的目标节点数组
    // 初始化 layer_offsets 和 block_offsets
    int layer_offsets_size = 0;
    int block_offsets_size = 0;
    int start=(current_nodeflow_index)*batch_size;
    if(start>=seed_nodes_size){
        return;
    }
    int temp_seed_nodes_size=(seed_nodes_size<=start+batch_size)?seed_nodes_size-start:batch_size;
    int start_point=temp_seed_nodes_size;
    // 处理 seed_nodes
    int node_mapping_size = temp_seed_nodes_size;
    int temp_node_mapping_size=0;
    int temp_edge_mapping_size=0;
    fill_array_from_array(seed_nodes+start,node_mapping,temp_seed_nodes_size,0);
    int temp_seed_nodes[max_node_id + 1];
    fill_array_from_array(seed_nodes+start,temp_seed_nodes,temp_seed_nodes_size,0);
    avx512_fill_map_from_goal(temp_seed_nodes,map,temp_seed_nodes_size,1);
    layer_offsets[layer_offsets_size++] = temp_seed_nodes_size;  // 初始种子节点层偏移量
    temp_node_mapping_size=temp_seed_nodes_size;
    int edge_mapping_size=0;
    int src_node_size=0;
    int neighbors[max_node_id +1];
    //double time_case_1 = 0, time_case_2 = 0, time_case_3 = 0, time_loop_check = 0, time_indptr_construction = 0;
    // 开始多跳采样
    for (int hop = 0; hop < num_hops; ++hop) {
        // 遍历当前种子节点
        int temp_sum=0;
        int temp_next_neighbors_num=0;
        int sampled_neighbors[max_node_id +1];
        for (int i = 0; i < temp_seed_nodes_size; ++i) {
            int node = temp_seed_nodes[i];
            int start = indptr[node];
            int end = indptr[node + 1];
            //cout<<node<<" "<<end-start<<endl;
            int input[end-start];
            fill_array_from_array(indices+start,input,end-start,0);
            int output[end-start];
            int out_egdes[end-start];
            //const int* const_map= static_cast<const int*>(map);
            auto start_loop_check = chrono::high_resolution_clock::now();
            int out_num=0;
            for(int jk=0;jk<end-start;jk++){
                if (map[input[jk]] == 0) {
                    output[out_num] = input[jk];
                    out_egdes[out_num]=jk+start;
                    out_num++;
                    map[input[jk]]=1;
                }
            }
            // auto end_loop_check = chrono::high_resolution_clock::now();
            // time_loop_check += chrono::duration<double>(end_loop_check - start_loop_check).count();
            // cout<<out_num<<endl;
            const int* const_output= static_cast<const int*>(output);
            //cout<<"jjj"<<endl;
            if(out_num<=num_neighbors){
               auto start_case_1 = chrono::high_resolution_clock::now();
               fill_array_with_one(src_node+src_node_size,temp_seed_nodes[i],out_num);
            // avx_copy_with_indices(data,const_output,data_out+src_node_size,out_num);
               src_node_size+=out_num; 
               avx_copy_with_indices(data,out_egdes,edge_mapping+edge_mapping_size,out_num);
               edge_mapping_size+=out_num;
               fill_array_from_array(const_output,node_mapping,out_num,node_mapping_size);
               temp_next_neighbors_num+=out_num;
               //edge_mapping_size+=out_num;
               node_mapping_size+=out_num;
            }
            else if(out_num<2*num_neighbors){
               auto start_case_2 = chrono::high_resolution_clock::now();
               avx512_fill_map_from_goal(const_output,map,out_num,0);
               //fill_array_from_array(const_output,neighbors,out_num,0);
               avx_negate_sample_neighbors(const_output,out_num,num_neighbors,node_mapping+node_mapping_size,out_egdes,edge_mapping+edge_mapping_size);
               edge_mapping_size+=num_neighbors;
               fill_array_with_one(src_node+src_node_size,temp_seed_nodes[i],num_neighbors);
               src_node_size+=num_neighbors;
               avx512_fill_map_from_goal(node_mapping+node_mapping_size,map,num_neighbors,1);
               temp_next_neighbors_num+=num_neighbors;
               node_mapping_size+=num_neighbors;
            }
            else{
               auto start_case_3 = chrono::high_resolution_clock::now();
               avx512_fill_map_from_goal(const_output,map,out_num,0);
               avx_mixed_precision_neighbor_sampling(const_output,out_num,num_neighbors,node_mapping+node_mapping_size,out_egdes,edge_mapping+edge_mapping_size);
               edge_mapping_size+=num_neighbors;
               fill_array_with_one(src_node+src_node_size,temp_seed_nodes[i],num_neighbors);
               src_node_size+=num_neighbors;
               avx512_fill_map_from_goal(node_mapping+node_mapping_size,map,num_neighbors,1);
               node_mapping_size+=num_neighbors;
               temp_next_neighbors_num+=num_neighbors;
            }
        }
        // 更新 node_mapping, edge_mapping, layer_offsets 和 block_offsets
        // 更新 layer_offsets
        layer_offsets[layer_offsets_size++] = node_mapping_size-temp_node_mapping_size;
        block_offsets[block_offsets_size++] = edge_mapping_size-temp_node_mapping_size;
        fill_array_from_array(node_mapping+temp_node_mapping_size,temp_seed_nodes,node_mapping_size-temp_node_mapping_size,0);//将该层节点作为下一层的种子节点
            //将该层节点作为下一层的种子
        temp_seed_nodes_size=node_mapping_size-temp_node_mapping_size;
        temp_node_mapping_size=node_mapping_size;
        temp_edge_mapping_size=edge_mapping_size;
    }
    node_mapping[data_size +5]=node_mapping_size;
    edge_mapping[data_size +5]=edge_mapping_size;
    int max_node = 0;
    for (int i = 0; i < src_node_size; ++i) {
        max_node = std::max(max_node, std::max(src_node[i], node_mapping[i+min(seed_nodes_size-start,batch_size)]));
    }
    node_mapping[data_size +6]=max_node;
    fill_array_from_array(node_mapping+min(seed_nodes_size-start,batch_size),dst_node,node_mapping_size-min(seed_nodes_size-start,batch_size),0);
    const int* const_src_node= static_cast<const int*>(src_node);
    const int* const_dst_node= static_cast<const int*>(dst_node);
    const int* const_edge_mapping= static_cast<const int*>(edge_mapping);
    int num_nodes = max_node + 1;

    // 初始化 indptr 数组
    memset(indptr_out, 0, (num_nodes) * sizeof(int));
    auto start_indptr_construction = chrono::high_resolution_clock::now();
    // 计算每个节点的出度
    for (int i = 0; i < edge_mapping_size; ++i) {
        indptr_out[const_src_node[i] + 1]++;  // 更新每个节点的出度
    }

    // 计算累积和以填充 indptr
    for (int i = 1; i <= num_nodes; ++i) {
        indptr_out[i] += indptr_out[i - 1];
    }

    // 填充 indices 和 data 数组
    for (int i = 0; i < edge_mapping_size; ++i) {
        int node = const_src_node[i];
        int pos = indptr_out[node]++;  // 计算当前边的索引位置
        indices_out[pos] = const_dst_node[i];  // 填充 indices
    }

    // 恢复 indptr 数组
    for (int i = num_nodes; i > 0; --i) {
        indptr_out[i] = indptr_out[i - 1];  // 恢复 indptr 数组
    }
    indptr_out[0] = 0;  // 将 indptr[0] 设置为 0
    int temp[num_hops+2];
    std::copy(layer_offsets, layer_offsets + num_hops+2, temp);
    construct_layer_off_data(temp,num_hops+1,layer_offsets);
    int temp1[num_hops+1];
    std::copy(layer_offsets, layer_offsets + num_hops+1, temp1);
    construct_layer_off_data(temp1,num_hops,block_offsets);
    //reverseElements(edge_mapping, edge_mapping_size);
}


void neighbor_sampler_th(
    const int* indptr,        // CSC 形式的 indptr 数组
    const int* indices,       // CSC 形式的 indices 数组
    const int* data,          // 边的信息（可以是边的ID或权重等）
    int indptr_size,          // indptr 数组的大小
    int indices_size,         // indices 数组的大小
    int data_size,            // data 数组的大小
    int batch_size,           // 每批种子节点的大小
    int num_neighbors,        // 每个节点的邻居采样数量
    const int* seed_nodes,    // 初始的种子节点
    int seed_nodes_size,      // 种子节点的大小
    int num_hops,             // 采样的跳数
    int current_nodeflow_index,
    int* node_mapping,        // 输出的 node_mapping 数组
    int* edge_mapping,        // 输出的 edge_mapping 数组
    int* layer_offsets,       // 输出的层偏移量数组
    int* block_offsets,       // 输出的块偏移量数组
    int* indptr_out,            // 输出的源节点数组
    int* indices_out,             // 输出的目标节点数组
    int* data_out             // 输出的边数组
) {
    auto time_start = chrono::high_resolution_clock::now();
    int max_node_id = indptr_size - 1;
    int map[max_node_id + 100];  // 用来记录哪些节点已经访问过
    memset(map, 0, (max_node_id + 1) * sizeof(int));
    int src_node[2*max_node_id];         // 输出的源节点数组
    int dst_node[2*max_node_id];             // 输出的目标节点数组
    // 初始化 layer_offsets 和 block_offsets
    int layer_offsets_size = 0;
    int block_offsets_size = 0;
    int start=(current_nodeflow_index)*batch_size;
    if(start>=seed_nodes_size){
        return;
    }
    int temp_seed_nodes_size=(seed_nodes_size<=start+batch_size)?seed_nodes_size-start:batch_size;
    int start_point=temp_seed_nodes_size;
    // 处理 seed_nodes
    int node_mapping_size = temp_seed_nodes_size;
    int temp_node_mapping_size=0;
    int temp_edge_mapping_size=0;
    fill_array_from_array(seed_nodes+start,node_mapping,temp_seed_nodes_size,0);
    int temp_seed_nodes[max_node_id + 1];
    fill_array_from_array(seed_nodes+start,temp_seed_nodes,temp_seed_nodes_size,0);
    avx512_fill_map_from_goal(temp_seed_nodes,map,temp_seed_nodes_size,1);
    layer_offsets[layer_offsets_size++] = temp_seed_nodes_size;  // 初始种子节点层偏移量
    temp_node_mapping_size=temp_seed_nodes_size;
    int edge_mapping_size=0;
    int src_node_size=0;
    int neighbors[2*max_node_id];
    int neighbors_count[max_node_id];
    int out_edgess[2*max_node_id];
    double time_case_1 = 0, time_case_2 = 0, time_case_3 = 0, time_loop_check = 0, time_indptr_construction = 0;
    // 开始多跳采样
    auto start_loop_check = chrono::high_resolution_clock::now();
    for (int hop = 0; hop < num_hops; ++hop) {
        // 遍历当前种子节点
        int temp_sum=0;
        int temp_node_num=0;
        int temp_neighbors_num=0;
        for (int i = 0; i < temp_seed_nodes_size; ++i) {
            int node = temp_seed_nodes[i];
            int start = indptr[node];
            int end = indptr[node + 1];
            //cout<<node<<" "<<end-start<<endl;
            int input[end-start];
            fill_array_from_array(indices+start,input,end-start,0);
            int output[end-start];
            int out_egdes[end-start];
            //const int* const_map= static_cast<const int*>(map);
            int out_num=0;
            auto start_loop_check = chrono::high_resolution_clock::now();
            for(int jk=0;jk<end-start;jk++){
                if (map[input[jk]] == 0) {
                    output[out_num] = input[jk];
                    out_egdes[out_num]=jk+start;
                    out_num++;
                    map[input[jk]]=1;
                }
            }
            auto end_loop_check = chrono::high_resolution_clock::now();
            time_loop_check += chrono::duration<double>(end_loop_check - start_loop_check).count();
            const int* const_output= static_cast<const int*>(output);
            //cout<<"jjj"<<endl;
            if(out_num<=num_neighbors){
               auto start_case_1 = chrono::high_resolution_clock::now();
               fill_array_with_one(src_node+src_node_size,temp_seed_nodes[i],out_num);
            // avx_copy_with_indices(data,const_output,data_out+src_node_size,out_num);
               src_node_size+=out_num; 
               avx_copy_with_indices(data,out_egdes,edge_mapping+edge_mapping_size,out_num);
               edge_mapping_size+=out_num;
               fill_array_from_array(const_output,node_mapping,out_num,node_mapping_size);
               //edge_mapping_size+=out_num;
               node_mapping_size+=out_num;
               auto end_case_1 = chrono::high_resolution_clock::now();
               time_case_1 += chrono::duration<double>(end_case_1 - start_case_1).count();
            }
            else if(out_num<2*num_neighbors){
                auto start_case_2 = chrono::high_resolution_clock::now();
               avx512_fill_map_from_goal(const_output,map,out_num,0);
               //fill_array_from_array(const_output,neighbors,out_num,0);
               avx_negate_sample_neighbors(const_output,out_num,num_neighbors,node_mapping+node_mapping_size,out_egdes,edge_mapping+edge_mapping_size);
               edge_mapping_size+=num_neighbors;
               fill_array_with_one(src_node+src_node_size,temp_seed_nodes[i],num_neighbors);
               src_node_size+=num_neighbors;
               avx512_fill_map_from_goal(node_mapping+node_mapping_size,map,num_neighbors,1);
               node_mapping_size+=num_neighbors;
               auto end_case_2 = chrono::high_resolution_clock::now();
               time_case_2 += chrono::duration<double>(end_case_2 - start_case_2).count();
            }
            else{
                auto start_case_3 = chrono::high_resolution_clock::now();
                fill_array_with_one(src_node+src_node_size,temp_seed_nodes[i],num_neighbors);
                src_node_size+=num_neighbors;
                fill_array_from_array(const_output,neighbors,out_num,temp_neighbors_num);
                fill_array_from_array(out_egdes,out_edgess,out_num,temp_neighbors_num);
                neighbors_count[temp_node_num++]=out_num;
                temp_neighbors_num+=out_num;
                auto end_case_3 = chrono::high_resolution_clock::now();
                time_case_3 += chrono::duration<double>(end_case_3 - start_case_3).count();
            }
        }
        // 清空邻居数组和邻居数数组
        int count_of_node=temp_node_num;
        if(count_of_node!=0){
           auto start_case_3 = chrono::high_resolution_clock::now();
           avx_th_neighbor_sampling_with_mixed_precision(neighbors, neighbors_count, count_of_node,
                      node_mapping+node_mapping_size,num_neighbors, out_edgess, edge_mapping+edge_mapping_size);
            temp_neighbors_num=0;
            temp_node_num=0;
            // 更新节点映射，存储下一个节点的邻居
            // 更新其它变量
            edge_mapping_size += count_of_node*num_neighbors;
            node_mapping_size += count_of_node*num_neighbors;
            auto end_case_3 = chrono::high_resolution_clock::now();
            time_case_3 += chrono::duration<double>(end_case_3 - start_case_3).count();
        }
        // 更新 node_mapping, edge_mapping, layer_offsets 和 block_offsets
        // 更新 layer_offsets
        layer_offsets[layer_offsets_size++] = node_mapping_size-temp_node_mapping_size;
        block_offsets[block_offsets_size++] = edge_mapping_size-temp_node_mapping_size;
        fill_array_from_array(node_mapping+temp_node_mapping_size,temp_seed_nodes,node_mapping_size-temp_node_mapping_size,0);//将该层节点作为下一层的种子节点
            //将该层节点作为下一层的种子节点
        temp_seed_nodes_size=node_mapping_size-temp_node_mapping_size;
        temp_node_mapping_size=node_mapping_size;
        temp_edge_mapping_size=edge_mapping_size;
    }
    node_mapping[data_size +5]=node_mapping_size;
    edge_mapping[data_size +5]=edge_mapping_size;
    int max_node = 0;
    for (int i = 0; i < src_node_size; ++i) {
        max_node = std::max(max_node, std::max(src_node[i], node_mapping[i+min(seed_nodes_size-start,batch_size)]));
    }
    node_mapping[data_size +6]=max_node;
    fill_array_from_array(node_mapping+min(seed_nodes_size-start,batch_size),dst_node,node_mapping_size-min(seed_nodes_size-start,batch_size),0);
    const int* const_src_node= static_cast<const int*>(src_node);
    const int* const_dst_node= static_cast<const int*>(dst_node);
    const int* const_edge_mapping= static_cast<const int*>(edge_mapping);
    int num_nodes = max_node + 1;

    // 初始化 indptr 数组
    auto start_indptr_construction = chrono::high_resolution_clock::now();
    for (int i = 0; i <= num_nodes; ++i) {
        indptr_out[i] = 0;  // 设置所有元素为 0
    }

    // 计算每个节点的出度
    for (int i = 0; i < edge_mapping_size; ++i) {
        indptr_out[const_src_node[i] + 1]++;  // 更新每个节点的出度
    }

    // 计算累积和以填充 indptr
    for (int i = 1; i <= num_nodes; ++i) {
        indptr_out[i] += indptr_out[i - 1];
    }

    // 填充 indices 和 data 数组
    for (int i = 0; i < edge_mapping_size; ++i) {
        int node = const_src_node[i];
        int pos = indptr_out[node]++;  // 计算当前边的索引位置
        indices_out[pos] = const_dst_node[i];  // 填充 indices
        data_out[pos] = const_edge_mapping[i];     // 填充 data
    }

    // 恢复 indptr 数组
    for (int i = num_nodes; i > 0; --i) {
        indptr_out[i] = indptr_out[i - 1];  // 恢复 indptr 数组
    }
    indptr_out[0] = 0;  // 将 indptr[0] 设置为 0
    int temp[num_hops+2];
    std::copy(layer_offsets, layer_offsets + num_hops+2, temp);
    construct_layer_off_data(temp,num_hops+1,layer_offsets);
    int temp1[num_hops+1];
    std::copy(layer_offsets, layer_offsets + num_hops+1, temp1);
    construct_layer_off_data(temp1,num_hops,block_offsets);
    auto end_indptr_construction = chrono::high_resolution_clock::now();
    time_indptr_construction += chrono::duration<double>(end_indptr_construction - start_indptr_construction).count();
    auto time_end=chrono::duration<double>(end_indptr_construction-time_start).count();
    ofstream time_file("timing_results.txt", std::ios::app);
    time_file << "Time for all: " << time_end << " microseconds\n";
    time_file << "Time for case 1: " << time_case_1 << " microseconds\n";
    time_file << "Time for case 2: " << time_case_2 << " microseconds\n";
    time_file << "Time for case 3: " << time_case_3 << " microseconds\n";
    time_file << "Time for loop check: " << time_loop_check << " microseconds\n";
    time_file << "Time for indptr construction: " << time_indptr_construction << " microseconds\n";
    time_file.close();
    //reverseElements(node_mapping, node_mapping_size);
    //reverseElements(edge_mapping, edge_mapping_size);
}

void MultiLayerNeighborSampler(
    const heterograph* csr_array,     //输入的csr图表示
    int* nodes_all_type, //各种类型的节点
    int* max_node_of_type,
    const int node_types,      //节点类型数量
    const int* nodes_offset,   //节点偏移量
    const int* fanouts,       // 每种边类型采样的数量
    int fanouts_size,         // 边的类型数量
    int dir,                  //根据出边还是入边进行采样
    int* sampled_nodes,        // 输出的 node数组
    int* sampled_nodes_offsets,   // 输出的 node偏移量
    int* node_mapping,        // 输出的 node_mapping 数组
    int* offsets,             // 输出的map偏移量数组
    heterograph* sample_heterograph_array     // 输出的csr图表示
) {
    auto time_start = chrono::high_resolution_clock::now();
    int number_of_nodes_specific_type=0;
    if(flag==0){
        for (int i = 0; i < node_types; i++) {
            int size = nodes_offset[i + 1] - nodes_offset[i];
            std::vector<int> temp(size, -1);
            map.push_back(temp);
            seed_nodes_mapped.push_back(0);
            cout<<size<<" ";
        }
        flag=1;
    }
    std::vector<std::vector<int>> new_nodes_vec(node_types);
    std::vector<int> map_size(node_types, 0);
    for (int etype = 0; etype < fanouts_size; ++etype){
       std::vector<int> csr_inprt,csr_indices;
       int temp_edges=0;
       int temp_src_vtype=csr_array[etype].src_vtype;
       int temp_dst_vtype=csr_array[etype].dst_vtype;
       int lhs_node_type=(dir==0)?temp_src_vtype:temp_dst_vtype;
       temp_edges=max_node_of_type[lhs_node_type];
       number_of_nodes_specific_type=0;
       //cout<<"长度为"<<temp_edges<<endl;
       std::vector<int> temp_indptr(temp_edges+1,0),Placeholders,neighbor_count(temp_edges,0);
       cout<<"11133333"<<endl;
       std::vector<std::vector<int>> sampled_graphs;
       int rhs_node_type=(dir==1)?temp_src_vtype:temp_dst_vtype;
    //    std::ofstream outFile("zj.txt",std::ios::app);
    //     outFile <<"源节点类型为"<<csr_array[etype].src_vtype<<"目标节点类型为"<<csr_array[etype].dst_vtype<<"节点数量为"<<csr_array[etype].num_edges<<endl;
    //     // 写入第一个数组
    //     outFile << "Array 1:" << std::endl;
    //     for (int i=0;i<csr_array[etype].num_edges;i++) {
    //         outFile << csr_array[etype].src_node[i] <<" "<<csr_array[etype].dst_node[i]<<endl;
    //     }
       build_csr_from_edges_with_max_node(csr_array[etype].src_node,csr_array[etype].dst_node,csr_array[etype].num_edges,temp_edges,
       csr_inprt,csr_indices);
    //    outFile << "种子节点为:" << std::endl;
    //     for (int i=0;i<nodes_offset[lhs_node_type+1]-nodes_offset[lhs_node_type];i++) {
    //         outFile << nodes_all_type[i+nodes_offset[lhs_node_type]]<<endl;
    //     }
    //     // 关闭文件
    //     outFile.close();
       cout<<"1114444"<<endl;
       number_of_nodes_specific_type=prepare_for_sample(nodes_all_type+nodes_offset[lhs_node_type],csr_inprt.data(),csr_indices.data(),fanouts[etype],nodes_offset[lhs_node_type+1]-nodes_offset[lhs_node_type],
       temp_indptr.data(),neighbor_count.data());
       cout<<number_of_nodes_specific_type<<endl;
       cout<<"1115555"<<endl;
       std::vector<int> sampled_indptr(number_of_nodes_specific_type,0);
       int sampled_indices[number_of_nodes_specific_type],sampled_data[number_of_nodes_specific_type];
       if(number_of_nodes_specific_type == 0 || fanouts[etype] == 0){
          for(int i=0;i<3;i++){
            sampled_graphs.push_back(Placeholders);
          }
       }
       else{
        avx_th_neighbor_edges_sampling(csr_indices.data(),neighbor_count.data(),temp_indptr.data(),nodes_all_type+nodes_offset[lhs_node_type],
        map[lhs_node_type],seed_nodes_mapped[lhs_node_type],map_size[lhs_node_type],temp_edges,sampled_indptr.data(),sampled_indices,
        fanouts[etype],csr_array[etype].data,sampled_data);
        cout<<"1116666"<<endl;
        seed_nodes_mapped[lhs_node_type]++;
        int num_threads_col=8;
        int num_cols=sampled_indptr[temp_edges];
        std::vector<std::vector<int>> src_nodes_local(num_threads_col); // 每个线程的局部节点集合
        std::vector<int> global_prefix_col(num_threads_col + 1, 0);    // 全局前缀数组
        cout<<"1111111"<<endl;
    // 并行处理
#pragma omp parallel num_threads(num_threads_col)
    {
        int thread_id = omp_get_thread_num();
        num_threads_col = omp_get_num_threads();

        int start_i = thread_id * (num_cols / num_threads_col) +min(thread_id, num_cols % num_threads_col);
        int end_i = (thread_id + 1) * (num_cols / num_threads_col) +min((thread_id + 1), num_cols % num_threads_col);

        assert(thread_id + 1 < num_threads_col || end_i == num_cols);

        // 局部收集新节点
        for (int i = start_i; i < end_i; ++i) {
            int picked_idx = sampled_indices[i];
            bool spot_claimed=1; 
            if(picked_idx<nodes_offset[picked_idx + 1] - nodes_offset[picked_idx]){
                spot_claimed = BoolCompareAndSwap(&map[rhs_node_type][picked_idx],-1,i);
            }
            if (spot_claimed) src_nodes_local[thread_id].push_back(picked_idx);
        }

        global_prefix_col[thread_id + 1] = src_nodes_local[thread_id].size();

#pragma omp barrier
#pragma omp master
        {
            global_prefix_col[0] = new_nodes_vec[rhs_node_type].size();
            for (int t = 0; t < num_threads_col; ++t) {
                global_prefix_col[t + 1] += global_prefix_col[t]; // 更新全局前缀数组
            }
        }

#pragma omp barrier
        // 更新全局映射关系
        int mapping_shift = global_prefix_col[thread_id];
        for (size_t i = 0; i < src_nodes_local[thread_id].size(); ++i) {
            map[rhs_node_type][src_nodes_local[thread_id][i]] = mapping_shift + i;
        }

#pragma omp barrier
        // 更新列索引到全局映射
        for (int i = start_i; i < end_i; ++i) {
            int picked_idx = sampled_indices[i];
            int mapped_idx = map[rhs_node_type][picked_idx];
            sampled_indices[i] = mapped_idx;
        }
    }

        // 拷贝线程的局部节点到全局节点向量
        new_nodes_vec[rhs_node_type].resize(global_prefix_col.back());
        map_size[rhs_node_type]=global_prefix_col.back();
        int offset = temp_edges;
        for (int thread_id = 0; thread_id < num_threads_col; ++thread_id) {
            fill_array_from_array(src_nodes_local[thread_id].data(),new_nodes_vec[rhs_node_type].data(),src_nodes_local[thread_id].size(),offset);
            offset += src_nodes_local[thread_id].size();
        }
       }
       int temp_temp=0;
       for(int kk=0;kk<sampled_indptr.size();kk++){
           fill_array_with_one(sample_heterograph_array[etype].src_node+temp_temp,kk,sampled_indptr[kk+1]-sampled_indptr[kk]);
           temp_temp+=sampled_indptr[kk+1]-sampled_indptr[kk];
       }
       sample_heterograph_array[etype].src_node=sampled_indptr.data();
       sample_heterograph_array[etype].dst_node=sampled_indices;
       sample_heterograph_array[etype].data=sampled_data;
       sample_heterograph_array[etype].src_vtype=csr_array[etype].src_vtype;
       sample_heterograph_array[etype].dst_vtype=csr_array[etype].dst_vtype;
       sample_heterograph_array[etype].num_nodes=number_of_nodes_specific_type;
       sample_heterograph_array[etype].num_edges=number_of_nodes_specific_type;
    }
    offsets=map_size.data();
    int j=0,temp_temp=0,temp_temp1=0;
    for(int i=0;i<node_types;i++){
        fill_array_from_array(new_nodes_vec[i].data(),sampled_nodes,new_nodes_vec[i].size(),temp_temp);
        fill_array_from_array(map[i].data(),node_mapping,map[i].size(),temp_temp1);
        temp_temp1+=map[i].size();
        temp_temp+=new_nodes_vec[i].size();
        sampled_nodes_offsets[i]=new_nodes_vec[i].size();
    }
    //reverseElements(node_mapping, node_mapping_size);
    //reverseElements(edge_mapping, edge_mapping_size);
}

}