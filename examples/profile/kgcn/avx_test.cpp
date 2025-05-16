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
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>
#include <future>
using namespace std;


// AVX 邻居采样函数

int count_common_elements_std(const std::vector<int>& neighbors, const std::vector<int>& belongs) {
    std::unordered_set<int> belongs_set(belongs.begin(), belongs.end());
    int count = 0;
    for (int neighbor : neighbors) {
        if (belongs_set.find(neighbor) != belongs_set.end()) {
            count++;
        }
    }
    return count;
}


int count_common_elements_avx512(const std::vector<int>& neighbors, const std::vector<int>& belongs) {
    int count = 0;
    size_t n = neighbors.size();
    size_t m = belongs.size();

    for (size_t i = 0; i < n; i += 16) {  
        __m512i vec_neighbors = _mm512_loadu_si512(&neighbors[i]);  

        for (size_t j = 0; j < m; j += 16) {  
            __m512i vec_belongs = _mm512_loadu_si512(&belongs[j]);  

            __mmask16 mask = _mm512_cmpeq_epi32_mask(vec_neighbors, vec_belongs);  
            count += __builtin_popcount(mask);  
        }
    }

    return count;
}

void construct_layer_off_data(const int* layer_sizes, int num_layers, int* layer_off_data) {
    // 初始化 layer_off_data，长度比 layer_sizes 多 1，因为需要存储最后的偏移量
    layer_off_data[0] = 0;
    // 从最外层到最内层，倒序计算
    for (int i = 0; i <=num_layers; ++i) {
        cout<<layer_sizes[i]<<" ";
    }
    cout<<endl;
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

void FindEdges_with_type(
    const int* indptr,        // CSC indptr 数组
    const int* indices,       // CSC indices 数组
    const int* data_type,     // 边的类型数组
    int num_nodes,            // 节点总数
    int target_type,          // 要筛选的目标边类型
    int* src_node,            // 输出：源节点数组（需要足够大）
    int* dst_node,            // 输出：目标节点数组（需要足够大）
    int& count                // 输出：总共写入的边数量
) {
    count = 0;
    for (int dst = 0; dst < num_nodes; ++dst) {
        int start = indptr[dst];
        int end = indptr[dst + 1];
        int num_edges = end - start;
        int i=0;
        while(i<num_edges){
            int temp_size=min(16,num_edges-i);
            if(temp_size==16){
               // 加载 data_type 和 indices 的16个元素
            __m512i data_vec = _mm512_loadu_si512(&data_type[start + i]);
            __m512i indices_vec = _mm512_loadu_si512(&indices[start + i]);

            // 生成符合条件的掩码
            __mmask16 mask = _mm512_cmpeq_epi32_mask(data_vec, _mm512_set1_epi32(target_type));

            // 用掩码压缩符合条件的 indices 数据并写入 src_node
            _mm512_mask_compressstoreu_epi32(&src_node[count], mask, indices_vec);

            // 用掩码压缩符合条件的 dst 数据并写入 dst_node
            __m512i dst_vec = _mm512_set1_epi32(dst);
            _mm512_mask_compressstoreu_epi32(&dst_node[count], mask, dst_vec);

            // 更新写入计数器
            count += _mm_popcnt_u32(mask);  // 掩码中为1的位数表示写入的元素个数
            }
            else{
                for (int k = i+start; k < num_edges+start; ++k) {
                    if (data_type[k] == target_type) {
                        src_node[count]=indices[k];
                        dst_node[count]=dst;
                        count++;
                    }
                } 
            }
            i+=temp_size;
        }
    }
}

void FilterEdgesByType(
    const int* indptr,        // CSC indptr 数组
    const int* indices,       // CSC indices 数组
    const int* data_type,     // 边的类型数组
    int num_nodes,            // 节点总数
    int target_type,          // 要筛选的目标边类型
    std::vector<int>& src_node, // 输出：源节点数组
    std::vector<int>& dst_node // 输出：目标节点数组
) {
    for (int dst = 0; dst < num_nodes; ++dst) {
        int start = indptr[dst];
        int end = indptr[dst + 1];
        for (int k = start; k < end; ++k) {
            if (data_type[k] == target_type) {
                src_node.push_back(indices[k]);
                dst_node.push_back(dst);
            }
        }
    }
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

void avx_mixed_precision_neighbor_sampling(const int* neighbors, int n, int num_samples, int* sampled_neighbors,int* edge_index,int* result_index) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    int i = 0;
    if (true) {
    std::vector<unsigned short int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::shuffle(numbers.begin(), numbers.end(), gen);
    // 使用 16 位指令的情况
    while (i < num_samples) {
        int temp_batch_size = std::min(32, num_samples - i); // 一次处理最多 32 个

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


void avx_neighbor_sampling(const int* neighbors, int n, int num_samples, int* sampled_neighbors,int* edge_index,int* result_index) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    int i = 0;
    std::vector<int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::shuffle(numbers.begin(), numbers.end(), gen);
    while (i < num_samples) {
        int temp_batch_size = std::min(16, num_samples - i);
        // 选择随机索引

        // 使用 32 位指令
        __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(numbers.data()+i));
        __m512i gathered_neighbors = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(neighbors), sizeof(int));
        __m512i gathered_edge = _mm512_i32gather_epi32(index_vec, reinterpret_cast<const void*>(edge_index), sizeof(int));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(sampled_neighbors + i), gathered_neighbors);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(result_index + i), gathered_edge);
        i += temp_batch_size;
    }
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


void avx_copy_with_indices(const int* source_array, const int* indices, int* target_array, int num_elements) {
    // 遍历索引数组并使用 AVX-512 指令进行元素复制
    for (int i = 0; i < num_elements; i += 16) {
        // 从索引数组中获取当前批次的索引
        __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));

        // 从源数组中根据索引加载对应的值
        __m512i values_vec = _mm512_i32gather_epi32(index_vec, source_array, 4);

        // 将加载的值存储到目标数组中
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(target_array + i), values_vec);
    }
}


void avx_copy_with_indices_th(const int* source_array, const int* indices, int* target_array, int num_elements, int num_threads) {
    // 确定每个线程要处理的块大小
    int chunk_size = num_elements / num_threads;
    int remainder = num_elements % num_threads;  // 处理不能均分的部分

    // 创建线程任务函数
    auto thread_task = [&](int thread_id) {
        int local_start = thread_id * chunk_size;
        int local_size = (thread_id == num_threads - 1) ? chunk_size + remainder : chunk_size;

        int i = local_start;
        for (; i <= local_start + local_size - 16; i += 16) {
            // 使用 AVX-512 指令加载索引和源数组值
            __m512i index_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(indices + i));
            __m512i values_vec = _mm512_i32gather_epi32(index_vec, source_array, 4);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(target_array + i), values_vec);
        }

        // 处理不足 16 的部分
        for (; i < local_start + local_size; ++i) {
            target_array[i] = source_array[indices[i]];
        }
    };

    // 启动多线程
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(thread_task, i);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
}

void build_csr_from_edges_with_max_node(
    const std::vector<int>& source, 
    const std::vector<int>& target, 
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

    size_t num_edges = target.size();
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


void build_csr_with_avx512_multithreaded(
    const std::vector<int>& source,
    const std::vector<int>& target,
    int max_node_id,
    std::vector<int>& inptr,
    std::vector<int>& indices) 
{
    size_t num_edges = source.size();
    size_t simd_width = 16; // AVX-512 每次处理 16 个 32 位整数
    size_t num_threads = std::thread::hardware_concurrency();

    // 对 source 数组进行排序并去重
    std::vector<int> unique_sources = source;
    std::sort(unique_sources.begin(), unique_sources.end());
    auto last = std::unique(unique_sources.begin(), unique_sources.end());
    unique_sources.erase(last, unique_sources.end());

    // 初始化 inptr 和 indices
    inptr.resize(max_node_id + 2, 0); // 注意 +2，因为最后一个是 max_node_id+1 位置
    indices.resize(num_edges);

    // 临时数组用于保存每个线程的结果
    std::vector<std::vector<int>> temp_indices(num_threads, std::vector<int>(num_edges, 0));
    std::vector<std::vector<int>> temp_offsets(num_threads, std::vector<int>(max_node_id + 1, 0));
    std::vector<int> threads_offsets(num_threads, 0);

    // 线程工作函数：处理每个节点值的邻居信息
    auto process_node_range = [&](size_t start_idx, size_t end_idx, size_t thread_id) {
        size_t count = 0;  
        for (size_t idx = start_idx; idx < end_idx; ++idx) {
            int value = unique_sources[idx];
            size_t local_count = 0;
            // 遍历 source 数组，筛选等于 value 的目标节点
            int i = 0;
            while (i < num_edges) {
                size_t remaining = std::min(simd_width, num_edges - i);
                if (remaining == 16) {
                    // 加载 source 和 target 的数据
                    __m512i src_vec = _mm512_loadu_si512(source.data() + i);
                    __m512i tgt_vec = _mm512_loadu_si512(target.data() + i);

                    // 生成掩码：source[i:j] == value
                    __mmask16 mask = _mm512_cmpeq_epi32_mask(src_vec, _mm512_set1_epi32(value));
                    int temp_count = _mm_popcnt_u32(mask);
                    // 压缩符合条件的 target 数据到 indices 数组
                    _mm512_mask_compressstoreu_epi32(temp_indices[thread_id].data() + threads_offsets[thread_id] + local_count, mask, tgt_vec);
                    local_count += temp_count;
                }
                else {
                    for (int j = i; j < num_edges; ++j) {
                        if (source[j] == value) {
                            temp_indices[thread_id][threads_offsets[thread_id] + local_count] = target[j];
                            local_count++;
                        }
                    }
                }
                i += remaining;
            }
            // 更新临时偏移量（每个节点的邻居数）
            temp_offsets[thread_id][value] = local_count;
            threads_offsets[thread_id] += local_count;
        }
        return count;
    };

    // 多线程并行处理节点值
    std::vector<std::thread> threads;
    size_t chunk_size = unique_sources.size() / num_threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = (t == num_threads - 1) ? unique_sources.size() : (start_idx + chunk_size);
        threads.emplace_back(process_node_range, start_idx, end_idx, t);
    }

    // 等待所有线程完成
    for (auto& th : threads) th.join();

    // 合并所有线程的偏移量（inptr 数组）
    size_t total_count = 0;
    for (size_t t = 0; t <= max_node_id; ++t) {
        for (size_t i = 0; i < num_threads; ++i) {
            inptr[t + 1] += temp_offsets[i][t];
        }
        inptr[t + 1] += inptr[t];
    }

    // 汇合所有线程的 indices 数组
    size_t current_offset = 0;
    for (size_t t = 0; t < num_threads; ++t) {
        fill_array_from_array(temp_indices[t].data(), indices.data(), threads_offsets[t], current_offset);
        current_offset += threads_offsets[t];
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
    int& node_mapping_size,   // node_mapping 数组的实际大小
    int& edge_mapping_size,   // edge_mapping 数组的实际大小
    int& layer_offsets_size,  // layer_offsets 数组的实际大小
    int& block_offsets_size   // block_offsets 数组的实际大小
) {
    int max_node_id = indptr_size - 1;
    int map[max_node_id + 1];  // 用来记录哪些节点已经访问过
    memset(map, 0, (max_node_id + 1) * sizeof(int));

    // 初始化 layer_offsets 和 block_offsets
    layer_offsets_size = 0;
    block_offsets_size = 0;
    int start=(current_nodeflow_index)*batch_size;
    if(start>=seed_nodes_size){
        return;
    }
    int temp_seed_nodes_size=(seed_nodes_size<=start+batch_size)?seed_nodes_size-start:batch_size;
    int start_point=temp_seed_nodes_size;
    // 处理 seed_nodes
    node_mapping_size = temp_seed_nodes_size;
    int temp_node_mapping_size=0;
    int temp_edge_mapping_size=0;
    fill_array_from_array(seed_nodes,node_mapping,temp_seed_nodes_size,0);
    int temp_seed_nodes[max_node_id + 1];
    fill_array_from_array(seed_nodes+start,temp_seed_nodes,temp_seed_nodes_size,0);
    avx512_fill_map_from_goal(temp_seed_nodes,map,temp_seed_nodes_size,1);
    layer_offsets[layer_offsets_size++] = temp_seed_nodes_size;  // 初始种子节点层偏移量
    temp_node_mapping_size=temp_seed_nodes_size;
    edge_mapping_size=0;
    int next_neighbors[max_node_id +1 ];
    int neighbors[max_node_id +1];
    int temp_next_neighbors_num=0;
    //cout<<"1"<<endl;
    // 开始多跳采样
    for (int hop = 0; hop < num_hops; ++hop) {
        std::vector<int> sampled_edge_ids;
        // 遍历当前种子节点
        int neighbors_size[temp_seed_nodes_size];
        memset(neighbors_size, 0, temp_seed_nodes_size * sizeof(int));
        int temp_neighbors_num=0;
        int temp_sum=0;
        int num_samples[temp_seed_nodes_size];
        int avx_node[temp_seed_nodes_size];
        //cout<<"zzz"<<endl;
        memset(num_samples, 0, temp_seed_nodes_size * sizeof(int));
        memset(avx_node, -1, temp_seed_nodes_size * sizeof(int));

        for (int i = 0; i < temp_seed_nodes_size; ++i) {
            int node = temp_seed_nodes[i];
            int start = indptr[node];
            int end = indptr[node + 1];
            //cout<<node<<" "<<end-start<<endl;
            int input[end-start];
            fill_array_from_array(indices+start,input,end-start,0);
            // for(int jk=start;jk<end;jk++){
            //     cout<<indices[jk]<<" ";
            // }
            // cout<<endl;
            // for(int jk=0;jk<end-start;jk++){
            //     cout<<input[jk]<<" ";
            // }
            // cout<<endl;
            //const int* const_input= static_cast<const int*>(input);
            int output[end-start];
            //const int* const_map= static_cast<const int*>(map);
            int out_num=0;
            for(int jk=0;jk<end-start;jk++){
                if (map[input[jk]] == 0) {
                    output[out_num] = input[jk];
                    out_num++;
                    map[input[jk]]=1;
                }
            }
            // cout<<out_num<<endl;
            // cout<<"jjjk"<<endl;
            const int* const_output= static_cast<const int*>(output);
            //cout<<"jjj"<<endl;
            if(out_num<=num_neighbors){
               //out_num=avx512_filter(const_input,const_map,output,end-start,data,edge_mapping+edge_mapping_size);
               for(int jk=0;jk<out_num;jk++){
                edge_mapping[edge_mapping_size++]=data[output[jk]];
            }
               fill_array_from_array(const_output,next_neighbors,out_num,temp_next_neighbors_num);
               fill_array_from_array(const_output,node_mapping,out_num,temp_next_neighbors_num+start_point);
               temp_next_neighbors_num+=out_num;
               //edge_mapping_size+=out_num;
               node_mapping_size+=out_num;
            }
            else{
               fill_array_from_array(const_output,neighbors,out_num,temp_neighbors_num);
               temp_neighbors_num+=out_num;
               neighbors_size[i]=out_num;
               num_samples[i]=num_neighbors;
               avx_node[i]=temp_seed_nodes[i];
            }
            temp_sum+=neighbors_size[i];
        }
            //cout<<"22222"<<endl;
            // 邻居采样
            if(temp_sum!=0){
               int sampled_neighbors[temp_sum];
            const int* const_neighbor_size= static_cast<const int*>(neighbors_size);
            const int* const_num_samples= static_cast<const int*>(num_samples);
            avx_th_neighbor_sampling(
                    neighbors, const_neighbor_size, temp_seed_nodes_size,
                    sampled_neighbors, const_num_samples
                );
                //cout<<"2.5"<<endl;
            const int* const_sampled_neighbors= static_cast<const int*>(sampled_neighbors); 
            fill_array_from_array(const_sampled_neighbors,next_neighbors,temp_sum,temp_next_neighbors_num);
            temp_next_neighbors_num+=temp_sum;
            // 记录边信息
            int avx_index=0;
            for(int iii=0;iii<temp_seed_nodes_size;iii++){
                avx_index=0;
                if(avx_node[iii]!=-1){
                   find_edges_avx512(const_sampled_neighbors+avx_index*num_neighbors,num_neighbors,indptr,indices,data,avx_node[iii],edge_mapping+edge_mapping_size);
                   edge_mapping_size+=num_neighbors;
                   node_mapping_size+=num_neighbors;
                   avx_index++;
                   cout<<"jiayixia"<<endl;
                }
            }
            //cout<<"3"<<endl;
            fill_array_from_array(next_neighbors+temp_next_neighbors_num-temp_sum,node_mapping,avx_index*num_neighbors,node_mapping_size-avx_index*num_neighbors);
            for (int indexi=temp_next_neighbors_num-temp_sum;indexi<temp_next_neighbors_num;indexi++) {
            map[next_neighbors[indexi]] = 1;
        }
            }
        // 更新 node_mapping, edge_mapping, layer_offsets 和 block_offsets
        // 更新 layer_offsets
        layer_offsets[layer_offsets_size++] = node_mapping_size-temp_node_mapping_size;
        block_offsets[block_offsets_size++] = edge_mapping_size-temp_node_mapping_size;
        fill_array_from_array(next_neighbors,temp_seed_nodes,node_mapping_size-temp_node_mapping_size,0);//将该层节点作为下一层的种子节点
        ofstream outFile("check.txt",std::ios::app);  // 打开文件，文件名为filename
        if (outFile.is_open()) {  // 检查文件是否成功打开
            for (int i = 0; i < node_mapping_size-temp_node_mapping_size; ++i) {
                outFile << temp_seed_nodes[i] <<" ";  // 将数组元素写入文件，每个元素占一行
            }
            outFile<<"over"<<endl;
            outFile.close();  // 写入完成后关闭文件
        } else {
            std::cerr << "Unable to open file for writing!" << std::endl;
        }
        temp_seed_nodes_size=node_mapping_size-temp_node_mapping_size;
        temp_node_mapping_size=node_mapping_size;
        temp_edge_mapping_size=edge_mapping_size;
        //cout<<"4"<<endl;
    }
    node_mapping[indptr_size - 1]=node_mapping_size;
    edge_mapping[indptr_size - 1]=edge_mapping_size;

}

void fill_array_from_array_th(const int* source_array, int* goal_array, int size, int start_place, int num_threads) {
    // 确定每个线程要处理的块大小
    int chunk_size = size / num_threads;
    int remainder = size % num_threads;

    // 创建 future 用于管理线程任务
    std::vector<std::future<void>> futures;

    for (int i = 0; i < num_threads; ++i) {
        futures.push_back(std::async(std::launch::async, [=]() {
            int local_start = i * chunk_size;
            int local_size = (i == num_threads - 1) ? chunk_size + remainder : chunk_size;

            int j = local_start;
            while (j <= local_start + local_size - 16) {
                // 使用 AVX-512 指令复制 16 个元素
                __m512i vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(source_array + j));
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(goal_array + j + start_place), vec);
                j += 16;
            }

            // 处理不足 16 个元素的剩余部分
            for (; j < local_start + local_size; ++j) {
                goal_array[j + start_place] = source_array[j];
            }
        }));
    }

    // 等待所有任务完成
    for (auto& f : futures) {
        f.get();
    }
}

// int main() {
//     const int num_elements = 10000000;
//     std::vector<int> source_array(num_elements);
//     std::vector<int> indices(num_elements);
//     std::vector<int> target_array(num_elements);
//     std::vector<int> target_array1(num_elements);
//     // 随机生成源数组和索引数组
//     std::mt19937 rng(42); // 设置随机数种子
//     for (int i = 0; i < num_elements; ++i) {
//         source_array[i] = rng() % 100; // 生成 0 到 99 的随机数
//         indices[i] = rng() % num_elements; // 生成合法的索引
//     }

//     // 测量普通方法的时间
//     auto start_regular = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_elements; ++i) {
//         target_array1[i] = indices[i];
//     }
//     auto end_regular = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_regular = end_regular - start_regular;
//     // 测量 AVX 方法的时间
//     auto start_avx = std::chrono::high_resolution_clock::now();
//     fill_array_from_array(indices.data(), target_array.data(),num_elements,0);
//     auto end_avx = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_avx = end_avx - start_avx;
//     // 输出时间
//     std::cout << "AVX Method Time: " << duration_avx.count() << " seconds\n";
//     std::cout << "Regular Method Time: " << duration_regular.count() << " seconds\n";
//     for(int i=0;i<num_elements;i++){
//         if(target_array[i]!=target_array1[i]){
//             cout<<"wrong answer"<<endl;
//         }
//     }
//     return 0;
// }

// int main() {
//     const int num_nodes = 10000000; // 节点数量
//     const int max_neighbors = 5000; // 每个节点的最大邻居数
//     const int num_samples = 4000; // 采样数量

//     // 随机生成邻居数组
//     //std::vector<std::vector<int>> neighbors(num_nodes, std::vector<int>(max_neighbors));
//     std::vector<int> neighbors(max_neighbors);
//     std::vector<int> edge(max_neighbors);
//     std::random_device rd;
//     std::mt19937 gen(rd());

//     for (int i = 0; i <max_neighbors; ++i) {
//         neighbors[i]= gen() % (num_nodes * 2); // 生成随机邻居 ID
//         edge[i]= gen() % (num_nodes * 2);
//     }
//     // 测试原始函数
//     cout<<"111"<<endl;
//     std::vector<int> sampled_neighbors_mixed_edge(num_samples+32);
//     auto start_mixed = std::chrono::high_resolution_clock::now();
//     std::vector<int> sampled_neighbors_mixed(num_samples+32);
//     auto end_mixed = std::chrono::high_resolution_clock::now();
//     for(int j=0;j<num_nodes;j++){
//         avx_mixed_precision_neighbor_sampling(neighbors.data(), max_neighbors, num_samples, sampled_neighbors_mixed.data(),edge.data(),sampled_neighbors_mixed_edge.data());
//     }
//     end_mixed = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_mixed = end_mixed - start_mixed;
//     std::cout << "Mixed Precision Sampling Time: " << duration_mixed.count() << " seconds\n";
//     std::vector<int> sampled_neighbors_original_edge(num_samples+32);
//     auto start_original = std::chrono::high_resolution_clock::now();
//     auto end_original=std::chrono::high_resolution_clock::now();
//     std::vector<int> sampled_neighbors_original(num_samples+16);
//     for(int j=0;j<num_nodes;j++){
//         avx_neighbor_sampling(neighbors.data(), max_neighbors, num_samples, sampled_neighbors_original.data(),edge.data(),sampled_neighbors_original_edge.data());
//     }
//     end_original = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_original = end_original - start_original;
//     // 测试混合精度函数
//     //cout<<"111"<<endl;
//     // 输出时间
//     std::cout << "Original Sampling Time: " << duration_original.count() << " seconds\n";

//     return 0;
// }
// int main() {
//     const int num_nodes = 1000000; // 节点数量
//     const int max_neighbors = 10000; // 每个节点的最大邻居数
//     const int num_samples = 4000; // 采样数量
//     int map[num_nodes];  // 用来记录哪些节点已经访问过
//     memset(map, 0, (num_nodes) * sizeof(int));
//     // 随机生成邻居数组
//     //std::vector<std::vector<int>> neighbors(num_nodes, std::vector<int>(max_neighbors));
//     std::vector<int> neighbors(max_neighbors);
//     std::random_device rd;
//     std::mt19937 gen(rd());
    
//     for (int i = 0; i <max_neighbors; ++i) {
//         neighbors[i]= gen() % (num_nodes-1); // 生成随机邻居 ID
//         //cout<<neighbors[i]<<" ";
//     }
//     // 测试原始函数
//     auto start_mixed = std::chrono::high_resolution_clock::now();
//     avx512_fill_map_from_goal(neighbors.data(),map,max_neighbors,1);
//     auto end_mixed = std::chrono::high_resolution_clock::now();
//     cout<<"111"<<endl;
//     std::chrono::duration<double> duration_mixed = end_mixed - start_mixed;
//     int map1[num_nodes];  // 用来记录哪些节点已经访问过
//     memset(map1, 0, (num_nodes) * sizeof(int));
//     auto start_original = std::chrono::high_resolution_clock::now();
//     for(int j=0;j<max_neighbors;j++){
//         map1[neighbors[j]]=1;
//     }
//     auto end_original = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_original = end_original - start_original;
//     // 测试混合精度函数
//     //cout<<"111"<<endl;
//     // 输出时间
//     for(int i=0;i<num_nodes;i++){
//         if(map[i]!=map1[i]){
//             cout<<"wrong answer";
//             break;
//         }
//     }
//     std::cout << "Original Sampling Time: " << duration_original.count() << " seconds\n";
//     std::cout << "Mixed Precision Sampling Time: " << duration_mixed.count() << " seconds\n";

//     return 0;
// }


// int main() {
//     // 示例参数
//     const int n = 100;   // n 次较小数组的复制需求
//     const int small_size = 800000; // 每个小数组的大小
//     const int num_threads = 8;   // 最大线程数
//     const int total_size = n * small_size; // 多线程处理时需要组合成的总数组大小
   
//     // 源数组和目标数组
//     std::vector<int> source_array(small_size);
//     std::vector<int> goal_array_single(small_size*n);  // 用于单线程的目标数组
//     std::vector<int> combined_source_array(total_size);  // 用于多线程的源数组
//     std::vector<int> goal_array_multi(total_size);       // 用于多线程的目标数组

//     // 填充源数组
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, 10000);
//     for (int i = 0; i < small_size; ++i) {
//         source_array[i] = dis(gen);
//     }
//     // 测量单线程处理 n 次的时间
//     auto start_single = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < n; ++i) {
//         fill_array_from_array(source_array.data(), goal_array_single.data(), small_size, small_size*i);
//     }
//     auto end_single = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_single = end_single - start_single;
//     std::cout << "单线程处理 " << n << " 次较小数组的时间: " << duration_single.count() << " 秒" << std::endl;

//     // 组合数组，模拟多线程处理多个较小数组
//     auto start_multi = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < n; ++i) {
//         fill_array_from_array_th(source_array.data(), goal_array_multi.data(), small_size, small_size*i, 4);
//     }
//     auto end_multi = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_multi = end_multi - start_multi;
//     std::cout << "多线程处理组合数组（包括数组组合时间）时间: " << duration_multi.count() << " 秒" << std::endl;

//     // 验证两种方法的结果是否一致
//     bool same = std::equal(goal_array_single.begin(), goal_array_single.end(), goal_array_multi.begin());
//     std::cout << "两种方法的结果是否一致: " << (same ? "是" : "否") << std::endl;

//     return 0;
// }


// int main() {
//     const int num_elements = 1000;  // 复制的总元素数量
//     const int num_threads = 8;          // 使用的线程数量
//     int n=1000000;
//     // 随机生成源数组和索引数组
//     std::vector<int> source_array(num_elements+16);
//     std::vector<int> indices(num_elements+16);
//     std::vector<int> source_array_total(num_elements*n);
//     std::vector<int> target_array_single(num_elements+16);
//     std::vector<int> target_array_multi(num_elements*n);

//     for (int i = 0; i < num_elements; ++i) {
//         source_array[i] = i;  // 使用递增的数值
//         indices[i] = rand() % num_elements;  // 随机生成索引
//     }

//     // 单线程 AVX-512 复制
//     auto start_single = std::chrono::high_resolution_clock::now();
//     for(int i=0;i<n;i++){
//         avx_copy_with_indices(indices.data(), source_array.data(), target_array_single.data(), num_elements);
//     }
//     auto end_single = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_single = end_single - start_single;

//     std::cout << "单线程 AVX-512 复制耗时: " << duration_single.count() << " 秒" << std::endl;

//     // 多线程 AVX-512 复制
//     auto start_multi = std::chrono::high_resolution_clock::now();
//     for(int i=0;i<n;i++){
//         fill_array_from_array(source_array.data(),source_array_total.data(),num_elements,i*num_elements);
//     }
//     cout<<"111"<<endl;
//     avx_copy_with_indices_th(indices.data(),source_array_total.data(), target_array_multi.data(), n*num_elements, );
//     auto end_multi = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> duration_multi = end_multi - start_multi;

//     std::cout << "多线程 AVX-512 复制耗时: " << duration_multi.count() << " 秒" << std::endl;

//     return 0;
// }

// int main(){
//     int layer_offsets[4]={2,3,4,0};
//     int temp[4];
//     std::copy(layer_offsets, layer_offsets + 4, temp);
//     construct_layer_off_data(temp,3,layer_offsets);
//     for(int i=0;i<4;i++){
//         cout<<layer_offsets[i]<<endl;
//     }
// }

// int main() {
//     constexpr int num_nodes = 800000;
//     constexpr int max_edges = 100000;
//     constexpr int target_type = 1;
//     int max_type=100;
//     std::vector<int> indptr(num_nodes + 1);
//     std::vector<int> indices(max_edges);
//     std::vector<int> data_type(max_edges);
//     std::mt19937 rng(42);
//     std::uniform_int_distribution<int> dist_edges(0, 100);

//     indptr[0] = 0;
//     for (int i = 1; i <= num_nodes; ++i) {
//         indptr[i] = indptr[i - 1] + dist_edges(rng);
//     }
//     int total_edges = indptr[num_nodes];
//     indices.resize(total_edges);
//     data_type.resize(total_edges);

//     std::uniform_int_distribution<int> dist_nodes(0, num_nodes - 1);
//     std::uniform_int_distribution<int> dist_types(0, max_type);

//     for (int i = 0; i < total_edges; ++i) {
//         indices[i] = dist_nodes(rng);
//         data_type[i] = dist_types(rng);
//     }

//     std::cout << "Running edge filtering tests...\n";

//     // 普通方法测试
//     auto start = std::chrono::high_resolution_clock::now();
//     std::vector<int> src_node_naive, dst_node_naive;
//     for(int type=0;type<max_type;type++){
//         src_node_naive.clear();
//         dst_node_naive.clear();
//        FilterEdgesByType(indptr.data(), indices.data(), data_type.data(),
//                       num_nodes, type, src_node_naive, dst_node_naive);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     auto naive_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     std::cout << "Naive method time: " << naive_duration << " ms\n";

//     // AVX-512 优化测试
//     start = std::chrono::high_resolution_clock::now();
//     std::vector<int> src_node_avx(total_edges), dst_node_avx(total_edges);
//     int count = 0;
//     for(int type=0;type<max_type;type++){
//        count=0;
//        std::vector<int> src_node_naive, dst_node_naive;
//        FindEdges_with_type(indptr.data(), indices.data(), data_type.data(),
//                             num_nodes, type, src_node_avx.data(), dst_node_avx.data(), count);
//     }
//     end = std::chrono::high_resolution_clock::now();
//     auto avx_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     std::cout << "AVX-512 method time: " << avx_duration << " ms\n";
//     cout<<"count为"<<count<<endl;
//     cout<<"count为"<<src_node_naive.size()<<endl;
//     // 验证结果
//     bool correct = true;
//     if (src_node_naive.size() != count || dst_node_naive.size() != count) {
//         correct = false;
//         cout<<"111"<<endl;
//     } else {
//         for (int i = 0; i < count; ++i) {
//             if (src_node_naive[i] != src_node_avx[i] || dst_node_naive[i] != dst_node_avx[i]) {
//                 correct = false;
//                 break;
//             }
//         }
//     }
//     std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << "\n";

//     return 0;
// }



// int main() {
//     // 参数配置
//     const size_t num_edges = 10000000;  // 边的数量
//     const int max_node_id = 150000000;  // 最大节点 ID

//     // 随机生成源节点和目标节点
//     std::vector<int> source(num_edges), target(num_edges);
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, max_node_id);

//     for (size_t i = 0; i < num_edges; ++i) {
//         source[i] = dis(gen);
//         target[i] = dis(gen);
//     }

//     // 正确性测试
//     std::vector<int> inptr1, indices1;
//     std::vector<int> inptr2(max_node_id+2), indices2(num_edges);

//     auto start = std::chrono::high_resolution_clock::now();
//     build_csr_from_edges_with_max_node(source, target, max_node_id, inptr1, indices1);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::cout << "Multi-threaded implementation time: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
//               << " ms" << std::endl;

//     start = std::chrono::high_resolution_clock::now();
//     build_csr_with_avx512_multithreaded(source, target, max_node_id, inptr2, indices2);
//     end = std::chrono::high_resolution_clock::now();
//     std::cout << "AVX-512 implementation time: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
//               << " ms" << std::endl;
//     int flag=1;
//     for(int i=0;i<max_node_id+1;i++){
//         if(inptr1[i]!=inptr2[i]){
//            flag=0;
//            break;
//         }
//     }
//     if(flag){
//        for(int i=0;i<num_edges;i++){
//         if(indices1[i]!=indices2[i]){
//             flag=0;
//             break;
//         }
//     }
//     }
//     if(flag){
//        std::cout << "Correctness test passed!" << std::endl;
//     }

//     return 0;
// }


int main() {
    const int SIZE = 1000000;  
    std::vector<int> neighbors, belongs;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 2000000);

    for (int i = 0; i < SIZE; ++i) {
        neighbors.push_back(dist(rng));
        belongs.push_back(dist(rng));
    }

    auto start = std::chrono::high_resolution_clock::now();
    int result_std = count_common_elements_std(neighbors, belongs);
    auto end = std::chrono::high_resolution_clock::now();
    double time_std = std::chrono::duration<double>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    int result_avx512 = count_common_elements_avx512(neighbors, belongs);
    end = std::chrono::high_resolution_clock::now();
    double time_avx = std::chrono::duration<double>(end - start).count();

    std::cout << "Common Elements (std): " << result_std << ", Time: " << time_std << "s\n";
    std::cout << "Common Elements (AVX-512): " << result_avx512 << ", Time: " << time_avx << "s\n";

    return 0;
}