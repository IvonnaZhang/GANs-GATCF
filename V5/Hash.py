import numpy as np
import multiprocessing as mp
import time

def generate_random_matrix(size):
    # 生成随机矩阵
    return np.random.randn(size, size)

def hash_function(matrix, hash_size):
    # 应用随机投影哈希函数
    random_projection = np.random.randn(matrix.shape[1], hash_size)
    return np.dot(matrix, random_projection)

def matrix_multiply_hashed(args):
    # 在哈希空间中执行矩阵乘法
    A, B = args
    return np.dot(A, B.T)

# 使用多进程并行计算哈希矩阵乘法
def parallel_matrix_multiply(A, B, hash_size, num_processes):
    # 哈希映射
    A_hashed = hash_function(A, hash_size)
    B_hashed = hash_function(B, hash_size)

    # 分块处理
    chunk_size = A_hashed.shape[0] // num_processes
    chunks = [(A_hashed[i:i + chunk_size], B_hashed) for i in range(0, A_hashed.shape[0], chunk_size)]

    # 创建进程池并计算
    pool = mp.Pool(processes=num_processes)
    start_time = time.time()  # 开始计时
    results = pool.map(matrix_multiply_hashed, chunks)
    pool.close()
    pool.join()
    end_time = time.time()  # 结束计时

    # 合并结果
    total_time = end_time - start_time
    return np.vstack(results), total_time

def main():
    # 生成随机矩阵
    size = 10000  # 矩阵维度
    A = generate_random_matrix(size)
    B = generate_random_matrix(size)

    # 定义哈希大小和进程数
    hash_size = 300  # 可调整
    num_processes = 4  # 根据CPU核心数调整

    # 并行计算哈希矩阵乘法结果并打印时间
    result, exec_time = parallel_matrix_multiply(A, B, hash_size, num_processes)
    print("Execution Time: {:.2f} seconds".format(exec_time))

if __name__ == '__main__':
    main()
