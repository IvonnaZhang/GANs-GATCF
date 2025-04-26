import numpy as np
import time

from tqdm import tqdm


def regular_matrix_multiply(A, B):
    n = A.shape[0]
    m = B.shape[1]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(m):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def strassen_matrix_multiply(A, B, base_size=128):
    n = A.shape[0]
    if n <= base_size:
        return regular_matrix_multiply(A, B)

    m = n // 2
    A11 = A[:m, :m]
    A12 = A[:m, m:]
    A21 = A[m:, :m]
    A22 = A[m:, m:]

    B11 = B[:m, :m]
    B12 = B[:m, m:]
    B21 = B[m:, :m]
    B22 = B[m:, m:]

    M1 = strassen_matrix_multiply(A11, B12 - B22)
    M2 = strassen_matrix_multiply(A11 + A12, B22)
    M3 = strassen_matrix_multiply(A21 + A22, B11)
    M4 = strassen_matrix_multiply(A22, B21 - B11)
    M5 = strassen_matrix_multiply(A11 + A22, B11 + B22)
    M6 = strassen_matrix_multiply(A12 - A22, B21 + B22)
    M7 = strassen_matrix_multiply(A11 - A21, B11 + B12)

    C11 = M5 + M4 - M2 + M6
    C12 = M1 + M2
    C21 = M3 + M4
    C22 = M5 + M1 - M3 - M7

    C = np.zeros((n, n))
    C[:m, :m] = C11
    C[:m, m:] = C12
    C[m:, :m] = C21
    C[m:, m:] = C22

    return C


def measure_time(method, A, B, iterations=5):
    times = []
    for _ in tqdm(range(iterations), desc=f"Measuring {method.__name__}"):
        start_time = time.time()
        result = method(A, B)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times)

# Generate two random 1000x1000 matrices
np.random.seed(0)
A = np.random.rand(100, 100)
B = np.random.rand(100, 100)

# Measure time
average_time_regular = measure_time(regular_matrix_multiply, A, B)
average_time_strassen = measure_time(strassen_matrix_multiply, A, B)

print(f"Average execution time for regular matrix multiplication: {average_time_regular:.2f} seconds.")
print(f"Average execution time for Strassen matrix multiplication: {average_time_strassen:.2f} seconds.")
