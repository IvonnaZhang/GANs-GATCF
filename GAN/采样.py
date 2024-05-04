import os

import numpy as np

def load_matrix_from_txt(file_path):
    matrix = np.loadtxt(file_path)
    return matrix

def preprocess_matrix(matrix):
    matrix[matrix < 0] = 0  # 将负数置为0
    return matrix

def split_train_valid_test(matrix, density=0.1):
    mask = np.random.rand(*matrix.shape)
    mask[mask > density] = 1
    mask[mask <= density] = 0

    train_matrix = matrix * (1 - mask)

    size = int(0.05 * np.prod(matrix.shape))

    tr_idx, tc_idx = mask.nonzero()
    p = np.random.permutation(len(tr_idx))
    tr_idx, tc_idx = tr_idx[p], tc_idx[p]

    vr_idx, vc_idx = tr_idx[:size], tc_idx[:size]
    tr_idx, tc_idx = tr_idx[size:], tc_idx[size:]

    valid_matrix = np.zeros_like(matrix)
    test_matrix = np.zeros_like(matrix)

    valid_matrix[vr_idx, vc_idx] = matrix[vr_idx, vc_idx]
    test_matrix[tr_idx, tc_idx] = matrix[tr_idx, tc_idx]

    return train_matrix, valid_matrix, test_matrix, mask

def save_matrix_to_txt(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%f', delimiter='\t')  # 保存为整数格式

if __name__ == "__main__":
    file_path = "rtMatrix.txt"  # 指定你的txt文件路径
    output_dir = "mask采样划分"
    output_train_path = os.path.join(output_dir, "train_matrix.txt")
    #output_train_path = "train_matrix.txt"  # 指定训练集保存路径
    output_valid_path = os.path.join(output_dir, "valid_matrix.txt")
    #output_valid_path = "valid_matrix.txt"  # 指定验证集保存路径
    output_test_path = os.path.join(output_dir, "test_matrix.txt")
    #output_test_path = "test_matrix.txt"    # 指定测试集保存路径

    matrix = load_matrix_from_txt(file_path)
    matrix = preprocess_matrix(matrix)
    train_matrix, valid_matrix, test_matrix, mask = split_train_valid_test(matrix)

    # # 统计非零元素个数
    # train_nonzero_count = np.count_nonzero(train_matrix)
    # valid_nonzero_count = np.count_nonzero(valid_matrix)
    # test_nonzero_count = np.count_nonzero(test_matrix)
    #
    # print("Train Nonzero Count:", train_nonzero_count)
    # print("Valid Nonzero Count:", valid_nonzero_count)
    # print("Test Nonzero Count:", test_nonzero_count)

    save_matrix_to_txt(train_matrix, output_train_path)
    print("已保存train_matrix")
    save_matrix_to_txt(valid_matrix, output_valid_path)
    print("已保存valid_matrix")
    save_matrix_to_txt(test_matrix, output_test_path)
    print("已保存test_matrix")
