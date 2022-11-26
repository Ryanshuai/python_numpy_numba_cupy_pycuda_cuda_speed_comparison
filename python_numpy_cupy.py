import numpy as np
import cupy as cp


def python_reduce(nums):
    sum = 0
    for num in nums:
        sum += num
    return sum


def python_matsum(mat_A, mat_B, mat_res):
    for i in range(mat_A.shape[0]):
        for j in range(mat_A.shape[1]):
            mat_res[i][j] = mat_A[i][j] + mat_B[i][j]
    return mat_res


def python_matmul(mat_A, mat_B, mat_res):
    for i in range(mat_res.shape[0]):
        for j in range(mat_res.shape[1]):
            for k in range(mat_A.shape[1]):
                mat_res[i][j] += mat_A[i][k] * mat_B[k][j]
    return mat_res


def numpy_reduce(nums: np.ndarray):
    return np.sum(nums)


def numpy_matadd(mat_A, mat_B):
    return mat_A + mat_B


def numpy_matmul(mat_A, mat_B):
    return mat_A @ mat_B


def cupy_reduce(nums: np.ndarray):
    return cp.sum(nums)


def cupy_matadd(mat_A, mat_B):
    return mat_A + mat_B


def cupy_matmul(mat_A, mat_B):
    return mat_A @ mat_B


if __name__ == '__main__':
    A = np.random.rand(1000).astype(np.float32)
    res = np.zeros(((A.size + 1024 - 1) // 1024), dtype=np.float32)
    print(python_reduce(A))
    print(numpy_reduce(A))
    A_cupy = cp.asarray(A)
    print(cupy_reduce(A_cupy))

    A = np.random.randn(100, 100).astype(np.float32)
    B = np.random.randn(100, 100).astype(np.float32)
    C = np.zeros((100, 100), dtype=np.float32)

    print(np.allclose(python_matsum(A, B, C), numpy_matadd(A, B)))
    cupy_sum_res = cupy_matadd(cp.asarray(A), cp.asarray(B)).get()
    print(np.allclose(numpy_matadd(A, B), cupy_sum_res))

    A = np.random.randn(100, 100).astype(np.float32)
    B = np.random.randn(100, 100).astype(np.float32)
    C = np.zeros((100, 100), dtype=np.float32)

    print(np.allclose(python_matmul(A, B, C), numpy_matmul(A, B), rtol=1e-4, atol=1e-4))

    cupy_res = cupy_matmul(cp.asarray(A), cp.asarray(B)).get()
    print(np.allclose(numpy_matmul(A, B), cupy_res, rtol=1e-4, atol=1e-4))
