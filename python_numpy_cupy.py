import cupy
import numpy as np
import cupy as cp
from time import time

TEST_TIMES = 10


def measure_time(func):
    def wrapper(nums, test_times):
        sum = func(nums)
        tic = time()
        for i in range(test_times):
            sum = func(nums)
        toc = time()
        return sum, (toc - tic) / test_times

    return wrapper


@measure_time
def python_reduce(nums):
    sum = 0
    for num in nums:
        sum += num
    return sum


@measure_time
def python_matsum(mat_1, mat_2, mat_res):
    for i in range(mat_1.shape[0]):
        for j in range(mat_1.shape[1]):
            mat_res[i][j] = mat_1[i][j] + mat_2[i][j]
    return mat_res


@measure_time
def numpy_matmul(mat1, mat2, mat_res):
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                mat_res[i][j] += mat1[i][k] * mat2[k][j]
    return mat_res


@measure_time
def numpy_reduce(nums: np.ndarray):
    return np.sum(nums)


@measure_time
def numpy_matsum(mat_1, mat_2, mat_res):
    mat_res = mat_1 + mat_2
    return mat_res


@measure_time
def cupy_reduce(nums: np.ndarray):
    return cp.sum(nums)


@measure_time
def cupy_matsum(mat_1, mat_2, mat_res):
    mat_res = mat_1 + mat_2
    return mat_res


@measure_time
def cupy_matmul(mat1, mat2, mat_res):
    mat_res = cp.matmul(mat1, mat2)
    return mat_res


arr = np.random.rand(100_000_000)
print("Python: ", python_reduce(arr, TEST_TIMES))
print("Numpy: ", numpy_reduce(arr, TEST_TIMES))
print("Cupy: ", cupy_reduce(cupy.asarray(arr), TEST_TIMES))
