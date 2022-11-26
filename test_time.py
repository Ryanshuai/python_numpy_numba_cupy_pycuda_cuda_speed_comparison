from time import perf_counter
import numpy as np
import cupy as cp
from numba import cuda as numba_cuda

from python_numpy_cupy import python_reduce, numpy_reduce, cupy_reduce
from python_numpy_cupy import python_matadd, numpy_matadd, cupy_matadd
from python_numpy_cupy import python_matmul, numpy_matmul, cupy_matmul
from numba_calculations import numba_reduce, numba_matadd, numba_matmul


def measure_time(func, *args, test_count=10):
    func(*args)
    start = perf_counter()
    for _ in range(test_count):
        func(*args)
    end = perf_counter()
    return (end - start) / test_count


if __name__ == '__main__':
    print("\nREDUCE: ******************************************************************************")
    print("Python reduce time:")
    for exp in range(3, 8):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_python = np.random.rand(n).astype(np.float32)
        print(measure_time(python_reduce, A_python, test_count=3))

    print("\n Numpy reduce time:")
    for exp in range(3, 10):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_numpy = np.random.rand(n).astype(np.float32)
        print(measure_time(numpy_reduce, A_numpy, test_count=10))

    print("\n Cupy reduce time:")
    for exp in range(3, 10):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_cupy = cp.asarray(np.random.rand(n).astype(np.float32))
        print(measure_time(cupy_reduce, A_cupy, test_count=10))

    print("\n Numba reduce time:")
    for exp in range(3, 10):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        BLOCK_SIZE = 512
        A_numba_cpu = np.random.rand(n).astype(np.float32)
        A_numba_gpu = numba_cuda.to_device(A_numba_cpu)
        res_numba_cpu = np.zeros(((A_numba_cpu.size + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=np.float32)
        res_numba_gpu = numba_cuda.to_device(res_numba_cpu)
        print(measure_time(numba_reduce, A_numba_gpu, res_numba_gpu, test_count=10))

    print("\nMATADD: ******************************************************************************")
    print("Python matadd time: ")
    for exp in range(2, 3):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_python = np.random.randn(n, n).astype(np.float32)
        B_python = np.random.randn(n, n).astype(np.float32)
        res_python = np.zeros((n, n), dtype=np.float32)
        print(measure_time(python_matadd, A_python, B_python, res_python, test_count=3))

    print("\n Numpy matadd time: ")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_numpy = np.random.randn(n, n).astype(np.float32)
        B_numpy = np.random.randn(n, n).astype(np.float32)
        res_numpy = np.zeros((n, n), dtype=np.float32)
        print(measure_time(numpy_matadd, A_numpy, B_numpy, test_count=10))

    print("\n Cupy matadd time: ")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_cupy = cp.asarray(np.random.randn(n, n).astype(np.float32))
        B_cupy = cp.asarray(np.random.randn(n, n).astype(np.float32))
        print(measure_time(cupy_matadd, A_cupy, B_cupy, test_count=10))

    print("\n Numba matadd time: ")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        BLOCK_SIZE = 512
        A_numba_cpu = np.random.randn(n, n).astype(np.float32)
        B_numba_cpu = np.random.randn(n, n).astype(np.float32)
        res_numba_cpu = np.zeros((n, n), dtype=np.float32)
        A_numba_gpu = numba_cuda.to_device(A_numba_cpu)
        B_numba_gpu = numba_cuda.to_device(B_numba_cpu)
        res_numba_gpu = numba_cuda.to_device(res_numba_cpu)
        print(measure_time(numba_matadd, A_numba_gpu, B_numba_gpu, res_numba_gpu, test_count=10))

    print("\nMATMUL: ******************************************************************************")
    print("Python matmul time: ")
    for exp in range(2, 3):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_python = np.random.randn(n, n).astype(np.float32)
        B_python = np.random.randn(n, n).astype(np.float32)
        res_python = np.zeros((n, n), dtype=np.float32)
        print(measure_time(python_matmul, A_python, B_python, res_python, test_count=3))

    print("\n Numpy matmul time: ")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_numpy = np.random.randn(n, n).astype(np.float32)
        B_numpy = np.random.randn(n, n).astype(np.float32)
        res_numpy = np.zeros((n, n), dtype=np.float32)
        print(measure_time(numpy_matmul, A_numpy, B_numpy, test_count=10))

    print("\n Cupy matmul time: ")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        A_cupy = cp.asarray(np.random.randn(n, n).astype(np.float32))
        B_cupy = cp.asarray(np.random.randn(n, n).astype(np.float32))
        print(measure_time(cupy_matmul, A_cupy, B_cupy, test_count=10))

    print("\n Numba matmul time: ")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"n = 1e{exp}:", end=' ')
        BLOCK_SIZE = 512
        A_numba_cpu = np.random.randn(n, n).astype(np.float32)
        B_numba_cpu = np.random.randn(n, n).astype(np.float32)
        res_numba_cpu = np.zeros((n, n), dtype=np.float32)
        A_numba_gpu = numba_cuda.to_device(A_numba_cpu)
        B_numba_gpu = numba_cuda.to_device(B_numba_cpu)
        res_numba_gpu = numba_cuda.to_device(res_numba_cpu)
        print(measure_time(numba_matmul, A_numba_gpu, B_numba_gpu, res_numba_gpu, test_count=10))
