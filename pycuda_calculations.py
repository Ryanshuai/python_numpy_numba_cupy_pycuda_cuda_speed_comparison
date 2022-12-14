import pycuda.driver as driver
from pycuda.compiler import SourceModule
import pycuda.autoinit  # need this for decode mod string
import numpy as np
import cupy as cp

# for reduction --------------------------------------------------------------------------------------------------------
BLOCK_SIZE = 512
reducation_mod = SourceModule("""
#include <stdio.h>
#define BLOCK_SIZE %(block_size)d

__global__ void optimizedReduction(float *out, float *in, unsigned size) {
    __shared__ float partial_sum[BLOCK_SIZE * 2];
    unsigned int thread_idx = threadIdx.x;
    unsigned int block_start_idx = blockIdx.x * blockDim.x * 2;

    if (block_start_idx + thread_idx < size) {
        partial_sum[thread_idx] = in[block_start_idx + thread_idx];
    } else {
        partial_sum[thread_idx] = 0;
    }

    if (block_start_idx + blockDim.x + thread_idx < size) {
        partial_sum[blockDim.x + thread_idx] = in[block_start_idx + blockDim.x + thread_idx];
    } else {
        partial_sum[blockDim.x + thread_idx] = 0;
    }

    for (unsigned int active_threads = blockDim.x; active_threads >= 1; active_threads /= 2) {
        __syncthreads();
        if (thread_idx < active_threads) {
            partial_sum[thread_idx] += partial_sum[thread_idx + active_threads];
        }
    }

    if (thread_idx == 0) {
        out[blockIdx.x] = partial_sum[0];
    }
}
""" % {"block_size": BLOCK_SIZE})


def pycuda_reduce(nums, nums_len, res):
    thread_per_block = BLOCK_SIZE
    blocks_per_grid = (nums_len + thread_per_block - 1) // thread_per_block
    optimizedReduction = reducation_mod.get_function("optimizedReduction")
    optimizedReduction(res, nums, np.int32(nums_len),
                       block=(thread_per_block, 1, 1), grid=(blocks_per_grid, 1), shared=BLOCK_SIZE * 2 * 4)


def test_pycuda():
    A = np.random.rand(1000).astype(np.float32)
    res = np.zeros(((A.size + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=np.float32)
    A_gpu = driver.mem_alloc(A.nbytes)
    driver.memcpy_htod(A_gpu, A)
    res_gpu = driver.mem_alloc(res.nbytes)
    pycuda_reduce(A_gpu, len(A), res_gpu)
    driver.memcpy_dtoh(res, res_gpu)
    print(np.allclose(np.sum(A), res.sum()))


# for matadd -----------------------------------------------------------------------------------------------------------

matadd_mod = SourceModule("""
__global__ void matAdd(int dim, const float *A, const float *B, float* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dim && y < dim) {
        int index = y * dim + x;
        C[index] = A[index] + B[index];
    }
}
""")

TILE_SIZE = 32


def pycuda_matadd(dim, mat_A_gpu, mat_B_gpu, mat_res_gpu):
    matadd = matadd_mod.get_function("matAdd")
    grid_dim = (dim + TILE_SIZE - 1) // TILE_SIZE, (dim + TILE_SIZE - 1) // TILE_SIZE
    matadd(np.int32(dim), mat_A_gpu, mat_B_gpu, mat_res_gpu, block=(TILE_SIZE, TILE_SIZE, 1), grid=grid_dim)


# for matmul -----------------------------------------------------------------------------------------------------------
matmul_mod = SourceModule("""
#include <stdio.h>
#define TILE_SIZE %(tile_size)d
__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {

    // Compute C = A x B
    //  A is m x k,  B is k x n, C is m x n

    __shared__ float A_block[TILE_SIZE * TILE_SIZE];
    __shared__ float B_block[TILE_SIZE * TILE_SIZE];

    unsigned int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int global_x = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int phase = (k + blockDim.x - 1) / blockDim.x;

    float sum = 0.0;
    for (int ph = 0; ph < phase; ph++) {
        unsigned int A_x = ph * blockDim.x + threadIdx.x;
        unsigned int A_y = global_y;
        unsigned int B_x = global_x;
        unsigned int B_y = ph * blockDim.x + threadIdx.y;

        if (A_y < m && A_x < k) {
            A_block[threadIdx.y * blockDim.x + threadIdx.x] = A[A_y * k + A_x];
        } else {
            A_block[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        if (B_y < k && B_x < n) {
            B_block[threadIdx.y * blockDim.x + threadIdx.x] = B[B_y * n + B_x];
        } else {
            B_block[threadIdx.y * blockDim.x + threadIdx.x] = 0;
        }

        __syncthreads();

        if (global_y < m && global_x < n) {
            for (int idx = 0; idx < blockDim.x; idx++) {
                sum += A_block[threadIdx.y * blockDim.x + idx] * B_block[idx * blockDim.x + threadIdx.x];
            }
        }

        __syncthreads();
    }
    if (global_y < m && global_x < n) {
        C[global_y * n + global_x] = sum;
    }
    /*************************************************************************/
}
""" % {"tile_size": TILE_SIZE})


def pycuda_matmul(m, k, n, mat_A_gpu, mat_B_gpu, mat_res_gpu):
    matmul = matmul_mod.get_function("mysgemm")
    grid_dim = (n + TILE_SIZE - 1) // TILE_SIZE, (m + TILE_SIZE - 1) // TILE_SIZE
    matmul(np.int32(m), np.int32(n), np.int32(k), mat_A_gpu, mat_B_gpu, mat_res_gpu,
           block=(TILE_SIZE, TILE_SIZE, 1), grid=grid_dim, shared=2 * TILE_SIZE * TILE_SIZE * 4)


if __name__ == '__main__':
    from time import perf_counter


    def gpu_measure_time(func, *args, test_count=10):
        func(*args)
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for _ in range(test_count):
            func(*args)
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end) / 1000 / test_count


    A = np.random.rand(1000).astype(np.float32)
    res = np.zeros(((A.size + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=np.float32)
    A_gpu = driver.mem_alloc(A.nbytes)
    driver.memcpy_htod(A_gpu, A)
    res_gpu = driver.mem_alloc(res.nbytes)
    pycuda_reduce(A_gpu, len(A), res_gpu)
    driver.memcpy_dtoh(res, res_gpu)
    print(np.allclose(np.sum(A), res.sum()))

    A = np.random.randn(1000, 1000).astype(np.float32)
    A_f = A.flatten()
    A_gpu = driver.mem_alloc(A_f.nbytes)
    driver.memcpy_htod(A_gpu, A_f)
    B = np.random.randn(1000, 1000).astype(np.float32)
    B_f = B.flatten()
    B_gpu = driver.mem_alloc(B_f.nbytes)
    driver.memcpy_htod(B_gpu, B_f)
    C = np.zeros((1000, 1000), dtype=np.float32)
    C_f = C.flatten()
    C_gpu = driver.mem_alloc(C_f.nbytes)
    pycuda_matadd(A.shape[0], A_gpu, B_gpu, C_gpu)
    driver.memcpy_dtoh(C, C_gpu)
    print(np.allclose(C, A + B))

    A = np.random.randn(1000, 1000).astype(np.float32)
    A_f = A.flatten()
    A_gpu = driver.mem_alloc(A_f.nbytes)
    driver.memcpy_htod(A_gpu, A_f)
    B = np.random.randn(1000, 1000).astype(np.float32)
    B_f = B.flatten()
    B_gpu = driver.mem_alloc(B_f.nbytes)
    driver.memcpy_htod(B_gpu, B_f)
    C = np.zeros((1000, 1000), dtype=np.float32)
    C_f = C.flatten()
    C_gpu = driver.mem_alloc(C_f.nbytes)
    pycuda_matmul(A.shape[0], A.shape[1], B.shape[1], A_gpu, B_gpu, C_gpu)
    driver.memcpy_dtoh(C, C_gpu)
    print(np.allclose(np.resize(C, (1000, 1000)), A @ B, rtol=1e-4, atol=1e-4))

    print("\nREDUCE: ******************************************************************************")
    for exp in range(3, 10):
        n = 10 ** exp
        print(f"1e{exp}:", end=' ')

        A_pycuda_cpu = np.random.rand(n).astype(np.float32)
        res_pycuda_cpu = np.zeros(((A_pycuda_cpu.size + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=np.float32)
        A_pycuda_gpu = driver.mem_alloc(A_pycuda_cpu.nbytes)
        res_pycuda_gpu = driver.mem_alloc(res_pycuda_cpu.nbytes)
        driver.memcpy_htod(A_pycuda_gpu, A_pycuda_cpu)
        print("Pycuda reduce time: ",
              gpu_measure_time(pycuda_reduce, A_pycuda_gpu, len(A_pycuda_cpu), res_pycuda_gpu, test_count=10))

    print("\nMATADD: ******************************************************************************")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"1e{exp}:", end=' ')

        A_pycuda_cpu = np.random.randn(n, n).astype(np.float32)
        A_pycuda_cpu_f = A_pycuda_cpu.flatten()
        A_pycuda_gpu = driver.mem_alloc(A_pycuda_cpu_f.nbytes)
        driver.memcpy_htod(A_pycuda_gpu, A_pycuda_cpu_f)
        B_pycuda_cpu = np.random.randn(n, n).astype(np.float32)
        B_pycuda_cpu_f = B_pycuda_cpu.flatten()
        B_pycuda_gpu = driver.mem_alloc(B_pycuda_cpu_f.nbytes)
        driver.memcpy_htod(B_pycuda_gpu, B_pycuda_cpu_f)
        C_pycuda_cpu = np.zeros((n, n), dtype=np.float32)
        C_pycuda_cpu_f = C_pycuda_cpu.flatten()
        C_pycuda_gpu = driver.mem_alloc(C_pycuda_cpu_f.nbytes)
        print("Pycuda matmul time: ",
              gpu_measure_time(pycuda_matadd, A_pycuda_cpu.shape[0], A_pycuda_gpu, B_pycuda_gpu, C_pycuda_gpu,
                               test_count=10))

    print("\nMATMUL: ******************************************************************************")
    for exp in range(2, 5):
        n = 10 ** exp
        print(f"1e{exp}:", end=' ')

        A_pycuda_cpu = np.random.randn(n, n).astype(np.float32)
        A_pycuda_cpu_f = A_pycuda_cpu.flatten()
        A_pycuda_gpu = driver.mem_alloc(A_pycuda_cpu_f.nbytes)
        driver.memcpy_htod(A_pycuda_gpu, A_pycuda_cpu_f)
        B_pycuda_cpu = np.random.randn(n, n).astype(np.float32)
        B_pycuda_cpu_f = B_pycuda_cpu.flatten()
        B_pycuda_gpu = driver.mem_alloc(B_pycuda_cpu_f.nbytes)
        driver.memcpy_htod(B_pycuda_gpu, B_pycuda_cpu_f)
        C_pycuda_cpu = np.zeros((n, n), dtype=np.float32)
        C_pycuda_cpu_f = C_pycuda_cpu.flatten()
        C_pycuda_gpu = driver.mem_alloc(C_pycuda_cpu_f.nbytes)
        print("Pycuda matmul time: ",
              gpu_measure_time(pycuda_matmul, A_pycuda_cpu.shape[0], A_pycuda_cpu.shape[1], B_pycuda_cpu.shape[1],
                               A_pycuda_gpu, B_pycuda_gpu, C_pycuda_gpu, test_count=10))
