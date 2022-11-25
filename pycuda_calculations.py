import pycuda.driver as driver
from pycuda.compiler import SourceModule
import numpy as np

# for reduction --------------------------------------------------------------------------------------------------------
BLOCK_SIZE = 32
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


def pycuda_reduction(nums, res):
    thread_per_block = BLOCK_SIZE
    blocks_per_grid = (nums.size + thread_per_block - 1) // thread_per_block
    optimizedReduction = reducation_mod.get_function("optimizedReduction")
    optimizedReduction(res, nums, np.int32(nums.size),
                       block=(thread_per_block, 1, 1), grid=(blocks_per_grid, 1), shared=BLOCK_SIZE * 2 * 4)


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
    grid_dim = (A.shape[0] + TILE_SIZE - 1) // TILE_SIZE, (A.shape[1] + TILE_SIZE - 1) // TILE_SIZE
    matadd(dim, mat_A_gpu, mat_B_gpu, mat_res_gpu, block=(TILE_SIZE, TILE_SIZE, 1), grid=grid_dim)


# for matmul -----------------------------------------------------------------------------------------------------------
matmul_mod = SourceModule("""
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


def pycuda_matmul(m, n, k, mat_A_gpu, mat_B_gpu, mat_res_gpu):
    matmul = matmul_mod.get_function("mysgemm")
    grid_dim = (n + TILE_SIZE - 1) // TILE_SIZE, (m + TILE_SIZE - 1) // TILE_SIZE
    matmul(m, n, k, mat_A_gpu, mat_B_gpu, mat_res_gpu, block=(TILE_SIZE, TILE_SIZE, 1), grid=grid_dim,
           shared=2 * TILE_SIZE * TILE_SIZE * 4)


if __name__ == '__main__':
    A = np.random.rand(1000).astype(np.float32)
    res = np.zeros(((A.size + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=np.float32)
    A_gpu = driver.mem_alloc(A.nbytes)
    driver.memcpy_htod(A_gpu, A)
    res_gpu = driver.mem_alloc(res.nbytes)
    pycuda_reduction(A_gpu, res_gpu, )
    driver.memcpy_dtoh(res, res_gpu)
    print(np.allclose(np.sum(A), res.sum()))

    # A = np.random.randn(1000, 1000).astype(np.float32)
    # A_gpu = driver.mem_alloc(A.nbytes)
    # driver.memcpy_htod(A_gpu, A)
    # B = np.random.randn(1000, 1000).astype(np.float32)
    # B_gpu = driver.mem_alloc(B.nbytes)
    # driver.memcpy_htod(B_gpu, B)
    # C = np.zeros((1000, 1000), dtype=np.float32)
    # C_gpu = driver.mem_alloc(C.nbytes)
    # pycuda_matadd(np.int32(A.shape[0]), A_gpu, B_gpu, C_gpu)
    # driver.memcpy_dtoh(C, C_gpu)
    # print(np.allclose(C, A + B))
    #
    # A = np.random.randn(1000, 1000).astype(np.float32)
    # A_gpu = driver.mem_alloc(A.nbytes)
    # driver.memcpy_htod(A_gpu, A)
    # B = np.random.randn(1000, 1000).astype(np.float32)
    # B_gpu = driver.mem_alloc(B.nbytes)
    # driver.memcpy_htod(B_gpu, B)
    # C = np.zeros((1000, 1000), dtype=np.float32)
    # C_gpu = driver.mem_alloc(C.nbytes)
    # pycuda_matmul(np.int32(A.shape[0]), np.int32(A.shape[1]), np.int32(B.shape[1]), A_gpu, B_gpu, C_gpu)
    # driver.memcpy_dtoh(C, C_gpu)
    # print(np.allclose(C, A @ B))
