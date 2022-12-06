import numpy as np
import numba
from numba import cuda

# for reduction --------------------------------------------------------------------------------------------------------
BLOCK_SIZE = 512
BLOCK_SIZE_2 = BLOCK_SIZE * 2


@cuda.jit(cache=True)
def numba_reduce_thread(nums, res):
    partial_sum = cuda.shared.array(shape=BLOCK_SIZE_2, dtype=numba.float32)
    thread_idx = cuda.threadIdx.x
    block_start_idx = cuda.blockIdx.x * cuda.blockDim.x * 2

    if block_start_idx + thread_idx < nums.size:
        partial_sum[thread_idx] = nums[block_start_idx + thread_idx]
    else:
        partial_sum[thread_idx] = 0

    if block_start_idx + thread_idx + cuda.blockDim.x < nums.size:
        partial_sum[thread_idx + cuda.blockDim.x] = nums[block_start_idx + thread_idx + cuda.blockDim.x]
    else:
        partial_sum[thread_idx + cuda.blockDim.x] = 0

    active_threads = cuda.blockDim.x
    while active_threads >= 1:
        cuda.syncthreads()
        if thread_idx < active_threads:
            partial_sum[thread_idx] += partial_sum[thread_idx + active_threads]
        active_threads //= 2

    if thread_idx == 0:
        res[cuda.blockIdx.x] = partial_sum[0]


def numba_reduce(nums, res):
    threads_per_block = BLOCK_SIZE
    blocks_per_grid = (nums.size + threads_per_block - 1) // threads_per_block

    numba_reduce_thread[blocks_per_grid, threads_per_block](nums, res)


# for matadd -----------------------------------------------------------------------------------------------------------
TILE_SIZE = 32


@cuda.jit(cache=True)
def numba_matadd_thread(mat_A, mat_B, mat_res):
    global_x, global_y = cuda.grid(2)

    if global_y < mat_res.shape[0] and global_x < mat_res.shape[1]:
        mat_res[global_y, global_x] = mat_A[global_y, global_x] + mat_B[global_y, global_x]


def numba_matadd(mat_A, mat_B, mat_res):
    threads_per_block = (TILE_SIZE, TILE_SIZE)
    blocks_per_grid = ((mat_res.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                       (mat_res.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])

    numba_matadd_thread[blocks_per_grid, threads_per_block](mat_A, mat_B, mat_res)


# for matmul -----------------------------------------------------------------------------------------------------------
@cuda.jit(cache=True)
def numba_matmul_thread(mat_A, mat_B, mat_res):
    A_block = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32)
    B_block = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32)

    global_x, global_y = cuda.grid(2)
    th_x, th_y = cuda.threadIdx.x, cuda.threadIdx.y

    phase = (mat_A.shape[1] + cuda.blockDim.x - 1) // cuda.blockDim.x

    sum = 0
    for ph in range(phase):
        A_x = ph * cuda.blockDim.x + th_x
        A_y = global_y
        B_x = global_x
        B_y = ph * cuda.blockDim.y + th_y

        if A_y < mat_A.shape[0] and A_x < mat_A.shape[1]:
            A_block[th_y, th_x] = mat_A[A_y, A_x]
        else:
            A_block[th_y, th_x] = 0

        if B_y < mat_B.shape[0] and B_x < mat_B.shape[1]:
            B_block[th_y, th_x] = mat_B[B_y, B_x]
        else:
            B_block[th_y, th_x] = 0

        cuda.syncthreads()

        if global_y < mat_res.shape[0] and global_x < mat_res.shape[1]:
            for i in range(cuda.blockDim.x):
                sum += A_block[th_y, i] * B_block[i, th_x]

        cuda.syncthreads()

    if global_y < mat_res.shape[0] and global_x < mat_res.shape[1]:
        mat_res[global_y, global_x] = sum


def numba_matmul(mat_A, mat_B, mat_res):
    threads_per_block = (TILE_SIZE, TILE_SIZE)
    blocks_per_grid = ((mat_res.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                       (mat_res.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])

    numba_matmul_thread[blocks_per_grid, threads_per_block](mat_A, mat_B, mat_res)

    return mat_res


if __name__ == '__main__':
    A = np.random.rand(1000).astype(np.float32)
    res = np.zeros(((A.size + BLOCK_SIZE - 1) // BLOCK_SIZE), dtype=np.float32)
    res_cuda = cuda.to_device(res)
    numba_reduce(cuda.to_device(A), res_cuda)
    res = res_cuda.copy_to_host()
    print(np.allclose(np.sum(A), np.sum(res)))

    A = np.random.rand(1000, 1000).astype(np.float32)
    B = np.random.rand(1000, 1000).astype(np.float32)
    C = np.zeros((1000, 1000), dtype=np.float32)
    C_cuda = cuda.to_device(C)
    numba_matadd(cuda.to_device(A), cuda.to_device(B), C_cuda)
    C = C_cuda.copy_to_host()
    print(np.allclose(C, A + B))

    A = np.random.rand(1000, 1000).astype(np.float32)
    B = np.random.rand(1000, 1000).astype(np.float32)
    C = np.zeros((1000, 1000), dtype=np.float32)
    C_cuda = cuda.to_device(C)
    numba_matmul(cuda.to_device(A), cuda.to_device(B), C_cuda)
    C = C_cuda.copy_to_host()
    print(np.allclose(C, A @ B))
