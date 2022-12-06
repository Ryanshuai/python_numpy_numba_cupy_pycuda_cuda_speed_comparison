# CS217 Project: CUDA Alternative


## 1.Overview of project:
The goals of my project are 1. Study how to use other CUDA libraries. 2. Compare the speed of different libraries on basic matrix operations.
1. Other CUDA libraries includes cupy, numba, and pycuda.
2. I compared the speed of python, numpy, cupy, numba, pycuda and cuda(C++).

## 2.Implementation:
I used python, numpy, cupy, numba, pycuda to implement 1. reduce, 2. matrix addition, and 3. matrix multiplication.
For cuda (C++), I use exactly the same code as the assignments.

## 3.Status of the project:
### feature is completed:
python, numpy, cupy, numba, pycuda and cuda on reduce, matrix addtion, and matrix multiplication.
### Current problem:
1. The pycuda code can only be called within the python file in which it is declared.

### technical challenges:
1. pycuda uses a string (C++ code) as function input, which makes it very tricky to locate the bug.
2. Although "import pycuda.autoinit" is not explicitly used by the python code, it must be added in the pycuda declaration python file, otherwise a "pycuda._driver.LogicError" error will be reported without any error location.
3. For numba code, in the kernel function, the input of block size cannot be an expression, it must be a certain number.
4. Because GPU execution runs asynchronously relative to CPU execution, using normal python timing functions will only account for CPU-side time. I used cp.cuda.get_elapsed_time() to fix this problem.


### limitations:
1. There is much more commonly used operation, I only introduced reduce, matrix addition, and matrix multiplication.
2. My computer has only 16G memory, so I cannot test larger matrix with size more than 1e5.
## 4.Timing: 
### timing for reduce operation:
![](D:\github_project\python_numpy_numba_cupy_pycuda_cuda_speed_comparison\reduce.png)

### timing for matrix addition operation:
![](D:\github_project\python_numpy_numba_cupy_pycuda_cuda_speed_comparison\matadd.png)

### timing for matrix multiplication operation:
![](D:\github_project\python_numpy_numba_cupy_pycuda_cuda_speed_comparison\matmul.png)


## 5.How to run:
in terminal: (install libraries)

> pip install cupy
> pip install numba
> pip install pycuda

in project root:

Check correctness of python, numpy and cupy.
> python python_numpy_cupy.py  

Check correctness numba.
> python numba_calculations.py

Check correctness pycuda and measure running time of pycuda.
> python pycuda_calculations.py

Measure running time of python, numpy, cupy and numba.
> python test_time.py
