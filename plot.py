import matplotlib.pyplot as plt

x_reduce_python = [1e3, 1e4, 1e5, 1e6, 1e7]
y_reduce_python = [0.00011849999999998435, 0.0010840999999999952, 0.011275466666666642, 0.11193423333333345,
                   1.1437677333333334]
plt.plot(x_reduce_python, y_reduce_python, "gold", label="python")

x_reduce_numpy = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_numpy = [5.699999999997374e-06, 6.82000000002958e-06, 3.2779999999998924e-05, 0.0003087599999999746,
                  0.0049526100000000465, 0.051971230000000014, 0.49556737999999995]
plt.plot(x_reduce_numpy, y_reduce_numpy, "royalblue", label="numpy")

x_reduce_cupy = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_cupy = [2.653999999999712e-05, 3.7440000000188435e-05, 3.47200000000214e-05, 3.3790000000166744e-05,
                 3.335999999976025e-05, 3.360000000007801e-05, 4.2249999999910415e-05]
plt.plot(x_reduce_cupy, y_reduce_cupy, "-g", label="cupy")

x_reduce_numba = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_numba = [5.8729999999229676e-05, 5.464999999986731e-05, 5.4749999999614826e-05, 6.817000000012286e-05,
                  5.5629999999950995e-05, 5.239999999986367e-05, 6.246999999746095e-05]
plt.plot(x_reduce_numba, y_reduce_numba, "-c", label="numba")

x_reduce_pycuda = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_pycuda = [4.6659999999976165e-05, 4.885999999997282e-05, 5.0920000000020946e-05, 7.345000000000824e-05,
                   1.925000000002619e-05, 1.911999999997249e-05, 4.552999999987151e-05]
plt.plot(x_reduce_pycuda, y_reduce_pycuda, "black", label="pycuda")

plt.title('reduce')
plt.xlabel('mat length', fontsize=14)
plt.ylabel('time', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper left")
plt.show()

x_matadd_python = [1e2, 1e3]
y_matadd_python = [0.00558166666666667, 0.5582099333333334]
plt.plot(x_matadd_python, y_matadd_python, "gold", label="python")

x_matadd_numpy = [1e2, 1e3, 1e4]
y_matadd_numpy = [2.1700000000013374e-06, 0.001481139999999992, 0.13732965000000003]
plt.plot(x_matadd_numpy, y_matadd_numpy, "royalblue", label="numpy")

x_matadd_cupy = [1e2, 1e3, 1e4]
y_matadd_cupy = [2.7199999999893976e-05, 2.3930000000049744e-05, 5.40599999999003e-05]
plt.plot(x_matadd_cupy, y_matadd_cupy, "-g", label="cupy")

x_matadd_numba = [1e2, 1e3, 1e4]
y_matadd_numba = [6.798999999997335e-05, 6.331999999993343e-05, 6.512999999976898e-05]
plt.plot(x_matadd_numba, y_matadd_numba, "-c", label="numba")

x_matadd_pycuda = [1e2, 1e3, 1e4]
y_matadd_pycuda = [2.667999999985682e-05, 2.4330000000105655e-05, 2.454000000042811e-05]
plt.plot(x_matadd_pycuda, y_matadd_pycuda, "black", label="pycuda")

plt.title('matadd')
plt.xlabel('mat length', fontsize=14)
plt.ylabel('time', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

x_matmul_python = [1e2, 2e2]
y_matmul_python = [0.6546607666666672, 5.0521908]
plt.plot(x_matmul_python, y_matmul_python, "gold", label="python")

x_matmul_numpy = [1e2, 1e3, 1e4]
y_matmul_numpy = [2.1620000000055484e-05, 0.005557010000000062, 4.9405981699999995]
plt.plot(x_matmul_numpy, y_matmul_numpy, "royalblue", label="numpy")

x_matmul_cupy = [1e2, 1e3, 1e4]
y_matmul_cupy = [2.865999999954738e-05, 3.6800000000880575e-05, 0.0001353399999999283]
plt.plot(x_matmul_cupy, y_matmul_cupy, "-g", label="cupy")

x_matmul_numba = [1e2, 1e3, 1e4]
y_matmul_numba = [7.092999999969152e-05, 6.374000000022306e-05, 0.0001355799999998908]
plt.plot(x_matmul_numba, y_matmul_numba, "-c", label="numba")

x_matmul_pycuda = [1e2, 1e3, 1e4]
y_matmul_pycuda = [2.394000000052188e-05, 6.0129999999958274e-05, 2.3519999999876972e-05]
plt.plot(x_matmul_pycuda, y_matmul_pycuda, "black", label="pycuda")

plt.title('matmul')
plt.xlabel('mat length', fontsize=14)
plt.ylabel('time', fontsize=14)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
