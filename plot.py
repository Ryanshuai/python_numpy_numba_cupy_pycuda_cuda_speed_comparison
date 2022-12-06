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
y_reduce_cupy = [2.847039997577667e-05, 3.8105601072311405e-05, 3.1436800956726076e-05, 2.4166400730609895e-05,
                 0.00011632640361785887, 0.001104588794708252, 0.010706432342529297]
plt.plot(x_reduce_cupy, y_reduce_cupy, "-g", label="cupy")

x_reduce_numba = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_numba = [7.29088008403778e-05, 9.031360149383545e-05, 5.926079750061035e-05, 6.338559985160828e-05,
                  0.0005272575855255127, 0.005344870376586914, 0.05438638305664063]
plt.plot(x_reduce_numba, y_reduce_numba, "-c", label="numba")

x_reduce_pycuda = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_pycuda = [4.253439903259277e-05, 3.461120128631592e-05, 3.478080034255981e-05, 4.8127999901771545e-05,
                   0.00046817278861999514, 0.005154710388183594, 0.03864739990234375]
plt.plot(x_reduce_pycuda, y_reduce_pycuda, "black", label="pycuda")

x_reduce_cuda = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
y_reduce_cuda = [0.004223, 0.003536, 0.003121, 0.003592, 0.003897, 0.002833, 0.018442]
plt.plot(x_reduce_cuda, y_reduce_cuda, "r", label="cuda")

plt.title('reduce')
plt.xlabel('mat length')
plt.ylabel('time')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper left")
plt.savefig('reduce.png', dpi=800)
plt.show()

x_matadd_python = [1e2, 1e3]
y_matadd_python = [0.00558166666666667, 0.5582099333333334]
plt.plot(x_matadd_python, y_matadd_python, "gold", label="python")

x_matadd_numpy = [1e2, 1e3, 1e4]
y_matadd_numpy = [2.1700000000013374e-06, 0.001481139999999992, 0.13732965000000003]
plt.plot(x_matadd_numpy, y_matadd_numpy, "royalblue", label="numpy")

x_matadd_cupy = [1e2, 1e3, 1e4]
y_matadd_cupy = [3.153280019760132e-05, 3.819519877433777e-05, 0.003117158317565918]
plt.plot(x_matadd_cupy, y_matadd_cupy, "-g", label="cupy")

x_matadd_numba = [1e2, 1e3, 1e4]
y_matadd_numba = [7.616000175476074e-05, 6.38975977897644e-05, 0.003133337593078613]
plt.plot(x_matadd_numba, y_matadd_numba, "-c", label="numba")

x_matadd_pycuda = [1e2, 1e3, 1e4]
y_matadd_pycuda = [2.29312002658844e-05, 3.7376001477241516e-05, 0.0036380672454833984]
plt.plot(x_matadd_pycuda, y_matadd_pycuda, "black", label="pycuda")

x_matadd_cuda = [1e2, 1e3, 1e4]
y_matadd_cuda = [0.002780, 0.004260, 0.007070]
plt.plot(x_matadd_cuda, y_matadd_cuda, "r", label="cuda")

plt.title('matadd')
plt.xlabel('mat length')
plt.ylabel('time')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('matadd.png', dpi=800)
plt.show()

x_matmul_python = [1e2, 2e2]
y_matmul_python = [0.6546607666666672, 5.0521908]
plt.plot(x_matmul_python, y_matmul_python, "gold", label="python")

x_matmul_numpy = [1e2, 1e3, 1e4]
y_matmul_numpy = [2.1620000000055484e-05, 0.005557010000000062, 4.9405981699999995]
plt.plot(x_matmul_numpy, y_matmul_numpy, "royalblue", label="numpy")

x_matmul_cupy = [1e2, 1e3, 1e4]
y_matmul_cupy = [7.774080038070679e-05, 0.0003435519933700561, 0.19716658935546877]
plt.plot(x_matmul_cupy, y_matmul_cupy, "-g", label="cupy")

x_matmul_numba = [1e2, 1e3, 1e4]
y_matmul_numba = [8.714240193367004e-05, 0.013781607055664063, 11.73740859375]
plt.plot(x_matmul_numba, y_matmul_numba, "-c", label="numba")

x_matmul_pycuda = [1e2, 1e3, 1e4]
y_matmul_pycuda = [5.27679979801178e-05, 0.0017118207931518554, 1.5892228515625]
plt.plot(x_matmul_pycuda, y_matmul_pycuda, "black", label="pycuda")

x_matmul_cuda = [1e2, 1e3, 1e4]
y_matmul_cuda = [0.003354, 0.006734, 1.549046]
plt.plot(x_matmul_cuda, y_matmul_cuda, "r", label="cuda")

plt.title('matmul')
plt.xlabel('mat length')
plt.ylabel('time')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('matmul.png', dpi=800)
plt.show()
