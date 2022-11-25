from time import time


def measure_time(func):
    def wrapper(nums, test_times):
        sum = func(nums)
        tic = time()
        for i in range(test_times):
            sum = func(nums)
        toc = time()
        return sum, (toc - tic) / test_times

    return wrapper
