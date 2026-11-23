import time
from functools import wraps

def logger(function):
    def wrapper(*args, **kwargs):
        print(f"----- {function.__name__}: start -----")
        output = function(*args, **kwargs)
        print(f"----- {function.__name__}: end -----")
        return output
    return wrapper      

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        if end - start > 60:
            print(f"{func.__name__} took {(end - start) / 60:.2f} minutes to complete")
        else:
            print(f"{func.__name__} took {end - start:.2f} seconds to complete")
        return result
    return wrapper