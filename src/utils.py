from time import time


def chrono(fn):
    def wrapper(*args):
        before = time()
        result = fn(*args)
        print(fn.__name__, ":", 1 / (time() - before), "FPS")
        return result

    return wrapper
