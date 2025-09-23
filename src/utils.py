from time import time
from typing import List


def chrono(fn):
    def wrapper(*args):
        before = time()
        result = fn(*args)
        print(fn.__name__, ":", 1 / (time() - before), "FPS")
        return result

    return wrapper


def parse_source(sources: List[str]):
    return list(map(lambda source: int(source) if source.isnumeric() else source, sources))
