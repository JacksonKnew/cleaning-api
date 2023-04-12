from typing import List, Generator, Any


def batch_list(L: list, N: int) -> Generator[list, None, None]:
    """batches list L into N size chunks. Returns a generator"""
    for i in range(0, len(L), N):
        yield L[i : i + N]


def flatten_list(L: List[List[Any]]) -> List[Any]:
    return [x for l in L for x in l]
