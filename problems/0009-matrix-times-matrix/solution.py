import numpy as np

def matrixmul(
    a:list[list[int|float]],
    b:list[list[int|float]]
)-> list[list[int|float]]:
    if len(a[1]) != len(b):
        return -1
    return (np.array(a) @ np.array(b)).tolist()