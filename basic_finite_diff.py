from typing import Callable

import functools
import numpy as np


def forward_diff(f: Callable[[float], float], x: float, h:float) -> float:
    return (f(x + h) - f(x)) / h

def backward_diff(f: Callable[[float], float], x: float, h:float) -> float:
    return (f(x) - f(x - h)) / h

def centered_diff(f: Callable[[float], float], x: float, h:float) -> float:
    return 0.5 * (forward_diff(f, x, h) + backward_diff(f, x, h))

# Trying this out for now, definitely very inefficient
# TODO: code up the general method by solving the system of equations
def create_diff_ordered(k):
    def D_k(f, x, h):
        # iteratively apply the forward difference method
        func = functools.partial(forward_diff, f, h=h)
        for _ in range(k-1):
            func = functools.partial(forward_diff, func, h=h)
        return func(x)
    return D_k

f = np.sin
x = 1
h = 0.1

D_2 = create_diff_ordered(2)

print(-np.sin(1) - D_2(f, x, h))

