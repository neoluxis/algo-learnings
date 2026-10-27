# 寻找一个 x_0 使得 x_0 ^ 2 \rightarrow \sqrt{n} ^ 2
import numpy as np
from typing import Literal

n = 10.0
T = 2e4  # 20000
dT = 0.999
eps = 1e-14


def func(x) -> np.float64:
    # return np.fabs(x**2 - n)
    return 0.5 * x ** 4 + x ** 3 - 2 * x **2 - 3 * x 


def SA(x_0=0):
    global T, dT, eps
    x = x_0
    f = func(x)
    while (T > eps):
        dx = x + (np.random.rand()*2 - 1) * T
        # while (dx < 0): 
        #     dx = x + (np.random.rand()*2 - 1) * T
        df = func(dx)
        if (df < f):
            x = dx
            f = df
        elif(np.exp((f - df)/T)> np.random.rand()):
            x = dx
            f = df
        T = T * dT
    return x

def main():
    res = SA(-3)
    print(f'{res = }, {func(res) = }')


if __name__ == "__main__":
    main()
