
import numpy as np

n = 10.0

T0 = 5.0
dT = 0.9999
MAX_ITER = 100_000


def func(x):
    return 0.5 * x**4 + x**3 - 2 * x**2 - 3 * x


def SA(x_0=-3.0):
    rng = np.random.default_rng()

    x = float(x_0)
    f = func(x)

    best_x = x
    best_f = f

    T = T0
    step0 = 1.0

    for _ in range(MAX_ITER):
        # 温度只控制接受概率
        # 步长独立设置，并随温度逐渐缩小
        step = step0 * np.sqrt(T / T0)

        dx = x + rng.uniform(-step, step)
        df = func(dx)

        # 接受更优解
        if df < f:
            x, f = dx, df

        # 以概率 exp(-delta_f / T) 接受较差解
        elif rng.random() < np.exp((f - df) / T):
            x, f = dx, df

        # 独立记录历史最优解
        if f < best_f:
            best_x, best_f = x, f

        T *= dT

    return best_x, best_f


def main():
    x, fx = SA(-3.0)

    print(f"x = {x:.12f}")
    print(f"f(x) = {fx:.12f}")
    print(f"参考解: x ≈ 1.2020235")
    print(f"参考函数值: f(x) ≈ -3.7152243")


if __name__ == "__main__":
    main()
