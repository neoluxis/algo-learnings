import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== 参数 ====================
size = 19
p_mut = 0.001  # 每个位点的变异概率
population = 20
select_k = 5
max_iter = 100
X_MIN, X_MAX = -2.5, 2.5
ELITE_COUNT = 1
RANDOM_SEED = None  # 需要复现实验时设为整数，例如 42

if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)


# 原始目标函数
def form(x):
    return (
        4.242780
        - 1.041148 * x
        - 3.212730 * x**2
        + 12.937858 * x**3
        + 5.588452 * x**4
        - 25.930953 * x**5
        - 8.359865 * x**6
        + 19.722596 * x**7
        + 6.453332 * x**8
        - 6.821465 * x**9
        - 2.413146 * x**10
        + 1.028840 * x**11
        + 0.398273 * x**12
        - 0.047035 * x**13
        - 0.020206 * x**14
    )


class Insect:
    """使用 size 位二进制染色体表示一个候选解。"""

    def __init__(self, chrosomes=None):
        if chrosomes is None:
            self.chrosomes = np.random.randint(0, 2, size=size, dtype=np.int8)
        else:
            chromosome = np.asarray(chrosomes, dtype=np.int8)
            if chromosome.shape != (size,):
                raise ValueError(f"染色体长度必须为 {size}")
            if not np.all((chromosome == 0) | (chromosome == 1)):
                raise ValueError("染色体只能包含 0 和 1")
            self.chrosomes = chromosome.copy()

    def pair(self, ptnr):
        """单点交叉，返回两个新个体，不修改父代。"""
        pos = np.random.randint(1, size)
        p1, p2 = self.chrosomes, ptnr.chrosomes
        c1 = np.concatenate((p1[:pos], p2[pos:]))
        c2 = np.concatenate((p2[:pos], p1[pos:]))
        return Insect(c1), Insect(c2)

    def mutate(self):
        """每个位点独立地以 p_mut 概率翻转。"""
        mask = np.random.random(size=size) < p_mut
        self.chrosomes[mask] = 1 - self.chrosomes[mask]

    def to_value(self):
        """把整数编码线性映射到 [X_MIN, X_MAX]，包含两个端点。"""
        i = 0
        for bit in self.chrosomes:
            i = (i << 1) | int(bit)
        return X_MIN + i * (X_MAX - X_MIN) / (2**size - 1)

    def __repr__(self):
        return (
            f"Insect(chros={''.join(map(str, self.chrosomes))}, "
            f"x={self.to_value():.8f}, score={score(self):.8f})"
        )


def score(p):
    """目标函数值越小越好。"""
    x = p.to_value()
    if not X_MIN <= x <= X_MAX:
        return np.inf
    value = float(form(x))
    return value if np.isfinite(value) else np.inf


def select(popul, k=select_k, n=population):
    """锦标赛选择：每次从候选个体中无放回抽取 k 个。"""
    if not popul:
        raise ValueError("种群为空")

    scores = np.array([score(p) for p in popul], dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(scores))
    if len(valid_indices) == 0:
        raise ValueError("种群中没有有效个体")

    k = min(max(1, k), len(valid_indices))
    selected = []
    for _ in range(n):
        group = np.random.choice(valid_indices, size=k, replace=False)
        winner_idx = group[np.argmin(scores[group])]
        selected.append(popul[winner_idx])
    return selected


def gen1(parents, n_children):
    """随机选父代交叉；仅对子代变异，不修改父代。"""
    if n_children <= 0:
        return []
    if len(parents) < 2:
        raise ValueError("交叉至少需要两个父代")

    children = []
    while len(children) < n_children:
        idx1, idx2 = np.random.choice(len(parents), size=2, replace=False)
        c1, c2 = parents[idx1].pair(parents[idx2])
        c1.mutate()
        c2.mutate()
        children.append(c1)
        if len(children) < n_children:
            children.append(c2)
    return children


def stat(popul, generation=None):
    """统计当前种群的有效个体数、最优值、均值和标准差。"""
    scores = np.array([score(p) for p in popul], dtype=float)
    valid = np.isfinite(scores)
    valid_scores = scores[valid]
    valid_individuals = [p for p, ok in zip(popul, valid) if ok]

    title = f"Generation {generation}" if generation is not None else "Population"
    print(f"==== {title} Stat ====")

    if len(valid_scores) == 0:
        print("No valid individuals")
        return {
            "count": 0,
            "best_score": np.inf,
            "best_x": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    idx = int(np.argmin(valid_scores))
    best_score = float(valid_scores[idx])
    best_x = float(valid_individuals[idx].to_value())
    mean = float(np.mean(valid_scores))
    std = float(np.std(valid_scores))

    print(f"Valid individuals: {len(valid_scores)}/{len(popul)}")
    print(f"Best score: {best_score:.10f}, x = {best_x:.10f}")
    print(f"Score mean: {mean:.10f}, std: {std:.10f}")

    return {
        "count": int(len(valid_scores)),
        "best_score": best_score,
        "best_x": best_x,
        "mean": mean,
        "std": std,
    }


def plot_stats(history, filename="ga_convergence.png"):
    """绘制逐代最优目标值、均值/标准差和最优 x。"""
    generations = np.arange(len(history))
    best_scores = np.array([h["best_score"] for h in history], dtype=float)
    means = np.array([h["mean"] for h in history], dtype=float)
    stds = np.array([h["std"] for h in history], dtype=float)
    best_xs = np.array([h["best_x"] for h in history], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(generations, best_scores, lw=2, label="Generation Best")
    axes[0].set_ylabel("Best Score")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(generations, means, lw=2, label="Population Mean")
    axes[1].fill_between(
        generations, means - stds, means + stds, alpha=0.2, label="Mean +/- Std"
    )
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(generations, best_xs, lw=2, label="Generation Best x")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Best x")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle("GA Population Statistics")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {filename}")


def run_ga():
    """运行遗传算法，返回最终种群、统计历史和历史最优个体。"""
    pop = [Insect() for _ in range(population)]
    history = []

    best_ever = Insect(min(pop, key=score).chrosomes)
    best_ever_score = score(best_ever)

    for g in range(max_iter):
        print(f"\\n======== Generation {g} ========")
        history.append(stat(pop, generation=g))

        current_best = min(pop, key=score)
        if score(current_best) < best_ever_score:
            best_ever = Insect(current_best.chrosomes)
            best_ever_score = score(best_ever)

        # 精英个体复制到下一代，避免最优解丢失
        elite_count = min(max(0, ELITE_COUNT), population)
        ranked = sorted(pop, key=score)
        elites = [Insect(p.chrosomes) for p in ranked[:elite_count]]

        parents = select(pop, k=select_k, n=population)
        children = gen1(parents, n_children=population - elite_count)
        pop = elites + children

    # 记录最后一次繁殖后的种群
    print(f"\\n======== Final Population ========")
    history.append(stat(pop, generation=max_iter))

    final_best = min(pop, key=score)
    if score(final_best) < best_ever_score:
        best_ever = Insect(final_best.chrosomes)
        best_ever_score = score(best_ever)

    print("\\n======== Best Found ========")
    print(f"x = {best_ever.to_value():.12f}")
    print(f"score = {score(best_ever):.12f}")
    print(f"chromosome = {''.join(map(str, best_ever.chrosomes))}")

    plot_stats(history)
    return pop, history, best_ever


if __name__ == "__main__":
    final_population, history, best = run_ga()
