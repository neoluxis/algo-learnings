import numpy as np
import numpy.random as random
import matplotlib

matplotlib.use("Agg")  # 无GUI后端，只生成图片文件，不弹窗
import matplotlib.pyplot as plt

size = 19
p_mut = 0.001
population = 20
select_k = 5
max_iter = 100
form = (
   lambda x: 2.260259 - 7.894738*x - 15.597642*x**2 + 93.885903*x**3 + 90.418917*x**4 - 333.138575*x**5 - 221.021778*x**6 + 550.535635*x**7 + 272.772461*x**8 - 500.892144*x**9 - 184.004071*x**10 + 264.579614*x**11 + 65.611795*x**12 - 78.660915*x**13 - 7.698198*x**14 + 10.489501*x**15 - 2.734180*x**16 + 0.317939*x**17 + 1.097048*x**18 - 0.139027*x**19 - 0.077158*x**20 - 0.047361*x**21 - 0.027821*x**22 + 0.013585*x**23 + 0.005528*x**24 - 0.000338*x**25 + 0.000042*x**26 - 0.000217*x**27 - 0.000100*x**28 + 0.000025*x**29 + 0.000010*x**30 - 0.000001*x**31 - 0.000000*x**32
)


class Insect:
    def __init__(self, chrosomes=None):
        self.chrosomes = (
            np.array(chrosomes)
            if chrosomes is not None
            else np.where(np.random.rand(size) > 0.5, 1, 0)
        )

    def pair(self, ptnr):
        pos = random.randint(0, size)
        # print(f'{pos = }')
        p1 = self.chrosomes.copy()
        p2 = ptnr.chrosomes.copy()
        c1 = np.concatenate([p1[:pos], p2[pos:]])
        c2 = np.concatenate([p2[:pos], p1[pos:]])
        return Insect(c1), Insect(c2)

    def mutate(self):
        if random.rand() > p_mut:
            return
        pos = random.randint(0, size)
        self.chrosomes[pos] = 1 - self.chrosomes[pos]

    def to_value(self):
        bits = "".join(map(str, self.chrosomes))
        i = int(bits, base=2)
        return -2.5 + i * 1e-5

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Insect(chros={''.join(map(str, self.chrosomes))}, val={self.to_value():.05f})"

    def __gt__(self, other):
        return self.to_value() > other.to_value()

    def __lt__(self, other):
        return self.to_value() < other.to_value()

    def __ge__(self, other):
        return self.to_value() >= other.to_value()

    def __le__(self, other):
        return self.to_value() <= other.to_value()


def score(p):
    if p.to_value() > 2.5 or p.to_value() < -2.5:
        return np.inf
    x = p.to_value()
    return form(x)


def gen1(popul):
    p_pair = np.array([score(p) for p in popul])
    child = []
    norm = np.linalg.norm(p_pair)
    if norm > 1e-12:  # 防止除0（零向量）
        p_pair = p_pair / norm
    for idx, m in enumerate(popul):
        for i, f in enumerate(popul):
            if idx == i:
                continue
            if random.rand() < p_pair[idx]:
                continue
            f.mutate(), m.mutate()
            child.extend(m.pair(f))
    return child


def select(popul, k=select_k):
    # Step1：过滤掉score为inf的个体
    candidates = [s for s in popul if score(s) != np.inf]
    if len(candidates) == 0:
        raise ValueError("All inf, no choice")

    sel = []
    # Step2：锦标赛循环选出population个
    for _ in range(population):
        # 随机无放回抽取k个候选
        group = np.random.choice(candidates, size=k)
        # 选出组内score最小的个体
        winner = min(group, key=lambda x: score(x))
        sel.append(winner)
    return sel


def stat(popul):
    """统计种群状态：有效个体、最优score、对应x，分数均值、标准差"""
    data = []
    for p in popul:
        s = score(p)
        if np.isinf(s):
            continue
        data.append({"x": p.to_value(), "score": s})
    if len(data) == 0:
        print("==== Population Stat ====")
        print("All individuals are invalid (score=inf)")
        return {
            "count": 0,
            "best_score": np.inf,
            "best_x": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    scores = np.array([d["score"] for d in data])
    xs = np.array([d["x"] for d in data])
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_x = xs[best_idx]

    stat_info = {
        "count": len(data),
        "best_score": best_score,
        "best_x": best_x,
        "mean": np.mean(scores),
        "std": np.std(scores),
    }
    # 打印
    print("==== Population Stat ====")
    print(f"Valid individuals: {stat_info['count']}")
    print(f"Best score: {best_score:.6f},  x = {best_x:.6f}")
    print(f"Score mean: {stat_info['mean']:.6f}, std: {stat_info['std']:.6f}")
    return stat_info


def plot_stats(history):
    gens = np.arange(len(history))
    best_scores = np.array([h["best_score"] for h in history])
    mean_scores = np.array([h["mean"] for h in history])
    std_scores = np.array([h["std"] for h in history])
    best_xs = np.array([h["best_x"] for h in history])

    # 清洗nan
    valid_mask = ~np.isnan(best_scores)
    gens = gens[valid_mask]
    best_scores = best_scores[valid_mask]
    mean_scores = mean_scores[valid_mask]
    std_scores = std_scores[valid_mask]
    best_xs = best_xs[valid_mask]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(gens, best_scores, color="#d62728", lw=2, label="Best Score")
    axes[0].set_ylabel("Best Score")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(gens, mean_scores, color="#1f77b4", lw=2, label="Mean Score")
    axes[1].fill_between(
        gens,
        mean_scores - std_scores,
        mean_scores + std_scores,
        alpha=0.2,
        color="#1f77b4",
        label="±Std",
    )
    axes[1].set_ylabel("Population Score")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(gens, best_xs, color="#2ca02c", lw=2, label="Best x")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Best x")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.suptitle("GA Population Statistics", fontsize=14)
    plt.tight_layout()
    # 保存图片，不要plt.show()
    plt.savefig("ga_convergence.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Plot saved to ga_convergence.png")


if __name__ == "__main__":
    pop = [Insect([0] * size) for _ in range(population)]
    history = []  # 用来存每一代stat

    for g in range(max_iter):
        print(f"\n======== Generation {g} ========")
        st = stat(pop)
        history.append(st)

        pop_parent = select(pop)
        child = gen1(pop_parent)
        pop = select(pop_parent + child)

    # 全部迭代结束后绘图
    plot_stats(history)
