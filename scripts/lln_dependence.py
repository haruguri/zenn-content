"""
独立が成り立たないとき、大数の法則に何が起きるか — 世論調査を例にシミュレーションで見る
再現スクリプト（図1〜図3と本文の表）。依存: numpy, matplotlib
乱数は default_rng(2026) を 1 回だけ初期化し、図1 → 図2 → 図3 → 表 の順に消費する。末尾の検算だけ別の rng。
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- 日本語フォント（候補順。無ければ例外）
for _f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "Yu Gothic", "Meiryo", "Hiragino Sans"]:
    if any(_f in x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
else:
    raise RuntimeError("日本語フォントが見つかりません")

SEED = 2026
P = 0.5          # 真の賛成率
N_MAX = 20_000   # 図1で聞く人数の上限
TRIALS = 8       # 図1の試行数
rng = np.random.default_rng(SEED)


# ============================================================
# 生成機構
# ============================================================
def sample_independent(n, p=P):
    """独立。各人が確率 p で賛成。"""
    return rng.random(n) < p


def sample_cluster(n, m, rho, p=P):
    """束抽出。1束 m 人。束ごとに賛成率 p_k を Beta(a, a) から引き、
    束の中の各人は p_k で賛成。束の中の相関（級内相関）は
    Var(p_k) / (p(1-p)) = rho になる（p = 0.5 のとき a = (1/rho - 1)/2）。"""
    if rho == 0:
        return sample_independent(n, p)
    k = -(-n // m)  # 束の数（切り上げ）
    a = (1 / rho - 1) / 2
    p_k = rng.beta(a, a, size=k)
    return (rng.random((k, m)) < p_k[:, None]).ravel()[:n]


# ============================================================
# 理論値: Var(標本平均) = σ²/n · (1 + (n-1)·平均相関)
# ============================================================
def deff_cluster(n, m, rho):
    """束抽出の (1 + (n-1)·平均相関)。同じ束の組の割合 (m-1)/(n-1) × rho。"""
    return 1 + (m - 1) * rho


def sd_pt(n, deff, p=P):
    """標本平均の標準偏差（ポイント）。"""
    return 100 * np.sqrt(p * (1 - p) / n * deff)


def running_mean(x):
    return np.cumsum(x) / np.arange(1, len(x) + 1)


# ============================================================
# 図1: 独立／束100人／束1,000人 で「ここまでの賛成率」を 8 本ずつ
# ============================================================
panels = [
    ("独立",                       lambda: sample_independent(N_MAX),        lambda n: deff_cluster(n, 1, 0.0)),
    ("束：1束100人・束の中の相関0.1",  lambda: sample_cluster(N_MAX, 100, 0.1),  lambda n: deff_cluster(n, 100, 0.1)),
    ("束：1束1,000人・束の中の相関0.1", lambda: sample_cluster(N_MAX, 1000, 0.1), lambda n: deff_cluster(n, 1000, 0.1)),
]
n_axis = np.arange(1, N_MAX + 1)
n_grid = np.unique(np.geomspace(10, N_MAX, 200).astype(int))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
for ax, (title, gen, deff) in zip(axes.ravel(), panels):
    for _ in range(TRIALS):
        ax.plot(n_axis, 100 * running_mean(gen()), lw=0.8, alpha=0.85)
    band = np.array([2 * sd_pt(n, deff(n)) for n in n_grid])
    ax.plot(n_grid, 50 + band, "r-", lw=1.2)
    ax.plot(n_grid, 50 - band, "r-", lw=1.2)
    ax.axhline(50, color="k", lw=0.6, ls=":")
    ax.axvline(1000, color="gray", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(15, 85)
    ax.set_title(title, fontsize=11)
    ax.text(1050, 83, f"1,000人の時点で ±{2*sd_pt(1000, deff(1000)):.1f}pt",
            fontsize=8.5, ha="left", va="top")
for ax in axes:
    ax.set_xlabel("聞いた人数 n（対数）")
axes[0].set_ylabel("ここまでの賛成率（%）")
fig.suptitle("束にして聞くと、賛成率の落ち着き方はどう変わるか（各8試行、赤は理論の±2σ）", fontsize=12)
fig.tight_layout()
fig.savefig("fig1_panels.png", dpi=130)
plt.close(fig)


# ============================================================
# 図2: 束の効き目は「1束の人数 × 束の中の相関」で決まる
# ============================================================
N0 = 1000
R2 = 400  # 各点の試行数
m_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
rho_list = [0.01, 0.03, 0.1, 0.3]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
for rho in rho_list:
    emp = []
    for m in m_list:
        means = np.array([sample_cluster(N0, m, rho).mean() for _ in range(R2)])
        emp.append(P * (1 - P) / means.var(ddof=1))   # 実測の「何人分か」
    th = [N0 / deff_cluster(N0, m, rho) for m in m_list]
    x_right = [(m - 1) * rho for m in m_list]
    (line,) = axL.plot(m_list, th, "-", lw=1.2, label=f"相関 {rho}")
    axL.plot(m_list, emp, "o", ms=4, color=line.get_color())
    axR.plot(x_right, th, "-", lw=1.2, color=line.get_color())
    axR.plot(x_right, emp, "o", ms=4, color=line.get_color())
for ax in (axL, axR):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("1,000人が何人分か")
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
axL.set_xlabel("1束の人数 m")
axL.set_title("相関ごとに別の線になる")
axL.legend(fontsize=9)
axR.set_xlabel("(m − 1) × 相関")
axR.set_title("横軸を (m−1)×相関 にすると1本に重なる")
xs = np.geomspace(0.01, 300, 100)
axR.plot(xs, N0 / (1 + xs), "k--", lw=0.8, label="1000 / (1 + x)")
axR.legend(fontsize=9)
fig.suptitle("束の効き目は、1束の人数と束の中の相関の積で決まる（線は理論、点は400試行の実測）", fontsize=11)
fig.tight_layout()
fig.savefig("fig2_m_rho.png", dpi=130)
plt.close(fig)


# ============================================================
# 図3 と表: 6 通りの束の作り方で、理論の「何人分か」と実測を突き合わせる
# ============================================================
R3 = 2000
setups = [
    ("独立",             lambda: sample_independent(N0),          deff_cluster(N0, 1, 0.0)),
    ("束 m=10, ρ=0.1",    lambda: sample_cluster(N0, 10, 0.1),     deff_cluster(N0, 10, 0.1)),
    ("束 m=100, ρ=0.01",  lambda: sample_cluster(N0, 100, 0.01),   deff_cluster(N0, 100, 0.01)),
    ("束 m=100, ρ=0.1",   lambda: sample_cluster(N0, 100, 0.1),    deff_cluster(N0, 100, 0.1)),
    ("束 m=100, ρ=0.5",   lambda: sample_cluster(N0, 100, 0.5),    deff_cluster(N0, 100, 0.5)),
    ("束 m=1000, ρ=0.1",  lambda: sample_cluster(N0, 1000, 0.1),   deff_cluster(N0, 1000, 0.1)),
]
print(f"{'束の作り方':16s} {'実測sd(pt)':>10s} {'理論sd(pt)':>10s} {'実測 何人分':>10s} {'理論 何人分':>10s}")
th_neff, emp_neff, labels = [], [], []
for name, gen, deff in setups:
    means = np.array([gen().mean() for _ in range(R3)])
    emp_sd = 100 * means.std(ddof=1)
    th_sd = sd_pt(N0, deff)
    e_neff = P * (1 - P) / means.var(ddof=1)
    t_neff = N0 / deff
    th_neff.append(t_neff); emp_neff.append(e_neff); labels.append(name)
    print(f"{name:16s} {emp_sd:10.2f} {th_sd:10.2f} {e_neff:10.1f} {t_neff:10.1f}")

fig, ax = plt.subplots(figsize=(6.2, 6))
ax.plot([5, 2000], [5, 2000], "k--", lw=0.8)
ax.plot(th_neff, emp_neff, "o", ms=6)
# 近い点同士はラベルを上下に振り分ける
offsets = [(7, -3), (7, 6), (7, -10), (7, -3), (7, -3), (7, -3)]
for x, y, lab, off in zip(th_neff, emp_neff, labels, offsets):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=8.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5, 2000); ax.set_ylim(5, 2000)
ax.set_xlabel("理論：1,000人が何人分か  n / (1 + (n−1)·平均相関)")
ax.set_ylabel("実測：1,000人が何人分か（2,000試行の分散から）")
ax.set_title("6通りの束の作り方が、1本の式に乗る", fontsize=11)
ax.grid(True, which="both", lw=0.3, alpha=0.5)
fig.tight_layout()
fig.savefig("fig3_neff.png", dpi=130)
plt.close(fig)


# ============================================================
# 検算: 束の生成機構が「各人の賛成確率 0.5・同じ束の2人の相関 rho」になっているか
# （図1〜3の乱数の消費順を変えないよう、ここだけ別の rng を使う）
# ============================================================
rng_check = np.random.default_rng(SEED + 1)
m_c, rho_c, k_c = 100, 0.1, 20_000
a_c = (1 / rho_c - 1) / 2
p_k = rng_check.beta(a_c, a_c, size=k_c)
x = (rng_check.random((k_c, m_c)) < p_k[:, None]).astype(float)   # k_c 束 × m_c 人
xc = x - P
same = ((xc.sum(1) ** 2 - (xc ** 2).sum(1)) / (m_c * (m_c - 1))).mean() / (P * (1 - P))
diff = np.corrcoef(x[:-1, 0], x[1:, 1])[0, 1]
print(f"\n検算（1束{m_c}人・相関{rho_c}・{k_c:,}束）")
print(f"  全体の賛成率        {x.mean():.4f}")
print(f"  同じ束の2人の相関   {same:.4f}")
print(f"  違う束の2人の相関   {diff:.4f}")
