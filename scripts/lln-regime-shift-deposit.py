"""Zenn「平均が途中で変わるとき、大数の法則に何が起きるか」の再現スクリプト。
図1：同じ跳ねで始まる2つの世界（サイト版コラム第4回の図1と同じ乱数・同じ数値）
図2：窓の長さの引き換え（平時のぶれ vs 切り替え後に気づくまでの日数）
図3：局面が繰り返す場合（マルコフ切替）の累積平均
各実験は default_rng(2026) をそれぞれ初期化する（図1をサイト版と一致させるため）。
出力はリポジトリの images/lln-regime-shift-deposit/ に fig1.png〜fig3.png、標準出力に検算の数値。
"""
import copy
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "lln-regime-shift-deposit")
os.makedirs(OUT, exist_ok=True)

CANDS = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Hiragino Sans"]
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f:
        font_manager.fontManager.addfont(f)
avail = {f.name for f in font_manager.fontManager.ttflist}
font = next((c for c in CANDS if c in avail), None)
if font is None:
    raise RuntimeError(f"日本語フォントが見つからない: {CANDS}")
plt.rcParams["font.family"] = font

SEED = 2026
N = 2_000
Q0, Q1 = 0.010, 0.030


def ma(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ---------------------------------------------------------------- 図1
PRE, SAME, AFTER = 250, 30, 180
LONG, SHORT = 250, 30
rng = np.random.default_rng(SEED)
pre = rng.binomial(N, Q0, PRE) / N
same = rng.binomial(N, Q1, SAME) / N
state = rng.bit_generator.state
rng_a = np.random.default_rng(); rng_a.bit_generator.state = copy.deepcopy(state)
rng_b = np.random.default_rng(); rng_b.bit_generator.state = copy.deepcopy(state)
x_a = np.r_[pre, same, rng_a.binomial(N, Q0, AFTER - SAME) / N]
x_b = np.r_[pre, same, rng_b.binomial(N, Q1, AFTER - SAME) / N]
t = np.arange(-PRE, AFTER)


def at(m, w, d):
    return 100 * m[d + PRE - (w - 1)]


fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100, sharey=True)
print("== 図1：同じ跳ねで始まる2つの世界 ==")
for ax, x, title in [(axes[0], x_a, "戻る世界、30日で1%に戻る"), (axes[1], x_b, "戻らない世界、3%のまま")]:
    mL, mS = ma(x, LONG), ma(x, SHORT)
    ax.scatter(t, 100 * x, s=6, color="gray", alpha=0.35, zorder=0, label="毎日の引出率")
    ax.plot(t[LONG - 1:], 100 * mL, color="#4C72B0", lw=2.4, label="直近1年の平均")
    ax.plot(t[SHORT - 1:], 100 * mS, color="#C44E52", lw=1.8, label="直近30日の平均")
    ax.axvspan(0, SAME, color="#f0e6a0", alpha=0.5, lw=0)
    ax.axvline(0, color="black", lw=0.9); ax.axvline(SAME, color="black", lw=0.9, ls="--")
    ax.axhline(100 * Q0, ls=":", color="gray", lw=0.8); ax.axhline(100 * Q1, ls=":", color="gray", lw=0.8)
    ax.text(SAME / 2, 4.25, "ここまで同じ", ha="center", fontsize=10)
    ax.set_xlim(-60, AFTER); ax.set_ylim(0, 4.5)
    ax.set_title(title, fontsize=12); ax.set_xlabel("跳ねた日を0とした日数")
    print(f"  {title}: " + "  ".join(f"{d}日目 長期 {at(mL, LONG, d-1):.2f}% 短期 {at(mS, SHORT, d-1):.2f}%" for d in (30, 90, 180)))
    if "戻らない" in title:
        # 理論：窓が跳ねる前を含むあいだ、長期の平均の期待値 = q1 + (PRE_in_window/LONG)(q0 - q1)
        for d in (90, 180):
            theory = Q1 + ((LONG - d) / LONG) * (Q0 - Q1)
            print(f"    {d}日目の長期の平均の期待値（理論） {100*theory:.2f}%")
axes[0].set_ylabel("1日あたりの引出率（%）")
axes[0].legend(loc="lower right", fontsize=10)
fig.suptitle("引出率が1%から3%へ跳ねてから30日、二つの世界は同じに見える", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png")); plt.close(fig)

# ---------------------------------------------------------------- 図2
Q1b = 0.015                                  # 小さめの切り替え（1%→1.5%）
WS = np.array([5, 10, 20, 30, 60, 90, 120, 180, 250])
REPS = 2_000
rng = np.random.default_rng(SEED)
sd_sim, delay_sim = [], []
for W in WS:
    calm = rng.binomial(N, Q0, (REPS, W)) / N
    sd_sim.append(calm.mean(axis=1).std())
    pre = rng.binomial(N, Q0, (REPS, W)) / N
    post = rng.binomial(N, Q1b, (REPS, 2 * W)) / N
    xs = np.c_[pre, post]
    cs = np.cumsum(np.c_[np.zeros(REPS), xs], axis=1)
    m = (cs[:, W:] - cs[:, :-W]) / W              # 窓の平均。m[:, k] は切り替え後 k 日目まで含む
    mid = (Q0 + Q1b) / 2
    hit = (m[:, 1:] > mid).argmax(axis=1) + 1
    delay_sim.append(np.median(hit))
sd_sim, delay_sim = np.array(sd_sim), np.array(delay_sim)
sd_th = np.sqrt(Q0 * (1 - Q0) / N / WS)
delay_th = WS / 2

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
ax2 = ax1.twinx()
ax1.plot(WS, 100 * sd_th, color="#4C72B0", lw=2, label="平時のぶれ（標準偏差・理論）")
ax1.plot(WS, 100 * sd_sim, "o", color="#4C72B0", ms=6, label="平時のぶれ（シミュレーション）")
ax2.plot(WS, delay_th, color="#C44E52", lw=2, ls="--", label="気づくまでの日数（理論 W/2）")
ax2.plot(WS, delay_sim, "s", color="#C44E52", ms=6, label="気づくまでの日数（シミュレーション中央値）")
ax1.set_xlabel("平均をとる窓の長さ W（日）")
ax1.set_ylabel("平時の窓平均の標準偏差（%ポイント）", color="#4C72B0")
ax2.set_ylabel("1%→1.5%の切り替え後、窓平均が1.25%を超えるまでの日数", color="#C44E52")
ax1.set_ylim(0, 0.11); ax2.set_ylim(0, 135)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=10)
ax1.set_title("窓を長くすると、ぶれは 1/√W で減り、気づくまでの日数は W に比例して増える", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2.png")); plt.close(fig)
print("== 図2：窓の長さの引き換え（1%→1.5%、反復2,000） ==")
for W, a, b, c, d in zip(WS, sd_sim, sd_th, delay_sim, delay_th):
    print(f"  W={W:4d}  ぶれ 実測 {100*a:.3f}  理論 {100*b:.3f}  |  気づくまで 実測 {c:5.1f}  理論 {d:5.1f}")

# ---------------------------------------------------------------- 図3
L0, L1 = 500, 50                             # 平均の滞在日数（平時・危機）
YEARS = 80; D = 250 * YEARS; TRIALS = 8
pi1 = L1 / (L0 + L1)
mu = (1 - pi1) * Q0 + pi1 * Q1
lam = 1 - 1 / L0 - 1 / L1
var_state = pi1 * (1 - pi1) * (Q1 - Q0) ** 2
var_binom = ((1 - pi1) * Q0 * (1 - Q0) + pi1 * Q1 * (1 - Q1)) / N
tau_int = (1 + lam) / (1 - lam)
rng = np.random.default_rng(SEED)
days = np.arange(1, D + 1)
CHECK = 2_000                                # 検算用の試行数（図に描くのは最初の8本）
s = (rng.random(CHECK) < pi1).astype(np.int8)    # 定常分布から始める
csum = np.zeros(CHECK)
snap = {}
paths = np.empty((TRIALS, D))
for i in range(D):
    if i > 0:
        u = rng.random(CHECK)
        s = np.where(s == 0, (u < 1 / L0), (u >= 1 / L1)).astype(np.int8)
    csum += rng.binomial(N, np.where(s == 1, Q1, Q0)) / N
    paths[:, i] = csum[:TRIALS] / (i + 1)
    if (i + 1) in (250, 2_500, 20_000):
        snap[i + 1] = csum / (i + 1)
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
for k in range(TRIALS):
    ax.plot(days / 250, 100 * paths[k], lw=1.2, alpha=0.85)
sd_n = np.sqrt((var_state * tau_int + var_binom) / days)
ax.fill_between(days / 250, 100 * (mu - 1.96 * sd_n), 100 * (mu + 1.96 * sd_n), color="#C44E52", alpha=0.12, lw=0,
                label="定常の混合平均 ±1.96×理論SD")
ax.axhline(100 * mu, color="#C44E52", lw=1.2, ls="--", label=f"定常の混合平均 {100*mu:.2f}%")
ax.axhline(100 * Q0, color="gray", lw=0.8, ls=":"); ax.axhline(100 * Q1, color="gray", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xlim(0.2, YEARS); ax.set_ylim(0.6, 2.4)
ax.set_xlabel("年数（対数目盛、1年 = 250日）"); ax.set_ylabel("累積の平均引出率（%）")
ax.set_title("平時（平均500日）と危機（平均50日）が繰り返す世界の累積平均、8試行", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3.png")); plt.close(fig)
print("== 図3：局面が繰り返す場合 ==")
print(f"  危機の日の割合 {pi1:.4f}  定常の混合平均 {100*mu:.3f}%  状態の自己相関 λ={lam:.3f}  積分自己相関時間 {tau_int:.1f}")
for y in (1, 10, 80):
    n = 250 * y
    v = snap[n]
    print(f"  {y:2d}年: 図の8本 {100*paths[:, n-1].min():.2f}〜{100*paths[:, n-1].max():.2f}%  "
          f"{CHECK}試行の平均 {100*v.mean():.3f}%  実測SD {100*v.std():.3f}  理論SD {100*sd_n[n-1]:.3f}  "
          f"（独立なら {100*np.sqrt(var_state + var_binom)/np.sqrt(n):.3f}）")
