"""Zenn: 関係が直線でないとき、回帰線を外側に伸ばした予測に何が起きるか（広告効果の逓減の仮想データ）
図1: 回帰線と本当の関係を、データがある範囲の外側（200万円）まで伸ばす（サイト第4回の図1と同じデータ）
図2: 同じデータに当てはめた3つの形（直線・2次式・対数）を外側まで伸ばす
検算: 直線の極限（最良の直線近似）、1つの世界、1,000個の世界、予測区間の的中率、曲がりの確認、3つの形
依存: numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "regression-extrapolation-diminishing")
os.makedirs(OUT, exist_ok=True)

S, T = 20, 104                              # 店舗数・週数
ALPHA, BETA, SIG = 300.0, 1.5, 60.0         # 切片・1本目の世界の傾き・誤差の標準偏差（万円）
X_LO, X_HI, X_WEEK_SD = 10.0, 100.0, 8.0    # 予算枠の範囲・週のゆらぎ
CAP, TAU = 225.0, 60.0                      # 逓減の形：売上の上乗せの上限と、効きが落ちる速さ

_files = sorted(fm.findSystemFonts())
for keys in (["NotoSansCJK", "NotoSansJP"], ["YuGoth"], ["meiryo", "Meiryo"], ["Hiragino"]):
    hit = [f for f in _files if any(k in f for k in keys)]
    if hit:
        reg = [f for f in hit if "Regular" in f]
        plt.rcParams["font.family"] = fm.FontProperties(fname=(reg or hit)[0]).get_name()
        break
else:
    raise RuntimeError("日本語フォントが見つかりません")
plt.rcParams["axes.unicode_minus"] = False


def m(x, diminishing=True):
    """広告費 x のときの売上の平均（本当の関係）。"""
    x = np.asarray(x, dtype=float)
    return ALPHA + CAP * (1 - np.exp(-x / TAU)) if diminishing else ALPHA + BETA * x


def generate(rng, diminishing=False):
    """乱数を引く順番は 枠→週のゆらぎ→誤差。広告費と誤差は1本目の世界と同じ値になる。"""
    n = S * T
    store = np.repeat(np.arange(S), T)
    c = rng.uniform(X_LO, X_HI, S)
    x = np.clip(c[store] + rng.normal(0, X_WEEK_SD, n), 5, None)
    e = rng.normal(0, SIG, n)
    return x, m(x, diminishing) + e


def fit(X, y):
    """最小二乗。係数、係数の分散共分散行列、残差、残差の分散を返す。"""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    return b, s2 * np.linalg.inv(X.T @ X), r, s2


def LIN(x):
    x = np.atleast_1d(np.asarray(x, dtype=float)); return np.c_[np.ones_like(x), x]


def QUAD(x):
    x = np.atleast_1d(np.asarray(x, dtype=float)); return np.c_[np.ones_like(x), x, x ** 2]


def LOG(x):
    x = np.atleast_1d(np.asarray(x, dtype=float)); return np.c_[np.ones_like(x), np.log(x)]


BLUE, RED, ORANGE, GREEN, BAND = "#4a6fa5", "#c0392b", "#e67e22", "#2e8b57", "#dfe6ee"
XMAX, X0 = 215.0, 200.0

# ---- 1つの世界（seed 2026） ----
worlds = {}
for key, dim in (("OFF", False), ("ON", True)):
    rng = np.random.default_rng(2026)
    x, y = generate(rng, dim)
    worlds[key] = (x, y, fit(LIN(x), y))
x, y, (b, V, r, s2) = worlds["ON"]
bq, Vq, rq, s2q = fit(QUAD(x), y)
bl, Vl, rl, s2l = fit(LOG(x), y)

# ---- 図1 ----
fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100, sharey=True)
for ax, key, dim, title in zip(axes, ("OFF", "ON"), (False, True),
                               ("前提がすべて成り立つ世界", "売上の増え方が小さくなっていく世界")):
    xx, yy, (bb, _, _, _) = worlds[key]
    ax.axvspan(xx.min(), xx.max(), color=BAND, alpha=0.6, lw=0, label="データがある範囲")
    ax.scatter(xx, yy, s=7, alpha=0.35, color=BLUE, linewidths=0, label="店舗×週（2,080件）")
    xs = np.linspace(0, XMAX, 300)
    ax.plot(xs, m(xs, dim), color="#333333", lw=2.2, ls="--", label="本当の関係")
    ax.plot(xs, bb[0] + bb[1] * xs, color=RED, lw=2.2, label=f"回帰線（傾き {bb[1]:.2f}）")
    p, t = bb[0] + bb[1] * X0, float(m(X0, dim))
    ax.plot([X0, X0], [t, p], color=RED, lw=1.2, ls=":")
    ax.plot(X0, p, "o", color=RED, ms=7); ax.plot(X0, t, "o", color="#333333", ms=7)
    ax.annotate(f"予測 {p:.0f}", (X0, p), textcoords="offset points", xytext=(-8, 9), ha="right", fontsize=10.5, color=RED)
    ax.annotate(f"本当 {t:.0f}", (X0, t), textcoords="offset points", xytext=(10, -20), ha="right", fontsize=10.5, color="#333333")
    ax.set_xlim(0, XMAX); ax.set_ylim(130, 850)
    ax.set_xlabel("広告費（万円/週）"); ax.set_title(title, fontsize=12); ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
axes[0].set_ylabel("売上（万円/週）")
fig.suptitle("図1：回帰線を、データがある範囲の外側まで伸ばす")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png")); plt.close(fig)

# ---- 図2 ----
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.axvspan(x.min(), x.max(), color=BAND, alpha=0.6, lw=0, label="データがある範囲")
ax.scatter(x, y, s=7, alpha=0.25, color=BLUE, linewidths=0)
xs = np.linspace(5, XMAX, 400)
ax.plot(xs, m(xs), color="#333333", lw=2.4, ls="--", label="本当の関係")
ax.plot(xs, LIN(xs) @ b, color=RED, lw=2.2, label="直線")
ax.plot(xs, QUAD(xs) @ bq, color=ORANGE, lw=2.2, label="2次式")
ax.plot(xs, LOG(xs) @ bl, color=GREEN, lw=2.2, label="対数")
ax.set_xlim(0, XMAX); ax.set_ylim(130, 850)
ax.set_xlabel("広告費（万円/週）"); ax.set_ylabel("売上（万円/週）")
ax.set_title("図2：同じデータに当てはめた3つの形を、外側まで伸ばす")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2.png")); plt.close(fig)

# ---- 直線の極限（広告費の分布を大きな標本で近似） ----
rngL = np.random.default_rng(1)
cL = rngL.uniform(X_LO, X_HI, 2_000_000)
xL = np.clip(cL + rngL.normal(0, X_WEEK_SD, cL.size), 5, None)
beta_star = np.cov(xL, m(xL))[0, 1] / xL.var()
alpha_star = m(xL).mean() - beta_star * xL.mean()

# ---- 1,000個の世界 ----
R = 1000
rng = np.random.default_rng(2026)
rng_new = np.random.default_rng(2027)       # 新しい1週の売上を引くための別の乱数
XS = (55.0, 100.0, 200.0)
slope = np.empty(R); se = np.empty(R)
pred = {k: np.empty(R) for k in XS}; hit_new = {k: np.empty(R, bool) for k in XS}; half = {k: np.empty(R) for k in XS}
pq = np.empty(R); pl = np.empty(R)
for i in range(R):
    x_, y_ = generate(rng, True)
    b_, V_, _, s2_ = fit(LIN(x_), y_)
    slope[i], se[i] = b_[1], np.sqrt(V_[1, 1])
    for k in XS:
        v = np.array([1.0, k])
        pred[k][i] = v @ b_
        half[k][i] = 1.96 * np.sqrt(v @ V_ @ v + s2_)
        y_new = float(m(k)) + rng_new.normal(0, SIG)
        hit_new[k][i] = abs(y_new - pred[k][i]) < half[k][i]
    pq[i] = (QUAD(X0) @ fit(QUAD(x_), y_)[0])[0]
    pl[i] = (LOG(X0) @ fit(LOG(x_), y_)[0])[0]

# ---- 検算 ----
print("== 検算1：本当の関係 ==")
print("1万円あたりの増え方 3.75·exp(−x/60)：" + "、".join(f"{k}万円で {CAP / TAU * np.exp(-k / TAU):.2f}" for k in (10, 55, 100, 200)))
print(f"直線の極限（最良の直線近似）：切片 {alpha_star:.1f}、傾き {beta_star:.3f}")
print("極限の直線と本当の関係の差：" + "、".join(f"{k}万円で {alpha_star + beta_star * k - float(m(k)):+.0f}" for k in (20, 55, 100, 150, 200)))
print("\n== 検算2：1つの世界 ==")
xo, yo, (bo, Vo, ro, s2o) = worlds["OFF"]
print(f"前提がすべて成り立つ世界：切片 {bo[0]:.1f}、傾き {bo[1]:.4f}（標準誤差 {np.sqrt(Vo[1, 1]):.4f}）、200万円の予測 {bo[0] + bo[1] * 200:.0f}（本当 {float(m(200, False)):.0f}）")
print(f"増え方が小さくなる世界：切片 {b[0]:.1f}、傾き {b[1]:.4f}（標準誤差 {np.sqrt(V[1, 1]):.4f}）、広告費の範囲 {x.min():.1f}〜{x.max():.1f}")
for k in (20, 55, 100, 120, 150, 200):
    v = np.array([1.0, k])
    print(f"  {k}万円：直線 {v @ b:.0f}、本当 {float(m(k)):.0f}、差 {v @ b - float(m(k)):+.0f}、95%予測区間の半幅 ±{1.96 * np.sqrt(v @ V @ v + s2):.0f}")
print(f"データ上の増え方の平均 {np.mean(CAP / TAU * np.exp(-x / TAU)):.3f}")
print("\n== 検算3：1,000個の世界 ==")
print(f"傾き：平均 {slope.mean():.3f}、標準偏差 {slope.std(ddof=1):.3f}、標準誤差の平均 {se.mean():.3f}")
for k in XS:
    print(f"{k:.0f}万円の予測：平均 {pred[k].mean():.1f}、最小 {pred[k].min():.0f}、最大 {pred[k].max():.0f}、本当 {float(m(k)):.1f}、"
          f"新しい1週の売上が95%予測区間に入った割合 {hit_new[k].mean():.3f}（半幅の平均 ±{half[k].mean():.0f}）")
print("\n== 検算4：曲がりはデータから分かるか ==")
for lo, hi in ((0, 25), (25, 40), (40, 70), (70, 100), (100, 1000)):
    k = (x >= lo) & (x < hi)
    lab = f"{lo}〜{hi}万円" if hi < 1000 else f"{lo}万円以上"
    print(f"残差の平均 {lab}：{r[k].mean():+.1f}（{k.sum()}件）")
print(f"2次の項の t 値 {bq[2] / np.sqrt(Vq[2, 2]):.1f}")
print("\n== 検算5：3つの形 ==")
print(f"残差の標準偏差：直線 {np.sqrt(s2):.2f}、2次式 {np.sqrt(s2q):.2f}、対数 {np.sqrt(s2l):.2f}（本当の関係からのずれ {np.std(y - m(x)):.2f}）")
for k in (100, 150, 200):
    print(f"{k}万円の予測：直線 {(LIN(k) @ b)[0]:.0f}、2次式 {(QUAD(k) @ bq)[0]:.0f}、対数 {(LOG(k) @ bl)[0]:.0f}、本当 {float(m(k)):.0f}")
print(f"1,000個の世界の200万円の予測：2次式 平均 {pq.mean():.0f}（{pq.min():.0f}〜{pq.max():.0f}）、対数 平均 {pl.mean():.0f}（{pl.min():.0f}〜{pl.max():.0f}）")
