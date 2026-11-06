"""Zenn: 回帰の前提がすべて成り立つ世界を作る（広告費と売上の仮想データ）
図1: 散布図＋本当の関係＋回帰線（サイト第1回の図1と同じ乱数の並び）
図2: 同じ世界を1,000回作り直したときの傾きの分布
検算: 傾きと標準誤差の式、理論値との突き合わせ、95%区間の的中率
依存: numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "regression-world-advertising")
os.makedirs(OUT, exist_ok=True)

S, T = 20, 104                 # 店舗数・週数
ALPHA, BETA, SIG = 300.0, 1.5, 60.0   # 切片・本当の傾き・誤差の標準偏差（万円）
X_LO, X_HI, X_WEEK_SD = 10.0, 100.0, 8.0   # 予算枠の範囲・週のゆらぎ

_files = fm.findSystemFonts()
for keys in (["NotoSansCJK", "NotoSansJP"], ["YuGoth"], ["meiryo", "Meiryo"], ["Hiragino"]):
    hit = [f for f in _files if any(k in f for k in keys)]
    if hit:
        plt.rcParams["font.family"] = fm.FontProperties(fname=hit[0]).get_name()
        break
else:
    raise RuntimeError("日本語フォントが見つかりません")
plt.rcParams["axes.unicode_minus"] = False


def generate(rng):
    """前提がすべて成り立つ世界。
    x: 店舗ごとの予算枠 c_s ~ U(10, 100) に週のゆらぎ N(0, 8) を足す
    y: 300 + 1.5 x + N(0, 60)。誤差は互いに独立・分散一定・x と無関係
    """
    n = S * T
    store = np.repeat(np.arange(S), T)
    budget = rng.uniform(X_LO, X_HI, S)[store]
    x = np.clip(budget + rng.normal(0, X_WEEK_SD, n), 5, None)
    y = ALPHA + BETA * x + rng.normal(0, SIG, n)
    return x, y


def ols(x, y):
    """単回帰の最小二乗。切片・傾き・傾きの標準誤差（誤差分散は残差から推定）"""
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = (resid ** 2).sum() / (len(y) - 2)
    return a, b, np.sqrt(s2 / sxx), sxx, s2


# ---- 図1（サイト第1回と同じ乱数の並び） ----
rng = np.random.default_rng(2026)
x, y = generate(rng)
a, b, se, sxx, s2 = ols(x, y)

fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
ax.scatter(x, y, s=7, alpha=0.35, color="#4a6fa5", linewidths=0, label="店舗×週（2,080件）")
xs = np.array([0.0, 125.0])
ax.plot(xs, ALPHA + BETA * xs, color="#333", lw=2.2, ls="--", label=f"本当の関係（傾き {BETA:.1f}）")
ax.plot(xs, a + b * xs, color="#c0392b", lw=2.2, label=f"回帰線（傾き {b:.2f}、標準誤差 {se:.2f}）")
ax.set_xlim(0, 125); ax.set_ylim(130, 650)
ax.set_xlabel("広告費（万円/週）"); ax.set_ylabel("売上（万円/週）")
ax.grid(alpha=0.3); ax.legend(loc="upper left")
ax.set_title("図1：前提がすべて成り立つ世界の20店舗×2年")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png")); plt.close(fig)

# ---- 検算1：傾きと標準誤差 ----
se_theory = SIG / np.sqrt(sxx)          # 誤差分散を既知として
r2 = 1 - s2 * (len(y) - 2) / ((y - y.mean()) ** 2).sum()
print("== 検算1：1つの世界 ==")
print(f"件数 n = {len(y)}")
print(f"傾き b_hat = {b:.4f}（本当の値 {BETA}）")
print(f"切片 a_hat = {a:.2f}（本当の値 {ALPHA}）")
print(f"残差の標準偏差 = {np.sqrt(s2):.2f}（本当の値 {SIG}）")
print(f"Σ(x - x̄)² = {sxx:,.0f}")
print(f"傾きの標準誤差（残差から） = {se:.4f}")
print(f"傾きの標準誤差（σ=60 を既知として σ/√Σ(x-x̄)²） = {se_theory:.4f}")
print(f"R² = {r2:.3f}")
print(f"傾きの95%区間 = [{b - 1.96*se:.3f}, {b + 1.96*se:.3f}]　本当の値を含む: {abs(b - BETA) < 1.96*se}")

# ---- 図2・検算2：同じ世界を1,000回 ----
R = 1000
rng2 = np.random.default_rng(2026)
bs = np.empty(R); ses = np.empty(R); hit = 0
for i in range(R):
    xx, yy = generate(rng2)
    _, bb, ss, _, _ = ols(xx, yy)
    bs[i], ses[i] = bb, ss
    hit += abs(bb - BETA) < 1.96 * ss

print("\n== 検算2：1,000個の世界 ==")
print(f"傾きの平均 = {bs.mean():.4f}（本当の値 {BETA}）")
print(f"傾きの標準偏差（1,000個のばらつき） = {bs.std(ddof=1):.4f}")
print(f"標準誤差の平均（各世界が自分で見積もった値） = {ses.mean():.4f}")
print(f"95%区間が本当の値を含んだ割合 = {hit / R:.3f}")

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
ax.hist(bs, bins=40, color="#4a6fa5", alpha=0.8, edgecolor="white")
ax.axvline(BETA, color="#333", ls="--", lw=2, label=f"本当の値 {BETA}")
ax.axvline(b, color="#c0392b", lw=2, label=f"図1の世界で出た傾き {b:.2f}")
ax.set_xlabel("回帰線の傾き（1万円あたりの売上の増え方）"); ax.set_ylabel("世界の数")
ax.set_title(f"図2：同じ世界を1,000回作り直したときの傾き（標準偏差 {bs.std(ddof=1):.3f}）")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2.png")); plt.close(fig)
