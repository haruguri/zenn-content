"""Zenn「裾が重いとき、大数の法則に何が起きるか──保険金を例にシミュレーションで見る」の図と検算。
numpy / matplotlib のみ。seed 2026。images/lln-heavy-tail-insurance/fig1.png〜fig3.png を書き出す。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "lln-heavy-tail-insurance")
os.makedirs(OUT, exist_ok=True)

def pick_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for c in ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Hiragino Sans"]:
        if c in names:
            return c
    raise RuntimeError("日本語フォントが見つかりません")
plt.rcParams["font.family"] = pick_font()
plt.rcParams["axes.unicode_minus"] = False

N, K, R, M = 10_000, 8, 2_000, 100.0

def lomax(rng, a, lam, size):
    """Lomax(α, λ)：P(X > x) = (1 + x/λ)^(-α)。平均 λ/(α−1)（α>1）、分散は α>2 で有限。"""
    return lam * (rng.random(size) ** (-1.0 / a) - 1.0)

MED15 = 50.0 * (2 ** (1 / 1.5) - 1)
LAM09 = MED15 / (2 ** (1 / 0.9) - 1)
DISTS = [
    ("指数（平均100）",            None, M,      lambda r, s: r.exponential(M, s)),
    ("Lomax α=2.5, λ=150",         2.5,  150.0,  lambda r, s: lomax(r, 2.5, 150.0, s)),
    ("Lomax α=1.5, λ=50",          1.5,  50.0,   lambda r, s: lomax(r, 1.5, 50.0, s)),
    ("Lomax α=0.9, λ=%.1f" % LAM09, 0.9, LAM09,  lambda r, s: lomax(r, 0.9, LAM09, s)),
]

# ---- 対応表 ----
print("## 対応表")
print("| 分布 | 平均 | 分散 | 中央値 | P(X > 10×中央値) |")
print("|---|---|---|---|---|")
for name, a, lam, _ in DISTS:
    if a is None:
        med = M * np.log(2); p = 2.0 ** -10; mean = "100"; var = "有限"
    else:
        med = lam * (2 ** (1 / a) - 1); p = (1 + 10 * med / lam) ** -a
        mean = "%.0f" % (lam / (a - 1)) if a > 1 else "なし"
        var = "有限" if a > 2 else "無限"
    print("| %s | %s | %s | %.1f | 1/%.0f |" % (name, mean, var, med, 1 / p))

# ---- 実験1・2（サイト版と同じ乱数の消費順）----
rng = np.random.default_rng(2026)
n = np.arange(1, N + 1)
shares, finals = [], []
fig, axs = plt.subplots(2, 2, figsize=(10, 7), dpi=100, sharex=True)
for ax, (name, a, lam, gen) in zip(axs.flat, DISTS):
    x = gen(rng, (K, N)); cm = np.cumsum(x, 1) / n
    big = gen(rng, (R, N)); bm = np.cumsum(big, 1) / n
    q = np.quantile(bm, [0.025, 0.975], axis=0)
    shares.append(np.median(np.maximum.accumulate(big, 1) / np.cumsum(big, 1), axis=0))
    finals.append((name, bm[:, -1], big))
    for p in cm: ax.plot(n, p, lw=1)
    ax.plot(n, q[0], "r--", lw=.9); ax.plot(n, q[1], "r--", lw=.9)
    if a is None or a > 1:
        ax.axhline(M, color="k", ls=":"); ax.set_ylim(0, 600)
    else:
        ax.set_ylim(0, 3000); name += "（縦軸の幅が違う）"
    ax.set_xscale("log"); ax.set_title(name, fontsize=11); ax.grid(alpha=.25)
fig.supxlabel("件数 n（対数）"); fig.supylabel("ここまでの標本平均（万円）")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png"))

fig2, ax = plt.subplots(figsize=(10, 5), dpi=100)
for (name, a, lam, _), s in zip(DISTS, shares):
    ax.plot(n, s * 100, lw=1.6, label=name)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.01, 100)
ax.set_xlabel("件数 n（対数）"); ax.set_ylabel("最大の1件 ÷ 合計（%、対数・中央値）")
ax.grid(alpha=.25, which="both"); ax.legend(fontsize=10)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "fig2.png"))

# ---- 実験3：標本平均の散らばり（四分位範囲）vs n ----
rng3 = np.random.default_rng(2026)
ns = [100, 1_000, 10_000, 100_000]
fig3, ax = plt.subplots(figsize=(8, 5), dpi=100)
print("\n## 実験3：四分位範囲の傾き（両対数）")
for name, a, lam, gen in DISTS:
    iqr = []
    for m in ns:
        reps = 2000 if m <= 10_000 else 400
        x = gen(rng3, (reps, m)); mean = x.mean(1)
        q = np.quantile(mean, [.25, .75]); iqr.append(q[1] - q[0])
    slope = np.polyfit(np.log10(ns), np.log10(iqr), 1)[0]
    theory = -0.5 if (a is None or a >= 2) else 1 / a - 1
    print("%s: 傾き %.3f（理論 %.3f）IQR=%s" % (name, slope, theory, np.round(iqr, 2)))
    ax.plot(ns, iqr, "o-", label="%s（傾き %.2f、理論 %.2f）" % (name, slope, theory))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("件数 n（対数）"); ax.set_ylabel("標本平均の四分位範囲（万円、対数）")
ax.grid(alpha=.25, which="both"); ax.legend(fontsize=9)
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "fig3.png"))

# ---- 検算 ----
print("\n## 検算（n = 10,000、2,000本）")
for (name, a, lam, _), (_, fin, big) in zip(DISTS, finals):
    r = big.max(1) / big.sum(1)
    print("%s: 標本平均の中央値 %.1f、95%%幅 %.1f〜%.1f、P(標本平均<100) %.2f、最大1件÷合計の中央値 %.3f" %
          (name, np.median(fin), *np.quantile(fin, [.025, .975]), (fin < M).mean(), np.median(r)))
# α<1：E[S_n/M_n] → 1/(1−α)（Darling 1952）
rng4 = np.random.default_rng(2026)
for a in [0.5, 0.9]:
    for m in [1_000, 100_000]:
        x = rng4.random((400, m)) ** (-1 / a)
        print("純Pareto α=%.1f, n=%d: E[合計÷最大] = %.2f（極限 1/(1−α) = %.2f）" % (a, m, (x.sum(1) / x.max(1)).mean(), 1 / (1 - a)))
