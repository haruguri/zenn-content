"""Zenn: 同じ店舗のデータが似ているとき、回帰の標準誤差に何が起きるか（店舗ごとの売上の水準の仮想データ）
図1: 同じ世界のデータを20回作り直したときの傾きの95%区間（サイト第3回の図3と同じ乱数の並び）
図2: 2つの世界を1,000回ずつ作り直したときの傾きの分布
検算: 分散の式の理論値、1,000個の世界、店舗でまとめた標準誤差、隣り合う週が似る世界・ばらつきが店舗で違う世界
依存: numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "regression-clustered-errors")
os.makedirs(OUT, exist_ok=True)

S, T = 20, 104                              # 店舗数・週数
ALPHA, BETA, SIG = 300.0, 1.5, 60.0         # 切片・本当の傾き・誤差の標準偏差（万円）
X_LO, X_HI, X_WEEK_SD = 10.0, 100.0, 8.0    # 予算枠の範囲・週のゆらぎ
A_SD = 40.0                                 # 店舗ごとの売上の水準の標準偏差（万円）
PHI = 0.7                                   # 参考：隣り合う週の誤差の相関

_files = fm.findSystemFonts()
for keys in (["NotoSansCJK", "NotoSansJP"], ["YuGoth"], ["meiryo", "Meiryo"], ["Hiragino"]):
    hit = [f for f in _files if any(k in f for k in keys)]
    if hit:
        plt.rcParams["font.family"] = fm.FontProperties(fname=hit[0]).get_name()
        break
else:
    raise RuntimeError("日本語フォントが見つかりません")
plt.rcParams["axes.unicode_minus"] = False


def generate(rng, store_level=False, ar=False, hetero=False):
    """store_level: 店舗の水準 a_s ~ N(0, 40) が売上に乗る。予算枠は a_s と無関係（前々回と同じ一様分布）。
    ar: 誤差が店舗の中で週をまたいで相関する（AR(1)、係数 0.7。ばらつきの大きさは 60 のまま）。
    hetero: 誤差の標準偏差が予算枠に比例する（枠 55 の店舗で 60）。
    乱数を引く順番は 枠→週のゆらぎ→誤差→店舗の水準。広告費は前々回の世界と同じ値になる。"""
    n = S * T
    store = np.repeat(np.arange(S), T)
    c = rng.uniform(X_LO, X_HI, S)
    x = np.clip(c[store] + rng.normal(0, X_WEEK_SD, n), 5, None)
    z = rng.normal(0, 1, n)
    if ar:
        zz = z.reshape(S, T); u = np.empty_like(zz); u[:, 0] = zz[:, 0]
        for t in range(1, T):
            u[:, t] = PHI * u[:, t - 1] + np.sqrt(1 - PHI ** 2) * zz[:, t]
        z = u.ravel()
    sd = SIG * c[store] / 55.0 if hetero else SIG
    a = rng.normal(0, A_SD, S) if store_level else np.zeros(S)
    y = ALPHA + BETA * x + a[store] + sd * z
    return x, y, store


def ols(x, y):
    """単回帰。係数（切片・傾き）、標準誤差、残差を返す。"""
    X = np.c_[np.ones(len(y)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s2 = r @ r / (len(y) - 2)
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X))), r


def cluster_se(x, r):
    """店舗でまとめた標準誤差（傾き）。店舗ごとに (x - x̄)·残差 を足してから2乗する。"""
    dx = x - x.mean()
    g = (dx * r).reshape(S, T).sum(axis=1)
    n = len(x)
    adj = S / (S - 1) * (n - 1) / (n - 2)
    return np.sqrt(adj * (g @ g)) / (dx @ dx)


def true_sd(x):
    """x を固定したときの傾きの本当の標準偏差（店舗の水準がある世界）。"""
    dx = x - x.mean()
    sxx = dx @ dx
    g = dx.reshape(S, T).sum(axis=1)
    return np.sqrt(SIG ** 2 / sxx + A_SD ** 2 * (g @ g) / sxx ** 2)


BLUE, RED, ORANGE = "#4a6fa5", "#c0392b", "#e67e22"
R = 1000
WORLDS = [("OFF", {}), ("STORE", dict(store_level=True)), ("AR", dict(ar=True)), ("HETERO", dict(hetero=True))]
res = {}
first = {}
for key, kw in WORLDS:
    rng = np.random.default_rng(2026)
    bs = np.empty(R); ses = np.empty(R); cses = np.empty(R); tsd = np.empty(R); bms = np.empty(R); sms = np.empty(R)
    for i in range(R):
        x, y, store = generate(rng, **kw)
        b, se, r = ols(x, y)
        bs[i], ses[i], cses[i], tsd[i] = b[1], se[1], cluster_se(x, r), true_sd(x)
        bm, sm, _ = ols(x.reshape(S, T).mean(axis=1), y.reshape(S, T).mean(axis=1))
        bms[i], sms[i] = bm[1], sm[1]
        if i == 0:
            first[key] = (x, y, b, se, r)
    res[key] = dict(b=bs, se=ses, cse=cses, tsd=tsd, bm=bms, sm=sms)

# ---- 図1：20回分の95%区間（サイト第3回の図3と同じ乱数の並び） ----
fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100, sharey=True)
for ax, (key, title) in zip(axes, [("OFF", "前提がすべて成り立つ世界"), ("STORE", "店舗ごとに売上の水準が違う世界")]):
    bs, ses = res[key]["b"][:20], res[key]["se"][:20]
    hit = np.abs(bs - BETA) < 1.96 * ses
    for i in range(20):
        col = BLUE if hit[i] else RED
        ax.plot([i + 1, i + 1], [bs[i] - 1.96 * ses[i], bs[i] + 1.96 * ses[i]], color=col, lw=3, solid_capstyle="butt")
        ax.plot(i + 1, bs[i], "o", color=col, ms=5)
    ax.axhline(BETA, color="#333", ls="--", lw=2.2, label=f"本当の値（傾き {BETA:.1f}）")
    ax.plot([], [], color=BLUE, lw=3, label="95%区間（本当の値を含む）")
    ax.plot([], [], color=RED, lw=3, label="95%区間（本当の値を含まない）")
    ax.set_title(f"{title}\n20回のうち {hit.sum()}回が本当の値を含む", fontsize=11)
    ax.set_xlabel("データを作り直した回（1〜20回目）"); ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0.6, 2.7); ax.grid(alpha=0.3, axis="y"); ax.legend(loc="upper left", fontsize=9)
axes[0].set_ylabel("回帰線の傾きと95%区間")
fig.suptitle("図1：同じ世界のデータを20回作り直したときの、傾きの95%区間")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png")); plt.close(fig)

# ---- 図2：1,000個の世界の傾き ----
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
bins = np.arange(0.3, 2.9, 0.04)
ax.hist(res["STORE"]["b"], bins=bins, color=ORANGE, alpha=0.75, label="店舗ごとに売上の水準が違う世界")
ax.hist(res["OFF"]["b"], bins=bins, color=BLUE, alpha=0.75, label="前提がすべて成り立つ世界")
ax.axvline(BETA, color="#333", ls="--", lw=2, label=f"本当の値 {BETA}")
ax.set_xlabel("回帰線の傾き（1万円あたりの売上の増え方）"); ax.set_ylabel("世界の数")
ax.set_title("図2：2つの世界を1,000回ずつ作り直したときの傾き")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2.png")); plt.close(fig)

# ---- 分散の式（理論値） ----
var_c = (X_HI - X_LO) ** 2 / 12
rho_u = A_SD ** 2 / (A_SD ** 2 + SIG ** 2)
rho_x = var_c / (var_c + X_WEEK_SD ** 2)
deff = 1 + (T - 1) * rho_u * rho_x

# ---- 検算 ----
print("== 検算1：1つの世界 ==")
for key, lab in [("OFF", "前提がすべて成り立つ世界"), ("STORE", "店舗の水準が違う世界")]:
    x, y, b, se, r = first[key]
    print(f"{lab}：傾き = {b[1]:.4f}（標準誤差 {se[1]:.4f}）、95%区間 {b[1] - 1.96 * se[1]:.2f}〜{b[1] + 1.96 * se[1]:.2f}")
x, y, b, se, r = first["STORE"]
rm = r.reshape(S, T).mean(axis=1)
print(f"店舗の水準が違う世界：残差の店舗ごとの平均 最小 {rm.min():.1f}、最大 {rm.max():.1f}")
print(f"広告費の分散：店舗の平均どうし {x.reshape(S, T).mean(axis=1).var(ddof=1):.0f}／全体 {x.var(ddof=1):.0f}")
print(f"店舗でまとめた標準誤差 = {cluster_se(x, r):.4f}、店舗ごとの平均20点の回帰：傾き {res['STORE']['bm'][0]:.4f}（標準誤差 {res['STORE']['sm'][0]:.4f}）")
print("\n== 検算2：分散の式 ==")
print(f"誤差の店舗内相関 = {A_SD ** 2:.0f}/{A_SD ** 2 + SIG ** 2:.0f} = {rho_u:.3f}、広告費の店舗内相関 = {var_c:.0f}/{var_c + X_WEEK_SD ** 2:.0f} = {rho_x:.3f}")
print(f"分散の倍率 1 + (T−1)·ρu·ρx = {deff:.1f}、標準誤差の倍率 {np.sqrt(deff):.2f}、2,080件 ÷ {deff:.1f} = {S * T / deff:.0f}件分")
print(f"x を固定した本当の標準偏差：1つ目の世界 {res['STORE']['tsd'][0]:.3f}、1,000個の平均 {res['STORE']['tsd'].mean():.3f}")
print("\n== 検算3：1,000個の世界 ==")
for key, lab in [("OFF", "前提がすべて成り立つ世界"), ("STORE", "店舗の水準が違う世界"), ("AR", "隣り合う週が似る世界"), ("HETERO", "ばらつきが店舗で違う世界")]:
    d = res[key]
    print(f"{lab}：傾きの平均 {d['b'].mean():.3f}、標準偏差 {d['b'].std(ddof=1):.3f}、最小 {d['b'].min():.2f}、最大 {d['b'].max():.2f}、"
          f"標準誤差の平均 {d['se'].mean():.3f}、95%区間の的中率 {np.mean(np.abs(d['b'] - BETA) < 1.96 * d['se']):.3f}")
print("\n== 検算4：店舗でまとめて数え直す ==")
for key, lab in [("OFF", "前提がすべて成り立つ世界"), ("STORE", "店舗の水準が違う世界"), ("AR", "隣り合う週が似る世界")]:
    d = res[key]
    print(f"{lab}：店舗でまとめた標準誤差の平均 {d['cse'].mean():.3f}、的中率 {np.mean(np.abs(d['b'] - BETA) < 1.96 * d['cse']):.3f}（1.96）／"
          f"{np.mean(np.abs(d['b'] - BETA) < 2.093 * d['cse']):.3f}（t分布・自由度19の2.093）、"
          f"店舗の平均20点の回帰の的中率 {np.mean(np.abs(d['bm'] - BETA) < 2.101 * d['sm']):.3f}")
