"""Zenn: 広告費を動かしているものが売上も動かすとき、回帰の傾きに何が起きるか（年末商戦の仮想データ）
図1: 前提がすべて成り立つ世界と、年末に予算も需要も増える世界の散布図（サイト第2回の図2と同じ乱数の並び）
図2: 3つの世界を1,000回ずつ作り直したときの傾きの分布
検算: 偏りの式の理論値、1,000個の世界、年末を入れた重回帰、新しい2年分での予測
依存: numpy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images", "regression-confounding-yearend")
os.makedirs(OUT, exist_ok=True)

S, T = 20, 104                              # 店舗数・週数
ALPHA, BETA, SIG = 300.0, 1.5, 60.0         # 切片・本当の傾き・誤差の標準偏差（万円）
X_LO, X_HI, X_WEEK_SD = 10.0, 100.0, 8.0    # 予算枠の範囲・週のゆらぎ
YEAREND_DEMAND, YEAREND_BUDGET_RATIO = 150.0, 0.3   # 年末4週の売上の上乗せ・予算の積み増し（需要の0.3倍）

_files = fm.findSystemFonts()
for keys in (["NotoSansCJK", "NotoSansJP"], ["YuGoth"], ["meiryo", "Meiryo"], ["Hiragino"]):
    hit = [f for f in _files if any(k in f for k in keys)]
    if hit:
        plt.rcParams["font.family"] = fm.FontProperties(fname=hit[0]).get_name()
        break
else:
    raise RuntimeError("日本語フォントが見つかりません")
plt.rcParams["axes.unicode_minus"] = False


def yearend(T):
    """年末4週（年内の週番号 48〜51）だけ 150、それ以外は 0。"""
    w = np.arange(T) % 52
    return np.where(w >= 48, YEAREND_DEMAND, 0.0)


def generate(rng, yearend_on=False, store_conf=False):
    """yearend_on: 年末4週は売上に +150、広告費に +45（需要の0.3倍）。
    store_conf: 店舗の規模 a_s ~ N(0, 40) が売上に乗り、予算枠を 55 + 0.5 a_s + N(0, 10) で配る。
    乱数を引く順番は前回と同じ（yearend_on だけなら、年末以外の週は前回の世界と同じ値になる）。"""
    n = S * T
    store = np.repeat(np.arange(S), T)
    week = np.tile(np.arange(T), S)
    a = np.zeros(S)
    if store_conf:
        a = rng.normal(0, 40, S)
        budget = np.clip(55 + 0.5 * a + rng.normal(0, 10, S), X_LO, X_HI)[store]
    else:
        budget = rng.uniform(X_LO, X_HI, S)[store]
    x = np.clip(budget + rng.normal(0, X_WEEK_SD, n), 5, None)
    d = yearend(T)[week] if yearend_on else np.zeros(n)
    x = np.clip(x + YEAREND_BUDGET_RATIO * d, 5, None)
    y = ALPHA + BETA * x + a[store] + d + rng.normal(0, SIG, n)
    return x, y, (week % 52) >= 48, d


def ols(X, y):
    """最小二乗。X は説明変数（1列または複数列）。係数（切片が先頭）と標準誤差を返す。"""
    X = np.c_[np.ones(len(y)), X]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s2 = r @ r / (len(y) - X.shape[1])
    return b, np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))


def rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


BLUE, ORANGE, PURPLE = "#4a6fa5", "#e67e22", "#8e44ad"

# ---- 図1（サイト第2回の図2と同じ乱数の並び） ----
rng = np.random.default_rng(2026); x0, y0, _, _ = generate(rng)
rng = np.random.default_rng(2026); x1, y1, ye, d1 = generate(rng, yearend_on=True)
b0, se0 = ols(x0, y0)
b1, se1 = ols(x1, y1)

fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100, sharey=True)
xs = np.array([0.0, 170.0])
for ax, (x, y, mask, b, se, title) in zip(axes, [
        (x0, y0, None, b0, se0, "前提がすべて成り立つ世界"),
        (x1, y1, ye, b1, se1, "年末に予算も需要も増える世界")]):
    if mask is None:
        ax.scatter(x, y, s=7, alpha=0.35, color=BLUE, linewidths=0, label="店舗×週（2,080件）")
    else:
        ax.scatter(x[~mask], y[~mask], s=7, alpha=0.35, color=BLUE, linewidths=0, label="年末以外の週")
        ax.scatter(x[mask], y[mask], s=10, alpha=0.75, color=ORANGE, linewidths=0, label="年末4週")
    ax.plot(xs, ALPHA + BETA * xs, color="#333", lw=2.2, ls="--", label=f"本当の関係（傾き {BETA:.1f}）")
    ax.plot(xs, b[0] + b[1] * xs, color="#c0392b", lw=2.2, label=f"回帰線（傾き {b[1]:.2f}、標準誤差 {se[1]:.2f}）")
    ax.set_xlim(0, 170); ax.set_ylim(130, 850)
    ax.set_xlabel("広告費（万円/週）"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    ax.set_title(title)
axes[0].set_ylabel("売上（万円/週）")
fig.suptitle("図1：前提がすべて成り立つ世界（左）と、年末に予算も需要も増える世界（右）")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1.png")); plt.close(fig)

# ---- 偏りの式（理論値） ----
p = 4 / 52
var_d = YEAREND_DEMAND ** 2 * p * (1 - p)
cov_xd = YEAREND_BUDGET_RATIO * var_d
var_x = (X_HI - X_LO) ** 2 / 12 + X_WEEK_SD ** 2 + YEAREND_BUDGET_RATIO ** 2 * var_d
bias_theory = cov_xd / var_x
cov_xd_s = np.cov(x1, d1)[0, 1]; var_x_s = x1.var(ddof=1)
var_a = 40.0 ** 2
bias_store_theory = (0.5 * var_a) / (0.25 * var_a + 10 ** 2 + X_WEEK_SD ** 2)

# ---- 図2：3つの世界を1,000回ずつ ----
R = 1000
res = {}
for key, kw in [("OFF", {}), ("YEAREND", dict(yearend_on=True)), ("STORE", dict(store_conf=True))]:
    rng2 = np.random.default_rng(2026); bs = np.empty(R); ses = np.empty(R)
    for i in range(R):
        xx, yy, _, _ = generate(rng2, **kw); bb, ss = ols(xx, yy); bs[i], ses[i] = bb[1], ss[1]
    res[key] = (bs, ses)

fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
bins = np.arange(1.2, 3.65, 0.025)
ax.hist(res["OFF"][0], bins=bins, color=BLUE, alpha=0.75, label="前提がすべて成り立つ世界")
ax.hist(res["YEAREND"][0], bins=bins, color=ORANGE, alpha=0.75, label="年末に予算も需要も増える世界")
ax.hist(res["STORE"][0], bins=bins, color=PURPLE, alpha=0.5, label="枠を店舗の規模で配る世界")
ax.axvline(BETA, color="#333", ls="--", lw=2, label=f"本当の値 {BETA}")
ax.axvline(BETA + bias_theory, color=ORANGE, ls=":", lw=2, label=f"年末の世界の理論値 {BETA + bias_theory:.2f}")
ax.set_xlabel("回帰線の傾き（1万円あたりの売上の増え方）"); ax.set_ylabel("世界の数")
ax.set_title("図2：3つの世界を1,000回ずつ作り直したときの傾き")
ax.legend(); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2.png")); plt.close(fig)

# ---- 年末を知っていれば ----
bd, sed = ols(np.c_[x1, ye.astype(float)], y1)
bx, sex = ols(x1[~ye], y1[~ye])

# ---- 新しい2年分での予測 ----
rng3 = np.random.default_rng(2027); x2, y2, ye2, _ = generate(rng3, yearend_on=True)
pf = b1[0] + b1[1] * x2
pt = ALPHA + BETA * x2
segs = [("全体", np.ones_like(ye2)), ("年末以外", ~ye2), ("年末4週", ye2)]
bands = [("年末以外・40未満", (~ye2) & (x2 < 40)), ("年末以外・40〜70", (~ye2) & (x2 >= 40) & (x2 < 70)),
         ("年末以外・70以上", (~ye2) & (x2 >= 70)), ("年末4週", ye2)]

# ---- 検算 ----
print("== 検算1：1つの世界 ==")
print(f"前提がすべて成り立つ世界：傾き = {b0[1]:.4f}（標準誤差 {se0[1]:.4f}）")
print(f"年末の世界：傾き = {b1[1]:.4f}（標準誤差 {se1[1]:.4f}）、切片 = {b1[0]:.2f}")
print(f"年末4週の平均：広告費 {x1[ye].mean():.1f}／それ以外 {x1[~ye].mean():.1f}、売上 {y1[ye].mean():.1f}／それ以外 {y1[~ye].mean():.1f}")
print("\n== 検算2：偏りの式 ==")
print(f"Var(d) = {var_d:.1f}、Cov(x, d) = {cov_xd:.1f}、Var(x) = {var_x:.1f}")
print(f"偏りの理論値 Cov(x, d)/Var(x) = {bias_theory:.3f} → 傾きの理論値 {BETA + bias_theory:.3f}")
print(f"この世界の標本値：Cov(x, d)/Var(x) = {cov_xd_s:.1f}/{var_x_s:.1f} = {cov_xd_s / var_x_s:.3f}")
print("\n== 検算3：1,000個の世界 ==")
for key, lab in [("OFF", "前提がすべて成り立つ世界"), ("YEAREND", "年末の世界"), ("STORE", "店舗の規模で枠を配る世界")]:
    bs, ses = res[key]
    hitr = np.mean(np.abs(bs - BETA) < 1.96 * ses)
    print(f"{lab}：傾きの平均 {bs.mean():.3f}、標準偏差 {bs.std(ddof=1):.3f}、最小 {bs.min():.2f}、最大 {bs.max():.2f}、"
          f"標準誤差の平均 {ses.mean():.3f}、95%区間の的中率 {hitr:.3f}")
print(f"店舗の世界の理論値：{BETA:.1f} + {bias_store_theory:.3f} = {BETA + bias_store_theory:.3f}")
print("\n== 検算4：年末を知っていれば ==")
print(f"年末の印を説明変数に入れた重回帰：広告費の傾き = {bd[1]:.4f}（標準誤差 {sed[1]:.4f}）、年末の係数 = {bd[2]:.1f}")
print(f"年末4週を除いた単回帰：傾き = {bx[1]:.4f}（標準誤差 {sex[1]:.4f}）")
print("\n== 検算5：新しい2年分での予測 ==")
for k, m in segs:
    print(f"{k}：外れの大きさ（2乗平均の平方根） 回帰線 {rmse((y2 - pf)[m]):.1f}／本当の関係の線 {rmse((y2 - pt)[m]):.1f}")
for k, m in bands:
    print(f"{k}（{m.sum()}件）：実際 − 予測の平均 回帰線 {np.mean((y2 - pf)[m]):+.1f}／本当の関係の線 {np.mean((y2 - pt)[m]):+.1f}")
