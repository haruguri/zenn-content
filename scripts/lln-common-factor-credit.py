"""Zenn 第2本（第3回の計算編）図と検算
図1: 2x2（サイトと同じ・Vasicek）  図2: 平均のSD vs n（等相関4本 + AR(1)）  図3: 非復元抽出（負の相関）
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import norm, multivariate_normal
import os

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(FONT): raise RuntimeError("no CJK font")
font_manager.fontManager.addfont(FONT); plt.rcParams["font.family"] = "Noto Sans CJK JP"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images", "lln-common-factor-credit")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "")  # OUT + "fig1.png"
SEED, P, N = 2026, 0.02, 10_000
k = norm.ppf(P)
xs = np.arange(1, N + 1)

def defcorr(ra):
    p12 = multivariate_normal(mean=[0, 0], cov=[[1, ra], [ra, 1]]).cdf([k, k])
    return (p12 - P * P) / (P * (1 - P))
def cond_pd(z, ra): return norm.cdf((k - np.sqrt(ra) * z) / np.sqrt(1 - ra))

ASSET = [0, 0.05, 0.10, 0.15]
RHO = {ra: (defcorr(ra) if ra else 0.0) for ra in ASSET}
print("== 資産相関 → 答えの相関（PD=2%） ==")
for ra in ASSET: print(f"  資産相関 {ra:<5} → 答えの相関 {RHO[ra]:.4f}")

# ---------- 図1 ----------
rng = np.random.default_rng(SEED)
grid = np.unique(np.logspace(0, 4, 60).astype(int))
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for ax, ra in zip(axes.ravel(), ASSET):
    M = 2000; z = rng.standard_normal(M)
    pk = cond_pd(z, ra) if ra else np.full(M, P)
    cum = np.zeros(M); lo=[]; hi=[]; prev=0
    for n in grid:
        cum += rng.binomial(n-prev, pk); prev = n; r = cum/n
        lo.append(np.quantile(r,.025)); hi.append(np.quantile(r,.975))
    lo, hi = np.array(lo)*100, np.array(hi)*100
    ax.fill_between(grid, lo, hi, color="#d62728", alpha=.07); ax.plot(grid, lo, "r-", lw=.9); ax.plot(grid, hi, "r-", lw=.9)
    for t in range(8):
        p_t = cond_pd(rng.standard_normal(), ra) if ra else P
        ax.plot(xs, np.cumsum(rng.random(N) < p_t)/xs*100, lw=1, alpha=.9)
    ax.axhline(P*100, color="k", lw=.6, ls=":")
    ax.set_title("独立" if not ra else f"資産相関 {ra}（答えの相関 {RHO[ra]:.3f}）", fontsize=11)
    ax.set_xscale("log"); ax.set_ylim(0, 8); ax.grid(alpha=.3)
for ax in axes[1]: ax.set_xlabel("貸出件数 n")
for ax in axes[:,0]: ax.set_ylabel("ここまでの貸倒率（%）")
fig.suptitle("真の貸倒率 2%、各8試行。赤はシミュレーション2,000本の 2.5〜97.5% 分位", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + "fig1.png", dpi=110)

# ---------- 図2: SD vs n ----------
def sd_equi(n, rho): return np.sqrt(P*(1-P)/n*(1+(n-1)*rho))
ns = np.unique(np.logspace(0, 4, 40).astype(int))
# AR(1) latent: X_i = phi X_{i-1} + sqrt(1-phi^2) e_i, default if X_i < k. 潜在相関 phi^|i-j| → 答えの相関は減衰
def ar1_sd(phi, reps=1500):
    rng2 = np.random.default_rng(SEED+1)
    e = rng2.standard_normal((reps, N)); X = np.empty_like(e); X[:,0] = e[:,0]
    c = np.sqrt(1-phi**2)
    for i in range(1, N): X[:,i] = phi*X[:,i-1] + c*e[:,i]
    Y = (X < k).astype(float); cum = np.cumsum(Y, axis=1)
    return np.array([cum[:, n-1].std()/n for n in ns])
phi = 0.98
ar_sd = ar1_sd(phi)
# AR(1) の隣接相関（答えの）
p_adj = multivariate_normal(mean=[0,0], cov=[[1,phi],[phi,1]]).cdf([k,k]); rho_adj = (p_adj-P*P)/(P*(1-P))
print(f"\n== AR(1) φ={phi}: 隣の貸出との答えの相関 {rho_adj:.3f} ==")

fig, ax = plt.subplots(figsize=(8, 5.5))
for ra in ASSET:
    r = RHO[ra]; ax.plot(ns, sd_equi(ns, r)*100, lw=1.6, label=("独立（相関0）" if not ra else f"等相関 {r:.3f}（資産相関 {ra}）"))
    if ra: ax.axhline(np.sqrt(r*P*(1-P))*100, color="gray", lw=.6, ls=":")
ax.plot(ns, ar_sd*100, "k--", lw=1.6, label=f"AR(1) 型：隣とは {rho_adj:.2f}、離れるほど減衰（シミュレーション）")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("貸出件数 n"); ax.set_ylabel("貸倒率のぶれ（標準偏差・%）")
ax.set_title("ぶれは n で縮むか：等相関は底値で止まり、減衰する相関は縮み続ける", fontsize=11)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT + "fig2.png", dpi=110)

print("\n== 検算：n=10,000 の標準偏差（%） 理論 vs 実測（2,000本） ==")
rng3 = np.random.default_rng(SEED+2)
for ra in ASSET:
    r = RHO[ra]
    if ra: pk = cond_pd(rng3.standard_normal(2000), ra)
    else:  pk = np.full(2000, P)
    meas = (rng3.binomial(N, pk)/N).std()*100
    print(f"  等相関 {r:.4f}: 理論 {sd_equi(N, r)*100:.3f}  底値 {np.sqrt(r*P*(1-P))*100:.3f}  実測 {meas:.3f}")
print(f"  AR(1) φ={phi}: 実測 {ar_sd[-1]*100:.3f}  （独立の {sd_equi(N,0)*100:.3f} の {ar_sd[-1]/sd_equi(N,0):.1f} 倍。n=1,000 では {ar_sd[np.searchsorted(ns,1000)]*100:.3f}）")

# ---------- 図3: 非復元抽出 ----------
Npop = 10_000; pop = np.zeros(Npop); pop[:int(P*Npop)] = 1
rng4 = np.random.default_rng(SEED+3)
ns3 = np.unique(np.logspace(0, 4, 40).astype(int)); ns3[-1] = Npop
sd_wo = []
for n in ns3:
    sd_wo.append(np.array([pop[rng4.permutation(Npop)[:n]].mean() for _ in range(1500)]).std())
sd_wo = np.array(sd_wo)
fpc = np.sqrt(P*(1-P)/ns3*(Npop-ns3)/(Npop-1))
fig, ax = plt.subplots(figsize=(8, 5))
ind = sd_equi(ns3, 0)
ax.plot(ns3, fpc/ind, lw=1.6, label=f"非復元抽出（母集団 {Npop:,} 件、相関 −1/(N−1)）：理論 √((N−n)/(N−1))")
ax.plot(ns3, sd_wo/ind, "o", ms=3, color="k", label="非復元抽出：シミュレーション")
ax.axhline(1, color="gray", lw=.8, ls=":", label="独立（復元抽出）＝1")
ax.set_xscale("log"); ax.set_ylim(0, 1.1); ax.set_xlabel("貸出件数 n"); ax.set_ylabel("ぶれの比（独立のときを 1）")
ax.set_title("負の相関は縮む側にしか効かない：n = N で誤差ゼロ", fontsize=11); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + "fig3.png", dpi=110)
print(f"\n== 負の相関：平均相関の下限 −1/(n−1)。n=10,000 で {-1/(N-1):.6f} ==")
print(f"  非復元 n=5,000: 理論 {fpc[np.searchsorted(ns3,5000)]*100:.3f}  実測 {sd_wo[np.searchsorted(ns3,5000)]*100:.3f}  n=N: 理論 0 実測 {sd_wo[-1]*100:.3f}")
