"""Synthetic experiments for tangent-space covariance features.

Experiment 1. Two classes of covariance matrices on the SPD manifold: geodesically Gaussian clouds around two centers
separated along a fixed tangent direction. All scales of the cloud are proportional to the parameter s, so the problem
in the tangent coordinates of the class center stays the same and only the cloud radius on the manifold changes.
Classifiers:
  * TSM + logistic regression (projection at the Riemannian mean of the sample);
  * log-Euclidean + logistic regression (projection at the identity, a misplaced reference point);
  * Euclidean + logistic regression (vectorization of the covariance itself);
  * MDM (nearest class Riemannian mean in the geodesic distance).
The distance distortion delta_R(R_i, R_j) / ||p_i - p_j||_2 over all pairs of training objects is compared with the
bounds c(rho) (cloud radius) and g(omega) (spectral spread of the tangent vectors), and the invariance of TSM under
R -> A R A^T is checked.

Experiment 1a. Sharpness of the constant: for a pair of tangent vectors with eigenvalues +-rho/sqrt(2) whose
eigenbases are rotated by the angle theta, the ratio delta_R / ||p_i - p_j||_2 tends to c(rho) as theta -> 0 while
staying strictly below c(rho).

Experiment 2. Estimation noise: the true covariances are replaced by sample covariances from T samples; the accuracy
of TSM and the error delta_R(R_hat, R) are compared with the bound of the theorem.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, json, time

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- matrix tools
def sym_fun(M, f):
    w, U = eigh(M)
    return (U * f(w)) @ U.T

def mlog(M):  return sym_fun(M, np.log)
def mexp(M):  return sym_fun(M, np.exp)
def msqrt(M): return sym_fun(M, np.sqrt)
def misqrt(M): return sym_fun(M, lambda w: 1.0 / np.sqrt(w))

def geo_dist(A, B):
    Ai = misqrt(A)
    return np.linalg.norm(mlog(Ai @ B @ Ai), "fro")

def pairwise_geo(Rs):
    """Pairwise geodesic distance matrix (batched eigvalsh)."""
    n = len(Rs); D = np.zeros((n, n))
    for i in range(n - 1):
        Ai = misqrt(Rs[i])
        w = np.linalg.eigvalsh(Ai @ Rs[i + 1:] @ Ai)
        D[i, i + 1:] = np.sqrt((np.log(w) ** 2).sum(axis=1))
    return D + D.T

def pairwise_euc(F):
    G = F @ F.T; sq = np.diag(G)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * G, 0.0))

def exp_at(R, V):          # Exp_R(V) for a tangent vector V given in whitened coordinates
    Rs = msqrt(R)
    return Rs @ mexp(V) @ Rs

def log_at(R, X):          # whitened Log_R(X) = log(R^{-1/2} X R^{-1/2})
    Ri = misqrt(R)
    return mlog(Ri @ X @ Ri)

def vec_sqrt2(S):
    h = S.shape[0]
    iu = np.triu_indices(h)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return S[iu] * w

def karcher_mean(Rs, iters=60, step=0.5, tol=1e-10):
    R = np.mean(Rs, axis=0)
    for _ in range(iters):
        Ri, Rs_ = misqrt(R), msqrt(R)
        G = np.mean([mlog(Ri @ X @ Ri) for X in Rs], axis=0)
        R = Rs_ @ mexp(step * G) @ Rs_
        if np.linalg.norm(G, "fro") < tol:
            break
    return R

def random_sym(rng, h):
    """Random symmetric matrix Z whose coordinates vec_sqrt2(Z) are independent N(0,1)."""
    G = rng.standard_normal((h, h))
    return (G + G.T) / 2.0

def g_fun(x):
    """g(x) = sinh(x/2)/(x/2), the stretch factor of the differential of the matrix exponential."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x); m = np.abs(x) > 1e-12
    out[m] = np.sinh(x[m] / 2.0) / (x[m] / 2.0)
    return out if out.ndim else float(out)

def c_rho(rho):
    return g_fun(np.sqrt(2.0) * np.asarray(rho, dtype=float))

def spread(S):
    """omega(S) = lambda_max - lambda_min, the log condition number of exp(S)."""
    w = np.linalg.eigvalsh(S)
    return w[-1] - w[0]

# ---------------------------------------------------------------- plot style
def plot_style():
    """Figure labels are typeset by pdflatex through the pgf backend."""
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": "\n".join([r"\usepackage[utf8]{inputenc}", r"\usepackage[T2A]{fontenc}",
                                   r"\usepackage[russian]{babel}", r"\usepackage{amsmath,amssymb}"]),
        "font.family": "serif", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.2, "lines.markersize": 4, "axes.grid": True, "grid.alpha": 0.3,
        "legend.framealpha": 0.9, "legend.edgecolor": "0.8",
    })

def savefig(fig, path):
    """Save a figure through the pgf backend (LaTeX fonts)."""
    fig.savefig(path, backend="pgf", bbox_inches="tight", pad_inches=0.02)

# ---------------------------------------------------------------- data model
def sample_problem(rng, h):
    """Random reference point R0 = exp(Z0) and unit class-separation direction U."""
    R0 = exp_at(np.eye(h), random_sym(rng, h))
    U = random_sym(rng, h); U /= np.linalg.norm(U, "fro")
    return R0, U

def make_dataset(rng, h, s, n_per_class, R0, U, noise_ratio=2.0):
    """Two classes: centers Exp_{R0}(+-s/2 U), objects Exp_{center}(sigma Z), sigma = s/noise_ratio."""
    sigma = s / noise_ratio
    centers = [exp_at(R0, +0.5 * s * U), exp_at(R0, -0.5 * s * U)]
    X, y = [], []
    for c, C in enumerate(centers):
        for _ in range(n_per_class):
            E = sigma * random_sym(rng, h)
            X.append(exp_at(C, E)); y.append(c)
    return np.array(X), np.array(y)

# ---------------------------------------------------------------- classifiers
def feats_tsm(Rtr, Rte):
    Rm = karcher_mean(Rtr)
    return (np.array([vec_sqrt2(log_at(Rm, X)) for X in Rtr]),
            np.array([vec_sqrt2(log_at(Rm, X)) for X in Rte]), Rm)

def feats_logeuc(Rtr, Rte):
    return (np.array([vec_sqrt2(mlog(X)) for X in Rtr]),
            np.array([vec_sqrt2(mlog(X)) for X in Rte]))

def feats_euc(Rtr, Rte):
    return (np.array([vec_sqrt2(X) for X in Rtr]),
            np.array([vec_sqrt2(X) for X in Rte]))

def lr_acc(Ftr, ytr, Fte, yte):
    # common scalar scale of the features (no coordinatewise standardization): the regularization
    # of the logistic regression then acts identically at any scale of the cloud
    scale = np.mean(np.linalg.norm(Ftr - Ftr.mean(axis=0), axis=1))
    clf = LogisticRegression(C=1.0, max_iter=5000)
    clf.fit(Ftr / scale, ytr)
    return clf.score(Fte / scale, yte)

def mdm_acc(Rtr, ytr, Rte, yte):
    means = [karcher_mean(Rtr[ytr == c]) for c in (0, 1)]
    pred = np.array([np.argmin([geo_dist(X, M) for M in means]) for X in Rte])
    return np.mean(pred == yte)

# ---------------------------------------------------------------- experiment 1
def experiment_radius(h=8, n_train=100, n_test=500, seeds=10,
                      s_grid=(0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5)):
    rows = []
    iu = np.triu_indices(2 * n_train, 1)
    for s in s_grid:
        acc = {k: [] for k in ("tsm", "logeuc", "euc", "mdm", "tsm_affine")}
        stat = {k: [] for k in ("rho95", "rho_max", "c_rho", "omega_max", "c_omega", "ratio_min",
                                "ratio_med", "ratio_q99", "ratio_max", "pair_frac", "affine_err")}
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            R0, U = sample_problem(rng, h)
            Rtr, ytr = make_dataset(rng, h, s, n_train, R0, U)
            Rte, yte = make_dataset(rng, h, s, n_test, R0, U)
            Ftr, Fte, Rm = feats_tsm(Rtr, Rte)
            acc["tsm"].append(lr_acc(Ftr, ytr, Fte, yte))
            d = np.array([geo_dist(Rm, X) for X in Rtr])
            stat["rho95"].append(np.quantile(d, 0.95)); stat["rho_max"].append(d.max())
            stat["c_rho"].append(c_rho(d.max()))
            # spectral spread of the tangent vectors and the refined bound g(omega)
            om = np.array([spread(log_at(Rm, X)) for X in Rtr])
            stat["omega_max"].append(om.max()); stat["c_omega"].append(g_fun(om.max()))
            # distance distortion over all pairs of training objects
            Dg = pairwise_geo(Rtr)[iu]; De = pairwise_euc(Ftr)[iu]
            ratio = Dg / De
            stat["ratio_min"].append(ratio.min()); stat["ratio_med"].append(np.median(ratio))
            stat["ratio_q99"].append(np.quantile(ratio, 0.99)); stat["ratio_max"].append(ratio.max())
            # pairwise bound g(max(omega_i, omega_j)): the ratio to it must be < 1
            stat["pair_frac"].append(np.max(ratio / g_fun(np.maximum(om[:, None], om[None, :])[iu])))
            Ftr2, Fte2 = feats_logeuc(Rtr, Rte)
            acc["logeuc"].append(lr_acc(Ftr2, ytr, Fte2, yte))
            Ftr2, Fte2 = feats_euc(Rtr, Rte)
            acc["euc"].append(lr_acc(Ftr2, ytr, Fte2, yte))
            acc["mdm"].append(mdm_acc(Rtr, ytr, Rte, yte))
            # invariance: the same after the transformation R -> A R A^T
            A = rng.standard_normal((h, h)) + 2.0 * np.eye(h)
            RtrA = np.array([A @ X @ A.T for X in Rtr]); RteA = np.array([A @ X @ A.T for X in Rte])
            FtrA, FteA, _ = feats_tsm(RtrA, RteA)
            acc["tsm_affine"].append(lr_acc(FtrA, ytr, FteA, yte))
            stat["affine_err"].append(np.max(np.abs(pairwise_euc(FtrA)[iu] - De)) / De.max())
        row = {"s": s}
        for k, v in list(stat.items()) + list(acc.items()):
            row[k] = float(np.mean(v)); row[k + "_std"] = float(np.std(v))
        rows.append(row)
        row["affine_err_max"] = float(np.max(stat["affine_err"]))
        row["pair_frac_max"] = float(np.max(stat["pair_frac"]))
        print("s=%.2f rho_max=%.2f c(rho)=%.1f omega_max=%.2f g(omega)=%.1f "
              "ratio[min/med/q99/max]=%.4f/%.3f/%.3f/%.3f pair_frac=%.3f (max %.3f)  "
              "tsm=%.3f tsm_A=%.3f logeuc=%.3f euc=%.3f mdm=%.3f  affine_err=%.1e (max %.1e)" % (
              s, row["rho_max"], row["c_rho"], row["omega_max"], row["c_omega"], row["ratio_min"],
              row["ratio_med"], row["ratio_q99"], row["ratio_max"], row["pair_frac"], row["pair_frac_max"], row["tsm"],
              row["tsm_affine"], row["logeuc"], row["euc"], row["mdm"], row["affine_err"],
              row["affine_err_max"]), flush=True)
    return rows

def plot_radius(rows, path):
    rho = np.array([r["rho_max"] for r in rows])
    xlab = r"cloud radius $\rho = \max_i \delta_{\mathcal{R}}(\bar{\mathbf{R}}, \mathbf{R}_i)$"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for key, lab, mk in [("tsm", "TSM", "o"), ("mdm", "MDM", "s"),
                         ("logeuc", "log-Euclidean", "^"), ("euc", "Euclidean", "v")]:
        m = np.array([r[key] for r in rows]); sd = np.array([r[key + "_std"] for r in rows])
        a1.errorbar(rho, m, yerr=sd, marker=mk, capsize=2, label=lab)
    a1.set_xlabel(xlab); a1.set_ylabel("test accuracy"); a1.set_title("(a)")
    a1.set_ylim(0.45, 1.02); a1.legend(loc="lower right")
    rr = np.linspace(0.9 * rho.min(), 1.02 * rho.max(), 200)
    a2.plot(rr, c_rho(rr), "k--", label=r"bound $c(\rho)$")
    a2.plot(rho, [r["c_omega"] for r in rows], "k-.", marker="x", label=r"bound $g(\omega)$")
    for key, lab, mk in [("ratio_max", "maximum over pairs", "o"),
                         ("ratio_q99", "99th percentile", "^"),
                         ("ratio_med", "median over pairs", "s")]:
        m = np.array([r[key] for r in rows]); sd = np.array([r[key + "_std"] for r in rows])
        a2.errorbar(rho, m, yerr=sd, marker=mk, capsize=2, label=lab)
    a2.set_yscale("log"); a2.set_xlabel(xlab); a2.set_title("(b)")
    a2.set_ylabel(r"$\delta_{\mathcal{R}}(\mathbf{R}_i, \mathbf{R}_j)\,/\,\|\mathbf{p}_i - \mathbf{p}_j\|_2$")
    a2.legend(loc="upper left")
    fig.tight_layout(w_pad=1.5); savefig(fig, path); plt.close(fig)
# ---------------------------------------------------------------- experiment 1a
def experiment_sharpness(h=2, rho_list=(1.0, 2.0, 4.0), thetas=np.logspace(-3, np.log10(np.pi / 2), 40)):
    """Pair S(+-theta/2) = Q D Q^T with D = diag(rho/sqrt2, -rho/sqrt2, 0, ...): the norm rho and the spread
    sqrt2*rho do not depend on theta, and delta_R/||p_i-p_j|| -> c(rho) as theta -> 0."""
    def rot(th):
        Q = np.eye(h); Q[0, 0] = Q[1, 1] = np.cos(th); Q[0, 1] = -np.sin(th); Q[1, 0] = np.sin(th)
        return Q
    out = {}
    for rho in rho_list:
        D = np.zeros((h, h)); D[0, 0] = rho / np.sqrt(2.0); D[1, 1] = -rho / np.sqrt(2.0)
        vals = []
        for th in thetas:
            Q1, Q2 = rot(-th / 2), rot(th / 2)
            S1, S2 = Q1 @ D @ Q1.T, Q2 @ D @ Q2.T
            vals.append(geo_dist(mexp(S1), mexp(S2)) / np.linalg.norm(S1 - S2, "fro"))
        out[rho] = {"theta": thetas.tolist(), "ratio": vals, "c_rho": float(c_rho(rho))}
        print("rho=%.1f c(rho)=%.4f  ratio at theta=pi/2: %.4f, at theta=1e-3: %.6f" % (
              rho, out[rho]["c_rho"], vals[-1], vals[0]), flush=True)
    return out

def plot_sharpness(out, path):
    fig, ax = plt.subplots(figsize=(3.7, 2.8))
    for k, (rho, r) in enumerate(out.items()):
        col = "C%d" % k
        ax.plot(r["theta"], r["ratio"], color=col, label=r"$\rho = %g$" % float(rho))
        ax.axhline(r["c_rho"], ls="--", lw=0.9, color=col)
    ax.set_xscale("log"); ax.set_xlabel(r"rotation angle of the eigenbasis $\theta$")
    ax.set_ylabel(r"$\delta_{\mathcal{R}}(\mathbf{R}_i, \mathbf{R}_j)\,/\,\|\mathbf{p}_i - \mathbf{p}_j\|_2$")
    hs, ls = ax.get_legend_handles_labels()
    hs.append(Line2D([0], [0], ls="--", lw=0.9, color="gray")); ls.append(r"bound $c(\rho)$")
    ax.legend(hs, ls, loc="center left")
    fig.tight_layout(); savefig(fig, path); plt.close(fig)
# ---------------------------------------------------------------- experiment 2
def experiment_noise(h_list=(8, 16), s=0.5, n_train=100, n_test=500, seeds=10,
                     T_grid=(20, 40, 80, 160, 320, 640, 1280)):
    out, acc_true = {}, {}
    for h in h_list:
        rows = []
        for T in T_grid:
            accs, errs, bound = [], [], None
            for seed in range(seeds):
                rng = np.random.default_rng(100 + seed)
                R0, U = sample_problem(rng, h)
                Rtr, ytr = make_dataset(rng, h, s, n_train, R0, U)
                Rte, yte = make_dataset(rng, h, s, n_test, R0, U)
                def sample_cov(R):
                    Y = msqrt(R) @ rng.standard_normal((h, T))
                    return Y @ Y.T / T
                Rtr_h = np.array([sample_cov(X) for X in Rtr]); Rte_h = np.array([sample_cov(X) for X in Rte])
                Ftr, Fte, _ = feats_tsm(Rtr_h, Rte_h)
                accs.append(lr_acc(Ftr, ytr, Fte, yte))
                errs.append(np.mean([geo_dist(Xh, X) for Xh, X in zip(Rte_h[:100], Rte[:100])]))
            # bound of the theorem at confidence level 0.95
            t = np.sqrt(2.0 * np.log(2.0 / 0.05) / T); eps = np.sqrt(h / T) + t
            bound = 2 * np.sqrt(h) * eps / (1 - eps) if eps < 1 else np.nan
            rows.append({"T": T, "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                         "err": float(np.mean(errs)), "err_std": float(np.std(errs)), "bound": float(bound)})
            print("h=%d T=%d  acc=%.3f  err=%.3f  bound=%.3f" % (h, T, rows[-1]["acc"], rows[-1]["err"], bound), flush=True)
        out[h] = rows
        # accuracy with the true covariances on the same problems (same seeds and generation order)
        vals = []
        for seed in range(seeds):
            rng = np.random.default_rng(100 + seed)
            R0, U = sample_problem(rng, h)
            Rtr, ytr = make_dataset(rng, h, s, n_train, R0, U); Rte, yte = make_dataset(rng, h, s, n_test, R0, U)
            Ftr, Fte, _ = feats_tsm(Rtr, Rte); vals.append(lr_acc(Ftr, ytr, Fte, yte))
        acc_true[h] = float(np.mean(vals)); print("h=%d acc with true covariances = %.3f" % (h, acc_true[h]), flush=True)
    return out, acc_true

def plot_noise(out, acc_true, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.4))   # taller by the height of the legends below the panels
    for k, (h, rows) in enumerate(out.items()):
        col = "C%d" % k
        T = np.array([r["T"] for r in rows])
        a1.errorbar(T, [r["acc"] for r in rows], yerr=[r["acc_std"] for r in rows], color=col,
                    marker="o", capsize=2, label=r"$h = %d$" % int(h))
        a1.axhline(acc_true[h], ls=":", color=col)
        a2.errorbar(T, [r["err"] for r in rows], yerr=[r["err_std"] for r in rows], color=col,
                    marker="o", capsize=2)
        a2.plot(T, [r["bound"] for r in rows], "--", color=col)
    # legends below the panels: the colour encodes h, the line style encodes the quantity
    hs, ls = a1.get_legend_handles_labels()
    hs.append(Line2D([0], [0], ls=":", color="gray")); ls.append("true covariances")
    a1.set_xscale("log"); a1.set_xlabel(r"number of samples $\mathcal{T}$")
    a1.set_ylabel("TSM accuracy"); a1.set_title("(a)")
    a1.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)
    a2.set_xscale("log"); a2.set_yscale("log"); a2.set_xlabel(r"number of samples $\mathcal{T}$"); a2.set_title("(b)")
    a2.set_ylabel(r"$\delta_{\mathcal{R}}(\hat{\mathbf{R}}_i, \mathbf{R}_i)$")
    hs2 = [Line2D([0], [0], color="C%d" % k, marker="o") for k in range(len(out))]
    ls2 = [r"$h = %d$" % int(h) for h in out]
    hs2 += [Line2D([0], [0], color="gray", marker="o"), Line2D([0], [0], ls="--", color="gray")]
    ls2 += ["empirical error", "bound of the theorem"]
    a2.legend(hs2, ls2, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)
    fig.tight_layout(w_pad=1.5); savefig(fig, path); plt.close(fig)
if __name__ == "__main__":
    t0 = time.time()
    plot_style()
    rows = experiment_radius()
    plot_radius(rows, os.path.join(OUT, "exp_radius.pdf"))
    sharp = experiment_sharpness()
    plot_sharpness(sharp, os.path.join(OUT, "exp_sharp.pdf"))
    out, acc_true = experiment_noise()
    plot_noise(out, acc_true, os.path.join(OUT, "exp_noise.pdf"))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json"), "w", encoding="utf-8") as f:
        json.dump({"radius": rows, "sharpness": {str(k): v for k, v in sharp.items()},
                   "noise": {str(k): v for k, v in out.items()}, "acc_true": acc_true},
                  f, ensure_ascii=False, indent=1)
    print("done in %.1f s" % (time.time() - t0))
