# -*- coding: utf-8 -*-
"""Проверка утверждений статьи на стандартной ЭЭГ: BCI Competition IV 2a (BNCI 001-2014), воображение движения
левой и правой руки, 22 канала, 250 Гц, обучающие сессии A01T..A03T. Для каждого испытуемого:
  1) инвариантность (теорема 1): случайное невырожденное смешивание каналов A = G + 2I не меняет точность TSM
     и попарные расстояния между признаками;
  2) искажение расстояний (теорема 2): отношение геодезического расстояния к евклидову по всем парам проб
     против границ g(omega) и c(rho); точность TSM, MDM, log-Euclidean, Euclidean (5-кратная кросс-валидация),
     отдельно при обучении на 20 пробах каждого класса;
  3) шум оценки ковариации (теорема 3): геодезическое расстояние между ковариацией по первым T' отсчетам пробы
     и ковариацией по всему окну против sqrt(h(h+1)/T') и границы теоремы при уровне доверия 0.95;
     точность TSM при коротких окнах.
Данные: файлы A01T.mat, A02T.mat, A03T.mat (BNCI Horizon 2020, набор 001-2014) в папке bci2a рядом со скриптом;
зеркало: https://lampx.tugraz.at/~bci/database/001-2014/ . Запуск: python bci2a_check.py"""
import os, sys, json, math, warnings
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # synthetic_experiments.py
import synthetic_experiments as se

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BCI2A_DIR", os.path.join(HERE, "bci2a"))
for _s in (1, 2, 3):
    if not os.path.isfile(os.path.join(DATA, f"A0{_s}T.mat")):
        sys.exit(f"нет файла A0{_s}T.mat в {DATA}: скачайте с https://lampx.tugraz.at/~bci/database/001-2014/")
SUBJECTS = [1, 2, 3]
FS = 250
T0, T1 = 2.5, 6.0                       # окно пробы после начала пробы (подсказка на 2 с), 3,5 с = 875 отсчетов
CH22 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
CENTRAL9 = ["FC1", "FCz", "FC2", "C3", "C1", "Cz", "C2", "C4", "CPz"]
rng = np.random.default_rng(0)
SOS = butter(4, [8.0, 30.0], btype="bandpass", fs=FS, output="sos")


def load_subject(subj):
    m = loadmat(os.path.join(DATA, f"A0{subj}T.mat"), struct_as_record=False, squeeze_me=True)
    trials, labels = [], []
    for run in np.atleast_1d(m["data"]):
        y = np.atleast_1d(run.y)
        if y.size == 0:
            continue                                      # калибровочные записи без проб
        Xr = np.asarray(run.X, dtype=float)[:, :22]
        if np.isnan(Xr).any():                              # NaN in raw export -> zero before filtering
            Xr = np.nan_to_num(Xr, nan=0.0)
        X = sosfiltfilt(SOS, Xr, axis=0)   # (отсчеты, 22 канала ЭЭГ), мкВ
        art = np.atleast_1d(run.artifacts) if hasattr(run, "artifacts") else np.zeros_like(y)
        for k, (t, c) in enumerate(zip(np.atleast_1d(run.trial), y)):
            if c in (1, 2) and not art[k]:
                a, b = int(t + T0 * FS), int(t + T1 * FS)
                trials.append(X[a:b].T); labels.append(int(c) - 1)
    return np.array(trials), np.array(labels)


def covs(X, T=None):
    Xs = X if T is None else X[:, :, :T]
    Xs = Xs - Xs.mean(axis=2, keepdims=True)
    return np.einsum("nct,ndt->ncd", Xs, Xs) / (Xs.shape[2] - 1)


def bound_thm3(h, T, conf=0.95):
    t = math.sqrt(2.0 * math.log(2.0 / (1.0 - conf)) / T)
    eps = math.sqrt(h / T) + t
    return 2.0 * math.sqrt(h) * eps / (1.0 - eps) if eps < 1 else float("nan")


def cv_accuracies(R, y, seed=0, folds=5, n_train=None):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    acc = dict(tsm=[], mdm=[], logeuc=[], euc=[])
    for tr, te in skf.split(R, y):
        if n_train is not None:
            tr = np.concatenate([rng.choice(tr[y[tr] == c], n_train, replace=False) for c in (0, 1)])
        Ftr, Fte, _ = se.feats_tsm(R[tr], R[te]); acc["tsm"].append(se.lr_acc(Ftr, y[tr], Fte, y[te]))
        Ftr, Fte = se.feats_logeuc(R[tr], R[te]); acc["logeuc"].append(se.lr_acc(Ftr, y[tr], Fte, y[te]))
        Ftr, Fte = se.feats_euc(R[tr], R[te]); acc["euc"].append(se.lr_acc(Ftr, y[tr], Fte, y[te]))
        acc["mdm"].append(se.mdm_acc(R[tr], y[tr], R[te], y[te]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def distortion_stats(R):
    Rm = se.karcher_mean(R)
    S = np.array([se.log_at(Rm, Ri) for Ri in R]); P = np.array([se.vec_sqrt2(Si) for Si in S])
    rho = float(max(se.geo_dist(Rm, Ri) for Ri in R))
    om = np.array([se.spread(Si) for Si in S]); omega = float(om.max())
    D = se.pairwise_geo(R); E = se.pairwise_euc(P)
    iu = np.triu_indices(len(R), 1); ratios = D[iu] / E[iu]
    pair_bound = se.g_fun(np.maximum(om[iu[0]], om[iu[1]]))
    return dict(rho=rho, c_rho=float(se.c_rho(rho)), omega=omega, g_omega=float(se.g_fun(omega)),
                ratio_min=float(ratios.min()), ratio_med=float(np.median(ratios)), ratio_q99=float(np.quantile(ratios, 0.99)),
                ratio_max=float(ratios.max()), pairwise_bound_ok=bool(np.all(ratios <= pair_bound * (1 + 1e-6)))), P


results = {}
for subj in SUBJECTS:
    X, y = load_subject(subj)
    c9 = [CH22.index(c) for c in CENTRAL9]
    res = dict(n_trials=int(len(y)), n_per_class=[int((y == 0).sum()), int((y == 1).sum())], T=int(X.shape[2]))
    for tag, Xc in (("h22", X), ("h9", X[:, c9, :])):
        h = Xc.shape[1]; R = covs(Xc); r = dict(h=h)
        r["acc"] = cv_accuracies(R, y)
        r["acc_20"] = {k: float(np.mean([cv_accuracies(R, y, seed=s, n_train=20)[k] for s in range(3)])) for k in ("tsm", "mdm", "logeuc", "euc")}
        ds, P = distortion_stats(R); r.update(ds)
        A = rng.standard_normal((h, h)) + 2.0 * np.eye(h)
        RA = np.einsum("ij,njk,lk->nil", A, R, A)
        r["acc_affine_tsm"] = cv_accuracies(RA, y)["tsm"]
        _, PA = distortion_stats(RA)
        D0 = se.pairwise_euc(P); DA = se.pairwise_euc(PA)
        r["affine_dist_relerr"] = float(np.max(np.abs(DA - D0)) / np.max(D0))
        noise = []
        for Tp in (50, 100, 200, 400, 600):
            Rp = covs(Xc, Tp)
            err = float(np.mean([se.geo_dist(Rp[i], R[i]) for i in range(len(R))]))
            noise.append(dict(T=Tp, err=err, iid=math.sqrt(h * (h + 1) / Tp), bound=bound_thm3(h, Tp), acc_tsm=cv_accuracies(Rp, y)["tsm"]))
        r["noise"] = noise
        res[tag] = r
    results[subj] = res
    a, b = res["h22"]["acc"], res["h9"]["acc"]
    print(f"A0{subj}T: проб {res['n_trials']} ({res['n_per_class']}), T = {res['T']} | h=22: TSM {a['tsm']:.3f} MDM {a['mdm']:.3f} logE {a['logeuc']:.3f} Euc {a['euc']:.3f}"
          f" | h=9: TSM {b['tsm']:.3f} MDM {b['mdm']:.3f} logE {b['logeuc']:.3f} Euc {b['euc']:.3f}", flush=True)

with open(os.path.join(HERE, "bci2a_check_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)


def mean(path):
    vals = []
    for s in results.values():
        v = s
        for k in path: v = v[k]
        vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


print("\n==== среднее по испытуемым ====")
for tag in ("h22", "h9"):
    print(f"-- {tag}")
    for m in ("tsm", "mdm", "logeuc", "euc"):
        mu, sd = mean((tag, "acc", m)); mu2, _ = mean((tag, "acc_20", m))
        print(f"   точность {m:7s}: {mu:.3f} ± {sd:.3f} | при 20 пробах на класс: {mu2:.3f}")
    mu, sd = mean((tag, "acc_affine_tsm")); print(f"   TSM после смешивания A: {mu:.3f} ± {sd:.3f}")
    print("   макс. относит. изменение попарных расстояний после A: %.1e" % max(s[tag]["affine_dist_relerr"] for s in results.values()))
    for k in ("rho", "c_rho", "omega", "g_omega", "ratio_min", "ratio_med", "ratio_q99", "ratio_max"):
        mu, sd = mean((tag, k)); print(f"   {k:10s}: {mu:.3g} ± {sd:.2g}")
    print("   попарная граница g(max(omega_i, omega_j)) выполнена у всех пар:", all(s[tag]["pairwise_bound_ok"] for s in results.values()))
    print("   шум оценки: T' | ошибка | sqrt(h(h+1)/T') | ошибка/iid | граница 0.95 | точность TSM")
    for k in range(5):
        e, _ = mean((tag, "noise", k, "err")); a, _ = mean((tag, "noise", k, "acc_tsm"))
        n0 = results[SUBJECTS[0]][tag]["noise"][k]
        print(f"      {n0['T']:4d} | {e:6.2f} | {n0['iid']:6.2f} | {e/n0['iid']:5.2f} | {n0['bound']:8.2f} | {a:.3f}")
