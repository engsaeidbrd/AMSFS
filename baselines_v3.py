# -*- coding: utf-8 -*-
"""
baselines_v3.py   (NEW file - AMSFS.py is NOT modified)
=========================================================
Real runs of the six additional baselines of Table 5:

    PSO | GWO | WOA | HHO | DT+Filter | Filter+SVM

They are run under EXACTLY the same protocol that AMSFS.py (run_comparison)
uses for RF-RFE-MI ("NoGurobi"), HFIA* and AMSFS:

  * one stratified 80/20 split, random_state = 42
  * StandardScaler fitted on the training part only
  * feature selection uses the training part only (KNN-CV, k=5,
    LOO if n_train <= 60 else stratified 4-fold  -> knn_cv_score of AMSFS.py)
  * final accuracy = evaluate_final() of AMSFS.py (SVM / KNN / RF, best one)
  * the same 14 datasets

Search budget of PSO / GWO / WOA / HHO is equal to HFIA*:
  population 14, 90 iterations (max. 14 x 91 = 1274 fitness evaluations),
  Fisher pre-filter min(220, d // 4), and the SAME fitness function as HFIA*:
  fitness = 1 - KNN-CV accuracy + 0.02 * (n_selected / n_filtered)

At the end the script merges the new columns with the three columns that
AMSFS.py already produced (Comparison_NoGurobi_HFIA_AMSFS.csv) and prints /
saves the complete Table 5 (accuracy), Table 6 (features), the running
times and the win / tie / loss counts of AMSFS.

Usage (from the folder that contains this file):
    python baselines_v3.py
Optional:
    python baselines_v3.py --datasets ALLAML colon      (only some datasets)
    python baselines_v3.py --rerun-main                 (run the 3 old methods again)
    python baselines_v3.py --data-dir data/Dataset_ASU
"""

import os
import sys
import time
import math
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# Paths (same as in ablation_study.py / chart.py)
# ----------------------------------------------------------------------------
AMSFS_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Dataset_ASU")
MAIN_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "Comparison_NoGurobi_HFIA_AMSFS.csv")

DATASETS = [
    'ALLAML', 'leukemia', 'Prostate_GE', 'lung', 'CARCINOM',
    'CLL_SUB_111', 'TOX_171', 'lymphoma', 'lung_discrete',
    'GLI_85', 'SMK_CAN_187', 'colon', 'nci9', 'GLIOMA'
]

# method key -> (display name, prefix used in CSV columns)
NEW_METHODS = [
    ('PSO', 'PSO'), ('GWO', 'GWO'), ('WOA', 'WOA'), ('HHO', 'HHO'),
    ('DTFilter', 'DT+Filter'), ('FilterSVM', 'Filter+SVM')
]

_LB, _UB = -6.0, 6.0          # bounds of the continuous (logit) positions


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--amsfs-path', default=AMSFS_PATH)
    p.add_argument('--data-dir', default=DATA_DIR)
    p.add_argument('--main-csv', default=MAIN_CSV)
    p.add_argument('--datasets', nargs='*', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pop', type=int, default=14)
    p.add_argument('--iters', type=int, default=90)
    p.add_argument('--rerun-main', action='store_true',
                   help='run RF-RFE-MI, HFIA* and AMSFS again (slow) instead of reading the CSV')
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
    return p.parse_args()


ARGS = _parse_args()
sys.path.append(ARGS.amsfs_path)
from AMSFS import (fisher_score, knn_cv_score, load_dataset,      # noqa: E402
                   evaluate_final, run_comparison)


# =============================================================================
# Binary wrapper metaheuristics (S-shaped transfer function)
# =============================================================================
class _BinarySearch(object):
    """
    Common part: Fisher pre-filter (as HFIA*), binarisation of a continuous
    position by the S-shaped (sigmoid) transfer function, the HFIA* fitness,
    an evaluation budget and best-so-far bookkeeping.
    """

    def __init__(self, pop_size=14, max_iter=90, fisher_keep=220, random_state=42):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.fisher_keep = fisher_keep
        self.random_state = random_state
        self.selected_indices_ = None
        self.n_evals_ = 0

    # ---------------- helpers ----------------
    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def _binarize(self, pos):
        p = self._sigmoid(pos)
        mask = self.rng.rand(self.nf) < p
        if mask.sum() < 3:
            mask[np.argsort(p)[::-1][:3]] = True
        return mask

    def _fitness(self, mask):
        ns = int(mask.sum())
        if ns < 3 or ns > max(5, int(0.45 * self.nf)):
            return 1.5
        key = mask.tobytes()
        if key in self._cache:
            return self._cache[key]
        acc = knn_cv_score(self.Xf, self.y, np.where(mask)[0], self.random_state)
        val = 1.0 - acc + 0.02 * (ns / float(self.nf))
        self._cache[key] = val
        self.n_evals_ += 1
        return val

    def _evaluate(self, pos):
        """continuous position -> (binary mask, fitness); keeps best-so-far."""
        mask = self._binarize(pos)
        fit = self._fitness(mask)
        if fit < self.best_fit:
            self.best_fit = fit
            self.best_mask = mask.copy()
        return mask, fit

    def _budget_left(self):
        return self.n_evals_ < self.cap

    def _init_positions(self):
        # initial logits ~ N(-1.7, 1): about 15-20 % of the genes are selected
        return self.rng.normal(-1.7, 1.0, (self.pop_size, self.nf))

    # ---------------- API ----------------
    def fit(self, X, y):
        d = X.shape[1]
        scores = fisher_score(X, y)
        top = np.argsort(scores)[::-1][:min(self.fisher_keep, d)]
        self.Xf = X[:, top]
        self.y = y
        self.nf = self.Xf.shape[1]
        self.rng = np.random.RandomState(self.random_state)
        self._cache = {}
        self.n_evals_ = 0
        self.cap = self.pop_size * (self.max_iter + 1)
        self.best_fit = np.inf
        self.best_mask = None

        self._search()

        mask = self.best_mask
        if mask is None or mask.sum() < 3:
            mask = np.zeros(self.nf, dtype=bool)
            mask[np.argsort(scores[top])[::-1][:8]] = True
        self.selected_indices_ = top[mask]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _search(self):
        raise NotImplementedError


class BinaryPSO(_BinarySearch):
    """Binary PSO (Kennedy & Eberhart, 1997): sigmoid of the velocity gives
    the probability of a bit being 1; w: 0.9 -> 0.4, c1 = c2 = 2, |v| <= 4."""

    def _search(self):
        N, nf, T = self.pop_size, self.nf, self.max_iter
        v = self.rng.normal(-1.7, 0.5, (N, nf))
        X = np.zeros((N, nf))
        fit = np.zeros(N)
        for i in range(N):
            m, fit[i] = self._evaluate(v[i])
            X[i] = m
        pbest, pfit = X.copy(), fit.copy()
        gbest = self.best_mask.astype(float).copy()
        for t in range(T):
            if not self._budget_left():
                break
            w = 0.9 - 0.5 * t / float(T)
            r1, r2 = self.rng.rand(N, nf), self.rng.rand(N, nf)
            v = w * v + 2.0 * r1 * (pbest - X) + 2.0 * r2 * (gbest - X)
            v = np.clip(v, -4.0, 4.0)
            for i in range(N):
                if not self._budget_left():
                    break
                m, f = self._evaluate(v[i])
                X[i] = m
                if f < pfit[i]:
                    pfit[i], pbest[i] = f, X[i].copy()
            gbest = self.best_mask.astype(float).copy()


class BinaryGWO(_BinarySearch):
    """Binary GWO (Mirjalili et al., 2014): standard alpha/beta/delta update on
    continuous positions, sigmoid transfer function to obtain the bits."""

    def _search(self):
        N, nf, T = self.pop_size, self.nf, self.max_iter
        pos = self._init_positions()
        fit = np.zeros(N)
        leaders = [np.zeros(nf), np.zeros(nf), np.zeros(nf)]
        lscore = [np.inf, np.inf, np.inf]

        def update_leaders(p, f):
            if f < lscore[0]:
                lscore[2], leaders[2] = lscore[1], leaders[1].copy()
                lscore[1], leaders[1] = lscore[0], leaders[0].copy()
                lscore[0], leaders[0] = f, p.copy()
            elif f < lscore[1]:
                lscore[2], leaders[2] = lscore[1], leaders[1].copy()
                lscore[1], leaders[1] = f, p.copy()
            elif f < lscore[2]:
                lscore[2], leaders[2] = f, p.copy()

        for i in range(N):
            _, fit[i] = self._evaluate(pos[i])
            update_leaders(pos[i], fit[i])
        for t in range(T):
            if not self._budget_left():
                break
            a = 2.0 * (1.0 - t / float(T))
            for i in range(N):
                if not self._budget_left():
                    break
                parts = []
                for L in leaders:
                    A = 2.0 * a * self.rng.rand(nf) - a
                    C = 2.0 * self.rng.rand(nf)
                    parts.append(L - A * np.abs(C * L - pos[i]))
                pos[i] = np.clip(np.mean(parts, axis=0), _LB, _UB)
                _, fit[i] = self._evaluate(pos[i])
                update_leaders(pos[i], fit[i])


class BinaryWOA(_BinarySearch):
    """Binary WOA (Mirjalili & Lewis, 2016): encircling / random search /
    spiral update (b = 1) on continuous positions + sigmoid transfer."""

    def _search(self):
        N, nf, T = self.pop_size, self.nf, self.max_iter
        pos = self._init_positions()
        fit = np.zeros(N)
        best_pos, best_f = pos[0].copy(), np.inf
        for i in range(N):
            _, fit[i] = self._evaluate(pos[i])
            if fit[i] < best_f:
                best_f, best_pos = fit[i], pos[i].copy()
        ar = np.arange(nf)
        for t in range(T):
            if not self._budget_left():
                break
            a = 2.0 - 2.0 * t / float(T)
            a2 = -1.0 - t / float(T)
            for i in range(N):
                if not self._budget_left():
                    break
                A = 2.0 * a * self.rng.rand(nf) - a
                C = 2.0 * self.rng.rand(nf)
                l = (a2 - 1.0) * self.rng.rand(nf) + 1.0
                p = self.rng.rand(nf)
                rand_pos = pos[self.rng.randint(0, N, size=nf), ar]
                enc = best_pos - A * np.abs(C * best_pos - pos[i])
                exp = rand_pos - A * np.abs(C * rand_pos - pos[i])
                spi = np.abs(best_pos - pos[i]) * np.exp(l) * np.cos(2.0 * np.pi * l) + best_pos
                new = np.where(p < 0.5, np.where(np.abs(A) < 1.0, enc, exp), spi)
                pos[i] = np.clip(new, _LB, _UB)
                _, fit[i] = self._evaluate(pos[i])
                if fit[i] < best_f:
                    best_f, best_pos = fit[i], pos[i].copy()


class BinaryHHO(_BinarySearch):
    """Binary HHO (Heidari et al., 2019): exploration, soft / hard besiege,
    progressive rapid dives with Levy flight; sigmoid transfer function."""

    def _levy(self, dim, beta=1.5):
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2.0) /
                 (math.gamma((1 + beta) / 2.0) * beta * 2 ** ((beta - 1) / 2.0))) ** (1.0 / beta)
        u = self.rng.randn(dim) * sigma
        v = self.rng.randn(dim)
        return 0.01 * u / (np.abs(v) ** (1.0 / beta) + 1e-12)

    def _search(self):
        N, nf, T = self.pop_size, self.nf, self.max_iter
        pos = self._init_positions()
        fit = np.zeros(N)
        rabbit, rfit = pos[0].copy(), np.inf
        for i in range(N):
            _, fit[i] = self._evaluate(pos[i])
            if fit[i] < rfit:
                rfit, rabbit = fit[i], pos[i].copy()
        for t in range(T):
            if not self._budget_left():
                break
            for i in range(N):
                if not self._budget_left():
                    break
                E = 2.0 * (2.0 * self.rng.rand() - 1.0) * (1.0 - t / float(T))
                if abs(E) >= 1.0:                                   # exploration
                    k = self.rng.randint(N)
                    if self.rng.rand() >= 0.5:
                        new = pos[k] - self.rng.rand() * np.abs(pos[k] - 2.0 * self.rng.rand() * pos[i])
                    else:
                        new = (rabbit - pos.mean(axis=0)) - self.rng.rand() * (
                            _LB + self.rng.rand() * (_UB - _LB))
                    new = np.clip(new, _LB, _UB)
                    _, f = self._evaluate(new)
                    pos[i], fit[i] = new, f
                else:                                               # exploitation
                    r = self.rng.rand()
                    J = 2.0 * (1.0 - self.rng.rand())
                    if r >= 0.5 and abs(E) >= 0.5:                  # soft besiege
                        new = (rabbit - pos[i]) - E * np.abs(J * rabbit - pos[i])
                        new = np.clip(new, _LB, _UB)
                        _, f = self._evaluate(new)
                        pos[i], fit[i] = new, f
                    elif r >= 0.5:                                  # hard besiege
                        new = rabbit - E * np.abs(rabbit - pos[i])
                        new = np.clip(new, _LB, _UB)
                        _, f = self._evaluate(new)
                        pos[i], fit[i] = new, f
                    else:                                           # rapid dives
                        if abs(E) >= 0.5:
                            Y = rabbit - E * np.abs(J * rabbit - pos[i])
                        else:
                            Y = rabbit - E * np.abs(J * rabbit - pos.mean(axis=0))
                        Y = np.clip(Y, _LB, _UB)
                        _, fy = self._evaluate(Y)
                        if fy < fit[i]:
                            pos[i], fit[i] = Y, fy
                        elif self._budget_left():
                            Z = np.clip(Y + self.rng.rand(nf) * self._levy(nf), _LB, _UB)
                            _, fz = self._evaluate(Z)
                            if fz < fit[i]:
                                pos[i], fit[i] = Z, fz
                if fit[i] < rfit:
                    rfit, rabbit = fit[i], pos[i].copy()


# =============================================================================
# Classical filter-based baselines (same subset size as RF-RFE-MI)
# =============================================================================
class FilterSVM(object):
    """ANOVA F-test filter (SelectKBest, top-k0) followed by SVM-RFE
    (linear SVC, step 0.12) down to n_features."""

    def __init__(self, n_features=40, prefilter=500, random_state=42):
        self.n_features = n_features
        self.prefilter = prefilter
        self.random_state = random_state
        self.selected_indices_ = None

    def fit(self, X, y):
        k0 = min(self.prefilter, X.shape[1])
        skb = SelectKBest(f_classif, k=k0).fit(X, y)
        keep = np.where(skb.get_support())[0]
        n = max(5, min(self.n_features, k0))
        rfe = RFE(SVC(kernel='linear', C=1.0, random_state=self.random_state),
                  n_features_to_select=n, step=0.12)
        rfe.fit(X[:, keep], y)
        self.selected_indices_ = keep[np.where(rfe.support_)[0]]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class DTFilter(object):
    """ANOVA F-test filter (SelectKBest, top-k0) followed by the importance
    ranking of a decision tree; the n_features most important are kept
    (ties are broken by the F-score)."""

    def __init__(self, n_features=40, prefilter=500, random_state=42):
        self.n_features = n_features
        self.prefilter = prefilter
        self.random_state = random_state
        self.selected_indices_ = None

    def fit(self, X, y):
        k0 = min(self.prefilter, X.shape[1])
        skb = SelectKBest(f_classif, k=k0).fit(X, y)
        keep = np.where(skb.get_support())[0]
        fs = np.nan_to_num(skb.scores_[keep])
        dt = DecisionTreeClassifier(random_state=self.random_state).fit(X[:, keep], y)
        imp = dt.feature_importances_
        order = np.lexsort((-fs, -imp))          # primary: importance, secondary: F-score
        n = max(5, min(self.n_features, k0))
        self.selected_indices_ = keep[order[:n]]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# =============================================================================
# Experiment
# =============================================================================
def _split_like_amsfs(X, y, seed):
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te


def _make_selector(key, X_tr, seed, pop, iters):
    fk = min(220, X_tr.shape[1] // 4)               # same pre-filter as HFIA*
    n_target = min(55, max(25, X_tr.shape[0] // 2))  # same size rule as RF-RFE-MI
    if key == 'PSO':
        return BinaryPSO(pop, iters, fk, seed)
    if key == 'GWO':
        return BinaryGWO(pop, iters, fk, seed)
    if key == 'WOA':
        return BinaryWOA(pop, iters, fk, seed)
    if key == 'HHO':
        return BinaryHHO(pop, iters, fk, seed)
    if key == 'DTFilter':
        return DTFilter(n_target, 500, seed)
    if key == 'FilterSVM':
        return FilterSVM(n_target, 500, seed)
    raise ValueError(key)


def run_new_baselines(data_dir, datasets, seed, pop, iters):
    rows = []
    total = len(datasets)
    for i, ds in enumerate(datasets, 1):
        print("\n" + "-" * 70)
        print(" [%d/%d]  Dataset: %s" % (i, total, ds))
        print("-" * 70)
        X, y = load_dataset(data_dir, ds)
        if X is None:
            print("  Status: SKIPPED (file not found or invalid)")
            continue
        print("  Shape: %d samples x %d features | Classes: %d"
              % (X.shape[0], X.shape[1], len(np.unique(y))))
        X_tr, X_te, y_tr, y_te = _split_like_amsfs(X, y, seed)

        rec = {'Dataset': ds}
        for key, disp in NEW_METHODS:
            print("  -> Running %s ..." % disp, end=" ", flush=True)
            t0 = time.time()
            sel = _make_selector(key, X_tr, seed, pop, iters)
            Xtr = sel.fit_transform(X_tr, y_tr)
            Xte = sel.transform(X_te)
            acc, f1, clf = evaluate_final(Xtr, Xte, y_tr, y_te, seed)
            dt = time.time() - t0
            rec[key + '_Acc'] = acc
            rec[key + '_F1'] = f1
            rec[key + '_Feat'] = len(sel.selected_indices_)
            rec[key + '_Time'] = dt
            print("Acc=%.2f%%  Feat=%d  (%.1fs)" % (acc * 100, len(sel.selected_indices_), dt))
        rows.append(rec)
    return pd.DataFrame(rows)


def _load_main(main_csv, data_dir, seed, rerun):
    if (not rerun) and os.path.exists(main_csv):
        print("Reading the three existing columns from", main_csv)
        return pd.read_csv(main_csv)
    if not rerun:
        print("WARNING: %s not found -> running RF-RFE-MI, HFIA*, AMSFS now (slow)." % main_csv)
    rows = run_comparison(data_dir, seed=seed)
    return pd.DataFrame(rows)


def print_and_save(main_df, new_df, out_dir):
    df = main_df.merge(new_df, on='Dataset', how='inner')
    order = [('NoGurobi', 'RF-RFE-MI'), ('HFIA', 'HFIA*'), ('PSO', 'PSO'), ('GWO', 'GWO'),
             ('WOA', 'WOA'), ('HHO', 'HHO'), ('DTFilter', 'DT+Filter'),
             ('FilterSVM', 'Filter+SVM'), ('AMSFS', 'AMSFS')]
    w = 11
    print("\n" + "=" * 125)
    print(" TABLE 5 - test accuracy (%%)  [single stratified 80/20 split, seed %d]" % ARGS.seed)
    print("=" * 125)
    print("%-15s" % "Dataset" + "".join("%*s" % (w, d) for _, d in order))
    for _, r in df.iterrows():
        print("%-15s" % r['Dataset'] + "".join("%*.2f" % (w, r[k + '_Acc'] * 100) for k, _ in order))
    print("-" * 125)
    avg = {k: df[k + '_Acc'].mean() * 100 for k, _ in order}
    print("%-15s" % "AVERAGE" + "".join("%*.2f" % (w, avg[k]) for k, _ in order))

    print("\n TABLE 6 - number of selected features")
    print("%-15s" % "Dataset" + "".join("%*s" % (w, d) for _, d in order))
    for _, r in df.iterrows():
        print("%-15s" % r['Dataset'] + "".join("%*d" % (w, int(r[k + '_Feat'])) for k, _ in order))
    print("%-15s" % "AVERAGE" + "".join("%*.1f" % (w, df[k + '_Feat'].mean()) for k, _ in order))

    print("\n RUNNING TIME (seconds, selection + final classifiers)")
    print("%-15s" % "Dataset" + "".join("%*s" % (w, d) for _, d in order))
    for _, r in df.iterrows():
        print("%-15s" % r['Dataset'] + "".join("%*.1f" % (w, r[k + '_Time']) for k, _ in order))
    print("%-15s" % "AVERAGE" + "".join("%*.2f" % (w, df[k + '_Time'].mean()) for k, _ in order))

    print("\n AMSFS versus every method (datasets): Win / Tie / Loss  and gap of the averages")
    for k, d in order:
        if k == 'AMSFS':
            continue
        diff = df['AMSFS_Acc'].values - df[k + '_Acc'].values
        win = int((diff > 1e-9).sum())
        tie = int((np.abs(diff) <= 1e-9).sum())
        loss = int((diff < -1e-9).sum())
        print("   vs %-11s  W/T/L = %2d/%2d/%2d   average gap = %+6.2f points"
              % (d, win, tie, loss, avg['AMSFS'] - avg[k]))
    print("=" * 125)

    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, 'Table5_full_v3.csv')
    df.to_csv(full, index=False)
    new_csv = os.path.join(out_dir, 'Baselines_new_six_v3.csv')
    new_df.to_csv(new_csv, index=False)
    print("Saved:", full)
    print("Saved:", new_csv)


def main():
    datasets = ARGS.datasets if ARGS.datasets else DATASETS
    print("=" * 100)
    print(" REAL RUNS: PSO | GWO | WOA | HHO | DT+Filter | Filter+SVM   (same protocol as AMSFS.py)")
    print(" population %d, iterations %d, seed %d" % (ARGS.pop, ARGS.iters, ARGS.seed))
    print("=" * 100)
    new_df = run_new_baselines(ARGS.data_dir, datasets, ARGS.seed, ARGS.pop, ARGS.iters)
    if new_df.empty:
        print("No dataset could be loaded. Check --data-dir.")
        return
    new_df.to_csv(os.path.join(ARGS.out_dir, 'Baselines_new_six_v3_partial.csv'), index=False)
    main_df = _load_main(ARGS.main_csv, ARGS.data_dir, ARGS.seed, ARGS.rerun_main)
    print_and_save(main_df, new_df, ARGS.out_dir)


if __name__ == "__main__":
    main()
