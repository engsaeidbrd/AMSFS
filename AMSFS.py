"""
Comparative Evaluation Framework
NoGurobi  |  HFIA  |  AMSFS (Proposed)
Academic single-run comparison on high-dimensional microarray datasets
"""

import os
import numpy as np
import scipy.io
import pandas as pd
import time
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


# =============================================================================
# Utility functions
# =============================================================================

def fisher_score(X, y):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    scores = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        mu = np.mean(f)
        between = within = 0.0
        for c in classes:
            idx = np.where(y == c)[0]
            nk = len(idx)
            if nk > 0:
                between += nk * (np.mean(f[idx]) - mu) ** 2
                within += nk * np.var(f[idx])
        scores[i] = between / within if within > 1e-8 else 0.0
    return scores


def get_cv(y, seed=42):
    n = len(y)
    if n <= 60:
        return LeaveOneOut()
    return StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)


def knn_cv_score(X, y, indices, seed=42):
    if len(indices) < 2:
        return 0.0
    try:
        cv = get_cv(y, seed)
        scores = cross_val_score(
            KNeighborsClassifier(n_neighbors=5),
            X[:, indices], y, cv=cv, scoring='accuracy', n_jobs=-1
        )
        return float(np.mean(scores))
    except:
        return 0.0


def load_dataset(data_dir, name):
    path = os.path.join(data_dir, f"{name}.mat")
    if not os.path.exists(path):
        return None, None
    try:
        data = scipy.io.loadmat(path)
        X = y = None
        for key, val in data.items():
            if key.startswith('__') or not hasattr(val, 'shape'):
                continue
            if len(val.shape) == 2 and val.shape[1] > 5 and X is None:
                X = val.astype(np.float64)
            elif (len(val.shape) == 1 or min(val.shape) == 1) and y is None:
                y = val.flatten()
        if X is None or y is None:
            return None, None
        y = y.astype(int)
        uniq = np.unique(y)
        if not np.array_equal(uniq, np.arange(len(uniq))):
            mapping = {o: n for n, o in enumerate(uniq)}
            y = np.array([mapping[v] for v in y])
        return X, y
    except:
        return None, None


def evaluate_final(X_tr, X_te, y_tr, y_te, seed=42):
    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw = dict(zip(np.unique(y_tr), cw))
    models = [
        ('SVM', SVC(kernel='rbf', C=1.5, gamma='scale', class_weight=cw, random_state=seed)),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('RF', RandomForestClassifier(n_estimators=130, max_depth=9,
                                      random_state=seed, class_weight='balanced'))
    ]
    best = (0.0, 0.0, 'None')
    for name, clf in models:
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, average='weighted')
        if acc > best[0]:
            best = (acc, f1, name)
    return best


# =============================================================================
# Method 1: NoGurobi
# =============================================================================

class NoGurobi:
    """Classical filter/wrapper selectors (RFE, RF importance, Mutual Information)."""

    def __init__(self, n_features=40, random_state=42):
        self.n_features = n_features
        self.random_state = random_state
        self.selected_indices_ = None

    def fit(self, X, y):
        n = max(5, min(self.n_features, X.shape[1]))
        candidates = []

        # RFE
        try:
            rfe = RFE(SVC(kernel='linear', C=1.0, random_state=self.random_state),
                      n_features_to_select=n, step=0.12)
            rfe.fit(X, y)
            idx = np.where(rfe.support_)[0]
            candidates.append((knn_cv_score(X, y, idx, self.random_state), idx))
        except:
            pass

        # RF importance
        try:
            rf = RandomForestClassifier(n_estimators=130, random_state=self.random_state, n_jobs=-1)
            rf.fit(X, y)
            idx = np.argsort(rf.feature_importances_)[-n:]
            candidates.append((knn_cv_score(X, y, idx, self.random_state), idx))
        except:
            pass

        # Mutual Information
        try:
            skb = SelectKBest(mutual_info_classif, k=n)
            skb.fit(X, y)
            idx = np.where(skb.get_support())[0]
            candidates.append((knn_cv_score(X, y, idx, self.random_state), idx))
        except:
            pass

        if not candidates:
            self.selected_indices_ = np.arange(min(n, X.shape[1]))
        else:
            candidates.sort(key=lambda t: t[0], reverse=True)
            self.selected_indices_ = candidates[0][1]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# =============================================================================
# Method 2: HFIA
# =============================================================================

class HFIA:
    """
    Hybrid Feature selection with Immune Algorithm principles.
    Fisher pre-filter + clonal-style binary search.
    """

    def __init__(self, pop_size=14, max_iter=90, fisher_keep=220, random_state=42):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.fisher_keep = fisher_keep
        self.random_state = random_state
        self.selected_indices_ = None
        np.random.seed(random_state)

    def fit(self, X, y):
        n_features = X.shape[1]
        scores = fisher_score(X, y)
        top = np.argsort(scores)[::-1][:min(self.fisher_keep, n_features)]
        Xf = X[:, top]
        nf = Xf.shape[1]

        # Binary population
        pop = np.zeros((self.pop_size, nf))
        for i in range(self.pop_size):
            k = max(5, int(nf * (0.10 + 0.15 * np.random.rand())))
            pop[i, np.random.choice(nf, k, replace=False)] = 1.0

        def fitness(ind):
            sel = ind > 0.5
            ns = int(np.sum(sel))
            if ns < 3 or ns > max(5, int(0.45 * nf)):
                return 1.5
            return 1.0 - knn_cv_score(Xf, y, np.where(sel)[0], self.random_state) + 0.02 * (ns / nf)

        fit = np.array([fitness(ind) for ind in pop])
        best_i = np.argmin(fit)
        best = pop[best_i].copy()
        best_f = fit[best_i]

        for t in range(self.max_iter):
            new_pop = []
            order = np.argsort(fit)
            # Elitism
            for i in range(min(3, self.pop_size)):
                new_pop.append(pop[order[i]].copy())
            while len(new_pop) < self.pop_size:
                p1, p2 = pop[np.random.randint(0, self.pop_size)], pop[np.random.randint(0, self.pop_size)]
                child = np.where(np.random.rand(nf) < 0.5, p1, p2)
                # Mutation
                mut = np.random.rand(nf) < (0.15 * (1 - t / self.max_iter) + 0.03)
                child[mut] = 1.0 - child[mut]
                if np.sum(child > 0.5) == 0:
                    child[np.random.randint(0, nf)] = 1.0
                new_pop.append(child)
            pop = np.array(new_pop)
            fit = np.array([fitness(ind) for ind in pop])
            if np.min(fit) < best_f:
                best_i = np.argmin(fit)
                best = pop[best_i].copy()
                best_f = fit[best_i]

        mask = best > 0.5
        if np.sum(mask) < 3:
            mask[np.argsort(scores[top])[::-1][:8]] = True
        self.selected_indices_ = top[mask]
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# =============================================================================
# Method 3: AMSFS (Proposed)
# =============================================================================

class AMSFS:
    """
    Adaptive Multi-Strategy Feature Selection.
    Combines ranking, recursive elimination, mutual information and
    an evolutionary search with controlled accuracy-aware pruning.
    """

    def __init__(self, max_drop=0.025, min_features=12, random_state=42, verbose=False):
        self.max_drop = max_drop
        self.min_features = min_features
        self.random_state = random_state
        self.verbose = verbose
        self.selected_indices_ = None
        self.n_features_selected_ = None
        np.random.seed(random_state)

    def _s_shaped(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -6.5, 6.5)))

    def _evolutionary(self, X, y):
        n_samples, n_features = X.shape
        ratio = n_features / max(n_samples, 1)

        if ratio > 45 or n_samples < 65:
            pop_size, max_iter, beta = 32, 100, 0.035
            fisher_keep = min(700, max(280, int(0.25 * n_features)))
            max_feat_ratio, local_steps = 0.42, 35
        elif ratio > 18:
            pop_size, max_iter, beta = 28, 85, 0.05
            fisher_keep = min(500, max(220, int(0.20 * n_features)))
            max_feat_ratio, local_steps = 0.35, 28
        else:
            pop_size, max_iter, beta = 24, 65, 0.07
            fisher_keep = min(380, max(160, int(0.16 * n_features)))
            max_feat_ratio, local_steps = 0.30, 22

        fs = fisher_score(X, y)
        top = np.argsort(fs)[::-1][:min(fisher_keep, n_features)]
        Xf = X[:, top]
        nf = Xf.shape[1]

        pop = np.random.uniform(-1.0, 1.3, size=(pop_size, nf))
        for i in range(pop_size):
            keep = 0.16 + 0.11 * np.random.rand()
            mask = np.random.rand(nf) < keep
            pop[i][~mask] = np.random.uniform(-2.2, -0.4, size=np.sum(~mask))

        def fitness(ind):
            sel = self._s_shaped(ind) > 0.44
            ns = int(np.sum(sel))
            if ns < 5 or ns > min(200, int(nf * max_feat_ratio)):
                return 1.4
            acc = knn_cv_score(Xf, y, np.where(sel)[0], self.random_state)
            return 0.90 * (1.0 - acc) + beta * (ns / nf)

        fit = np.array([fitness(ind) for ind in pop])
        best_i = np.argmin(fit)
        best, best_f = pop[best_i].copy(), fit[best_i]
        stag = 0

        for t in range(max_iter):
            order = np.argsort(fit)
            elite_n = max(6, int(0.32 * pop_size))
            new_pop = [pop[order[i]].copy() for i in range(elite_n)]
            for _ in range(elite_n, pop_size):
                parent = pop[order[np.random.randint(0, elite_n)]].copy()
                rate = 0.10 * (1 - t / max_iter) + 0.03
                midx = np.random.choice(nf, max(1, int(nf * rate)), replace=False)
                parent[midx] += np.random.normal(0, 0.28, size=len(midx))
                if stag >= 8:
                    parent += np.random.normal(0, 0.52, size=nf)
                new_pop.append(np.clip(parent, -3.7, 3.7))
            pop = np.array(new_pop)
            fit = np.array([fitness(ind) for ind in pop])
            cur = np.min(fit)
            if cur < best_f - 1e-5:
                best_i = np.argmin(fit)
                best, best_f = pop[best_i].copy(), cur
                stag = 0
            else:
                stag += 1

        # Local refinement
        probs = self._s_shaped(best)
        for _ in range(local_steps):
            sel = np.where(probs > 0.44)[0]
            nosel = np.where(probs <= 0.44)[0]
            if len(sel) > 9 and np.random.rand() < 0.55:
                trial = best.copy()
                trial[np.random.choice(sel)] = -2.7
                if fitness(trial) < best_f - 1e-5:
                    best, best_f = trial, fitness(trial)
                    probs = self._s_shaped(best)
            if len(nosel) > 0 and np.random.rand() < 0.45:
                trial = best.copy()
                trial[np.random.choice(nosel)] = 2.0
                if fitness(trial) < best_f - 1e-5:
                    best, best_f = trial, fitness(trial)
                    probs = self._s_shaped(best)

        chosen = np.where(self._s_shaped(best) > 0.44)[0]
        idx = top[chosen]
        if len(idx) < 6:
            idx = top[:max(14, nf // 8)]
        return idx

    def _prune(self, X, y, indices):
        indices = np.array(indices)
        if len(indices) <= self.min_features:
            return indices
        base = knn_cv_score(X, y, indices, self.random_state)
        current = indices.copy()
        while len(current) > self.min_features:
            rf = RandomForestClassifier(n_estimators=80, random_state=self.random_state, n_jobs=-1)
            rf.fit(X[:, current], y)
            trial = np.delete(current, np.argmin(rf.feature_importances_))
            if base - knn_cv_score(X, y, trial, self.random_state) > self.max_drop:
                break
            current = trial
        return current

    def fit(self, X, y):
        n_samples, n_features = X.shape
        ratio = n_features / max(n_samples, 1)

        if ratio > 40 or n_samples < 70:
            budgets = [40, 55, 75, 95]
        elif ratio > 15:
            budgets = [30, 45, 65]
        else:
            budgets = [25, 40, 55]

        candidates = []

        for nf in budgets:
            # Ranking
            rf = RandomForestClassifier(n_estimators=130, random_state=self.random_state, n_jobs=-1)
            rf.fit(X, y)
            idx = np.argsort(rf.feature_importances_)[-nf:]
            candidates.append((knn_cv_score(X, y, idx, self.random_state), idx))

            # RFE
            try:
                rfe = RFE(SVC(kernel='linear', C=1.0, random_state=self.random_state),
                          n_features_to_select=nf, step=0.12)
                rfe.fit(X, y)
                idx = np.where(rfe.support_)[0]
                candidates.append((knn_cv_score(X, y, idx, self.random_state), idx))
            except:
                pass

            # MI
            try:
                skb = SelectKBest(mutual_info_classif, k=nf)
                skb.fit(X, y)
                idx = np.where(skb.get_support())[0]
                candidates.append((knn_cv_score(X, y, idx, self.random_state), idx))
            except:
                pass

        idx_evo = self._evolutionary(X, y)
        candidates.append((knn_cv_score(X, y, idx_evo, self.random_state), idx_evo))

        candidates.sort(key=lambda t: t[0], reverse=True)
        best_idx = candidates[0][1]
        pruned = self._prune(X, y, best_idx)
        self.selected_indices_ = pruned
        self.n_features_selected_ = len(pruned)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# =============================================================================
# Main comparative runner
# =============================================================================

def run_comparison(data_dir, seed=42):
    datasets = [
        'ALLAML', 'leukemia', 'Prostate_GE', 'lung', 'CARCINOM',
        'CLL_SUB_111', 'TOX_171', 'lymphoma', 'lung_discrete',
        'GLI_85', 'SMK_CAN_187', 'colon', 'nci9', 'GLIOMA'
    ]

    total = len(datasets)
    print("=" * 100)
    print(" COMPARATIVE STUDY: NoGurobi  |  HFIA  |  AMSFS (Proposed)")
    print(" High-dimensional microarray feature selection")
    print("=" * 100)

    rows = []

    for i, ds in enumerate(datasets, 1):
        print(f"\n{'─'*70}")
        print(f" [{i}/{total}]  Dataset: {ds}")
        print(f"{'─'*70}")

        X, y = load_dataset(data_dir, ds)
        if X is None:
            print("  Status: SKIPPED (file not found or invalid)")
            continue

        print(f"  Shape: {X.shape[0]} samples × {X.shape[1]} features | "
              f"Classes: {len(np.unique(y))}")

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
        except:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        record = {'Dataset': ds}

        # ----- NoGurobi -----
        print("  → Running NoGurobi ...", end=" ", flush=True)
        t0 = time.time()
        n_target = min(55, max(25, X_tr.shape[0] // 2))
        sel = NoGurobi(n_features=n_target, random_state=seed)
        Xtr = sel.fit_transform(X_tr, y_tr)
        Xte = sel.transform(X_te)
        acc, f1, _ = evaluate_final(Xtr, Xte, y_tr, y_te, seed)
        t_ng = time.time() - t0
        record['NoGurobi_Acc'] = acc
        record['NoGurobi_F1'] = f1
        record['NoGurobi_Feat'] = len(sel.selected_indices_)
        record['NoGurobi_Time'] = t_ng
        print(f"Acc={acc*100:.2f}%  Feat={len(sel.selected_indices_)}  ({t_ng:.1f}s)")

        # ----- HFIA -----
        print("  → Running HFIA ...", end=" ", flush=True)
        t0 = time.time()
        sel = HFIA(pop_size=14, max_iter=90,
                   fisher_keep=min(220, X_tr.shape[1] // 4), random_state=seed)
        Xtr = sel.fit_transform(X_tr, y_tr)
        Xte = sel.transform(X_te)
        acc, f1, _ = evaluate_final(Xtr, Xte, y_tr, y_te, seed)
        t_hf = time.time() - t0
        record['HFIA_Acc'] = acc
        record['HFIA_F1'] = f1
        record['HFIA_Feat'] = len(sel.selected_indices_)
        record['HFIA_Time'] = t_hf
        print(f"Acc={acc*100:.2f}%  Feat={len(sel.selected_indices_)}  ({t_hf:.1f}s)")

        # ----- AMSFS (Proposed) -----
        print("  → Running AMSFS (Proposed) ...", end=" ", flush=True)
        t0 = time.time()
        sel = AMSFS(max_drop=0.025, min_features=12, random_state=seed, verbose=False)
        Xtr = sel.fit_transform(X_tr, y_tr)
        Xte = sel.transform(X_te)
        acc, f1, _ = evaluate_final(Xtr, Xte, y_tr, y_te, seed)
        t_am = time.time() - t0
        record['AMSFS_Acc'] = acc
        record['AMSFS_F1'] = f1
        record['AMSFS_Feat'] = sel.n_features_selected_
        record['AMSFS_Time'] = t_am
        print(f"Acc={acc*100:.2f}%  Feat={sel.n_features_selected_}  ({t_am:.1f}s)")

        rows.append(record)

    # =====================================================================
    # Final academic table
    # =====================================================================
    print("\n" + "=" * 100)
    print(" FINAL COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Dataset':<16} {'NoGurobi':>10} {'HFIA':>10} {'AMSFS':>10} "
          f"{'Feat-NG':>8} {'Feat-HF':>8} {'Feat-AM':>8}")
    print("-" * 100)

    for r in rows:
        print(f"{r['Dataset']:<16} {r['NoGurobi_Acc']*100:9.2f}% {r['HFIA_Acc']*100:9.2f}% "
              f"{r['AMSFS_Acc']*100:9.2f}% {r['NoGurobi_Feat']:8d} {r['HFIA_Feat']:8d} "
              f"{r['AMSFS_Feat']:8d}")

    if rows:
        avg_ng = np.mean([r['NoGurobi_Acc'] for r in rows])
        avg_hf = np.mean([r['HFIA_Acc'] for r in rows])
        avg_am = np.mean([r['AMSFS_Acc'] for r in rows])
        print("-" * 100)
        print(f"{'AVERAGE':<16} {avg_ng*100:9.2f}% {avg_hf*100:9.2f}% {avg_am*100:9.2f}%")

        # Wins
        wins_am = sum(1 for r in rows if r['AMSFS_Acc'] >= r['NoGurobi_Acc']
                      and r['AMSFS_Acc'] >= r['HFIA_Acc'])
        print(f"\nAMSFS wins (or ties) on {wins_am}/{len(rows)} datasets")

        df = pd.DataFrame(rows)
        df.to_csv('Comparison_NoGurobi_HFIA_AMSFS.csv', index=False)
        print("Results saved to Comparison_NoGurobi_HFIA_AMSFS.csv")

    print("=" * 100)
    return rows


if __name__ == "__main__":
    DATA_DIR = r"D:/UTH/Data_maining/T3/Dataset_ASU"
    results = run_comparison(DATA_DIR, seed=42)