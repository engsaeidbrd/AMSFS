"""
Quick checks of the code (a few seconds). No real data is needed.

Usage (from the repository root):
    python tests/test_small.py            # quick checks
    python tests/test_small.py --full     # also runs AMSFS once on a small synthetic dataset (several minutes)
"""
import os
import sys
import tempfile
import numpy as np
import scipy.io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from AMSFS import AMSFS, load_dataset, fisher_score, knn_cv_score  # noqa: E402


def make_synthetic(folder, n=40, d=60, k=2, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(1, k + 1, n)
    X = rng.randn(n, d)
    for c in range(1, k + 1):
        X[y == c, :8] += rng.randn(8) * 1.5
    scipy.io.savemat(os.path.join(folder, "synthetic.mat"), {"X": X, "Y": y.reshape(-1, 1)})


def test_loader_and_scores():
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic(tmp)
        X, y = load_dataset(tmp, "synthetic")
        assert X is not None and X.shape == (40, 60)
        assert set(np.unique(y)) == {0, 1}
        scores = fisher_score(X, y)
        assert len(scores) == 60 and np.all(np.isfinite(scores))
        acc = knn_cv_score(X, y, np.arange(8), 42)
        assert 0.0 <= acc <= 1.0


def test_amsfs_full():
    with tempfile.TemporaryDirectory() as tmp:
        make_synthetic(tmp)
        X, y = load_dataset(tmp, "synthetic")
        model = AMSFS(max_drop=0.025, min_features=12, random_state=42)
        model.fit(X, y)
        idx = model.selected_indices_
        assert idx is not None and len(idx) >= 2
        assert np.all((idx >= 0) & (idx < X.shape[1]))


if __name__ == "__main__":
    test_loader_and_scores()
    print("quick checks: OK")
    if "--full" in sys.argv:
        test_amsfs_full()
        print("AMSFS run: OK")
