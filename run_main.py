"""
Runs the three methods of the paper (RF-RFE-MI, HFIA*, AMSFS) on all datasets
with the protocol of Section 4.1 and writes results/Comparison_NoGurobi_HFIA_AMSFS.csv.

Usage (from the repository root):
    python experiments/run_main.py --data-dir data/Dataset_ASU
"""
import os
import sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from AMSFS import run_comparison  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data", "Dataset_ASU"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(out_dir)  # run_comparison writes its CSV to the current folder
    try:
        run_comparison(args.data_dir, seed=args.seed)
    finally:
        os.chdir(cwd)
