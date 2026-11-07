# AMSFS: Adaptive Multi-Strategy Feature Selection for high-dimensional microarray data

This repository contains the code of the paper

> This repository accompanies a manuscript currently under review. Full citation details will be added after acceptance.

It includes the proposed method (AMSFS), the in-house baseline RF-RFE-MI, our re-implementation
of HFIA (HFIA\*), the binary metaheuristics PSO, GWO, WOA and HHO, the filter baselines
DT+Filter and Filter+SVM, and the SHAP analysis of Section 4.6.

## Repository layout

    src/
      AMSFS.py              AMSFS, RF-RFE-MI (NoGurobi), HFIA*, evaluation helpers, run_comparison()
      baselines_v3.py       PSO, GWO, WOA, HHO (binary), DT+Filter, Filter+SVM; merges all columns of Table 5-7
    experiments/
      run_main.py           runs RF-RFE-MI, HFIA* and AMSFS on all datasets -> results/
      chart_shap.py         SHAP analysis of Section 4.6 (Figures 2 and 3)
    tests/
      test_small.py         quick checks on a synthetic dataset (--full also runs AMSFS once)
    data/
      README_data.md        where to put the datasets and how they are loaded
    results/                output CSV files
    requirements.txt

## Installation

Python 3.9 or 3.10 is recommended.

    python -m venv venv
    venv\Scripts\activate          (Windows)   or   source venv/bin/activate   (Linux/macOS)
    pip install -r requirements.txt

## Quick test (no real data needed)

    python tests/test_small.py            (a few seconds)
    python tests/test_small.py --full     (also runs AMSFS once, several minutes)

## Reproducing the results

1. Put the 14 datasets in `data/Dataset_ASU/` (see `data/README_data.md`).
2. Run the three main methods (writes `results/Comparison_NoGurobi_HFIA_AMSFS.csv`, about 1.5 hours):

       python experiments/run_main.py --data-dir data/Dataset_ASU

3. Run the six additional baselines and merge all columns (writes `results/Table5_full_v3.csv`
   and `results/Baselines_new_six_v3.csv`, and prints Tables 5, 6 and 7 and the win/tie/loss counts; about 1 hour):

       python src/baselines_v3.py --data-dir data/Dataset_ASU

   If step 2 has not been run, step 3 runs the three main methods again.

4. Run the SHAP analysis of Section 4.6 (writes figures to `results/`):

       python experiments/chart_shap.py

All experiments use one stratified 80/20 split with `random_state = 42`. Feature selection,
including the fitness evaluation, the pruning and the KNN cross-validation, uses the training part only.
The StandardScaler is fitted on the training part only.

## Expected results

With the settings above, the paper reports the following average test accuracies (Table 5):
RF-RFE-MI 86.16 %, HFIA\* 85.21 %, AMSFS 90.86 %. Individual values may differ slightly between
library versions. Please report any difference you observe.
Small differences in the number of selected features may occur with other versions of scikit-learn (for example, SMK_CAN_187 selected 115 instead of 122 features with scikit-learn 1.5.2 and numpy 1.26.4).

## Notes

- The classifier that is reported in Table 5 is the best of SVM, KNN and Random Forest
  on the test set (see Section 4.1 of the paper). The three values are also stored in the output files.
- The ablation study of Section 4.7 is not included in this repository. The earlier ablation script evaluated
  the selected features on the same data that was used for selection, so it was removed.
- The running time depends on the hardware. The paper uses an Intel Core i7 with 16 GB RAM.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Licence

MIT (see `LICENSE`). The datasets are not included and have their own licences (see `data/README_data.md`).
