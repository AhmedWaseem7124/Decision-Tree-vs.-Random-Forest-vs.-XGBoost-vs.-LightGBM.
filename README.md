# Day 05 — Tree Ensemble Benchmark
## Decision Tree · Random Forest · XGBoost · LightGBM

> A rigorous, single-file ML benchmark that trains four tree-based architectures on the
> **California Housing** dataset, measures bias-variance tradeoff, training efficiency,
> residual distributions, and feature-importance alignment — then exports a
> publication-quality dark-themed dashboard.

---

## 📁 Project Structure

```
.
├── day05_tree_rf_xgboost_lightgbm.py   # Complete benchmark script (single file)
├── day05_tree_ensemble_artifact.png    # Generated visualization dashboard
└── README.md                           # This file
```

---

## 🎯 Objectives

| Goal | Detail |
|------|--------|
| **Data Integrity** | Strict 80/20 train/test split (`random_state=42`), zero test-set leakage |
| **Fair Benchmark** | All models evaluated on identical splits with identical metrics |
| **Bias-Variance** | Expose how a single deep tree overfits vs. ensemble regularization |
| **Speed vs. Quality** | Quantify LightGBM's histogram-binning speed advantage |
| **Explainability** | Rank and compare feature importances across 3 ensemble methods |

---

## 📊 Dataset

**California Housing** — from `sklearn.datasets.fetch_california_housing`

| Property | Value |
|----------|-------|
| Total samples | 20,640 |
| Features | 8 numerical |
| Target | Median house value (`MedHouseVal`, in $100k) |
| Train split | 16,512 samples (80%) |
| Test split | 4,128 samples (20%) |

### Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `MedInc` | Median income in block group |
| 2 | `HouseAge` | Median house age in block group |
| 3 | `AveRooms` | Average number of rooms per household |
| 4 | `AveBedrms` | Average number of bedrooms per household |
| 5 | `Population` | Block group population |
| 6 | `AveOccup` | Average number of household members |
| 7 | `Latitude` | Block group latitude |
| 8 | `Longitude` | Block group longitude |

---

## 🏗️ Model Architectures

### Model 1 — Decision Tree (Single)
```
Architecture : Single CART tree (greedy recursive splitting)
Hyperparams  : max_depth=12, random_state=42
Weakness     : High variance — memorizes training noise
               Zero bias correction across models
```

### Model 2 — Random Forest (Bagging Ensemble)
```
Architecture : Bootstrap-aggregated ensemble of independent trees
Hyperparams  : n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
Key trick    : Feature sub-sampling (sqrt) per split → decorrelates trees
Strength     : Dramatically reduces variance vs. single tree
```

### Model 3 — XGBoost (Level-wise Gradient Boosting)
```
Architecture : Additive boosting with exact greedy split finding
Hyperparams  : n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1
Key trick    : Second-order Taylor expansion of loss; L1/L2 regularization
Level-wise   : Splits all leaves at a given depth simultaneously (safer)
```

### Model 4 — LightGBM (Leaf-wise + Histogram Binning)
```
Architecture : Gradient-based one-side sampling + exclusive feature bundling
Hyperparams  : n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1
Key trick    : Histogram binning compresses continuous features → O(#bins)
Leaf-wise    : Grows the leaf with maximum gain → faster convergence
Speed        : Typically 5–10× faster than XGBoost on tabular data
```

---

## 📈 Metrics Logged

For every model, the script computes and records:

| Metric | Formula / Method |
|--------|-----------------|
| **Train Time** | `time.perf_counter()` wall-clock (seconds) |
| **Inference Latency** | Mean prediction time over 1,000 samples × 20 repeats (ms) |
| **Train RMSE** | `√(MSE(y_train, ŷ_train))` — measure of training fit |
| **Test RMSE** | `√(MSE(y_test, ŷ_test))` — generalization error |
| **Test R²** | `r2_score(y_test, ŷ_test)` — coefficient of determination |
| **Feature Importances** | `model.feature_importances_` normalized to sum = 1 |
| **Residuals** | `y_test − ŷ_test` per sample |

---

## 🖼️ Dashboard Panels

The output PNG (`day05_tree_ensemble_artifact.png`) contains a **2×2 dark-themed** Matplotlib dashboard:

### Panel ① — Bias–Variance Tradeoff
- Side-by-side **Train RMSE vs. Test RMSE** for all 4 models
- Clearly exposes the **overfitting gap** of the single Decision Tree
- Annotated bar values for precision reading

### Panel ② — Efficiency Benchmark
- **Scatter/bubble plot**: Training Time (log scale) on X vs. Test R² on Y
- **Bubble size** encodes inference latency per 1,000 samples
- Reveals LightGBM's Pareto-optimal position (fast AND accurate)

### Panel ③ — Residual Distribution
- **KDE + histogram** overlay for Decision Tree vs. LightGBM residuals
- Centered at zero = unbiased; narrow spread = low variance
- Standard deviation annotated for each model

### Panel ④ — Feature Importance Alignment
- **Grouped bar chart** for the top-5 features across RF, XGBoost, LightGBM
- Validates that all ensemble methods agree on `MedInc`, `Latitude`, `Longitude`
- Highlights where models disagree (interpretability insight)

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib scipy scikit-learn xgboost lightgbm
```

> **Python version**: 3.9+ recommended

### Run the Script

```bash
python day05_tree_rf_xgboost_lightgbm.py
```

### Expected Output

```
════════════════════════════════════════════════════════════════
  DAY 05 — Tree Ensemble Benchmark
  Decision Tree · Random Forest · XGBoost · LightGBM
════════════════════════════════════════════════════════════════

  California Housing Dataset
  ════════════════════════════════════════════════════════════════
  Total samples      : 20,640
  Features           : 8  (MedInc, HouseAge, ...)
  Train samples      : 16,512
  Test  samples      : 4,128
  ...

  ▶  Training  Decision Tree ...  done  [0.123s | RMSE_test=0.6821 | R²=0.7812]
  ▶  Training  Random Forest ...  done  [3.421s | RMSE_test=0.5012 | R²=0.8756]
  ▶  Training  XGBoost       ...  done  [2.134s | RMSE_test=0.4633 | R²=0.8932]
  ▶  Training  LightGBM      ...  done  [0.456s | RMSE_test=0.4521 | R²=0.8981]

  MODEL LEADERBOARD  (ranked by Test R²)
  ...

  ✔  Dashboard saved → day05_tree_ensemble_artifact.png
```

---

## 🔑 Key Insights

### Why Single Trees Overfit
A decision tree with `max_depth=12` can have up to **4,096 leaf nodes**. With 16,512 training samples, many leaves contain just a handful of points — the model memorizes training noise, showing near-zero training RMSE but high test RMSE.

### Why Random Forest Reduces Variance
By averaging 100 independently-trained trees on bootstrap samples, the ensemble prediction error converges to the **irreducible noise floor** — each tree's errors cancel out (provided they are decorrelated via feature sub-sampling).

### Why Boosting Reduces Both Bias and Variance
Gradient boosting fits successive trees to **residuals** of previous predictions, progressively reducing bias. The learning rate (`η = 0.1`) shrinks each tree's contribution, providing implicit regularization that also controls variance.

### LightGBM's Speed Advantage
LightGBM replaces exact split-finding (O(#data × #features)) with histogram-based binning (O(#bins × #features), default 255 bins), reducing computation by **~16–64×**. Leaf-wise growth further accelerates convergence vs. XGBoost's level-wise approach.

---

## 📐 Architecture Comparison Matrix

| Property | Decision Tree | Random Forest | XGBoost | LightGBM |
|----------|:-------------:|:-------------:|:-------:|:--------:|
| **Type** | Single model | Bagging | Boosting | Boosting |
| **Parallelism** | None | Per-tree | Per-split | Histogram |
| **Split finding** | Exact greedy | Exact greedy | Exact greedy | Histogram bins |
| **Growth strategy** | Level-wise | Level-wise | Level-wise | **Leaf-wise** |
| **Variance** | Very high | Low | Low | Low |
| **Bias** | Low | Low | Very low | Very low |
| **Training speed** | Fast | Slow | Medium | **Fast** |
| **Memory** | Low | High | Medium | **Low** |

---

## ⚙️ Reproducibility

All random states are fixed at `42`:
- `train_test_split(random_state=42)`
- `DecisionTreeRegressor(random_state=42)`
- `RandomForestRegressor(random_state=42)`
- `XGBRegressor(random_state=42)`
- `LGBMRegressor(random_state=42)`

Results should be **deterministic** across runs (subject to CPU/threading non-determinism in LightGBM's parallel mode).

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24 | Numerical arrays & RMSE calculation |
| `matplotlib` | ≥3.7 | 2×2 dashboard visualization |
| `scipy` | ≥1.10 | Gaussian KDE for residual distribution |
| `scikit-learn` | ≥1.3 | Dataset, splits, DecisionTree, RandomForest, metrics |
| `xgboost` | ≥2.0 | XGBRegressor (level-wise gradient boosting) |
| `lightgbm` | ≥4.0 | LGBMRegressor (leaf-wise + histogram) |

---

## 🧠 Learning Goals

After studying this benchmark you should be able to:

1. ✅ Explain the **bias-variance tradeoff** with concrete RMSE numbers
2. ✅ Articulate why **bagging** reduces variance without reducing bias
3. ✅ Explain the difference between **level-wise** and **leaf-wise** tree growth
4. ✅ Justify when to prefer **LightGBM over XGBoost** (speed-sensitive tasks, large datasets)
5. ✅ Interpret **feature importance** and its consistency across methods
6. ✅ Read residual plots and diagnose model quality from distributional properties

---

*Part of the 30-Day Machine Learning Engineering Series — Day 05*
