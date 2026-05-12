# Post-Pandemic-Housing-SHAP

**Comparative Analysis of Decision Tree Regression and Multiple Linear Regression with SHapley Additive exPlanations Analysis for Interpretable Post-Pandemic Housing Price Prediction using Housing Market Data**

**Group 4 | 3CSE Machine Learning**

---

## Table of Contents

1. [Quick Start Guide (Reproducibility Steps)](#1-quick-start-guide)
   - [Phase 1: Preprocessing](#phase-1-preprocessing)
   - [Phase 2: Multiple Linear Regression (MLR)](#phase-2-multiple-linear-regression-mlr)
   - [Phase 3: GPU-Accelerated XGBoost](#phase-3-gpu-accelerated-xgboost)
   - [Phase 4: SHAP Analysis](#phase-4-shap-analysis)
   - [Model Performance Summary](#model-performance-summary)
2. [Detailed Explanation](#2-detailed-explanation)
   - [Environment & RAPIDS Installation Guide](#environment--rapids-installation-guide)
   - [Data Sources](#data-sources)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Multiple Linear Regression](#multiple-linear-regression)
   - [GPU-Accelerated XGBoost](#gpu-accelerated-xgboost)
   - [SHAP Analysis](#shap-analysis)
   - [File Outputs Reference](#file-outputs-reference)

---

## 1. Quick Start Guide

This guide walks through the complete reproducibility pipeline in four sequential phases. The project can be reproduced by running the notebooks in the order listed below.

### Prerequisites

- **Python 3.10+**
- **Google Colab** account (for Phases 1, 2, and 4)
- **NVIDIA GPU** with CUDA 12.x+ and a **RAPIDS** environment (for Phase 3)
- See the [Environment & RAPIDS Installation Guide](#environment--rapids-installation-guide) for detailed setup instructions.

---

### Phase 1: Preprocessing

**Notebook:** `Notebooks/Preprocessing.ipynb`

Converts 14 raw Zillow CSV files into a cleaned, feature-engineered dataset ready for modeling.

**Steps:**
1. Open `Notebooks/Preprocessing.ipynb` in **Google Colab**.
2. Connect to a runtime and mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload the `Dataset/` folder (containing all `.csv` files) to your Google Drive at:
   ```
   /content/drive/MyDrive/3RD YEAR 2ND SEM/MACHINE LEARNING/3CSE Machine Learning Group 4/FINAL PROJECT/Dataset/
   ```
   > **Note:** If your folder path differs, update the `base_path` variable in the notebook accordingly.
4. Run all cells sequentially. The pipeline:
   - Loads all 14 CSV files
   - Melts horizontal date columns into vertical rows
   - Merges all 12 data tables into one master dataframe
   - Engineers features: `space_premium`, `is_post_pandemic`, `state_encoded`
   - Applies a chronological 80/20 train-test split
   - Standardizes features using `StandardScaler`
5. **Outputs** (saved to `Dataset/Model_Ready_Exports/`):
   - `X_train.parquet` — Training features (scaled)
   - `X_test.parquet` — Test features (scaled)
   - `y_train.parquet` — Training target (ZHVI)
   - `y_test.parquet` — Test target (ZHVI)
   - `feature_scaler.pkl` — Fitted StandardScaler

---

### Phase 2: Multiple Linear Regression (MLR)

**Notebook:** `Baseline MLR.ipynb`

Trains an Ordinary Least Squares (OLS) linear regression model as the baseline.

**Steps:**
1. Open `Baseline MLR.ipynb` in **Google Colab**.
2. Mount Google Drive. Update the `dataset_path` variable if needed:
   ```python
   dataset_path = '/content/drive/MyDrive/ML Project/Dataset/'
   ```
   > **Note:** Point `dataset_path` to the same folder containing the `Model_Ready_Exports/` directory from Phase 1.
3. Run all cells. The pipeline:
   - Loads the exported parquet files and scaler
   - Applies `log1p` transformation to the target (ZHVI)
   - Fits `LinearRegression` from sklearn
   - Evaluates on the holdout set (inverse-transformed to dollar values)
   - Runs Gauss-Markov diagnostics (VIF, residual plots, normality check)
4. **Outputs:**
   - `mlr_model_final.joblib` — Trained MLR model
   - `mlr_baseline_summary.json` — Metrics and model parameters

---

### Phase 3: GPU-Accelerated XGBoost

**Notebook:** `XGB Zillow.ipynb`

Trains an XGBoost regressor with GPU acceleration and Optuna hyperparameter tuning.

#### Environment Setup (RAPIDS)
This notebook requires a **RAPIDS** environment with an NVIDIA GPU. See the [RAPIDS Installation Guide](#environment--rapids-installation-guide) below before proceeding.

**Steps:**
1. Activate your RAPIDS conda environment:
   ```bash
   conda activate rapids-26.04
   ```
2. Open `XGB Zillow.ipynb` locally (or in your RAPIDS-enabled Jupyter kernel).
3. The notebook uses **relative paths** — no Google Drive mount needed. Ensure the `Dataset/Model_Ready_Exports/` folder (from Phase 1) is in the correct relative location, or update the paths at the top of the notebook:
   ```python
   DATASET_PATH = "Dataset/Model_Ready_Exports/"
   JSON_PATH = "XGB/xgb_json/"
   DB_PATH = "XGB/xgb_database/"
   SUMMARY_PATH = "XGB/xgb_summary/"
   ```
4. Run all cells. The pipeline:
   - Loads data into GPU memory via **cuDF**
   - Adds a `Time_Index` feature for temporal ordering
   - Enforces `float32` precision (50% memory reduction)
   - Performs a sequential (chronological) 80/20 train-validation split
   - Creates **QuantileDMatrix** objects for GPU-optimized histogram binning
   - Runs **Optuna** hyperparameter tuning (50 trials, 3-fold TimeSeriesSplit)
   - Trains the final model with best hyperparameters on full training data
   - Evaluates on the holdout set
   - Generates gain-based feature importance analysis
5. **Outputs:**
   - `XGB/xgb_json/xgboost_booster.json` — Trained XGBoost booster
   - `XGB/xgb_json/best_params.json` — Best hyperparameters
   - `XGB/xgb_json/tuning_results.json` — Full tuning results
   - `XGB/xgb_database/tuning_progress.db` — Optuna study database
   - `XGB/xgb_summary/project_report.txt` — Summary report
   - `XGB/xgb_summary/residual_plot.png` — Residual diagnostics plot

---

### Phase 4: SHAP Analysis

**Notebook:** `Notebooks/SHAP_Analysis.ipynb`

Applies SHAP (SHapley Additive exPlanations) to interpret both MLR and XGBoost models.

**Steps:**
1. Open `Notebooks/SHAP_Analysis.ipynb` in **Google Colab**.
2. Mount Google Drive. Update the `base_path` if needed:
   ```python
   base_path = '/content/drive/MyDrive/3CSE Machine Learning Group 4/'
   ```
   The notebook expects this folder structure:
   ```
   {base_path}FINAL PROJECT/Dataset/Model_Ready_Exports/  (from Phase 1)
   {base_path}FINAL PROJECT/XGB/xgb_json/                  (from Phase 3)
   ```
3. Run all cells. The pipeline:
   - Loads test data, scaler, and the XGBoost booster
   - Reconstructs the MLR model from pre-extracted coefficients (from Phase 2)
   - Samples 2,000 rows for SHAP computation
   - Runs **LinearExplainer** on MLR → generates bar + beeswarm plots
   - Runs **TreeExplainer** on XGBoost → generates bar + beeswarm + dependence plots
4. **Outputs** (saved to `XGB/`):
   - `shap_mlr_bar.png` — MLR feature importance (bar plot)
   - `shap_mlr_beeswarm.png` — MLR SHAP values (beeswarm plot)
   - `shap_xgb_bar.png` — XGBoost feature importance (bar plot)
   - `shap_xgb_beeswarm.png` — XGBoost SHAP values (beeswarm plot)
   - `shap_xgb_dependence_*.png` — XGBoost dependence plots for top 5 features

---

### Model Performance Summary

| Metric               | MLR (Baseline) | XGBoost (GPU-Accelerated) |
|----------------------|----------------|---------------------------|
| **R² (Holdout)**     | —              | **0.9289**               |
| **RMSE (Holdout)**   | —              | **$48,100.72**           |
| **MAE (Holdout)**    | —              | **$26,839.65**           |
| **MAPE (Holdout)**   | —              | **7.61%**                |
| **Optuna Best MAPE** | —              | 9.05% (Trial 36 of 50)   |

> **Note:** MLR metrics were computed during Phase 2 execution. Run the notebook to populate exact values.

---

## 2. Detailed Explanation

### Environment & RAPIDS Installation Guide

#### Google Colab (Phases 1, 2, 4)
Phases 1 (Preprocessing), 2 (MLR), and 4 (SHAP) are designed for **Google Colab**. No local installation is required — just a Google account and the notebooks uploaded to Colab. Ensure your runtime has sufficient memory (at least 16 GB RAM recommended for the full dataset).

#### Local RAPIDS Environment (Phase 3 — XGBoost)
Phase 3 requires **RAPIDS** for GPU-accelerated XGBoost. An NVIDIA GPU with CUDA 12.x+ is required.

RAPIDS provides several installation methods — conda/mamba, pip, Docker, and SDK Manager. The official [**RAPIDS Installation Guide**](https://docs.rapids.ai/install) includes a **Release Selector** tool where you choose your preferred method, packages, and environment to generate the exact install command for your setup.

**Tested configuration for this project:**

```bash
mamba create -n rapids-26.04 \
    -c rapidsai -c conda-forge -c nvidia \
    rapids=26.04 python=3.12 'cuda-version>=13.0,<=13.1'

conda activate rapids-26.04
pip install shap optuna xgboost joblib
```

If your CUDA version differs, use the [Release Selector](https://docs.rapids.ai/install) to generate the correct command.

**Verify GPU availability:**
```python
import cupy as cp
import xgboost as xgb

print(f"CUDA GPUs: {cp.cuda.runtime.getDeviceCount()}")
print(f"XGBoost version: {xgb.__version__}")

# Test GPU training
test_model = xgb.XGBRegressor(tree_method='hist', device='cuda')
print("✓ GPU training verified")
```

#### Troubleshooting RAPIDS Installation & CUDA Version Compatibility

RAPIDS package versions are tied to specific CUDA versions. The project was developed and tested on:

| Component | Version |
|-----------|---------|
| GPU | NVIDIA GTX 1660 Ti |
| CUDA Driver | 13.2 |
| RAPIDS | 26.04 |
| Python | 3.12 |

**Before installing, check your CUDA version:**

```bash
nvidia-smi
```

Look for the **CUDA Version** line in the top-right of the output (e.g., `CUDA Version: 13.2`).

**Match the `cuda-version` specifier to your installed CUDA:**

| Your CUDA Version | Use `cuda-version=` | Compatible RAPIDS |
|-------------------|---------------------|-------------------|
| 13.x | `cuda-version=13.x` | 26.04 |
| 12.x | `cuda-version=12.x` | 26.04 / 24.12 |
| 11.x | `cuda-version=11.x` | 24.10 / 24.08 |

For example, if you have CUDA 12.5:
```bash
mamba create -n rapids-26.04 \
    -c rapidsai -c conda-forge -c nvidia \
    rapids=26.04 python=3.12 cuda-version=12.5
```

**Find your exact RAPIDS version** at: https://docs.rapids.ai/install

**Common Issues & Fixes:**
- **`libcuda.so` not found**: Ensure NVIDIA drivers are installed. Run `nvidia-smi` to verify.
- **Out of memory on GPU**: Reduce `n_trials` in Optuna or use fewer TimeSeriesSplit folds.
- **conda environment solve takes too long**: Use `mamba` instead of `conda` (much faster solver).
- **RAPIDS package not found for your CUDA**: Try an older RAPIDS release or upgrade your NVIDIA driver.
- **Fallback to CPU**: If GPU setup fails, XGBoost can still run on CPU by changing `device` from `'cuda'` to `'cpu'` and `tree_method` from `'hist'` to `'approx'`.

---

### Data Sources

All 14 datasets are Zillow metro-level CSV files sourced from [Zillow Research Data](https://www.zillow.com/research/data/). They are located in the `Dataset/` directory.

| File | Description |
|------|-------------|
| `metro_zhvi.csv` | **Zillow Home Value Index** (ZHVI) — median home value (target variable) |
| `metro_zori.csv` | Zillow Observed Rent Index (ZORI) — median rent |
| `metro_forsale.csv` | Number of homes for sale |
| `metro_new_listings.csv` | Number of new listings |
| `metro_med_days_to_pending.csv` | Median days to pending |
| `metro_shrlist_prcut.csv` | Share of listings with price cuts |
| `metro_income_needed.csv` | Income needed to afford a home |
| `metro_new_homeowner_affordability.csv` | New homeowner affordability index |
| `metro_market_heat_index.csv` | Market heat index (supply-demand ratio) |
| `metro_zhvi_1_bdrm.csv` | ZHVI — 1-bedroom homes |
| `metro_zhvi_2_bdrm.csv` | ZHVI — 2-bedroom homes |
| `metro_zhvi_3_bdrm.csv` | ZHVI — 3-bedroom homes |
| `metro_zhvi_4_bdrm.csv` | ZHVI — 4-bedroom homes |
| `metro_zhvi_5plus_bdrm.csv` | ZHVI — 5+ bedroom homes |

---

### Preprocessing Pipeline

The full pipeline is implemented in `Notebooks/Preprocessing.ipynb`.

#### 1. Data Reshaping (Melt)
Each CSV file stores dates as horizontal columns (e.g., `2022-01-01`, `2022-02-01`). The `melt()` operation transforms these into vertical rows with `RegionID`, `Date`, and `value` columns, creating a normalized time-series format.

#### 2. Master Merge
All 12 reshaped data tables are merged sequentially using `RegionID` and `Date` as composite keys, with `metro_zhvi.csv` as the anchor table. This produces a single wide-format dataframe.

#### 3. Feature Engineering
- **space_premium**: Ratio of 4-bedroom ZHVI to 2-bedroom ZHVI
  ```math
  \text{space\_premium} = \frac{\text{ZHVI}_{4BR}}{\text{ZHVI}_{2BR}}
  ```
  Captures the premium for larger homes during the WFH era.

- **is_post_pandemic**: Binary flag (`1` for dates ≥ 2022-04-01, `0` otherwise). Marks the shift from pandemic low-rate era to post-pandemic interest rate hike era.

- **state_encoded**: Target-encoded state feature. Each state's ZHVI mean (calculated from training data only to prevent data leakage) replaces the categorical `StateName`.

#### 4. Chronological Split
Data is sorted by date and split at `2023-01-01`:
- **Training**: All dates before 2023-01-01
- **Test**: All dates from 2023-01-01 onwards

This chronological split preserves temporal ordering and prevents future leakage.

#### 5. Standardization
Features are standardized to mean=0, variance=1 using `StandardScaler`. The scaler is fitted on training data only and saved for later use.

#### 6. Exported Files
The pipeline exports ready-to-use files to `Dataset/Model_Ready_Exports/`:
- `X_train.parquet` / `X_test.parquet` — Standardized features
- `y_train.parquet` / `y_test.parquet` — Raw ZHVI target values
- `feature_scaler.pkl` — Fitted StandardScaler

**Features used for modeling (10 total):**
| Feature | Description |
|---------|-------------|
| `rent` | Zillow Observed Rent Index |
| `for_sale` | Number of homes for sale |
| `days_pending` | Median days to pending |
| `price_cuts` | Share of listings with price cuts |
| `market_heat_index` | Market heat index |
| `new_homeowner_affordability` | New homeowner affordability |
| `space_premium` | 4BR-to-2BR price ratio |
| `is_post_pandemic` | Post-pandemic binary flag |
| `state_encoded` | Target-encoded state mean |
| `Time_Index` | Sequential time index (added in Phase 3) |

---

### Multiple Linear Regression

The MLR baseline is trained in `Baseline MLR.ipynb`.

#### Model
- **Algorithm**: Ordinary Least Squares (OLS) via `sklearn.linear_model.LinearRegression`
- **Target Transformation**: `log1p` (natural log of 1 + ZHVI) to stabilize variance and reduce skewness

  ```math
  \ln(y_{\text{train}} + 1) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
  ```

- **Predictions**: Inverse-transformed via `expm1` to return to dollar values:
  ```math
  \hat{y}_{\text{actual}} = e^{\hat{y}_{\text{log}}} - 1
  ```

#### Gauss-Markov Diagnostics
The notebook checks the four key OLS assumptions:
1. **Linearity & Homoscedasticity** — Residuals vs. Fitted plot
2. **Multicollinearity** — Variance Inflation Factor (VIF) for each feature (rule of thumb: VIF > 5–10 indicates problematic collinearity)
3. **Normality of Residuals** — Histogram with KDE overlay
4. **Independence** — Assumed via temporal ordering

#### Metrics Evaluated
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

#### Exported Artifacts
- `mlr_model_final.joblib` — Serialized model for reuse
- `mlr_baseline_summary.json` — Metrics, coefficients, and parameters

---

### GPU-Accelerated XGBoost

Trained in `XGB Zillow.ipynb`. This notebook leverages NVIDIA GPU acceleration via RAPIDS (cuDF, cuPy) and XGBoost's `hist` tree method.

#### GPU Optimization Techniques
- **cuDF DataFrames**: Data loaded directly into GPU memory, avoiding CPU-GPU transfer overhead
- **float32 Precision**: All data cast to 32-bit floats, reducing memory usage by 50% vs. float64
- **QuantileDMatrix**: Pre-calculates histogram bin boundaries once outside the Optuna loop, saving minutes of computation across 50 trials
- **GPU-based MAPE**: Computed in CuPy to avoid costly CPU transfers during tuning

#### Time-Series Split
Data is ordered by `Time_Index` and split **sequentially** (not randomly) to preserve temporal order:
- **80% Training** (first 78,825 rows)
- **20% Validation** (last 19,707 rows)
- **Holdout**: ~13,923 rows (from Phase 1 test set, untouched until final evaluation)

#### Optuna Hyperparameter Tuning

**Search Space:**
| Parameter | Range | Purpose |
|-----------|-------|---------|
| `max_depth` | 3–12 | Tree depth control |
| `min_child_weight` | 5–50 | **Critical** — prevents single-zip code leaves |
| `gamma` | 0.1–5.0 | Minimum loss reduction for split |
| `learning_rate` | 0.001–0.5 (log) | Step size shrinkage |
| `subsample` | 0.6–1.0 | Row sampling ratio |
| `colsample_bytree` | 0.6–1.0 | Column sampling ratio |
| `reg_alpha` | 0–50 | L1 regularization (elevated for target-encoded features) |
| `reg_lambda` | 0–50 | L2 regularization |
| `objective` | `reg:absoluteerror` | MAE-optimized objective |

**Tuning Configuration:**
- **50 trials** with **3-fold TimeSeriesSplit** (expanding window)
- **Early stopping**: 50 rounds
- **Sampler**: TPESampler (seed=42, multivariate, grouped)
- **Metric**: MAPE (minimized)

**Best Found:**
- MAPE (validation cross-validation): **9.05%** (Trial 36)
- Trees: ~100 (from early stopping)

#### Final Model Performance (Holdout)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| R² | **0.9289** | Model explains 92.89% of variance |
| RMSE | **$48,100.72** | Typical prediction error magnitude |
| MAE | **$26,839.65** | Average absolute error in dollars |
| MAPE | **7.61%** | Average percentage error |

#### Top 5 Features (Gain-Based Importance)
1. **new_homeowner_affordability** — 89.17 gain
2. **market_heat_index** — 61.86 gain
3. **rent** — 54.77 gain
4. **state_encoded** — 48.33 gain
5. **Time_Index** — 36.83 gain

---

### SHAP Analysis

The SHAP analysis is implemented in `Notebooks/SHAP_Analysis.ipynb`, applying interpretability to both models.

#### MLR: LinearExplainer
Uses `shap.LinearExplainer` with interventional feature perturbation. Since MLR is a linear model, SHAP values decompose each prediction as:
```math
\text{SHAP}_j(x) = \beta_j \cdot (x_j - \mathbb{E}[x_j])
```
where \(\beta_j\) is the coefficient for feature \(j\).

#### XGBoost: TreeExplainer
Uses `shap.TreeExplainer` which leverages the tree structure to compute exact SHAP values in polynomial time (rather than the exponential cost of generic Shapley values).

#### Sampling
Due to the large dataset, **2,000 rows** are randomly sampled (seed=42) from the test set for SHAP computation.

#### Output Plots
| Plot | Description |
|------|-------------|
| **Bar plot** (`shap_mlr_bar.png`, `shap_xgb_bar.png`) | Mean absolute SHAP value per feature — global feature importance |
| **Beeswarm plot** (`shap_mlr_beeswarm.png`, `shap_xgb_beeswarm.png`) | SHAP value distribution per feature — direction and spread of impact |
| **Dependence plots** (`shap_xgb_dependence_*.png`) | SHAP value vs. feature value for top 5 XGBoost features — reveals non-linear relationships |

---

### File Outputs Reference

| File | Source Phase | Description |
|------|-------------|-------------|
| `Dataset/Model_Ready_Exports/X_train.parquet` | 1 | Training features (standardized) |
| `Dataset/Model_Ready_Exports/X_test.parquet` | 1 | Test features (standardized) |
| `Dataset/Model_Ready_Exports/y_train.parquet` | 1 | Training target (ZHVI) |
| `Dataset/Model_Ready_Exports/y_test.parquet` | 1 | Test target (ZHVI) |
| `Dataset/Model_Ready_Exports/feature_scaler.pkl` | 1 | Fitted StandardScaler |
| (Colab export) `mlr_model_final.joblib` | 2 | Trained linear regression model |
| (Colab export) `mlr_baseline_summary.json` | 2 | MLR metrics and parameters |
| `XGB/xgb_json/xgboost_booster.json` | 3 | Trained XGBoost booster |
| `XGB/xgb_json/best_params.json` | 3 | Best Optuna hyperparameters |
| `XGB/xgb_json/tuning_results.json` | 3 | Full tuning results |
| `XGB/xgb_database/tuning_progress.db` | 3 | Optuna study database |
| `XGB/xgb_summary/project_report.txt` | 3 | Full project summary report |
| `XGB/xgb_summary/residual_plot.png` | 3 | Residual diagnostics |
| `XGB/shap_mlr_bar.png` | 4 | MLR SHAP bar plot |
| `XGB/shap_mlr_beeswarm.png` | 4 | MLR SHAP beeswarm plot |
| `XGB/shap_xgb_bar.png` | 4 | XGBoost SHAP bar plot |
| `XGB/shap_xgb_beeswarm.png` | 4 | XGBoost SHAP beeswarm plot |
| `XGB/shap_xgb_dependence_*.png` | 4 | XGBoost SHAP dependence plots |

---

## Citation

If you use this work, please cite:

```
Group 4, 3CSE Machine Learning. "Post-Pandemic-Housing-SHAP: Comparative Analysis
of Decision Tree Regression and Multiple Linear Regression with SHAP Analysis for
Interpretable Post-Pandemic Housing Price Prediction." 2026.
```

---

**License:** Academic Project — Machine Learning Course, 3CSE