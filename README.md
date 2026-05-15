# Alpha Research with AI

## Project Description

A quantitative alpha research framework combining classical factor analysis with machine learning. Covers the full pipeline: alpha factor construction, risk modelling, ML-based alpha combination, and portfolio optimisation.

Related articles:
- [How to Build Quant Algorithmic Trading Model in Python](https://yuki678.medium.com/how-to-build-quant-algorithmic-trading-model-in-python-12abab49abe3?sk=56d5b2b038ce6aefa6c2049cff9e89b6)
- [How to generate an AI Alpha Factor in Python](https://yuki678.medium.com/how-to-generate-an-ai-alpha-factor-in-python-6509c5cb5bf6?sk=d8cfa3b0f87f69bcae75eced08fd7916)

---

## Getting Started

### Prerequisites

- Python 3.11
- [Zipline-Reloaded](https://github.com/stefan-jansen/zipline-reloaded) requires a US equity data bundle ingested locally (see [Data Setup](#data-setup) below)
- Graphviz system binary (required for decision-tree visualisation):
  ```bash
  brew install graphviz        # macOS
  sudo apt install graphviz   # Debian/Ubuntu
  ```

### Installation

```bash
git clone <repo-url>
cd ai-alpha

# Full install (alpha research + backtesting + zipline)
pip install -r requirements.txt

# Backtesting only (lighter, no zipline)
pip install -r requirements_backtest.txt
```

### Data Setup

The notebooks run against Zipline's `quandl` bundle. Ingest it once before launching any notebook:

```bash
zipline ingest -b quandl
```

Sector mappings used by `datautil.Sector` are read from `data/sector/data.npy`. Make sure this file is present before running the pipeline engine.

### Running the Notebooks

Launch Jupyter from the repo root so that the `myalpha` package is on the path:

```bash
jupyter notebook
```

Open the notebooks in this order:

| Notebook | Purpose |
|---|---|
| `Alpha Research Base.ipynb` | Alpha factor construction and evaluation without ML |
| `Alpha Research AI.ipynb` | Adds a Random Forest ensemble to combine alpha factors |
| `Backtest Base.ipynb` | Backtests the resulting alpha via Zipline |

---

## Repository Structure

```
ai-alpha/
├── myalpha/               # Core library
│   ├── alphas.py          # Zipline CustomFactor definitions
│   ├── datautil.py        # Pipeline engine + pricing helpers
│   ├── riskmodel.py       # PCA-based risk model
│   ├── mlutil.py          # NoOverlapVoter ensemble + training utils
│   ├── performance.py     # AlphaLens wrappers + alpha vector composition
│   └── optimization.py   # CVXPY portfolio optimiser
├── sandbox/               # Standalone backtest scripts and tutorials
│   ├── Backtest AI.ipynb
│   ├── Backtest Trial.ipynb
│   └── Zipline Tutorial.ipynb
├── data/
│   └── sector/data.npy    # Sector classification array (asset_id → sector)
├── requirements.txt           # Full dependencies (includes zipline-reloaded)
└── requirements_backtest.txt  # Lighter set for backtesting only
```

---

## Module Reference

### `myalpha/alphas.py`
Zipline `CustomFactor` and pipeline factor definitions:
- `momentum_1yr` — 1-year cross-sectional momentum, sector-demeaned
- `mean_reversion_5day_sector_neutral_smoothed` — smoothed 5-day mean reversion
- `overnight_sentiment_smoothed` — trailing overnight return sentiment
- `MarketDispersion` — cross-sectional return dispersion
- `MarketVolatility` — 252-day annualised market volatility

### `myalpha/datautil.py`
Helpers for running Zipline pipelines and pulling pricing data:
- `PricingLoader` — wires `USEquityPricingLoader` into the pipeline engine
- `Sector` — classifier that maps asset ids to GICS sectors from `data/sector/data.npy`
- `build_pipeline_engine` — constructs a `SimplePipelineEngine`
- `get_universe_tickers` — runs a pipeline screen for a given date
- `get_pricing` — fetches OHLCV history from a `DataPortal`

### `myalpha/riskmodel.py`
PCA risk model:
- `fit_pca` — fits a PCA model to return data
- `factor_betas` / `factor_returns` / `factor_cov_matrix` / `idiosyncratic_var_matrix` — risk model components
- `get_risk_factors` — convenience wrapper returning all components at once
- `predict_portfolio_risk` — computes predicted portfolio volatility given weights

### `myalpha/mlutil.py`
Ensemble classifier for alpha signal combination:
- `train_valid_test_split` — time-series-safe train/validation/test split
- `NoOverlapVoter` — `VotingClassifier` subclass trained on non-overlapping data subsets to reduce look-ahead bias. Requires `oob_score=True` in the classifier parameters.
- `train_model` — end-to-end training helper; returns `[model, train_score, test_score, oob_score, retrain_score, retrain_oob]`
- `plot_tree_classifier` — renders a decision tree to PNG via Graphviz
- `rank_features_by_importance` — prints feature importances ranked descending

### `myalpha/performance.py`
AlphaLens wrappers and alpha vector utilities:
- `get_sharpe_ratio` — annualised Sharpe from a returns DataFrame
- `get_factor_returns` / `get_qr_factor_returns` — long-short and quantile factor returns
- `plot_factor_returns` / `plot_qr_factor_returns` / `plot_factor_rank_autocorrelation` — standard AlphaLens plots
- `build_factor_data` — runs `get_clean_factor_and_forward_returns` over a dict of factor series
- `show_sample_results` — full evaluation display including the ML alpha score
- `get_alpha_vector_mean_lastday` — simple equal-weight alpha vector from the last date
- `get_alpha_vector2` — Sharpe-weighted alpha vector; raises `ValueError` if all weights are zero

### `myalpha/optimization.py`
CVXPY-based mean-variance portfolio optimiser:
- `OptimalHoldings` — minimises negative alpha subject to risk, factor exposure, and weight constraints
- `OptimalHoldingsRegualization` — adds an L2 regularisation term on weights
- `OptimalHoldingsStrictFactor` — minimises deviation from the alpha vector directly

All three raise `ValueError` if the solver fails to find a feasible solution.

---

## Changelog

### 2026-05-16

#### Dependencies updated
All packages updated from 2018–2019 pins to current stable versions:

| Package | Old | New | Notes |
|---|---|---|---|
| `zipline` | 1.3.0 | `zipline-reloaded` 3.1.1 | Original Quantopian project abandoned; community fork is drop-in replacement |
| `alphalens` | 0.3.6 | `alphalens-reloaded` 0.4.6 | Same situation |
| `numpy` | 1.16.2 | 2.2.6 | |
| `pandas` | 0.22.0 | 2.2.3 | |
| `matplotlib` | 2.1.2 | 3.10.9 | |
| `scipy` | 1.0.1 | 1.14.1 | |
| `scikit-learn` | 0.19.1 | 1.8.0 | |
| `cvxpy` | 1.0.3 | 1.8.2 | |
| `statsmodels` | 0.9.0 | 0.14.6 | |
| `plotly` | 2.2.3 | 6.7.0 | |
| `patsy` | 0.5.1 | 1.0.2 | |
| Other packages | various | latest stable | `graphviz`, `tqdm`, `tables`, `requests`, `pytz`, etc. |

#### Code fixes (API breakages from dependency upgrades)
- **`performance.py`** — `DataFrame.iteritems()` removed in pandas 2.0; replaced with `.items()`
- **`datautil.py`** — `pd.Timestamp(freq=)` parameter removed in pandas 2.0; removed the argument
- **`optimization.py`** — CVXPY 1.1+ requires `@` for matrix/vector products; replaced all `*` operators accordingly
- **`mlutil.py`** — `RandomForestClassifier` was used but never imported; added to imports

#### Security and correctness fixes
- **`datautil.py`** — `Sector.__init__` loaded sector data via a cwd-relative path; replaced with an absolute path anchored to the module file, with `allow_pickle=False`
- **`optimization.py`** — `find()` accessed `weights.value` without checking solver status; added an explicit status check that raises `ValueError` on infeasible/failed solves
- **`optimization.py`** — `factor_cov_matrix` passed to `cvx.quad_form` without symmetry enforcement; added `(M + M.T) / 2` symmetrisation
- **`datautil.py`** — `get_pricing` used `bar_count = end_loc - start_loc`, dropping the first day of every requested window; corrected to `+ 1`
- **`mlutil.py`** — `NoOverlapVoterAbstract.__init__` allocated `n_skip_samples + 1` classifiers but `non_overlapping_estimators` only trained `n_skip_samples`; loop range and stride corrected to `n_skip_samples + 1`
- **`riskmodel.py`** — `idiosyncratic_var_matrix` used `ddof=0` while `factor_cov_matrix` used `ddof=1`; unified to `ddof=1, axis=0`
- **`riskmodel.py`** — `predict_portfolio_risk` returned `result[0]` (pandas column selection, fragile) instead of a scalar; changed to `float(result.iloc[0, 0])`
- **`performance.py`** — `get_alpha_vector2` divided by `np.sum(shape_ratio_value)` without checking for zero; added guard that raises `ValueError`
- **`performance.py`** — `get_alpha_vector_mean_lastday` and `get_alpha_vector2` mutated the caller's DataFrame in place; added `.copy()` in both
- **`performance.py`** — `plot_factor_rank_autocorrelation` shadowed the `factor_data` dict variable inside a comprehension, then iterated the original dict in a second loop while ignoring the computed `unixt_factor_data`; fixed variable names and used the unix-timestamp data for the autocorrelation call
- **`alphas.py`** — `MarketVolatility.window_length = 1` caused the factor to always output `0.0` (variance of a single value is zero); changed to `252`
- **`mlutil.py`** — `train_model` accessed `clf.oob_score_` without verifying that `oob_score=True` was set; added a guard with a clear error message
