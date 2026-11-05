# California Housing Model Lab

A reproducible, end-to-end housing-value regression benchmark that compares a simple baseline, regularized linear regression, a tree ensemble, and a small neural model on the public California Housing dataset. It is a practical model-training project: run it locally, inspect the learned estimators and outputs, then explain the evaluation and where the model fails.

## Why this is worth publishing

House-value prediction is a useful supervised-learning task because it makes the whole workflow visible: data provenance, feature construction, preprocessing, model comparison, held-out evaluation, residual analysis, and interpretation. The point is not to sell a house-price oracle; it is to show when a simple model is enough, whether a nonlinear model helps, and how evaluation can change when nearby geographic areas are held out.

This is distinct from the Boston Housing coursework project: it uses a different public dataset, a spatially blocked holdout, a reproducible multi-model comparison, and packaged diagnostic artifacts.

## Dataset and limits

The benchmark uses the 20,640-row California Housing dataset distributed by scikit-learn and sourced from StatLib. It describes 1990 California census block groups, with eight aggregate predictors such as median income, average rooms, population, latitude, and longitude. The target is median house value in units of $100,000 and is capped near $500,001 in the source. The data are historical and geographically aggregated; the model is not an appraisal, investment recommendation, or estimate of current prices. It contains no private user data.

Dataset reference: [scikit-learn `fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), citing Pace and Barry, “Sparse Spatial Autoregressions,” *Statistics & Probability Letters* 33(3), 1997.

## What it compares

- Training-median `DummyRegressor` baseline.
- Standardized `Ridge` regression (regularized linear model).
- `HistGradientBoostingRegressor` tree ensemble.
- Small `(64, 32)`-unit `MLPRegressor` neural network.

All models share the same training-only imputation and engineered ratios (rooms per occupant and bedrooms per room). The linear and neural pipelines also standardize predictors; tree models do not need scaling. The fixed seed is 42.

## Evaluation protocol

A 25% holdout is formed by keeping entire 0.5-degree longitude/latitude tiles out of training with `GroupShuffleSplit`. This is more demanding than a random row split because nearby census blocks are less likely to leak across train and holdout. The exact train/holdout counts and number of disjoint tiles are written to `reports/benchmark.json`.

The held-out metrics are RMSE and MAE in units of $100,000, and R². `reports/error_by_target_band.csv` summarizes errors across target-value quartiles; `reports/largest_holdout_errors.csv` exposes the 20 largest absolute residuals with their aggregate inputs. After the models are evaluated, permutation importance on up to 1,000 holdout rows is saved as a descriptive explanation only; it is not used for model selection or tuning. Fit time and serialized model size are reported for context, but are machine-dependent.

## Results from the committed run

Seed 42 trained on 14,674 rows and evaluated on 5,966 rows from 44 held-out geographic tiles (132 training tiles). The target and all reported errors are in units of $100,000:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Small MLP (64, 32) | 0.7000 | 0.4897 | 0.6907 |
| Ridge | 0.7131 | 0.5104 | 0.6790 |
| Histogram gradient boosting | 0.7635 | 0.5298 | 0.6320 |
| Training-median baseline | 1.4188 | 1.0929 | -0.2709 |

The MLP is best among these tested models, but it is not a production recommendation: its holdout RMSE is about $70,000 and MAE about $49,000 on historical values. The highest target-value quartile has about $97,000 MAE, making the limits visible rather than hiding them in one overall score. The source target is capped, and the experiment is not a current-price estimator.

Permutation importance ranked longitude, median income, and latitude highest for the selected MLP on the holdout; this is descriptive, not causal. The full per-feature values, target-band errors, and largest residual examples are committed under `reports/`. Fit times and artifact sizes are environment-specific.

## Run it

Requires Python 3.10+ and `uv`.

```bash
uv sync --extra test
uv run python -m california_housing_lab.train
uv run python -m pytest -q
uv build
```

The first training run downloads the public dataset through scikit-learn if it is not cached. Generated files go into `reports/`: benchmark JSON, metrics CSV, residual summaries, permutation importance, and a serialized fitted pipeline for each model. You can change output path or seed with `--output PATH` and `--seed INTEGER`.

## How to explain it

“I trained four regression approaches on the same public housing dataset. I kept 0.5-degree geographic tiles out of training to make the test less optimistic than a random row split, and fit imputation/scaling only on the training data. I compare RMSE, MAE, and R², inspect errors by price band and the worst residuals, then use permutation importance to describe which inputs the strongest holdout model relies on. Because this is 1990, aggregated census data with a capped target, the benchmark demonstrates ML workflow and trade-offs—not present-day property prices.”

## Reproducibility

`uv.lock` pins the tested dependency graph. The fixed split seed, model seeds, test suite, generated benchmark JSON, and GitHub Actions workflow make the experiment repeatable. The downloaded dataset itself is not committed; scikit-learn fetches the documented public source.

## Live demo

Try the [interactive GitHub Pages demo](https://krrish-ghub.github.io/california-housing-model-lab/).
