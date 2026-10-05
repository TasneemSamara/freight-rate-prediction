# Freight Rate Prediction

Predict `posted_rate` for US freight loads using a CatBoost regressor trained on
`log(rate / distance)` (log dollars per mile).

## Results (October holdout)

| Metric | Value |
|---|---|
| Body MAE (99% of loads) | **$50.48** |
| Median absolute error  | $19.13 |
| Overall MAE (incl. 1% tail) | $89.41 |

## Quickstart

```bash
git clone <this-repo>
cd freight-rate-prediction
pip install -r requirements.txt
python train.py
python scorer/score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december-chart-inputs.csv
```

## Data

Place the following files in `data/`:
- `train-test.csv` — 48,000 historical loads (Jan–Oct 2025)
- `validation.csv` — 12,000 loads to predict (Nov–Dec 2025)
- `december-chart-inputs.csv` — 31 rows, one per December day

## Pipeline

1. **Clean** — abs(weight), drop distance <= 0 or rate <= 0
2. **Feature engineering** — `dow`, `is_weekend`, `quarter`, impute weight with median
3. **Target** — `log(posted_rate / distance)`
4. **Holdout** — train Jan–Sep, test Oct (chronological, no leakage)
5. **Final model** — 3 seeds x fixed iterations on 100% of data; median across seeds
6. **December** — fill missing coords from training lookup; impute `quote_signal` with recent Lexington median

## Model

`CatBoostRegressor(loss_function='MAE', learning_rate=0.05, depth=8, l2_leaf_reg=3.0)`,
3 seeds (42, 1, 2025), median-averaged predictions.

## Why this target

Distance alone explains ~90% of rate variance. Dividing distance out before
logging forces the model to learn the *marginal* pricing problem
(dollars per mile), which drops body MAE from $88.97 to $53.95 on the same
features.

## Why not seasonal features

`day`, `doy`, and `month` were tested and each worsened holdout MAE by
$10-15, because the model has never seen December in training and would
extrapolate calendar patterns from Jan-Oct. The December chart therefore
varies only by day-of-week.
