"""
Freight Rate Prediction — CatBoost with log(rate/distance) target.
Usage: python train.py
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "data", "train-test.csv")
VAL   = os.path.join(HERE, "data", "validation.csv")
DEC   = os.path.join(HERE, "data", "december-chart-inputs.csv")
OUT_VAL = os.path.join(HERE, "outputs", "validation_predictions.csv")
OUT_DEC = os.path.join(HERE, "outputs", "december-chart-inputs.csv")

SEEDS = (42, 1, 2025)
FEATURES = ["distance", "weight",
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "pickup", "delivery", "equipment",
            "dow", "is_weekend", "quarter", "quote_signal"]
CATS = ["pickup", "delivery", "equipment"]
CAT_IDX = [FEATURES.index(c) for c in CATS]


def clean(df):
    df = df.copy()
    df["weight"] = df["weight"].abs()
    df = df[df["distance"] > 0].reset_index(drop=True)
    df = df[df["posted_rate"] > 0].reset_index(drop=True)
    return df


def add_feats(df, wmed):
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["quarter"]    = d.dt.quarter
    df["dow"]        = d.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["weight"]     = df["weight"].fillna(wmed)
    return df


def main():
    train = pd.read_csv(TRAIN)
    val   = pd.read_csv(VAL)
    dec   = pd.read_csv(DEC)

    train = clean(train)
    wmed = train["weight"].median()

    train = add_feats(train, wmed)
    val   = add_feats(val, wmed)

    train["date"] = pd.to_datetime(train["date"])
    train["rpm"]     = train["posted_rate"] / train["distance"]
    train["log_rpm"] = np.log(train["rpm"])
    train = train.sort_values("date").reset_index(drop=True)
    val["date"] = pd.to_datetime(val["date"])

    # --- October holdout (measurement only) ---
    cut = pd.Timestamp("2025-10-01")
    tr = train[train["date"] <  cut]
    ho = train[train["date"] >= cut]

    best_iters = []
    for seed in SEEDS:
        m = CatBoostRegressor(
            iterations=4000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
            loss_function="MAE", eval_metric="MAE",
            od_type="Iter", od_wait=150, random_seed=seed,
            verbose=0, allow_writing_files=False)
        cut2 = int(len(tr) * 0.9)
        a, b = tr.iloc[:cut2], tr.iloc[cut2:]
        m.fit(Pool(a[FEATURES], a["log_rpm"], cat_features=CAT_IDX),
              eval_set=Pool(b[FEATURES], b["log_rpm"], cat_features=CAT_IDX),
              use_best_model=True)
        pred = np.exp(m.predict(Pool(ho[FEATURES], cat_features=CAT_IDX))) * ho["distance"].values
        err = np.abs(ho["posted_rate"].values - pred)
        body = ho["posted_rate"].values <= np.percentile(ho["posted_rate"], 99)
        print(f"seed {seed:>4}: overall ${err.mean():6.2f}  body ${err[body].mean():5.2f}  median ${np.median(err):5.2f}")
        best_iters.append(m.get_best_iteration())

    n_iter = int(np.median(best_iters) * 1.15)
    print(f"Final iterations: {n_iter}")

    # --- Final model on 100% of data ---
    models = []
    for seed in SEEDS:
        m = CatBoostRegressor(
            iterations=n_iter, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
            loss_function="MAE", random_seed=seed, verbose=0,
            allow_writing_files=False)
        m.fit(Pool(train[FEATURES], train["log_rpm"], cat_features=CAT_IDX))
        models.append(m)

    def predict(df):
        preds = []
        for m in models:
            raw = m.predict(Pool(df[FEATURES], cat_features=CAT_IDX))
            preds.append(np.exp(raw) * df["distance"].values)
        return np.clip(np.median(np.vstack(preds), axis=0), 1.0, None)

    # --- Validation ---
    val["predicted_rate"] = predict(val)
    val[["load_id", "predicted_rate"]].to_csv(OUT_VAL, index=False)
    print(f"Wrote {OUT_VAL}")

    # --- December ---
    dec["date"] = pd.to_datetime(dec["date"])
    dec = add_feats(dec, wmed)
    pickup_coords   = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    delivery_coords = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    dec = dec.merge(pickup_coords,   left_on="pickup",   right_index=True, how="left")
    dec = dec.merge(delivery_coords, left_on="delivery", right_index=True, how="left")

    lex_qs = train.loc[
        (train["pickup"] == "Lexington") &
        (train["date"] >= train["date"].max() - pd.Timedelta(days=60)),
        "quote_signal"
    ].median()
    if pd.isna(lex_qs):
        lex_qs = train["quote_signal"].median()
    dec["quote_signal"] = lex_qs

    dec["predicted_rate"] = predict(dec)
    dec[["pickup", "delivery", "distance", "equipment",
         "weight", "date", "predicted_rate"]].to_csv(OUT_DEC, index=False)
    print(f"Wrote {OUT_DEC}")


if __name__ == "__main__":
    main()
