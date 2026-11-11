"""Train and compare all regression models for transformer Health Index prediction.

Usage (from the repository root):
    python -m src.train                      # uses data/Health index1.csv
    python -m src.train --data path/to.csv   # custom path
"""
import argparse
import re
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split

from . import config
from .data import load_data, preprocess, split
from .evaluate import comparison_plot, evaluate, parity_plot
from .models import get_models


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=config.DATA_PATH, help="Path to the dataset CSV")
    args = parser.parse_args()

    out_fig = config.FIGURES_DIR / "parity_generated"
    out_fig.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)

    X, y = preprocess(load_data(args.data))
    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Features: {X.shape[1]} | Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    results, fitted = [], {}

    # Model 1: Simple linear regression on the single most-correlated feature
    best_feat = X.corrwith(y).abs().idxmax()
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X[[best_feat]], y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    slr = LinearRegression().fit(Xs_tr, ys_tr)
    results.append(evaluate("Simple Linear Regression", ys_tr, slr.predict(Xs_tr), ys_te, slr.predict(Xs_te)))
    print(f"Simple Linear Regression uses '{best_feat}'")

    # Models 2-13
    for name, (est, grid) in get_models().items():
        t0 = time.time()
        if grid:
            search = GridSearchCV(clone(est), grid, scoring="r2", cv=config.CV_FOLDS, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            extra = f" best={search.best_params_}"
        else:
            model = clone(est).fit(X_train, y_train)
            extra = ""
        p_tr, p_te = model.predict(X_train), model.predict(X_test)
        res = evaluate(name, y_train, p_tr, y_test, p_te)
        results.append(res)
        fitted[name] = model
        parity_plot(name, y_train, p_tr, y_test, p_te, out_fig / f"{slug(name)}.png")
        print(f"{name:<30} test R²={res['Test_R2']:.4f}  [{res['Condition']}]  ({time.time() - t0:.1f}s){extra}")

    results_df = pd.DataFrame(results).set_index("Model").sort_values("Test_R2", ascending=False)
    results_df.to_csv(config.RESULTS_DIR / "model_comparison_generated.csv")
    comparison_plot(results_df, config.FIGURES_DIR / "comparison_generated.png")

    best = results_df.index[0]
    # Save the best model that uses all features (the single-feature baseline is not deployable)
    best_saved = next(m for m in results_df.index if m in fitted)
    joblib.dump({"model": fitted[best_saved], "features": list(X.columns)}, config.MODELS_DIR / "best_model.joblib")

    print("\n" + results_df.to_string())
    print(f"\nBest model: {best}")
    print(f"Saved to models/best_model.joblib: {best_saved}")


if __name__ == "__main__":
    main()
