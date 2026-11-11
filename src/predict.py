"""Predict the Health Index for new transformer oil samples.

Usage:
    python -m src.predict --input new_samples.csv [--output predictions.csv]

The input CSV must contain the 14 feature columns listed in src/config.py.
Run `python -m src.train` first to create models/best_model.joblib.
"""
import argparse

import joblib
import pandas as pd

from . import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default=config.MODELS_DIR / "best_model.joblib")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.input)
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    df["Predicted Health index"] = bundle["model"].predict(df[bundle["features"]]).round(2)
    df.to_csv(args.output, index=False)
    print(df[["Predicted Health index"]].describe())
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
