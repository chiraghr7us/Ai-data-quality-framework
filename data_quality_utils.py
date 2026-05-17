import numpy as np
import pandas as pd

def corrupt_dataset(df, null_rate=0.05, drop_column_prob=0.3, schema_drift_prob=0.2, seed=1):
    rng = np.random.default_rng(seed)
    corrupted = df.copy()

    for col in corrupted.select_dtypes(include=['float64', 'object']).columns[:3]:
        mask = rng.random(len(corrupted)) < null_rate
        corrupted.loc[mask, col] = np.nan

    if rng.random() < drop_column_prob:
        drop_col = rng.choice(corrupted.columns)
        corrupted = corrupted.drop(columns=[drop_col])
        print(f"[corruption] dropped column: {drop_col}")

    if rng.random() < schema_drift_prob:
        cols = list(corrupted.columns)
        target = rng.choice(cols)
        corrupted = corrupted.rename(columns={target: target + "_v2"})
        print(f"[corruption] renamed column: {target} -> {target}_v2")

    return corrupted