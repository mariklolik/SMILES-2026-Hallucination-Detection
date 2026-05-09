from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    n_splits = 5
    n_repeats = 5
    folds: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for r in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + r
        )
        for idx_train, idx_test in skf.split(np.zeros((len(y), 1)), y):
            folds.append(
                (idx_train.astype(np.int64), None, idx_test.astype(np.int64))
            )
    return folds
