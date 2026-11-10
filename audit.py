"""Data audit for Rawaj Pay transfer extracts.

You write this file. Keep the signature exactly as it is: the grader calls it by name.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _missing_mask(series: pd.Series) -> pd.Series:
    """Treat NaN/None and empty or whitespace-only strings as missing."""
    blank_strings = series.map(lambda value: isinstance(value, str) and not value.strip())
    return series.isna() | blank_strings


def _sort_values(values: list) -> list:
    """Sort ordinary values naturally, with a stable fallback for mixed types."""
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    """Audit a transfer extract before anyone trains a model on it."""
    for column in (target, id_col, time_col):
        if column not in df.columns:
            raise KeyError(column)

    n_rows = int(len(df))

    # Use the same definition of missingness everywhere in the audit.
    missing_masks = {column: _missing_mask(df[column]) for column in df.columns}
    missing = {
        column: float(mask.mean()) if n_rows else 0.0
        for column, mask in missing_masks.items()
    }

    # Duplicate IDs: ignore missing/blank IDs, and return each repeated ID once.
    valid_ids = df.loc[~missing_masks[id_col], id_col]
    duplicate_ids = _sort_values(valid_ids[valid_ids.duplicated(keep=False)].unique().tolist())

    # Count rows that exactly repeat an earlier complete row.
    duplicate_rows = int(df.duplicated(keep="first").sum())

    # Constant columns ignore missing values; the target is always excluded.
    constant_columns = []
    for column in df.columns:
        if column == target:
            continue
        non_missing_values = df.loc[~missing_masks[column], column]
        if non_missing_values.nunique(dropna=True) <= 1:
            constant_columns.append(column)
    constant_columns = _sort_values(constant_columns)

    target_missing = missing_masks[target]
    labelled = df.loc[~target_missing]
    unlabelled_rows = int(target_missing.sum())
    positive_rate = (
        float((labelled[target] == 1).mean()) if len(labelled) else 0.0
    )

    # Parse only non-missing timestamps. Invalid non-missing strings count as bad.
    time_missing = missing_masks[time_col]
    parsed_times = pd.to_datetime(df.loc[~time_missing, time_col], errors="coerce")
    bad_timestamps = int(parsed_times.isna().sum())
    good_times = parsed_times.dropna()
    if len(good_times):
        time_range = [pd.Timestamp(good_times.min()).isoformat(),
                      pd.Timestamp(good_times.max()).isoformat()]
    else:
        time_range = [None, None]

    # Leakage checks are meaningful only with enough labelled examples of both classes.
    leakage_suspects = []
    if len(labelled) >= 20:
        # The task's labels are binary. Filter explicitly so unexpected label values
        # cannot make ROC AUC fail or silently become a third class.
        labelled_positions = np.flatnonzero((~target_missing).to_numpy(dtype=bool))
        labelled_values = df[target].to_numpy()[labelled_positions]
        binary_positions = labelled_positions[np.isin(labelled_values, [0, 1])]
        y_all = df[target].to_numpy()[binary_positions].astype(int)
        if len(binary_positions) >= 20 and np.unique(y_all).size == 2:
            for column in df.columns:
                if column in (target, id_col, time_col):
                    continue

                # For the leakage exception, missing is itself a distinct value.
                # Count distinct values over the whole column, treating all missing
                # representations as one value. This is deliberately different from
                # constant_columns, which ignores missing values.
                all_column_missing = missing_masks[column].to_numpy(dtype=bool)
                distinct_nonmissing = df.loc[~missing_masks[column], column].nunique(dropna=True)
                distinct_with_missing = int(distinct_nonmissing) + int(all_column_missing.any())
                if distinct_with_missing <= 1:
                    continue

                is_suspect = False
                # Use positional arrays here so duplicate DataFrame index labels cannot
                # accidentally duplicate or misalign rows in the leakage calculation.
                column_missing = all_column_missing[binary_positions]
                column_values = df[column].to_numpy()[binary_positions]

                # (a) A numeric/boolean value that nearly determines the label.
                is_numeric_or_bool = (
                    pd.api.types.is_numeric_dtype(df[column].dtype)
                    or pd.api.types.is_bool_dtype(df[column].dtype)
                )
                if is_numeric_or_bool:
                    value_mask = ~column_missing
                    if int(value_mask.sum()) >= 20:
                        y_values = y_all[value_mask]
                        # ROC AUC is undefined if only one label remains.
                        if np.unique(y_values).size == 2:
                            x_values = column_values[value_mask]
                            try:
                                auc = float(roc_auc_score(y_values, x_values))
                                if auc >= 0.98 or auc <= 0.02:
                                    is_suspect = True
                            except (TypeError, ValueError):
                                # A malformed numeric column should not break the whole audit.
                                pass

                # (b) Whether a value is missing can itself reveal the label.
                missing_indicator = column_missing.astype(int)
                try:
                    missing_auc = float(roc_auc_score(y_all, missing_indicator))
                    if missing_auc >= 0.98 or missing_auc <= 0.02:
                        is_suspect = True
                except (TypeError, ValueError):
                    pass

                if is_suspect:
                    leakage_suspects.append(column)

    leakage_suspects = _sort_values(leakage_suspects)

    return {
        "n_rows": n_rows,
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }
