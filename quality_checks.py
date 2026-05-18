"""
quality_checks.py

Four independent data-quality checks (schema, completeness, freshness,
category drift) plus a composite trust-score calculator. Each check
returns a dict with at minimum a boolean "passed" key, so they can be
composed into a scorecard or run in a batch validation loop.

Validated against both known-clean and known-corrupted test fixtures
(see notebooks/ for the full validation results) — not just written
and assumed to work.
"""
import pandas as pd


def check_schema(df, expected_columns):
    """Flags missing or unexpected columns against an expected contract."""
    actual = set(df.columns)
    expected = set(expected_columns)
    missing = expected - actual
    extra = actual - expected
    return {
        "passed": len(missing) == 0,
        "missing_columns": list(missing),
        "extra_columns": list(extra),
    }


def check_nulls(df, threshold=0.02):
    """
    Flags any column whose null rate exceeds `threshold`.

    Note: threshold should be calibrated relative to the expected
    injection/contamination rate of the data being checked. A 2% default
    is appropriate for single-record validation; batch-scale validation
    against data with a known ~5% corruption rate should use a higher
    threshold (e.g. 0.06) explicitly, rather than changing this default
    globally — see docs/build_guide.md, Phase 6, for why.
    """
    null_rates = df.isnull().mean()
    flagged = null_rates[null_rates > threshold]
    return {
        "passed": len(flagged) == 0,
        "null_rates": null_rates.to_dict(),
        "flagged_columns": flagged.to_dict(),
    }


def check_freshness(df, timestamp_col, max_age_hours=6):
    """Flags rows whose timestamp exceeds max_age_hours from now."""
    from datetime import datetime
    age_hours = (datetime.now() - pd.to_datetime(df[timestamp_col])).dt.total_seconds() / 3600
    stale = age_hours > max_age_hours
    return {
        "passed": stale.sum() == 0,
        "stale_row_count": int(stale.sum()),
        "stale_row_pct": float(stale.mean()),
    }


def check_category_drift(baseline_df, new_df, category_col, threshold=0.05):
    """
    Compares category proportions between a baseline and a new batch.

    Threshold should be calibrated to the actual category distribution's
    spread — a generic 15-point threshold is meaningless if real category
    shares range from under 5% to nearly 30%, as in this project's dataset.
    """
    baseline_dist = baseline_df[category_col].value_counts(normalize=True)
    new_dist = new_df[category_col].value_counts(normalize=True)
    combined = pd.DataFrame({'baseline': baseline_dist, 'new': new_dist}).fillna(0)
    combined['abs_diff'] = (combined['baseline'] - combined['new']).abs()
    drifted = combined[combined['abs_diff'] > threshold]
    return {
        "passed": len(drifted) == 0,
        "drifted_categories": drifted.to_dict('index'),
    }


def calculate_trust_score(schema_result, null_result, freshness_result):
    """
    Composite 0-100 trust score. Schema and freshness failures are
    binary penalties (a broken contract or stale data is a hard fail);
    null-rate failure is a proportional penalty capped at 30, since
    completeness issues exist on a spectrum of severity.
    """
    score = 100
    if not schema_result['passed']:
        score -= 30
    null_rate_avg = sum(null_result['null_rates'].values()) / len(null_result['null_rates'])
    score -= min(null_rate_avg * 100, 30)
    if not freshness_result['passed']:
        score -= 20
    return max(round(score, 1), 0)
