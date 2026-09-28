"""Train a reproducible bot-detection model and write submission.csv.

The script intentionally reads only local competition files.  Event features are
computed after joining each event to its cookie's observation window, then the
window predicate is applied before any aggregation.  This prevents leakage from
events before or after the day being scored.

Run: python train.py
Requires: Python 3.10+; pandas, numpy, scikit-learn.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score

from metric import precision_at_recall

SEED = 2026
DATA = Path("data")


def _safe(value: object) -> str:
    """Make category values safe and stable feature-name suffixes."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).lower()).strip("_")[:60]


def _add_count_table(features: pd.DataFrame, events: pd.DataFrame, column: str,
                     prefix: str, categories: list[str] | None = None) -> pd.DataFrame:
    """Add per-cookie counts for a small categorical variable."""
    # Values such as WEB/Web/web are the same semantic platform.  Normalize
    # before crosstab so feature names stay unique and their evidence is pooled.
    values = events[column].fillna("__missing__").map(_safe)
    table = pd.crosstab(events.cookie_id, values)
    if categories is not None:
        table = table.reindex(columns=categories, fill_value=0)
    table.columns = [f"{prefix}_{c}_n" for c in table.columns]
    return features.join(table, how="left")


def make_features(meta: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return one numeric feature row per cookie in *meta*.

    `meta` may contain train and test rows together.  This is useful because it
    makes the unsupervised category vocabulary identical for both data sets;
    target labels are never used here.
    """
    base = meta[["cookie_id", "cookie_created_at", "window_start_ts", "window_end_ts"]].copy()
    ev = events.merge(base[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id", how="inner")
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)].copy()
    ev.sort_values(["cookie_id", "event_ts"], inplace=True)

    age_hours = (base.window_start_ts - base.cookie_created_at).dt.total_seconds() / 3600
    f = pd.DataFrame({"cookie_id": base.cookie_id, "cookie_age_hours": age_hours.clip(lower=0)})
    f = f.set_index("cookie_id")
    g = ev.groupby("cookie_id", sort=False)

    # Volume, diversity and missingness distinguish repeated automatic browsing
    # from varied human sessions, including cookies without events in the window.
    aggregate = g.agg(
        n_events=("eid", "size"), item_nunique=("item_id", "nunique"),
        category_nunique=("item_category", "nunique"), location_nunique=("item_location", "nunique"),
        query_nunique=("search_query", "nunique"), ua_nunique=("user_agent", "nunique"),
        platform_nunique=("platform", "nunique"), seller_type_nunique=("seller_type", "nunique"),
        item_missing_rate=("item_id", lambda s: s.isna().mean()),
        query_missing_rate=("search_query", lambda s: s.isna().mean()),
        pointer_present_rate=("pointer_x", lambda s: s.notna().mean()),
        search_page_mean=("search_page", "mean"), search_page_max=("search_page", "max"),
        pointer_x_std=("pointer_x", "std"), pointer_y_std=("pointer_y", "std"),
    )
    f = f.join(aggregate, how="left")

    ev["offset_s"] = (ev.event_ts - ev.window_start_ts).dt.total_seconds()
    ev["gap_s"] = g.event_ts.diff().dt.total_seconds()
    temporal = ev.groupby("cookie_id").agg(
        first_event_offset_s=("offset_s", "min"), last_event_offset_s=("offset_s", "max"),
        active_span_s=("offset_s", lambda s: s.max() - s.min()),
        gap_mean_s=("gap_s", "mean"), gap_median_s=("gap_s", "median"), gap_min_s=("gap_s", "min"),
        gap_std_s=("gap_s", "std"),
    )
    f = f.join(temporal, how="left")
    f["events_per_active_hour"] = f.n_events / (1 + f.active_span_s / 3600)
    f["items_per_event"] = f.item_nunique / f.n_events
    f["queries_per_event"] = f.query_nunique / f.n_events

    # Event types and compact categorical dimensions are one-hot count features.
    for column, prefix in [("event_name", "event"), ("platform", "platform"),
                           ("seller_type", "seller"), ("item_category", "category")]:
        f = _add_count_table(f, ev, column, prefix)

    # Consecutive event pairs preserve a lightweight representation of navigation.
    ev["next_event"] = g.event_name.shift(-1)
    pairs = ev.loc[ev.next_event.notna(), ["cookie_id", "event_name", "next_event"]].copy()
    pairs["pair"] = pairs.event_name.astype(str) + "__to__" + pairs.next_event.astype(str)
    f = _add_count_table(f, pairs.rename(columns={"pair": "transition"}), "transition", "transition")

    ua = ev.user_agent.fillna("").str.lower()
    for token in ["bot", "headless", "selenium", "webdriver", "curl", "python", "googlebot", "iphone", "android"]:
        f[f"ua_has_{token}"] = ev.assign(flag=ua.str.contains(token, regex=False).astype(int)).groupby("cookie_id").flag.max()
    query = ev.search_query.fillna("")
    query_stats = ev.assign(
        query_length=query.str.len(), query_digit_rate=query.str.count(r"\d") / query.str.len().replace(0, np.nan)
    ).groupby("cookie_id").agg(query_length_mean=("query_length", "mean"), query_length_max=("query_length", "max"),
                               query_digit_rate_mean=("query_digit_rate", "mean"))
    f = f.join(query_stats, how="left")

    f = f.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()
    # Numeric conversion makes the model input explicit and catches accidental objects.
    for col in f.columns:
        if col != "cookie_id":
            f[col] = pd.to_numeric(f[col], errors="coerce").fillna(0).astype("float32")
    return f


def report_validation(X: pd.DataFrame, train: pd.DataFrame) -> None:
    """Time-based diagnostic: last three labeled days are never used for fitting."""
    dates = sorted(train.window_start_ts.unique())
    valid_dates = dates[-3:]
    valid = train.window_start_ts.isin(valid_dates).to_numpy()
    cols = [c for c in X.columns if c != "cookie_id"]
    baseline = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    # Baseline: one model family on the complete engineered feature table.
    baseline.fit(X.loc[~valid, cols], train.loc[~valid, "target"])
    baseline_pred = baseline.predict_proba(X.loc[valid, cols])[:, 1]
    extra = ExtraTreesClassifier(
        n_estimators=700, min_samples_leaf=1, max_features=0.8,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    ).fit(X.loc[~valid, cols], train.loc[~valid, "target"])
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=350, max_leaf_nodes=20, l2_regularization=2.0,
        random_state=SEED,
    ).fit(X.loc[~valid, cols], train.loc[~valid, "target"],
          sample_weight=np.where(train.loc[~valid, "target"].to_numpy() == 1, 6.0, 1.0))
    pred = 0.7 * extra.predict_proba(X.loc[valid, cols])[:, 1] + 0.3 * hgb.predict_proba(X.loc[valid, cols])[:, 1]
    y = train.loc[valid, "target"]
    print("Validation (last 3 days):")
    print(f"  Baseline P@R>=0.70: {precision_at_recall(y, baseline_pred):.4f}")
    print(f"  P@R>=0.70: {precision_at_recall(y, pred):.4f}")
    print(f"  PR-AUC:     {average_precision_score(y, pred):.4f}")
    print(f"  ROC-AUC:    {roc_auc_score(y, pred):.4f}")


def main() -> None:
    train = pd.read_csv(DATA / "train.csv", parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
    test = pd.read_csv(DATA / "test.csv", parse_dates=["cookie_created_at", "window_start_ts", "window_end_ts"])
    events = pd.read_csv(DATA / "events.csv.gz", parse_dates=["event_ts"])
    all_features = make_features(pd.concat([train.drop(columns="target"), test], ignore_index=True), events)
    X_train = all_features.iloc[:len(train)].reset_index(drop=True)
    X_test = all_features.iloc[len(train):].reset_index(drop=True)
    report_validation(X_train, train)

    cols = [c for c in X_train.columns if c != "cookie_id"]
    # Extra Trees captures threshold/interaction rules; HGB adds a smoother,
    # complementary ranking.  Blending only ranks probabilities, the submission
    # itself remains a calibrated score in [0, 1].
    extra = ExtraTreesClassifier(
        n_estimators=1000, min_samples_leaf=1, max_features=0.8,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    ).fit(X_train[cols], train.target)
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=350, max_leaf_nodes=20, l2_regularization=2.0,
        random_state=SEED,
    ).fit(X_train[cols], train.target,
          sample_weight=np.where(train.target.to_numpy() == 1, 6.0, 1.0))
    score = 0.7 * extra.predict_proba(X_test[cols])[:, 1] + 0.3 * hgb.predict_proba(X_test[cols])[:, 1]
    submission = pd.DataFrame({"cookie_id": test.cookie_id, "score": np.clip(score, 0, 1)})
    assert submission.cookie_id.is_unique and len(submission) == len(test)
    assert submission.score.notna().all() and submission.score.between(0, 1).all()
    submission.to_csv("submission.csv", index=False)
    print(f"Wrote submission.csv: {len(submission)} rows, score range {score.min():.4f}..{score.max():.4f}")


if __name__ == "__main__":
    main()
