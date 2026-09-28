# Avito bot-detection solution

## Reproduction

Use Python 3.10+ (the solution was developed with Python 3.14), install the
dependencies and run:

```bash
pip install -r requirements.txt
python train.py
```

The command reads the issued files from `data/` and creates `submission.csv` in
the repository root.  All random components use `SEED = 2026`.

## Approach

For every cookie, events are first restricted to its own half-open observation
window `[window_start_ts, window_end_ts)`.  I aggregate event volume, event-type
counts, unique ads/categories/locations/queries, item and query missingness,
platform and seller-type counts, User-Agent flags, pointer presence and spread,
search-page statistics, query length, and temporal activity features (first and
last event, active duration and inter-event gaps).  Counts of consecutive event
types describe short navigation patterns.

The final score blends Extra Trees (70%) and class-balanced histogram gradient
boosting (30%).
Both are classical local scikit-learn models; there are no external services or
language models.  The blend was chosen to combine rule-like behavioural splits
with smoother score ranking.

## Validation and diagnostics

The validation fold is temporal: the last three labeled daily windows are held
out, while all earlier labeled days are used for fitting.  This mirrors scoring
on later unseen days and avoids randomly mixing neighbouring windows.  `train.py`
prints the official `Precision @ Recall >= 0.70`, PR-AUC and ROC-AUC for that
split using the supplied `metric.py` implementation. On the fixed development
split, a Random Forest baseline on the engineered feature table obtains
**0.6222** P@R≥0.70 and the final blend
obtains **0.6871**.  The exact figures are diagnostics rather than estimates of
the hidden-test score. The final model extends the baseline with event
composition, timing, navigation, UA and interaction features.

## Limitations

The labels identify known automated sources, so the model can be less sensitive
to entirely new bot behaviour.  Features are aggregated per one-day window and
therefore do not use behavioural history outside the allowed window.
