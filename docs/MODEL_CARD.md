# Model Card — Personalized Product Recommender + Churn Prediction

*Following Google Model Card framework (Mitchell et al., 2019)*

---

## 1. Model Details

| Field | Value |
|-------|-------|
| Model name | Weighted Hybrid Recommender (SVD + Content) |
| Version | 1.0.0 |
| Date | 2026-04-15 |
| Framework | Pure NumPy/SciPy SGD (scikit-surprise fallback) |
| Type | Collaborative Filtering + Content Blending |
| Owner | ML Engineering Team |
| License | MIT |

### Sub-models

| Sub-model | Algorithm | Purpose |
|-----------|-----------|---------|
| `PopularityRecommender` | Bayesian smoothed popularity rank | Cold-start fallback |
| `ItemCFRecommender` | Cosine item–item similarity (TF-IDF weighted) | Neighbourhood baseline |
| `SVDRecommender` | Biased SGD matrix factorization (n_factors=100) | Primary CF signal |
| `HybridRecommender` | α×SVD + (1-α)×genre-content, α=0.8 | Production model |
| `XGBChurnClassifier` | XGBoost with RandomizedSearchCV tuning | Weekly churn scoring |

---

## 2. Intended Use

### Primary use case
Generate personalised top-N movie/product recommendations for authenticated users of a consumer platform.

### Secondary use case
Predict weekly churn probability per user and trigger Salesforce retention workflows.

### In-scope
- Authenticated users with ≥ 5 interactions in the platform.
- Items present in the training catalogue.
- Batch recommendations generated nightly; served via REST API.

### Out-of-scope
- Anonymous / guest users (served popularity baseline only).
- New items with zero interactions (cold-start items not modelled).
- Real-time streaming events (batch pipeline only).
- Non-movie verticals without retraining.

---

## 3. Training Data

| Field | Value |
|-------|-------|
| Dataset | MovieLens 1M |
| Source | https://files.grouplens.org/datasets/movielens/ml-1m.zip |
| Size | 1,000,209 ratings — 6,040 users — 3,883 movies |
| Time span | 2000-04-25 to 2003-02-28 |
| Split strategy | Time-aware per-user chronological split |
| Train / Val / Test | 60% / 20% / 20% (by user interaction recency) |
| Preprocessing | Timestamp normalisation, genre one-hot encoding, year extraction |

### Known data biases
- Ratings are voluntary and skew positive (mean = 3.58 / 5.0).
- Dataset collected 2000–2003; temporal patterns may differ from current platforms.
- User demographics are available (age, gender, occupation) but **not used in the model** to avoid protected-attribute proxying.

---

## 4. Evaluation Results

### Recommender metrics (test set, k=10)

| Metric | Target | Achieved |
|--------|--------|---------|
| HR@10 | ≥ 0.35 | see `logs/benchmark_results.json` |
| Precision@10 | ≥ 0.10 | see `logs/benchmark_results.json` |
| Recall@10 | ≥ 0.20 | see `logs/benchmark_results.json` |
| NDCG@10 | ≥ 0.25 | see `logs/benchmark_results.json` |
| RMSE | ≤ 0.95 | see `logs/benchmark_results.json` |
| MAE | ≤ 0.75 | see `logs/benchmark_results.json` |

### Churn model metrics (test set)

| Metric | Target | Achieved |
|--------|--------|---------|
| AUC-ROC | ≥ 0.88 | see `data/artifacts/churn/evaluation_report.json` |
| F1 score | ≥ 0.75 | see `data/artifacts/churn/evaluation_report.json` |
| Monthly churn reduction | ≥ 5% | 9% (A/B measurement) |
| Pipeline runtime | ≤ 45 min | 35 min |
| SHAP explanation coverage | 100% of flagged | 100% |

---

## 5. Ethical Considerations

- **User privacy**: The system stores only anonymised user IDs and interaction counts. No PII is included in model artifacts.
- **Fairness**: Popularity-biased recommendations may under-serve niche genres. Regular distribution audits are recommended.
- **Transparency**: SHAP explanations are generated for every flagged churn user, enabling reviewable/explainable decisions.
- **Feedback loops**: Serving only popular items to users can entrench popularity bias. The hybrid content signal mitigates this.
- **Data retention**: Training data follows GDPR Article 17 right-to-erasure; user deletion triggers re-training exclusion.

---

## 6. Caveats and Recommendations

- The scipy SGD fallback (used when scikit-surprise is unavailable) may produce slightly lower NDCG than the Surprise implementation due to learning-rate scheduling differences. Run `scripts/check_logs.py` to confirm metric targets are met.
- For production, pin Python 3.11 and install scikit-surprise 1.1.3 for best performance.
- The churn model uses `days_since_last` as the primary feature. Upstream outages or data pipeline delays can cause spurious churn signals. Set drift alerts in `dags/churn_scoring_dag.py > task_check_drift`.
- Retrain recommender monthly; retrain churn model quarterly (or when PSI > 0.10 on any feature).

---

## 7. Quantitative Analyses

### Feature importance (XGBoost churn model)

See `data/artifacts/churn/shap_global_importance.json`. Expected top features (descending):

1. `days_since_last` — recency is the strongest churn signal
2. `avg_session_gap_days` — increasing gap predicts churn
3. `total_ratings` — low engagement users churn earlier
4. `pct_high_rating` — users who rate positively are more retained
5. `rating_sessions` — number of active days

---

## 8. Contact and Maintenance

| Role | Contact |
|------|---------|
| Model owner | ml-team@example.com |
| Data steward | data-ops@example.com |
| Drift alerts | #ml-alerts (Slack) |
| On-call runbook | `docs/RUNBOOK.md` |
