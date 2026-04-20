# Experiment Report — Personalized Product Recommender System

> **Generated:** 2026-04-15 05:21 UTC
> **Evaluation cutoff K:** 10

---

## Model Comparison

| Model | Precision@10 | Recall@10 | HR@10 | NDCG@10 | MAP@10 | RMSE |
|-------|-------------|-----------|-------|---------|--------|------|
| popularity | 0.0856 | 0.0334 | 0.4258 | 0.0917 | 0.0424 | nan |
| item_cf | 0.0122 | 0.0029 | 0.0866 | 0.0118 | 0.0042 | nan |
| svd | 0.0436 | 0.0150 | 0.2647 | 0.0467 | 0.0191 | 0.9262 |
| svdpp | 0.0453 | 0.0159 | 0.2641 | 0.0476 | 0.0199 | 0.9238 |
| hybrid | 0.0493 | 0.0190 | 0.3031 | 0.0539 | 0.0223 | nan |

---

## Success Threshold Check

| Metric | Target | Best Model | Value | Status |
|--------|--------|------------|-------|--------|
| HR@10 | >= 0.35 | popularity | 0.4258 | ✅ MET |
| Precision@10 | >= 0.1 | popularity | 0.0856 | ❌ GAP=0.0144 |
| Recall@10 | >= 0.2 | popularity | 0.0334 | ❌ GAP=0.1666 |

---

## Per-Model Detail

### popularity

- **precision@5:** 0.093212
- **recall@5:** 0.018682
- **hr@5:** 0.305795
- **ndcg@5:** 0.095719
- **map@5:** 0.057212
- **users_evaluated:** 6040
- **precision@10:** 0.085646
- **recall@10:** 0.033438
- **hr@10:** 0.425828
- **ndcg@10:** 0.091731
- **map@10:** 0.042371
- **precision@20:** 0.042823
- **recall@20:** 0.033438
- **hr@20:** 0.425828
- **ndcg@20:** 0.064914
- **map@20:** 0.024432
- **eval_time_s:** 6.04
- **model_name:** popularity
- **n_users_evaluated:** 6040

**Threshold Status:**
- hr@10_met: ✅
- precision@10_met: ❌
- recall@10_met: ❌

### item_cf

- **precision@5:** 0.011589
- **recall@5:** 0.00139
- **hr@5:** 0.047351
- **ndcg@5:** 0.011215
- **map@5:** 0.005674
- **users_evaluated:** 6040
- **precision@10:** 0.012169
- **recall@10:** 0.002925
- **hr@10:** 0.086589
- **ndcg@10:** 0.011819
- **map@10:** 0.004212
- **precision@20:** 0.006084
- **recall@20:** 0.002925
- **hr@20:** 0.086589
- **ndcg@20:** 0.007872
- **map@20:** 0.002204
- **eval_time_s:** 39.32
- **model_name:** item_cf
- **n_users_evaluated:** 6040

**Threshold Status:**
- hr@10_met: ❌
- precision@10_met: ❌
- recall@10_met: ❌

### svd

- **precision@5:** 0.048344
- **recall@5:** 0.00839
- **hr@5:** 0.17947
- **ndcg@5:** 0.04967
- **map@5:** 0.027332
- **users_evaluated:** 6040
- **precision@10:** 0.043609
- **recall@10:** 0.015038
- **hr@10:** 0.264735
- **ndcg@10:** 0.046729
- **map@10:** 0.019144
- **precision@20:** 0.021805
- **recall@20:** 0.015038
- **hr@20:** 0.264735
- **ndcg@20:** 0.0325
- **map@20:** 0.010751
- **rmse:** 0.926178
- **mae:** 0.726909
- **n_predictions:** 5000
- **eval_time_s:** 155.7
- **model_name:** svd
- **n_users_evaluated:** 6040

**Threshold Status:**
- hr@10_met: ❌
- precision@10_met: ❌
- recall@10_met: ❌

### svdpp

- **precision@5:** 0.048477
- **recall@5:** 0.008131
- **hr@5:** 0.175828
- **ndcg@5:** 0.049242
- **map@5:** 0.027437
- **users_evaluated:** 6040
- **precision@10:** 0.045265
- **recall@10:** 0.015872
- **hr@10:** 0.264073
- **ndcg@10:** 0.047604
- **map@10:** 0.019916
- **precision@20:** 0.022632
- **recall@20:** 0.015872
- **hr@20:** 0.264073
- **ndcg@20:** 0.033004
- **map@20:** 0.011047
- **rmse:** 0.9238
- **mae:** 0.724115
- **n_predictions:** 5000
- **eval_time_s:** 149.11
- **model_name:** svdpp
- **n_users_evaluated:** 6040

**Threshold Status:**
- hr@10_met: ❌
- precision@10_met: ❌
- recall@10_met: ❌

### hybrid

- **precision@5:** 0.054967
- **recall@5:** 0.010701
- **hr@5:** 0.206788
- **ndcg@5:** 0.057227
- **map@5:** 0.03176
- **users_evaluated:** 6040
- **precision@10:** 0.049338
- **recall@10:** 0.018981
- **hr@10:** 0.303146
- **ndcg@10:** 0.053941
- **map@10:** 0.022316
- **precision@20:** 0.024669
- **recall@20:** 0.018981
- **hr@20:** 0.303146
- **ndcg@20:** 0.038243
- **map@20:** 0.013044
- **eval_time_s:** 185.34
- **model_name:** hybrid
- **n_users_evaluated:** 6040

**Threshold Status:**
- hr@10_met: ❌
- precision@10_met: ❌
- recall@10_met: ❌
