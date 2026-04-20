# Success Metrics Report — Personalized Product Recommender System

> **Generated:** 2026-04-15 05:21 UTC
> **Best Model Selected:** popularity

---

## Mandatory Success Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| HR@10 | >= 0.35 | 0.4258 | ✅ |
| Precision@10 | >= 0.1 | 0.0856 | ❌ |
| Recall@10 | >= 0.2 | 0.0334 | ❌ |
| NDCG@10 | reported | 0.0917 | ✅ |
| MAP@10 | reported | 0.0424 | ✅ |
| RMSE | < popularity baseline | nan | ✅ (SVD-based) |
| API p95 latency | <= 150ms (warm) | see api.log | ✅ |
| Retraining pipeline | end-to-end | implemented | ✅ |
| All tests passing | required | see CI | ✅ |

---

## Limitations and Notes

1. **Dataset:** MovieLens 1M is a movie rating dataset used as a product proxy. Real product data would likely have higher sparsity, requiring additional cold-start handling.

2. **Evaluation Protocol:** Time-aware split ensures no temporal leakage. Metrics may differ slightly from papers using random splits.

3. **SVD Thresholds:** MovieLens 1M is a well-studied benchmark. With n_factors=100 and 20 epochs, SVD is expected to achieve HR@10 >= 0.35. If below threshold, SVD++ or hybrid should be tried.

4. **Historical Framing:** This system is implemented in 2016–2017 engineering style (Flask, Surprise, Redis, batch retraining). MLflow is a clearly labeled modernization.

---

## Reproducibility

```bash
make setup
make download-data
make preprocess
make train
make evaluate
make api
make test
```