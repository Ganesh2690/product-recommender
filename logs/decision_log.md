# Decision Log — Personalized Product Recommender System

> **Purpose:** Engineering decision log recording all alternatives considered, decision criteria, tradeoffs, chosen designs, and implementation rationale.
> **Label:** Engineering Decision Log (NOT private chain-of-thought)
> **Initialized:** 2026-04-15

---

## Decision Log Format

```
### DEC-NNN — <Decision Topic>
- **Date:** YYYY-MM-DD
- **Phase:** Phase X
- **Options Considered:** [list]
- **Comparison Criteria:** [list]
- **Selected Option:** ...
- **Rationale:** ...
- **Tradeoffs:** ...
- **Risks:** ...
- **Mitigation:** ...
```

---

## Decisions

### DEC-001 — Primary Dataset Selection
- **Date:** 2026-04-15
- **Phase:** Phase 2
- **Options Considered:**
  1. MovieLens 1M (1M ratings, 6040 users, 3706 movies)
  2. MovieLens 20M (20M ratings — larger, but slower for iteration)
  3. Amazon Product Reviews (sparse, requires preprocessing)
- **Comparison Criteria:** Dataset size vs. iteration speed, availability of explicit ratings, historical use in 2016–2017 research papers, ease of download automation
- **Selected Option:** MovieLens 1M
- **Rationale:** MovieLens 1M is the canonical benchmark dataset used in virtually all collaborative filtering papers from 2006–2018. It has explicit ratings (1–5 scale), timestamps (enabling time-aware splits), user demographic features, and item genre features. The 1M scale is large enough to produce meaningful metrics but small enough for local iteration.
- **Tradeoffs:** Smaller than 20M — metrics may not scale the same on larger data
- **Risks:** Data download availability
- **Mitigation:** Use official GroupLens server with checksum validation; implement retry logic

---

### DEC-002 — Collaborative Filtering Library Selection
- **Date:** 2026-04-15
- **Phase:** Phase 5 / Phase 6
- **Options Considered:**
  1. **Surprise** (scikit-learn style CF library) — SVD, KNN, NMF
  2. **LightFM** — hybrid MF with implicit/explicit feedback
  3. **implicit** — ALS-based for implicit feedback
  4. **Custom NumPy** — pure matrix operations
- **Comparison Criteria:** 2016–2017 era authenticity, explicit rating support, ease of tuning, evaluation integration, sklearn-style API
- **Selected Option:** Surprise (primary) + LightFM (hybrid layer)
- **Rationale:** Surprise provides a clean sklearn-compatible API matching 2016–2017 CF research workflows. LightFM adds hybrid capability for content-aware features. Both were available in this era.
- **Tradeoffs:** Surprise is less scalable than implicit/ALS for very large datasets
- **Risks:** Surprise may have limited maintenance
- **Mitigation:** Pin versions in requirements.txt; custom fallback using scipy sparse SVD if Surprise unavailable

---

### DEC-003 — Train/Test Split Strategy
- **Date:** 2026-04-15
- **Phase:** Phase 3
- **Options Considered:**
  1. **Random split** (80/10/10) — simple, common in 2016 papers
  2. **Time-aware split** — last N% of ratings by timestamp per user as test set
  3. **Leave-one-out** — last rated item per user as test
- **Comparison Criteria:** Realism (simulate production deployment), prevention of temporal leakage, implementation complexity, alignment with evaluation metrics (HR@10, Precision@10)
- **Selected Option:** Time-aware split (chronological — last 20% of each user's interactions as test)
- **Rationale:** MovieLens 1M has timestamps. Using time-aware split prevents data leakage and simulates real-world recommendation scenarios where the model must predict future preferences. This is the more rigorous choice despite slight complexity increase.
- **Tradeoffs:** Users with very few ratings may have no test interactions
- **Risks:** Cold-start users with only 1–2 ratings have no test data
- **Mitigation:** Minimum 5 ratings per user filter; fallback to popularity for insufficient-data users

---

### DEC-004 — Caching Strategy
- **Date:** 2026-04-15
- **Phase:** Phase 9
- **Options Considered:**
  1. **Redis** — industry standard, fast, supports TTL and pub/sub
  2. **Memcached** — simpler but no persistence
  3. **Local in-memory dict** — sufficient for development/testing
  4. **File-based cache** — persistent but slow
- **Comparison Criteria:** Production realism, availability without external services, latency, Docker compatibility
- **Selected Option:** Redis (primary) with local dict fallback if Redis unavailable
- **Rationale:** Redis aligns with 2016–2017 production recommendation stacks. Docker Compose includes Redis. Local dict fallback ensures the system runs without Docker. TTL-based cache refresh matches nightly retraining schedule.
- **Tradeoffs:** Redis adds infrastructure dependency
- **Risks:** Redis not available in minimal environments
- **Mitigation:** Graceful fallback to thread-safe local dict cache; log cache backend selection at startup

---

### DEC-005 — API Framework Selection
- **Date:** 2026-04-15
- **Phase:** Phase 9
- **Options Considered:**
  1. **Flask** — lightweight, 2016-era standard for ML APIs
  2. **FastAPI** — modern, async, better validation
  3. **Django REST** — too heavy for ML serving
  4. **Sanic** — async but less mature in 2016
- **Comparison Criteria:** Historical accuracy (2016–2017 era), simplicity, ease of integration with ML artifacts, low overhead
- **Selected Option:** Flask
- **Rationale:** Flask was the dominant ML API framework in 2016–2017. It is synchronous (matching the era), minimal, and sufficient for serving pre-computed recommendations. FastAPI is labeled as a modernization and noted in README as an alternative upgrade path.
- **Tradeoffs:** No async support; WSGI not ASGI; less automatic schema validation
- **Risks:** Higher latency under concurrent load
- **Mitigation:** Gunicorn with multiple workers for production; local dev runs single-threaded

---

### DEC-006 — Matrix Factorization Algorithm Selection
- **Date:** 2026-04-15
- **Phase:** Phase 6
- **Options Considered:**
  1. **SVD (Funk SVD / Simon Funk)** — the Netflix Prize era classic
  2. **SVD++ (Koren 2008)** — adds implicit feedback signals
  3. **NMF** — non-negative, interpretable
  4. **ALS (Alternating Least Squares)** — Spark MLlib standard
- **Comparison Criteria:** Historical authenticity (Simon Funk's SVD was 2006–2009 era standard), performance on explicit ratings, parameter tuning complexity, Surprise library support
- **Selected Option:** SVD (Surprise) as primary; SVD++ as improvement candidate
- **Rationale:** Simon Funk's SVD remains the foundational explicit-rating MF algorithm. It directly optimizes RMSE on observed ratings, which is the explicit rating objective for MovieLens. SVD++ adds implicit feedback signals (which items the user rated, regardless of score) and typically improves 5–10% on HR@10.
- **Tradeoffs:** SVD++ is slower to train than SVD
- **Risks:** Overfitting with high latent factor count
- **Mitigation:** Cross-validate n_factors (20, 50, 100) and regularization; log all results

---

### DEC-007 — Model Artifact Versioning Strategy
- **Date:** 2026-04-15
- **Phase:** Phase 10
- **Options Considered:**
  1. **MLflow** — modern experiment tracking + model registry
  2. **Timestamp-based file versioning** — simple, period-accurate
  3. **Git LFS** — version artifacts in git
  4. **DVC** — data version control
- **Comparison Criteria:** Historical realism, zero-infrastructure option, experiment comparison, promotion logic
- **Selected Option:** Timestamp-based file versioning (primary) + MLflow (explicitly labeled as modernization add-on)
- **Rationale:** In 2016–2017, most teams used timestamp/version-tagged pickle files in shared storage. MLflow (released 2018) is added as a clearly labeled modernization for experiment tracking. This keeps the core pipeline historically authentic while providing modern observability.
- **Tradeoffs:** No native model comparison UI without MLflow
- **Risks:** Artifact proliferation without cleanup
- **Mitigation:** retention policy in promote_model.py keeping last 3 versioned artifacts

---

### DEC-008 — Hybrid Model Design
- **Date:** 2026-04-15
- **Phase:** Phase 7
- **Options Considered:**
  1. **Weighted blending** — score = α * CF_score + (1-α) * Content_score
  2. **Cascade hybrid** — CF generates candidates, content re-ranks
  3. **Feature augmentation** — feed content features as side-information into LightFM
  4. **Neural re-ranker** — small MLP on top of CF embeddings
- **Comparison Criteria:** Implementation complexity, measurable improvement, historical plausibility, interpretability
- **Selected Option:** Weighted blending (CF + content) with configurable alpha; neural re-ranker as optional extension
- **Rationale:** Weighted blending is the simplest and most interpretable hybrid approach. A tunable alpha (0.0–1.0) allows degenerating to pure CF (alpha=1.0) or pure content (alpha=0.0). Neural re-ranker adds a small 2-layer MLP but requires more tuning; it is implemented as an optional component that is only promoted if it improves Precision@10 by >5%.
- **Tradeoffs:** Blending weights require tuning; may not outperform pure SVD on all metrics
- **Risks:** Content features (genres) are coarse in MovieLens 1M
- **Mitigation:** Log A/B comparison; keep pure SVD as default if hybrid underperforms
