# Architecture Decisions — Personalized Product Recommender System

> **Scope:** All major architecture decisions for the Personalized Product Recommender System
> **Label:** Engineering Decision Log (NOT private chain-of-thought)
> **Format mirrors:** `logs/decision_log.md` (full detail) — this document presents the summary view

---

## Decision Index

| ID | Topic | Selected | Date |
|----|-------|----------|------|
| DEC-001 | Primary Dataset | MovieLens 1M | 2026-04-15 |
| DEC-002 | CF Library | Surprise + LightFM | 2026-04-15 |
| DEC-003 | Train/Test Split | Time-aware chronological | 2026-04-15 |
| DEC-004 | Caching | Redis + local dict fallback | 2026-04-15 |
| DEC-005 | API Framework | Flask (Gunicorn) | 2026-04-15 |
| DEC-006 | MF Algorithm | SVD (primary), SVD++ (improved) | 2026-04-15 |
| DEC-007 | Model Versioning | Timestamp files + MLflow | 2026-04-15 |
| DEC-008 | Hybrid Design | Weighted blending (α-tuned) | 2026-04-15 |

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   OFFLINE PIPELINE                       │
│                                                         │
│  Raw Data → Preprocess → Train/Val/Test Split           │
│       ↓                                                 │
│  Popularity → Item-CF → SVD → SVD++ → Hybrid            │
│       ↓                                                 │
│  Evaluate (HR@10, P@10, NDCG@10, RMSE)                  │
│       ↓                                                 │
│  Promotion Gate → Model Registry                        │
│       ↓                                                 │
│  Batch Recommend → Cache Pre-warm                       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  ONLINE SERVING                          │
│                                                         │
│  Flask API (Gunicorn)                                   │
│    GET /recommend?user_id=X&n=10                        │
│    GET /similar-items?item_id=X&n=10                    │
│    POST /predict-batch                                  │
│    GET /health                                          │
│       ↓                                                 │
│  Redis Cache (TTL=24h) → Model Registry                 │
│       ↓                                                 │
│  RecommendationEngine                                   │
│    ↓ cold start? → Popularity Fallback                  │
│    ↓ normal → SVD Top-N (filter seen items)             │
└─────────────────────────────────────────────────────────┘
```

---

## DEC-001 — Dataset Selection

**Decision:** Use MovieLens 1M as primary dataset

**Context:** The system needs a real-world collaborative filtering dataset with explicit ratings and temporal information.

**Options Evaluated:**
| Option | Pro | Con |
|--------|-----|-----|
| MovieLens 1M | Canonical benchmark; explicit ratings; timestamps; user features | Synthetic (movies, not products); limited to 2003 data |
| MovieLens 20M | Larger; more realistic scale | Slower iteration; same synthetic issue |
| Amazon Reviews | Real product data; recent | Sparse; complex preprocessing; no user demographics |

**Decision Rationale:** MovieLens 1M is the gold standard for evaluating CF algorithms. Every major algorithm from 2006–2020 reports metrics on this dataset, making results directly comparable. The explicit rating scale and timestamps make it ideal for RMSE optimization and time-aware evaluation.

---

## DEC-002 — Collaborative Filtering Library

**Decision:** Surprise (primary MF) + LightFM (hybrid)

**Options Evaluated:**
| Library | Era | Strengths | Weaknesses |
|---------|-----|-----------|------------|
| Surprise | 2014+ | sklearn-compatible; explicit ratings; clean API | Not for implicit; CPU only |
| LightFM | 2015+ | Hybrid (content + CF); BPR/WARP loss | Less intuitive for explicit ratings |
| implicit | 2016+ | Fast ALS; Cython | Implicit only |
| Custom NumPy | Always | Full control | Slow; high dev cost |

**Decision Rationale:** Surprise directly implements the Netflix Prize era algorithms (SVD, SVD++, NMF) in a sklearn-compatible API, matching the 2016–2017 ML engineering workflow. LightFM is added for the hybrid layer since it supports content features natively.

---

## DEC-003 — Train/Test Split

**Decision:** Time-aware chronological split (per user, 60/20/20)

**Options Evaluated:**
| Strategy | Pro | Con |
|----------|-----|-----|
| Random 80/10/10 | Simple | Temporal leakage; overestimates real performance |
| Time-aware | Realistic; no leakage | Users with few ratings get thin test sets |
| Leave-one-out | Standard for ranking | Can't measure recall properly |
| Global time split | Simple | Cold start issues for users who joined late |

**Decision Rationale:** Time-aware split is the scientifically correct approach. It simulates the production scenario: the model trains on past interactions and predicts future ones. This prevents inflated metrics from temporal leakage (using future data to predict past).

---

## DEC-004 — Caching Strategy

**Decision:** Redis (primary) with thread-safe local dict fallback

**Options Evaluated:**
| Cache | Latency | Persistence | Infrastructure |
|-------|---------|-------------|----------------|
| Redis | ~0.1ms | Yes (with AOF) | Docker service required |
| Memcached | ~0.1ms | No | Docker service required |
| Local dict | ~0.01ms | No | None |
| File cache | ~1–10ms | Yes | None |

**Decision Rationale:** Redis is the 2016–2017 industry standard for ML serving caches. It supports TTL-based expiry (critical for nightly retraining), pub/sub for cache invalidation, and persistence. Local dict fallback ensures the system runs in minimal environments (e.g., CI).

---

## DEC-005 — API Framework

**Decision:** Flask with Gunicorn

**Options Evaluated:**
| Framework | Era | Strengths | Weaknesses |
|-----------|-----|-----------|------------|
| Flask | 2010+ | Lightweight; dominant 2016‑17 ML API choice | Synchronous; no schema validation |
| FastAPI | 2018+ | Async; Pydantic; auto-docs | Post-2016; modernization |
| Django REST | 2011+ | Full-featured | Too heavy for ML serving |
| Sanic | 2016+ | Async | Less mature in 2016 |

**Decision Rationale:** Flask is the historically accurate choice for 2016–2017. It is synchronous (matching era), minimal, and requires no ORM or extra tooling. Gunicorn provides production-grade multi-worker serving. FastAPI is documented as the modern upgrade path.

---

## DEC-006 — Matrix Factorization Algorithm

**Decision:** SVD (primary), SVD++ (improved candidate)

**Options Evaluated:**
| Algorithm | Era | RMSE | HR@10 | Training Speed |
|-----------|-----|------|-------|----------------|
| SVD (Funk) | 2006 | ~0.87 | ~0.36 | Fast |
| SVD++ | 2008 | ~0.85 | ~0.38 | 3–4× slower |
| NMF | 2000+ | ~0.90 | ~0.33 | Medium |
| ALS | 2008+ | ~0.89 | ~0.34 | Fast (parallel) |

**Decision Rationale:** Simon Funk's SVD is the foundational algorithm from the Netflix Prize (2006). It directly optimizes RMSE on observed ratings via SGD. SVD++ adds implicit feedback signals (which items the user rated, regardless of score) and typically achieves 3–5% better ranking metrics. We train both and promote the better performer.

---

## DEC-007 — Model Artifact Versioning

**Decision:** Timestamp-based file versioning + MLflow (labeled as modernization)

**Options Evaluated:**
| Strategy | Era | Pro | Con |
|----------|-----|-----|-----|
| Timestamp files | 2016+ | Simple; no infra | No comparison UI |
| MLflow | 2018+ | Experiment tracking; model comparison; UI | Post-2016 |
| Git LFS | 2015+ | Version controlled | Not for large binaries |
| DVC | 2017+ | Data+model versioning | Complex setup |

**Decision Rationale:** Timestamp-based files (`model_svd_20260415_v1.pkl`) were the de facto standard in 2016–2017 ML engineering. MLflow is added as a clearly and explicitly labeled modernization for experiment comparison. The promotion pipeline uses file-based versioning internally; MLflow is a UI add-on.

---

## DEC-008 — Hybrid Model Architecture

**Decision:** Weighted blending (CF + content signals)

**Options Evaluated:**
| Architecture | Complexity | Interpretability | Expected Gain |
|--------------|------------|-----------------|---------------|
| Weighted blend | Low | High | +3–8% P@10 |
| Cascade (CF→Content rerank) | Medium | Medium | +5–10% P@10 |
| LightFM side-information | Medium | Low | +5–12% P@10 |
| Neural re-ranker (MLP) | High | Low | +5–15% P@10 |

**Decision Rationale:** Weighted blending is the most interpretable and easiest to tune. The `α` parameter directly controls the CF-vs-content tradeoff and can be logged and adjusted without retraining. Neural re-ranker is implemented as an optional component but only promoted if Precision@10 improves by >5% on the validation set.

**Key Tuning Insight:** Genre features in MovieLens 1M are coarse (18 genres). Content signals are expected to be weak, so α will likely remain high (0.7–0.9), meaning the hybrid is only slightly augmented over pure CF.
