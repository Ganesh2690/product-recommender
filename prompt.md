You are GitHub Copilot operating as a senior ML engineer, MLOps engineer, data engineer, QA engineer, and technical documentation lead.
Your task is to implement the following project end to end inside this repository:

Project name: Personalized Product Recommender System (2016–2017)

You must execute the project from dataset download to final validation, ensuring that all scope items are covered, success metrics are measured, and every phase is documented in logs.

1. Core objective
Build a complete, production-style Personalized Product Recommender System that reflects a realistic 2016–2017 ML engineering stack, while remaining runnable in a modern GitHub repository.

The implementation must include:

Dataset acquisition and validation

Data preprocessing

Exploratory data analysis

Baseline recommender

Improved recommender

Hybrid recommendation logic

Offline evaluation

Batch recommendation generation

Real-time inference API

Caching layer

Retraining pipeline

Experiment tracking

Test coverage

Reproducible run instructions

Structured decision logging at every phase

Use the project scope below as the governing implementation contract.

2. Mandatory implementation rule: full execution logging
2.1 Logging requirement
You must create and maintain a project-wide execution log system from the very beginning.

Required log artifacts
Create these files/directories first before doing any project work:

logs/

logs/master_execution_log.md

logs/decision_log.md

logs/run_log.jsonl

logs/data_pipeline.log

logs/model_training.log

logs/evaluation.log

logs/api.log

logs/test.log

logs/checkpoint_status.md

Also create:

docs/IMPLEMENTATION_JOURNAL.md

docs/ARCHITECTURE_DECISIONS.md

2.2 What must be logged
At every phase, record:

Phase name

Timestamp

Objective of the phase

What options were considered

Which option was selected

Why that option was selected

What was implemented

Files created or modified

Commands run

Outputs generated

Errors encountered

Fixes applied

Validation performed

Next step

This is not optional.

2.3 Important constraint about “thinking logs”
Do not claim access to hidden/private internal chain-of-thought.

Instead, create a transparent engineering reasoning log that explicitly records:

alternatives considered,

decision criteria,

tradeoffs,

chosen design,

and implementation rationale.

Label it as:

“engineering decision log”
not “private chain-of-thought”.

If logging is not active at any phase, stop work immediately, initialize/fix logging, and then continue.

2.4 Logging enforcement
Before each major step, do this:

Check whether all required log files exist.

If not, create them.

Append a new entry before making changes.

Append a completion entry after finishing the step.

Update logs/checkpoint_status.md.

Every script in the repo must also log execution details to the relevant log file.

3. Period-accurate project framing
Build the system as if it were originally designed in 2016–2017, but implemented in a way that can run today.

Historical modeling style to preserve
Prefer these approaches:

collaborative filtering

matrix factorization

item-item similarity

user-item embeddings

lightweight neural re-ranking

batch + online serving architecture

Historically plausible stack
Prefer these tools and patterns:

Python

pandas

NumPy

scikit-learn

Surprise or LightFM

Keras / TensorFlow-style neural re-ranker

Flask API

Redis caching

PostgreSQL or SQLite for local simulation

Docker

batch retraining scripts

Jenkins-style CI logic, but implemented via GitHub Actions if needed

MLflow is allowed as a modern add-on for experiment tracking, but clearly label it as a modernization

Do not turn this into an LLM or RAG project.

4. Dataset requirements
4.1 Primary dataset
Use MovieLens 1M as the primary dataset.

Preferred source:

https://grouplens.org/datasets/movielens/1m/

4.2 Optional secondary dataset
Optionally support:

Amazon Product Reviews dataset
if needed for hybrid/content signals.

4.3 Required data tasks
You must:

download the dataset automatically via script

verify file integrity and expected columns

create a reproducible local data directory structure

log the source URL, download timestamp, file size, and schema

create a data dictionary in docs/

5. Scope of implementation
Implement the following scope completely.

5.1 Executive goal
Build a recommender system that:

produces personalized top-N recommendations,

supports offline evaluation,

exposes a real-time API,

caches recommendations,

supports retraining,

and tracks performance against defined success metrics.

5.2 Functional requirements
Implement all of the following:

Data layer
download and store raw data

preprocess ratings, users, and items

split data into train/validation/test, preferably time-aware if timestamps exist

generate user-item interaction matrices

Modeling layer
Implement at least:

Popularity baseline

Item-item collaborative filtering baseline

Matrix factorization model

Optional neural re-ranker or hybrid model

Recommendation serving
generate top-N recommendations per user

support exclusion of already-consumed items

support fallback logic for cold-start users

support fallback logic for missing users/items

API
Create a Flask API with endpoints such as:

GET /health

GET /recommend?user_id=<id>&n=10

GET /similar-items?item_id=<id>&n=10

POST /predict-batch

Caching
integrate Redis if available

otherwise implement a local fallback cache

log cache hits, misses, and refreshes

Batch pipeline
nightly or manual retraining script

batch generation of top-N recommendations

model artifact versioning

evaluation gate before promotion

Monitoring
log inference latency

log evaluation metrics

log model version

log recommendation generation status

log errors and fallbacks

Testing
Add tests for:

data loading

preprocessing

train/test split

recommendation generation

API endpoints

cold-start fallback

metric calculation

cache behavior

6. Success metrics to meet
Measure and report these metrics.

6.1 Minimum mandatory metrics
At minimum compute:

Precision@K

Recall@K

MAP@K or NDCG@K

Hit Rate@10

RMSE or MAE if using explicit ratings

API latency

cache effectiveness

6.2 Target success thresholds
Aim for these thresholds:

HR@10 >= 0.35

Precision@10 >= 0.10

Recall@10 >= 0.20

RMSE better than popularity baseline

GET /recommend p95 latency <= 150 ms locally with cache warm

successful retraining pipeline end to end

all tests passing

If exact thresholds are not met, do not hide it.
Instead:

log the failure,

diagnose the issue,

try improvements,

compare alternatives,

document why the final result is acceptable or what gap remains.

7. Required repository structure
Create and maintain a clean structure like this:

text
recommender-system/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
├── logs/
│   ├── master_execution_log.md
│   ├── decision_log.md
│   ├── run_log.jsonl
│   ├── data_pipeline.log
│   ├── model_training.log
│   ├── evaluation.log
│   ├── api.log
│   ├── test.log
│   └── checkpoint_status.md
├── docs/
│   ├── IMPLEMENTATION_JOURNAL.md
│   ├── ARCHITECTURE_DECISIONS.md
│   ├── DATA_DICTIONARY.md
│   ├── EXPERIMENT_REPORT.md
│   └── SUCCESS_METRICS_REPORT.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_matrix_factorization.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── config.py
│   ├── logging_utils.py
│   ├── data/
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── models/
│   │   ├── popularity.py
│   │   ├── item_cf.py
│   │   ├── matrix_factorization.py
│   │   ├── hybrid.py
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── benchmark.py
│   │   └── report.py
│   ├── serving/
│   │   ├── app.py
│   │   ├── cache.py
│   │   └── schemas.py
│   ├── pipelines/
│   │   ├── train_pipeline.py
│   │   ├── batch_recommend.py
│   │   └── promote_model.py
│   └── utils/
│       ├── io.py
│       ├── timing.py
│       └── validation.py
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_models.py
│   ├── test_metrics.py
│   ├── test_api.py
│   └── test_cache.py
├── scripts/
│   ├── run_all.sh
│   ├── run_train.sh
│   ├── run_api.sh
│   ├── run_tests.sh
│   └── check_logs.sh
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── gunicorn.conf.py
└── .github/
    ├── workflows/
    │   └── ci.yml
    └── copilot-instructions.md
8. Required implementation phases
Execute in this exact order unless a justified dependency requires adjustment.
Log before and after every phase.

Phase 0 — Logging bootstrap
create all log files

create logging utilities

create a reusable project logger

verify each script writes to logs

create a “logging health check” script

Phase 1 — Repository setup
create structure

add dependency files

add README skeleton

add environment configuration

add Makefile or shell scripts if useful

Phase 2 — Dataset acquisition
implement automated downloader

validate dataset

generate data dictionary

log dataset provenance

Phase 3 — EDA and data quality
inspect sparsity

user/item cardinality

rating distribution

missing values

popularity skew

long-tail analysis

log observations and implications

Phase 4 — Baseline model
implement popularity baseline

evaluate it

log why this baseline matters

Phase 5 — Collaborative filtering baseline
item-item or user-user CF

evaluate against baseline

log comparison and tradeoffs

Phase 6 — Matrix factorization model
implement SVD / ALS / Surprise-based MF

tune key parameters

measure improvements

log why this is the main production candidate

Phase 7 — Hybrid / neural re-ranking layer
implement lightweight hybrid logic or neural re-ranker

justify whether it improves performance enough to keep

if not beneficial, log the evidence and keep architecture simple

Phase 8 — Recommendation service
implement recommendation engine abstraction

support top-N

filter seen items

add fallback logic

add serialization of artifacts

Phase 9 — API and cache
implement Flask service

add health checks

add cache

measure latency

log p50/p95 response times

Phase 10 — Retraining and promotion pipeline
implement train pipeline

model registry/versioning

evaluation gate

promotion only if thresholds are passed

Phase 11 — Testing
run unit tests and integration tests

log failures and fixes

ensure repeatability

Phase 12 — Final validation and reporting
produce success metrics report

compare all models

identify final selected model

verify all scope items covered

verify all log files populated

if any phase lacks logs, backfill clearly marked entries

9. Decision-making expectations
For all important decisions, explicitly compare alternatives.

Examples:

Surprise SVD vs LightFM vs implicit ALS

cosine item-item CF vs Pearson similarity

Redis vs local file cache

time-based split vs random split

Flask vs FastAPI

explicit ratings objective vs implicit feedback ranking objective

For every such decision, record:

options

comparison criteria

selected option

reason

risk

mitigation

Write this into both:

logs/decision_log.md

docs/ARCHITECTURE_DECISIONS.md

10. Logging implementation requirements in code
Create a reusable logging utility that every script imports.

10.1 Required logger features
The logger must support:

console logging

file logging

JSONL structured logs

phase-level checkpoint logging

command logging

exception logging with traceback

model metrics logging

dataset provenance logging

runtime measurement logging

10.2 Required helper methods
Implement helpers such as:

start_phase(phase_name, objective)

log_decision(topic, options, selected, rationale, risks=None)

log_command(cmd, purpose)

log_metric(metric_name, value, context)

log_artifact(path, description)

log_error(error, context)

end_phase(phase_name, outcome, next_step)

check_logging_health()

11. Verification rule: logging health checks
At the end of each phase, run a check that confirms:

the relevant log file was updated

the master log was updated

the JSONL log received entries

the checkpoint file was updated

If any of these fail:

stop,

fix logging,

re-run the failed portion,

document the issue and resolution.

Create scripts/check_logs.sh or equivalent Python script for this.

12. README requirements
The README must include:

project overview

architecture

dataset source

setup instructions

training instructions

evaluation instructions

API usage examples

success metrics achieved

limitations

historical 2016–2017 framing

modernizations added for GitHub usability

logging and audit trail explanation

13. Final deliverables
By the end, the repository must contain:

working project code

reproducible data download

trained final model artifact

evaluation report

success metrics report

tested API

retraining pipeline

complete logs

architecture decision records

README with execution steps

14. Copilot execution mode
You must work iteratively.

For every major phase:

log phase start

inspect current repo state

implement the phase

run validation

log output

verify logs exist

update checkpoint

propose next phase

Do not skip ahead.

Do not leave placeholders like “TODO”.
Do not silently fail.
Do not omit logs.

15. Most critical compliance requirement
If at any point logs are not being recorded for a phase, you must:

stop implementation immediately,

create or repair the logging system,

record that logging had failed,

explain what was missing,

backfill the phase notes as accurately as possible,

only then continue.

16. First action to take now
Start by doing only the following:

inspect repository contents,

create the required folder/file structure,

implement the logging framework,

run a logging health check,

write the first entries into all required log files,

show me the exact files created and the next recommended step.

Do not begin model training until logging is fully operational.

Recommended companion file
create a repository instruction file because GitHub supports repository-level custom instructions for Copilot.


Also create .github/copilot-instructions.md and/or AGENTS.md for this repository so that every future Copilot session follows the same logging, validation, and project execution rules consistently.

Log every major engineering decision, alternatives considered, selected approach, implementation reason, validation result, and corrective action.”