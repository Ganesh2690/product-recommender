# Data Dictionary — MovieLens 1M

## Overview

The MovieLens 1M dataset was released by GroupLens Research in 2003 and contains
1,000,209 anonymous ratings of approximately 3,706 movies made by 6,040 MovieLens
users who joined MovieLens in 2000.

Source: https://files.grouplens.org/datasets/movielens/ml-1m.zip
License: Non-commercial research use only.

---

## File: ratings.dat

**Format:** `UserID::MovieID::Rating::Timestamp`

| Column       | Type     | Range  | Description                                                 |
|--------------|----------|--------|-------------------------------------------------------------|
| `user_id`    | integer  | 1–6040 | Unique user identifier                                      |
| `movie_id`   | integer  | 1–3952 | Unique movie identifier (maps to `movies.dat`)             |
| `rating`     | float    | 1–5    | Explicit integer rating in half-star increments             |
| `timestamp`  | integer  | —      | Unix epoch seconds; represents when rating was submitted    |

**Statistics:**
- Total rows: 1,000,209
- Unique users: 6,040
- Unique movies: 3,706 (out of 3,952 listed in movies.dat)
- Sparsity: ~95.5%
- Mean rating: ~3.58
- Rating distribution: skewed toward 4 (most common)

---

## File: movies.dat

**Format:** `MovieID::Title::Genres`

| Column      | Type    | Description                                              |
|-------------|---------|----------------------------------------------------------|
| `movie_id`  | integer | Unique movie identifier                                  |
| `title`     | string  | Movie title with release year in parentheses, e.g. `Toy Story (1995)` |
| `genres`    | string  | Pipe-separated genre tags, e.g. `Animation|Children's|Comedy` |

**Available genres (18):**
Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi,
Thriller, War, Western

**Notes:**
- `year` is extracted from the title substring `(YYYY)` during preprocessing
- Genres are one-hot encoded into binary feature columns `genre_<name>`
- Movies not seen in ratings.dat are still included in movies.dat

---

## File: users.dat

**Format:** `UserID::Gender::Age::Occupation::Zip-code`

| Column       | Type    | Values / Range | Description                                      |
|--------------|---------|----------------|--------------------------------------------------|
| `user_id`    | integer | 1–6040         | Unique user identifier                           |
| `gender`     | string  | M / F          | Self-reported gender                             |
| `age`        | integer | see below      | Age group code                                   |
| `occupation` | integer | 0–20           | Occupation code (see table below)                |
| `zip_code`   | string  | —              | US ZIP code (5-digit); not used in models        |

**Age codes:**
| Code | Description     |
|------|-----------------|
| 1    | Under 18        |
| 18   | 18–24           |
| 25   | 25–34           |
| 35   | 35–44           |
| 45   | 45–49           |
| 50   | 50–55           |
| 56   | 56+             |

**Occupation codes:**
| Code | Description          |
|------|----------------------|
| 0    | other / not specified |
| 1    | academic/educator    |
| 2    | artist               |
| 3    | clerical/admin       |
| 4    | college/grad student |
| 5    | customer service     |
| 6    | doctor/health care   |
| 7    | executive/managerial |
| 8    | farmer               |
| 9    | homemaker            |
| 10   | K-12 student         |
| 11   | lawyer               |
| 12   | programmer           |
| 13   | retired              |
| 14   | sales/marketing      |
| 15   | scientist            |
| 16   | self-employed        |
| 17   | technician/engineer  |
| 18   | tradesman/craftsman  |
| 19   | unemployed           |
| 20   | writer               |

---

## Processed Artifacts

After running `python -m src.data.preprocess`, the following files are created in `data/processed/`:

| File                   | Format  | Description                                        |
|------------------------|---------|----------------------------------------------------|
| `ratings.parquet`      | Parquet | Cleaned ratings with `datetime` column added       |
| `movies.parquet`       | Parquet | Movies with `year`, `genre_list`, one-hot columns  |
| `users.parquet`        | Parquet | Users with standardised integer types              |
| `user_item_matrix.npz` | SciPy sparse CSR | User × item matrix of ratings          |
| `id_maps.json`         | JSON    | `user2idx`, `idx2user`, `movie2idx`, `idx2movie` dicts |

After running `python -m src.data.split`:

| File                | Format  | Description                              |
|---------------------|---------|------------------------------------------|
| `train.parquet`     | Parquet | 60% of ratings (earliest per user)       |
| `val.parquet`       | Parquet | 20% of ratings (middle per user)         |
| `test.parquet`      | Parquet | 20% of ratings (latest per user)         |

---

## Notes on Temporal Splitting

The project uses **time-aware chronological splitting** per user to prevent temporal leakage:

1. Sort each user's ratings by timestamp ascending.
2. Take first 60% as train, next 20% as validation, last 20% as test.
3. Verify: `max(train.timestamp) <= min(test.timestamp)` for every user.

Random splits are explicitly forbidden (`src/config.py: ALLOW_RANDOM_SPLIT = False`).
