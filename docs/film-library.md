# Film library and manual profile

## Overview

FilmFinder exposes a browsable film catalogue at `/films`. Visitors can
search, filter and open film details. Signed-in users can rate films and keep
Watchlist / Do Not Watch lists that stay consistent with Discover and My Lists.

The catalogue is derived from the existing production artifacts and the raw
`movie_data.csv` file. It is stored in SQLite and queried with pagination; the
full catalogue is never sent to the browser or loaded into the API process.

## Source data

The repository does not contain `movie.csv`; the actual raw file is:

```text
movie_data.csv   ~163 MB, gitignored
```

Useful columns:

```text
_id, movie_id, movie_title, overview, release_date,
vote_average, vote_count, tmdb_id, year_released
```

Verified coverage:

| Metric | Value |
|---|---:|
| Films in the FilmFinder catalogue | 40 981 |
| Library films released up to 2019 | 38 705 |
| Films released after 2019 | 2 276 |
| Library films with a summary | 38 679 |
| Library films with a precise release date | 38 703 |
| Library films with a TMDB poster | 38 233 |
| Distinct library genres | 19 |

Films released after 2019 remain in the engine catalogue and can appear in
Discover, but they are excluded from the library list with the `in_library`
flag. This is explicit and does not delete source data.

Summaries come from `overview`. When a film has multiple CSV rows, the row with
the longest non-empty overview is kept. Release dates fall back to the
catalogue year in the UI when no precise date is available. Genres, titles and
years come from `movie_metadata.json`; posters and backdrops come from
`tmdb_metadata.json`. No TMDB request is made by the library or by API
requests.

The CSV-to-catalogue matching reuses the same stable identifier matching as
the TMDB enrichment script (`_id`, then `movie_id`). No independent matching
logic is used.

## Storage

Two SQLite tables are added by `0002_catalog_films`:

```text
catalog_films
  internal_movie_id  PK
  movie_id           unique
  title
  title_normalized   indexed
  year               indexed
  release_date       indexed
  genres             JSON
  summary
  tmdb_id, poster_path, backdrop_path
  ratings_count      indexed
  rating_mean
  weighted_rating    indexed
  in_library         indexed
  synced_at

catalog_film_genres
  film_id + genre    composite PK
  genre              indexed
```

The normalized title is produced with the existing Letterboxd title
normalization (lowercase, accents removed, punctuation collapsed). Search uses
`LIKE '%query%'` on that column.

FTS5 was evaluated but is not required at this scale: on 40 981 rows, a
normalized-title search takes roughly 35–45 ms, while date sorting is under
1 ms and `COUNT(*)` is around 1.5 ms. The trigram tokenizer would be a useful
later optimization if search becomes a bottleneck.

## Popularity and Highest Rated

- **Most Popular** uses `ratings_count`, the number of FilmFinder ratings. No
  synthetic popularity score is invented.
- **Highest Rated** uses a Bayesian weighted rating:

```text
weighted = v / (v + 100) * mean + 100 / (v + 100) * global_mean
```

with `v = ratings_count`, `m = 100`, and `global_mean = 6.5445`. This prevents
films with very few ratings from automatically topping the list.
- **Release Date** sorts by year, then precise release date, then internal
  movie ID. The internal ID makes the ordering stable.

## Synchronization

Run the migration once, then synchronize the catalogue:

```bash
alembic -c apps/backend/alembic.ini upgrade head

python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
```

The sync is idempotent: it rebuilds both catalogue tables in one transaction.
It reports processed films, library count, post-2019 count, summaries, posters,
release dates and genre rows. Use `--dry-run` to inspect without writing and
`--limit N` for a small run.

If the CSV is unavailable on the VPS, run the sync locally and import the
`catalog_films` / `catalog_film_genres` tables into the production SQLite
database, or copy `movie_data.csv` temporarily and run the script there. The
catalogue is derived data and can be regenerated at any time.

## API

```http
GET /api/v1/films
GET /api/v1/films/facets
GET /api/v1/films/{film_id}
```

`GET /api/v1/films` parameters:

```text
page (>= 1)
page_size (1..50, default 50)
search
sort: popular | rating | release_date
order: asc | desc (release date)
genre
```

The endpoint is public, but a valid session adds `user_rating` and
`user_list_status` to each item. `GET /api/v1/films/{film_id}` returns the
summary, precise release date, genres, community rating and the same personal
state.

Authenticated profile endpoints:

```http
GET    /api/v1/me/ratings
PUT    /api/v1/me/ratings/{film_id}
DELETE /api/v1/me/ratings/{film_id}
GET    /api/v1/me/profile-summary
POST   /api/v1/recommendations/from-saved-ratings
```

Ratings are integers from 1 to 10. The UI exposes five stars with half-star
steps, which map directly to that range. Deleting a rating removes the
`user_films` row completely, as decided for the manual profile flow. Setting a
rating also removes any Watchlist / Do Not Watch decision for that film.

The manual recommendation endpoint enforces a server-side minimum of **10
ratings**. The existing Letterboxd import and `POST /api/v1/recommendations/refresh`
keep their previous behavior; only the manual button uses the strict endpoint.

## Lists and Discover

No second list system was introduced. The existing `user_swipes` table is the
single source of truth:

| Library / My Lists | Swipe action | Recommendation effect |
|---|---|---|
| Watchlist | `interested` | not excluded |
| Do Not Watch | `not_interested` | excluded |
| Seen | `already_seen` | excluded |

A user has at most one `user_swipes` row per film, so the most recent decision
always replaces the previous one. Adding a film to Watchlist or Do Not Watch
never creates a rating. Rating a film clears its list decision.

My Lists displays the same data as Watchlist / Do Not Watch. Discover writes
the same actions, so the three views stay consistent without a second source of
truth.

## Performance and security

- Catalogue queries use SQL pagination and indexes; only 50 rows are sent per
  page.
- No ML artifact is loaded per request; the engine remains loaded once.
- The catalogue sync is an offline operation using streaming CSV reading.
- API mutations require a session and CSRF; user identity always comes from the
  session, never from the path or request body.
- Caddy must not expose the SQLite database or the ML artifacts.
- SQLite backups cover the catalogue tables; they can also be regenerated.
