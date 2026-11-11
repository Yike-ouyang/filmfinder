import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Filmfinder — Data Preprocessing

    Version simplifiée et rapide du pipeline. Elle produit uniquement les
    quatre sorties consommées par le notebook de modélisation :

    - `ratings_matrix.npz`
    - `user_mapping.parquet`
    - `movie_mapping.parquet`
    - `movie_context.parquet`

    Les fichiers sources sont lus localement, jamais modifiés.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Choix de conception

    - Déduplication déterministe des films par `titre normalisé + année`,
      avec une contrainte de durée (écart strictement inférieur à 10 minutes
      lorsque les deux durées sont connues). Les films incompatibles restent
      séparés.
    - Rapprochement `movie_data.csv` → catalogue : identifiants directs
      lorsque le recouvrement est suffisant, sinon `titre + année`. Les
      correspondances ambiguës sont écartées.
    - Les colonnes `description` / `overview` ne sont pas chargées : elles ne
      servent ni à la matrice ni au contexte.
    - Les notes sont lues par morceaux, nettoyées puis dédupliquées
      globalement. Les doublons utilisateur-film sont résolus par la dernière
      note si un timestamp fiable existe, sinon par la moyenne des notes
      distinctes.
    - `FAST_MODE=True` désactive les rapports détaillés. Les contrôles
      essentiels restent actifs.
    """)
    return


@app.cell
def _():
    import os
    import re
    import time
    import unicodedata
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, save_npz

    PIPELINE_START = time.perf_counter()

    return (
        PIPELINE_START,
        Path,
        csr_matrix,
        mo,
        np,
        os,
        pd,
        re,
        save_npz,
        time,
        unicodedata,
    )


@app.cell
def _(Path, os):
    NOTEBOOK_DIR = Path.cwd()
    INPUT_DIR = Path(
        os.environ.get(
            "FILMFINDER_INPUT_DIR",
            "/Data/yike-ouyang/ml_training/data",
        )
    )

    CONFIG = {
        "input_dir": INPUT_DIR,
        "output_dir": Path(
            os.environ.get(
                "FILMFINDER_OUTPUT_DIR",
                NOTEBOOK_DIR / "outputs",
            )
        ),
        "fast_mode": True,
        "min_ratings_per_movie": 10,
        "min_ratings_per_user": 15,
        "rating_min": 0.5,
        "rating_max": 5.0,
        "year_tolerance_years": 2,
        "duration_tolerance_minutes": 10,
        "chunk_size": 1_000_000,
        "max_filter_iterations": 100,
        "timestamp_coverage_threshold": 0.95,
    }

    REQUIRED_FILES = {
        "movies": "movies.csv",
        "movie_data": "movie_data.csv",
        "ratings": "ratings_export.csv",
    }
    OPTIONAL_FILES = {
        "studios": "studios.csv",
        "genres": "genres.csv",
    }
    COLUMN_CANDIDATES = {
        "movies": {
            "id": ["id", "movie_id"],
            "title": ["name", "movie_title", "title"],
            "year": ["date", "release_date", "year"],
            "duration": ["minute", "minutes", "runtime", "duration"],
        },
        "studios": {
            "movie_id": ["movie_id", "film_id", "id"],
            "studio": ["studio", "name", "company"],
        },
        "genres": {
            "movie_id": ["movie_id", "film_id", "id"],
            "genre": ["genre", "genres", "name"],
        },
        "movie_data": {
            "movie_id": ["movie_id", "id"],
            "title": ["movie_title", "title", "name"],
            "year": ["release_date", "date", "year"],
        },
        "ratings": {
            "user_id": ["user_id", "user", "userId"],
            "movie_id": ["movie_id", "film_id", "movieId"],
            "rating": ["rating_val", "rating", "score"],
            "timestamp": [
                "timestamp",
                "date",
                "watched_date",
                "created_at",
                "updated_at",
            ],
        },
    }

    return (
        COLUMN_CANDIDATES,
        CONFIG,
        OPTIONAL_FILES,
        REQUIRED_FILES,
    )


@app.cell
def _(COLUMN_CANDIDATES, CONFIG, OPTIONAL_FILES, REQUIRED_FILES, pd):
    def _resolve_columns(path, specification):
        header = pd.read_csv(path, nrows=0)
        available = {
            str(column).strip().lstrip("\ufeff").casefold(): str(column)
            .strip()
            .lstrip("\ufeff")
            for column in header.columns
        }
        resolved = {}
        for role, candidates in specification.items():
            required = role != "timestamp"
            found = None
            for candidate in candidates:
                if candidate.casefold() in available:
                    found = available[candidate.casefold()]
                    break
            if found is None and required:
                raise KeyError(
                    f"Missing column for {path.name}: {role}. "
                    f"Expected one of {candidates}; found {list(header.columns)}."
                )
            resolved[role] = found
        return resolved

    SOURCE_PATHS = {}
    SOURCE_COLUMNS = {}

    for _key, _filename in REQUIRED_FILES.items():
        _path = CONFIG["input_dir"] / _filename
        if not _path.is_file():
            raise FileNotFoundError(f"Missing required file: {_path}")
        SOURCE_PATHS[_key] = _path
        SOURCE_COLUMNS[_key] = _resolve_columns(
            _path,
            COLUMN_CANDIDATES[_key],
        )

    for _key, _filename in OPTIONAL_FILES.items():
        _path = CONFIG["input_dir"] / _filename
        if not _path.is_file():
            continue
        SOURCE_PATHS[_key] = _path
        SOURCE_COLUMNS[_key] = _resolve_columns(
            _path,
            COLUMN_CANDIDATES[_key],
        )

    return SOURCE_COLUMNS, SOURCE_PATHS


@app.cell
def _(CONFIG, SOURCE_COLUMNS, SOURCE_PATHS, np, pd, re, unicodedata):
    def _normalize_text(value):
        if pd.isna(value):
            return pd.NA
        text = unicodedata.normalize("NFKD", str(value))
        text = "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )
        text = re.sub(r"\s+", " ", text.casefold()).strip()
        return text if text else pd.NA

    def _parse_year(value):
        if pd.isna(value):
            return pd.NA
        match = re.search(r"(18|19|20)\d{2}", str(value))
        return int(match.group(0)) if match else pd.NA

    def _parse_duration(value):
        if pd.isna(value):
            return pd.NA
        match = re.search(r"\d+(?:[.,]\d+)?", str(value))
        if not match:
            return pd.NA
        return float(match.group(0).replace(",", "."))

    def _to_identifier(series):
        return series.astype("string").str.strip().replace({"": pd.NA})

    _columns = SOURCE_COLUMNS["movies"]
    _movies_raw = pd.read_csv(
        SOURCE_PATHS["movies"],
        usecols=[
            _columns["id"],
            _columns["title"],
            _columns["year"],
            _columns["duration"],
        ],
        dtype="string",
    )

    movies = pd.DataFrame({
        "source_raw_id": _to_identifier(_movies_raw[_columns["id"]]),
        "title_raw": _movies_raw[_columns["title"]],
        "date_raw": _movies_raw[_columns["year"]],
        "duration_raw": _movies_raw[_columns["duration"]],
    })

    if movies["source_raw_id"].isna().any():
        _missing = int(movies["source_raw_id"].isna().sum())
        raise ValueError(f"movies.csv contains {_missing} missing source ids.")
    if movies["source_raw_id"].duplicated().any():
        _duplicates = int(movies["source_raw_id"].duplicated().sum())
        raise ValueError(
            f"movies.csv contains {_duplicates} duplicated source ids."
        )

    movies["_row_id"] = np.arange(len(movies), dtype=np.int64)
    movies["source_id"] = "gsimonx37:" + movies["source_raw_id"]
    movies["title_norm"] = movies["title_raw"].map(_normalize_text)
    movies["year"] = pd.to_numeric(
        movies["date_raw"].map(_parse_year),
        errors="coerce",
    )
    movies["duration_min"] = pd.to_numeric(
        movies["duration_raw"].map(_parse_duration),
        errors="coerce",
    )

    _canonical_of = {}
    _keyed = movies.loc[
        movies["title_norm"].notna() & movies["year"].notna()
    ].copy()
    _keyed["_group_key"] = (
        _keyed["title_norm"].astype(str)
        + "|"
        + _keyed["year"].astype("Int64").astype(str)
    )
    _duration_tolerance = CONFIG["duration_tolerance_minutes"]

    for _group_key, _group in _keyed.groupby("_group_key", sort=False):
        _with_duration = _group.dropna(subset=["duration_min"])
        if len(_with_duration):
            _ordered = _with_duration.sort_values(
                ["duration_min", "source_raw_id", "_row_id"]
            )
            _representative = _ordered.iloc[len(_ordered) // 2]
        else:
            _representative = _group.sort_values(
                ["source_raw_id", "_row_id"]
            ).iloc[0]

        _representative_id = int(_representative["_row_id"])
        _representative_duration = _representative["duration_min"]

        for _, _member in _group.iterrows():
            _member_id = int(_member["_row_id"])
            _member_duration = _member["duration_min"]

            if _member_id == _representative_id:
                _canonical_of[_member_id] = _representative_id
            elif pd.isna(_member_duration) and pd.isna(
                _representative_duration
            ):
                _canonical_of[_member_id] = _representative_id
            elif (
                pd.notna(_member_duration)
                and pd.notna(_representative_duration)
                and abs(
                    float(_member_duration)
                    - float(_representative_duration)
                )
                < _duration_tolerance
            ):
                _canonical_of[_member_id] = _representative_id
            else:
                _canonical_of[_member_id] = _member_id

    for _row_id in movies["_row_id"].to_numpy():
        _canonical_of.setdefault(int(_row_id), int(_row_id))

    movies["canonical_row_id"] = (
        movies["_row_id"].map(_canonical_of).astype("int64")
    )
    _row_to_source = movies.set_index("_row_id")["source_id"]
    movies["canonical_movie_id"] = movies["canonical_row_id"].map(
        _row_to_source
    )

    canonical_movies = movies.loc[
        movies["_row_id"] == movies["canonical_row_id"]
    ].copy()

    MOVIE_ID_MAPPING = (
        movies[["source_raw_id", "canonical_movie_id"]]
        .drop_duplicates(subset=["source_raw_id"])
        .reset_index(drop=True)
    )

    CATALOG = (
        canonical_movies[
            [
                "canonical_movie_id",
                "source_raw_id",
                "title_raw",
                "title_norm",
                "year",
                "duration_min",
            ]
        ]
        .rename(columns={"title_raw": "title"})
        .reset_index(drop=True)
    )

    for _relation, _value_role in (
        ("studios", "studio"),
        ("genres", "genre"),
    ):
        if _relation not in SOURCE_PATHS:
            CATALOG[_relation] = pd.Series(
                [None] * len(CATALOG),
                index=CATALOG.index,
                dtype=object,
            )
            continue

        _relation_columns = SOURCE_COLUMNS[_relation]
        _relation_raw = pd.read_csv(
            SOURCE_PATHS[_relation],
            usecols=[
                _relation_columns["movie_id"],
                _relation_columns[_value_role],
            ],
            dtype="string",
        )
        _relation_raw = _relation_raw.rename(
            columns={
                _relation_columns["movie_id"]: "source_raw_id",
                _relation_columns[_value_role]: "value",
            }
        )
        _relation_raw["source_raw_id"] = (
            _relation_raw["source_raw_id"]
            .str.strip()
            .replace({"": pd.NA})
        )
        _relation_raw = _relation_raw.dropna(subset=["source_raw_id"])
        _relation_raw = _relation_raw.merge(
            MOVIE_ID_MAPPING,
            on="source_raw_id",
            how="inner",
            validate="many_to_one",
        )
        _aggregated = (
            _relation_raw.groupby("canonical_movie_id")["value"]
            .apply(
                lambda values: sorted({
                    str(value).strip()
                    for value in values
                    if pd.notna(value) and str(value).strip()
                })
            )
            .rename(_relation)
        )
        CATALOG = CATALOG.merge(
            _aggregated,
            left_on="canonical_movie_id",
            right_index=True,
            how="left",
        )

    MOVIES_STATS = {
        "raw": int(len(movies)),
        "canonical": int(len(CATALOG)),
        "merged": int(len(movies) - len(CATALOG)),
    }

    return CATALOG, MOVIE_ID_MAPPING, MOVIES_STATS


@app.cell
def _(CATALOG, CONFIG, MOVIE_ID_MAPPING, SOURCE_COLUMNS, SOURCE_PATHS, pd, re, unicodedata):
    def _normalize_text(value):
        if pd.isna(value):
            return pd.NA
        text = unicodedata.normalize("NFKD", str(value))
        text = "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )
        text = re.sub(r"\s+", " ", text.casefold()).strip()
        return text if text else pd.NA

    def _parse_year(value):
        if pd.isna(value):
            return pd.NA
        match = re.search(r"(18|19|20)\d{2}", str(value))
        return int(match.group(0)) if match else pd.NA

    _columns = SOURCE_COLUMNS["movie_data"]
    _movie_data_raw = pd.read_csv(
        SOURCE_PATHS["movie_data"],
        usecols=[
            _columns["movie_id"],
            _columns["title"],
            _columns["year"],
        ],
        dtype="string",
    )

    movie_data = pd.DataFrame({
        "source_movie_id": (
            _movie_data_raw[_columns["movie_id"]]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA})
        ),
        "title_norm": _movie_data_raw[_columns["title"]].map(
            _normalize_text
        ),
        "year": pd.to_numeric(
            _movie_data_raw[_columns["year"]].map(_parse_year),
            errors="coerce",
        ),
    })
    movie_data = (
        movie_data.dropna(subset=["source_movie_id"])
        .drop_duplicates(subset=["source_movie_id"], keep="first")
        .reset_index(drop=True)
    )

    _catalog_ids = set(
        MOVIE_ID_MAPPING["source_raw_id"].dropna().astype(str)
    )
    _sample_ids = set(movie_data["source_movie_id"].dropna().astype(str))
    _overlap = (
        len(_catalog_ids & _sample_ids) / len(_sample_ids)
        if _sample_ids
        else 0.0
    )
    _use_ids = _overlap >= 0.5

    if _use_ids:
        _direct_mapping = MOVIE_ID_MAPPING.rename(
            columns={"source_raw_id": "source_movie_id"}
        )
        MOVIE_DATA_MAP = movie_data[["source_movie_id"]].merge(
            _direct_mapping,
            on="source_movie_id",
            how="inner",
            validate="one_to_one",
        )
        MOVIE_DATA_MAP["match_method"] = "id"
        _ambiguous = 0
    else:
        _catalog_keys = (
            CATALOG[["canonical_movie_id", "title_norm", "year"]]
            .dropna(subset=["title_norm", "year"])
            .copy()
        )
        _catalog_keys["_key"] = (
            _catalog_keys["title_norm"].astype(str)
            + "|"
            + _catalog_keys["year"].astype("Int64").astype(str)
        )
        _key_sizes = _catalog_keys.groupby("_key")[
            "canonical_movie_id"
        ].nunique()
        _unique_keys = _key_sizes[_key_sizes == 1].index
        _ambiguous_keys = _key_sizes[_key_sizes > 1].index
        _catalog_unique = _catalog_keys.loc[
            _catalog_keys["_key"].isin(_unique_keys),
            ["_key", "canonical_movie_id"],
        ]

        _candidates = movie_data.dropna(
            subset=["title_norm", "year"]
        ).copy()
        _candidates["_key"] = (
            _candidates["title_norm"].astype(str)
            + "|"
            + _candidates["year"].astype("Int64").astype(str)
        )
        MOVIE_DATA_MAP = _candidates[
            ["source_movie_id", "_key"]
        ].merge(
            _catalog_unique,
            on="_key",
            how="inner",
            validate="many_to_one",
        )
        MOVIE_DATA_MAP = MOVIE_DATA_MAP[
            ["source_movie_id", "canonical_movie_id"]
        ]
        MOVIE_DATA_MAP["match_method"] = "title_year"
        _ambiguous = int(
            _candidates["_key"].isin(_ambiguous_keys).sum()
        )

    MOVIE_DATA_MAP = (
        MOVIE_DATA_MAP.drop_duplicates(subset=["source_movie_id"])
        .reset_index(drop=True)
    )

    MATCH_STATS = {
        "movie_data_rows": int(len(movie_data)),
        "matched": int(len(MOVIE_DATA_MAP)),
        "method": "id" if _use_ids else "title_year",
        "overlap": float(_overlap),
        "ambiguous": int(_ambiguous),
    }

    return MATCH_STATS, MOVIE_DATA_MAP


@app.cell
def _(CONFIG, MOVIE_DATA_MAP, SOURCE_COLUMNS, SOURCE_PATHS, pd):
    _columns = SOURCE_COLUMNS["ratings"]
    _timestamp_column = _columns.get("timestamp")

    _usecols = [
        _columns["user_id"],
        _columns["movie_id"],
        _columns["rating"],
    ]
    _rename = {
        _columns["user_id"]: "user_id",
        _columns["movie_id"]: "source_movie_id",
        _columns["rating"]: "rating",
    }
    if _timestamp_column is not None:
        _usecols.append(_timestamp_column)
        _rename[_timestamp_column] = "timestamp"

    _accepted = MOVIE_DATA_MAP[
        ["source_movie_id", "canonical_movie_id"]
    ]
    _parts = []
    _read_rows = 0

    for _chunk in pd.read_csv(
        SOURCE_PATHS["ratings"],
        usecols=_usecols,
        dtype="string",
        chunksize=CONFIG["chunk_size"],
    ):
        _read_rows += len(_chunk)
        _chunk = _chunk.rename(columns=_rename)
        _chunk["user_id"] = (
            _chunk["user_id"].str.strip().replace({"": pd.NA})
        )
        _chunk["source_movie_id"] = (
            _chunk["source_movie_id"].str.strip().replace({"": pd.NA})
        )
        _chunk["rating"] = pd.to_numeric(
            _chunk["rating"],
            errors="coerce",
        )
        _chunk = _chunk.dropna(
            subset=["user_id", "source_movie_id", "rating"]
        )
        _chunk = _chunk.loc[
            (_chunk["rating"] >= CONFIG["rating_min"])
            & (_chunk["rating"] <= CONFIG["rating_max"])
        ]
        if _chunk.empty:
            continue
        _chunk["rating"] = _chunk["rating"].astype("float32")

        _chunk = _chunk.merge(
            _accepted,
            on="source_movie_id",
            how="inner",
            validate="many_to_one",
        )
        if _chunk.empty:
            continue

        _keep = ["user_id", "canonical_movie_id", "rating"]
        if _timestamp_column is not None:
            _keep.append("timestamp")
        _parts.append(_chunk[_keep])

    if not _parts:
        RATINGS_CLEAN = pd.DataFrame(
            columns=["user_id", "canonical_movie_id", "rating"]
        )
        _aggregation = "empty"
    else:
        _ratings = pd.concat(_parts, ignore_index=True)
        del _parts

        _ratings = _ratings.drop_duplicates(
            subset=["user_id", "canonical_movie_id", "rating"]
        )

        if (
            _timestamp_column is not None
            and "timestamp" in _ratings.columns
        ):
            _timestamps = pd.to_datetime(
                _ratings["timestamp"],
                errors="coerce",
            )
            if (
                _timestamps.notna().mean()
                >= CONFIG["timestamp_coverage_threshold"]
            ):
                _ratings = _ratings.assign(_timestamp=_timestamps)
                _ratings = (
                    _ratings.sort_values(
                        ["_timestamp", "rating"],
                        na_position="first",
                    )
                    .drop_duplicates(
                        subset=["user_id", "canonical_movie_id"],
                        keep="last",
                    )
                    .drop(columns=["_timestamp", "timestamp"])
                )
                _aggregation = "latest_timestamp"
            else:
                _ratings = _ratings.groupby(
                    ["user_id", "canonical_movie_id"],
                    as_index=False,
                    sort=False,
                )["rating"].mean()
                _aggregation = "mean_of_distinct_ratings"
        else:
            _ratings = _ratings.groupby(
                ["user_id", "canonical_movie_id"],
                as_index=False,
                sort=False,
            )["rating"].mean()
            _aggregation = "mean_of_distinct_ratings"

        _ratings["rating"] = _ratings["rating"].astype("float32")
        RATINGS_CLEAN = _ratings.reset_index(drop=True)

    RATINGS_STATS = {
        "read_rows": int(_read_rows),
        "clean_rows": int(len(RATINGS_CLEAN)),
        "aggregation": _aggregation,
        "has_timestamp": bool(_timestamp_column is not None),
    }

    return RATINGS_CLEAN, RATINGS_STATS


@app.cell
def _(CONFIG, RATINGS_CLEAN, pd):
    _ratings = RATINGS_CLEAN.copy()
    _history = []
    _stable = False
    _iteration = 0

    for _iteration in range(1, CONFIG["max_filter_iterations"] + 1):
        _before = len(_ratings)

        _film_counts = _ratings.groupby("canonical_movie_id").size()
        _kept_films = _film_counts[
            _film_counts >= CONFIG["min_ratings_per_movie"]
        ].index
        _ratings = _ratings.loc[
            _ratings["canonical_movie_id"].isin(_kept_films)
        ]

        _user_counts = _ratings.groupby("user_id").size()
        _kept_users = _user_counts[
            _user_counts >= CONFIG["min_ratings_per_user"]
        ].index
        _ratings = _ratings.loc[
            _ratings["user_id"].isin(_kept_users)
        ]

        _history.append({
            "iteration": _iteration,
            "rows": len(_ratings),
            "users": int(_ratings["user_id"].nunique()),
            "movies": int(_ratings["canonical_movie_id"].nunique()),
            "removed_rows": _before - len(_ratings),
        })

        if len(_ratings) == _before:
            _stable = True
            break
        if _ratings.empty:
            break

    RATINGS_FILTERED = _ratings.reset_index(drop=True)
    FILTER_HISTORY = pd.DataFrame(_history)
    FILTER_SUMMARY = {
        "before": int(len(RATINGS_CLEAN)),
        "after": int(len(RATINGS_FILTERED)),
        "iterations": int(_iteration),
        "stable": bool(_stable),
    }

    return FILTER_HISTORY, FILTER_SUMMARY, RATINGS_FILTERED


@app.cell
def _(
    CATALOG,
    CONFIG,
    RATINGS_FILTERED,
    csr_matrix,
    np,
    pd,
    save_npz,
):
    if RATINGS_FILTERED.empty:
        raise ValueError(
            "No ratings left after filtering; the matrix cannot be built."
        )

    _user_ids = np.sort(
        RATINGS_FILTERED["user_id"].astype(str).unique()
    )
    _movie_ids = np.sort(
        RATINGS_FILTERED["canonical_movie_id"].astype(str).unique()
    )

    _user_index = pd.Index(_user_ids)
    _movie_index = pd.Index(_movie_ids)
    _user_codes = _user_index.get_indexer(
        RATINGS_FILTERED["user_id"].astype(str)
    )
    _movie_codes = _movie_index.get_indexer(
        RATINGS_FILTERED["canonical_movie_id"].astype(str)
    )

    if (_user_codes < 0).any() or (_movie_codes < 0).any():
        raise ValueError("Some ratings could not be mapped to matrix indices.")

    RATINGS_MATRIX = csr_matrix(
        (
            RATINGS_FILTERED["rating"].to_numpy(dtype=np.float32),
            (_user_codes, _movie_codes),
        ),
        shape=(len(_user_ids), len(_movie_ids)),
    )

    USER_MAPPING = pd.DataFrame({
        "user_id": _user_ids,
        "row_index": np.arange(len(_user_ids), dtype=np.int64),
    })
    MOVIE_MAPPING = pd.DataFrame({
        "canonical_movie_id": _movie_ids,
        "col_index": np.arange(len(_movie_ids), dtype=np.int64),
    })

    _context_columns = [
        "canonical_movie_id",
        "title",
        "year",
        "duration_min",
        "genres",
        "studios",
    ]
    _missing_context = [
        column
        for column in _context_columns
        if column not in CATALOG.columns
    ]
    if _missing_context:
        raise KeyError(
            f"Missing movie context columns: {_missing_context}"
        )

    MOVIE_CONTEXT = (
        MOVIE_MAPPING
        .merge(
            CATALOG[_context_columns],
            on="canonical_movie_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("col_index")
        .reset_index(drop=True)
    )

    if MOVIE_CONTEXT["canonical_movie_id"].tolist() != (
        MOVIE_MAPPING["canonical_movie_id"].tolist()
    ):
        raise ValueError("Movie context is not aligned with the matrix columns.")

    OUTPUT_DIR = CONFIG["output_dir"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_npz(OUTPUT_DIR / "ratings_matrix.npz", RATINGS_MATRIX)
    USER_MAPPING.to_parquet(
        OUTPUT_DIR / "user_mapping.parquet",
        index=False,
    )
    MOVIE_MAPPING.to_parquet(
        OUTPUT_DIR / "movie_mapping.parquet",
        index=False,
    )
    MOVIE_CONTEXT.to_parquet(
        OUTPUT_DIR / "movie_context.parquet",
        index=False,
    )

    return (
        MOVIE_CONTEXT,
        MOVIE_MAPPING,
        OUTPUT_DIR,
        RATINGS_MATRIX,
        USER_MAPPING,
    )


@app.cell
def _(
    CONFIG,
    FILTER_HISTORY,
    FILTER_SUMMARY,
    MATCH_STATS,
    MOVIE_CONTEXT,
    MOVIE_MAPPING,
    MOVIES_STATS,
    OUTPUT_DIR,
    PIPELINE_START,
    RATINGS_FILTERED,
    RATINGS_MATRIX,
    RATINGS_STATS,
    USER_MAPPING,
    mo,
    np,
    pd,
    time,
):
    _film_counts = RATINGS_FILTERED.groupby("canonical_movie_id").size()
    _user_counts = RATINGS_FILTERED.groupby("user_id").size()

    checks = {
        "canonical ids non-null": bool(
            MOVIE_MAPPING["canonical_movie_id"].notna().all()
            and MOVIE_CONTEXT["canonical_movie_id"].notna().all()
        ),
        "unique user-film pairs": bool(
            not RATINGS_FILTERED.duplicated(
                subset=["user_id", "canonical_movie_id"]
            ).any()
        ),
        "matrix dimensions": bool(
            RATINGS_MATRIX.shape
            == (len(USER_MAPPING), len(MOVIE_MAPPING))
        ),
        "mapping alignment": bool(
            np.array_equal(
                USER_MAPPING["row_index"].to_numpy(),
                np.arange(len(USER_MAPPING)),
            )
            and np.array_equal(
                MOVIE_MAPPING["col_index"].to_numpy(),
                np.arange(len(MOVIE_MAPPING)),
            )
            and MOVIE_CONTEXT["canonical_movie_id"].tolist()
            == MOVIE_MAPPING["canonical_movie_id"].tolist()
        ),
        "activity thresholds": bool(
            (
                _film_counts.empty
                or (
                    _film_counts
                    >= CONFIG["min_ratings_per_movie"]
                ).all()
            )
            and (
                _user_counts.empty
                or (
                    _user_counts
                    >= CONFIG["min_ratings_per_user"]
                ).all()
            )
        ),
        "all ratings mapped": bool(
            RATINGS_FILTERED["canonical_movie_id"].isin(
                MOVIE_MAPPING["canonical_movie_id"]
            ).all()
        ),
        "nnz equals final ratings": bool(
            RATINGS_MATRIX.nnz == len(RATINGS_FILTERED)
        ),
    }

    _failed = [name for name, passed in checks.items() if not passed]
    if _failed:
        raise AssertionError(f"Consistency checks failed: {_failed}")

    _elapsed = time.perf_counter() - PIPELINE_START
    _density = (
        RATINGS_MATRIX.nnz
        / (RATINGS_MATRIX.shape[0] * RATINGS_MATRIX.shape[1])
    )

    _summary = mo.md(
        f"""
        ## Résumé du preprocessing

        | Indicateur | Valeur |
        |---|---|
        | Films sources | {MOVIES_STATS['raw']:,} |
        | Films canoniques | {MOVIES_STATS['canonical']:,} |
        | Films fusionnés | {MOVIES_STATS['merged']:,} |
        | Lignes `movie_data` | {MATCH_STATS['movie_data_rows']:,} |
        | Correspondances acceptées | {MATCH_STATS['matched']:,} |
        | Méthode de rapprochement | `{MATCH_STATS['method']}` |
        | Notes lues | {RATINGS_STATS['read_rows']:,} |
        | Notes après nettoyage | {RATINGS_STATS['clean_rows']:,} |
        | Agrégation des doublons | `{RATINGS_STATS['aggregation']}` |
        | Notes après filtrage | {FILTER_SUMMARY['after']:,} |
        | Utilisateurs | {len(USER_MAPPING):,} |
        | Films retenus | {len(MOVIE_MAPPING):,} |
        | Dimensions matrice | {RATINGS_MATRIX.shape} |
        | Valeurs non nulles | {RATINGS_MATRIX.nnz:,} |
        | Densité | {_density:.4%} |
        | Itérations de filtrage | {FILTER_SUMMARY['iterations']} |
        | Stabilisation | {FILTER_SUMMARY['stable']} |
        | Durée totale | {_elapsed:.1f} s |

        Sorties écrites dans `{OUTPUT_DIR}` :
        `ratings_matrix.npz`, `user_mapping.parquet`,
        `movie_mapping.parquet`, `movie_context.parquet`.
        """
    )

    if CONFIG["fast_mode"]:
        _output = _summary
    else:
        _output = mo.vstack([
            _summary,
            mo.md("### Historique du filtrage"),
            mo.ui.table(FILTER_HISTORY, pagination=True),
        ])

    _output
    return


if __name__ == "__main__":
    app.run()
