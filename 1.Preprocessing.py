import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Filmfinder — Data Preprocessing

    This notebook builds a reproducible user–film rating matrix from two
    Letterboxd datasets. It audits the sources, normalizes text, deduplicates
    the film catalogue, attaches studios and genres, matches the rating
    catalogue to canonical films, cleans ratings, filters low activity and
    exports a CSR matrix with aligned metadata.

    Source files are never modified. All outputs are written to the configured
    output directory.
    """)
    return


@app.cell
def _():
    import json
    import os
    import re
    import unicodedata
    from collections import defaultdict
    from datetime import datetime, timezone
    from itertools import combinations
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, save_npz

    try:
        import pyarrow
    except ImportError:
        pyarrow = None

    return (
        Path,
        combinations,
        csr_matrix,
        datetime,
        defaultdict,
        json,
        mo,
        np,
        os,
        pd,
        plt,
        pyarrow,
        re,
        save_npz,
        timezone,
        unicodedata,
    )


@app.cell
def _(pd):
    class ProcessingLog:
        def __init__(self):
            self._rows = {}

        def add(self, step, rule, before=None, after=None, removed=None, details=""):
            self._rows[step] = {
                "step": step,
                "rule": rule,
                "before": before,
                "after": after,
                "removed": removed,
                "details": details,
            }

        def frame(self):
            return pd.DataFrame(list(self._rows.values()))

    return (ProcessingLog,)


@app.cell
def _(ProcessingLog):
    log = ProcessingLog()
    return (log,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Sources and configuration

    Kaggle datasets are downloaded only when the required CSV files cannot be
    found locally. Authentication relies on the existing Kaggle environment;
    no credentials are stored in this notebook.
    """)
    return


@app.cell
def _(Path, os):
    NOTEBOOK_DIR = Path.cwd()
    INPUT_DIR = Path(
        os.environ.get("FILMFINDER_INPUT_DIR", NOTEBOOK_DIR / "data")
    )
    OUTPUT_DIR = Path(
        os.environ.get("FILMFINDER_OUTPUT_DIR", NOTEBOOK_DIR / "outputs")
    )

    CONFIG = {
        "notebook_dir": NOTEBOOK_DIR,
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "download_if_missing": True,
        "min_ratings_per_movie": 10,
        "min_ratings_per_user": 15,
        "duration_tolerance_minutes": 10,
        "year_tolerance_years": 2,
        "rating_min": 0.5,
        "rating_max": 5.0,
        "max_filter_iterations": 100,
        "timestamp_coverage_threshold": 0.95,
    }

    DATASETS = {
        "catalogue": {
            "slug": "gsimonx37/letterboxd",
            "version": None,
            "files": {
                "movies.csv": "movies",
                "studios.csv": "studios",
                "genres.csv": "genres",
            },
        },
        "ratings": {
            "slug": "samlearner/letterboxd-movie-ratings-data",
            "version": 6,
            "files": {
                "movie_data.csv": "movie_data",
                "ratings_export.csv": "ratings",
            },
        },
    }

    SEARCH_DIRS = [
        INPUT_DIR,
        NOTEBOOK_DIR,
        NOTEBOOK_DIR / "data",
        NOTEBOOK_DIR / "input",
        Path(
            "/users/eleves-b/2023/yike.ouyang/.cache/kagglehub/"
            "datasets/samlearner/letterboxd-movie-ratings-data/versions/6"
        ),
    ]

    return CONFIG, DATASETS, SEARCH_DIRS


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Source discovery and download

    Each dataset is searched as a whole. If at least one expected file is
    missing, the dataset is downloaded with `kagglehub` and searched
    recursively in the returned directory.
    """)
    return


@app.cell
def _(CONFIG, DATASETS, Path, SEARCH_DIRS, pd):
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the datasets. "
            "Install it with `pip install kagglehub`."
        ) from exc

    def find_file(filename, roots):
        for root in roots:
            if not root.exists():
                continue
            direct = root / filename
            if direct.is_file():
                return direct
        for root in roots:
            if not root.exists():
                continue
            matches = sorted(root.rglob(filename))
            if matches:
                return matches[0]
        return None

    def download_dataset(slug, version=None):
        try:
            if version is not None:
                return Path(
                    kagglehub.dataset_download(slug, version=version)
                )
            return Path(kagglehub.dataset_download(slug))
        except TypeError:
            return Path(kagglehub.dataset_download(slug))
        except Exception as exc:
            raise RuntimeError(
                f"Unable to download Kaggle dataset {slug}. "
                "Check the Kaggle authentication available in the "
                "execution environment."
            ) from exc

    SOURCE_PATHS = {}
    download_rows = []
    local_roots = [root for root in SEARCH_DIRS if root.exists()]

    for dataset_name, specification in DATASETS.items():
        found = {
            source_filename: find_file(source_filename, local_roots)
            for source_filename in specification["files"]
        }

        mode = "local"
        dataset_root = None

        if not all(found.values()):
            if not CONFIG["download_if_missing"]:
                missing_files = [
                    source_filename
                    for source_filename, source_path in found.items()
                    if source_path is None
                ]
                raise FileNotFoundError(
                    f"Missing files for {dataset_name}: {missing_files}"
                )

            dataset_root = download_dataset(
                specification["slug"],
                specification["version"],
            )
            mode = "kagglehub"
            found = {
                source_filename: find_file(
                    source_filename,
                    [dataset_root],
                )
                for source_filename in specification["files"]
            }

        for source_filename, source_path in found.items():
            if source_path is None:
                raise FileNotFoundError(
                    f"Could not locate {source_filename} for dataset "
                    f"{dataset_name}."
                )
            SOURCE_PATHS[specification["files"][source_filename]] = source_path

        download_rows.append({
            "dataset": dataset_name,
            "slug": specification["slug"],
            "version": specification["version"],
            "mode": mode,
            "root": None if dataset_root is None else str(dataset_root),
        })

    SOURCE_PATHS_DF = pd.DataFrame(download_rows)
    SOURCE_PATHS_DF
    return SOURCE_PATHS, SOURCE_PATHS_DF


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Audit of the source files

    Column names are resolved explicitly. The next cells fail early with a
    clear message when an expected column is absent.
    """)
    return


@app.cell
def _(SOURCE_PATHS, pd):
    def read_source(path):
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [
            str(column).strip().lstrip("\ufeff")
            for column in frame.columns
        ]
        return frame

    movies_raw = read_source(SOURCE_PATHS["movies"])
    studios_raw = read_source(SOURCE_PATHS["studios"])
    genres_raw = read_source(SOURCE_PATHS["genres"])
    movie_data_raw = read_source(SOURCE_PATHS["movie_data"])
    ratings_raw = read_source(SOURCE_PATHS["ratings"])

    return (
        genres_raw,
        movie_data_raw,
        movies_raw,
        ratings_raw,
        studios_raw,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Schema resolution

    The datasets do not necessarily share the same key names. In particular,
    `id` and `movie_id` are checked separately and never assumed interchangeable.
    """)
    return


@app.cell
def _(
    genres_raw,
    log,
    movie_data_raw,
    movies_raw,
    pd,
    ratings_raw,
    studios_raw,
):
    def pick_column(frame, candidates, label, required=True):
        lower_columns = {
            str(column).casefold(): column for column in frame.columns
        }
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
            match = lower_columns.get(candidate.casefold())
            if match is not None:
                return match
        if required:
            raise KeyError(
                f"Missing column for {label}. Expected one of "
                f"{candidates}; found {list(frame.columns)}."
            )
        return None

    SCHEMA = {
        "movies": {
            "id": pick_column(movies_raw, ["id", "movie_id"], "movies.id"),
            "title": pick_column(
                movies_raw,
                ["name", "movie_title", "title"],
                "movies.title",
            ),
            "description": pick_column(
                movies_raw,
                ["description", "overview"],
                "movies.description",
            ),
            "year": pick_column(
                movies_raw,
                ["date", "release_date", "year"],
                "movies.year",
            ),
            "duration": pick_column(
                movies_raw,
                ["minute", "minutes", "runtime", "duration"],
                "movies.duration",
            ),
        },
        "studios": {
            "movie_id": pick_column(
                studios_raw,
                ["movie_id", "film_id", "id"],
                "studios.movie_id",
            ),
            "studio": pick_column(
                studios_raw,
                ["studio", "name", "company"],
                "studios.studio",
            ),
        },
        "genres": {
            "movie_id": pick_column(
                genres_raw,
                ["movie_id", "film_id", "id"],
                "genres.movie_id",
            ),
            "genre": pick_column(
                genres_raw,
                ["genre", "genres", "name"],
                "genres.genre",
            ),
        },
        "movie_data": {
            "movie_id": pick_column(
                movie_data_raw,
                ["movie_id", "id"],
                "movie_data.movie_id",
            ),
            "title": pick_column(
                movie_data_raw,
                ["movie_title", "title", "name"],
                "movie_data.title",
            ),
            "overview": pick_column(
                movie_data_raw,
                ["overview", "description"],
                "movie_data.overview",
            ),
            "year": pick_column(
                movie_data_raw,
                ["release_date", "date", "year"],
                "movie_data.year",
            ),
            "duration": pick_column(
                movie_data_raw,
                ["minute", "minutes", "runtime", "duration"],
                "movie_data.duration",
                required=False,
            ),
            "genres": pick_column(
                movie_data_raw,
                ["genres", "genre"],
                "movie_data.genres",
                required=False,
            ),
        },
        "ratings": {
            "user_id": pick_column(
                ratings_raw,
                ["user_id", "user", "userId"],
                "ratings.user_id",
            ),
            "movie_id": pick_column(
                ratings_raw,
                ["movie_id", "film_id", "movieId"],
                "ratings.movie_id",
            ),
            "rating": pick_column(
                ratings_raw,
                ["rating_val", "rating", "score"],
                "ratings.rating",
            ),
            "timestamp": pick_column(
                ratings_raw,
                [
                    "timestamp",
                    "date",
                    "watched_date",
                    "created_at",
                    "updated_at",
                ],
                "ratings.timestamp",
                required=False,
            ),
        },
    }

    def audit_frame(name, frame, key_columns):
        duplicate_keys = (
            int(frame.duplicated(subset=key_columns).sum())
            if key_columns
            else None
        )
        return {
            "file": name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "column_names": ", ".join(map(str, frame.columns)),
            "missing_values": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_keys": duplicate_keys,
            "key_columns": ", ".join(key_columns) if key_columns else None,
        }

    audit_df = pd.DataFrame([
        audit_frame(
            "movies.csv",
            movies_raw,
            [SCHEMA["movies"]["id"]],
        ),
        audit_frame(
            "studios.csv",
            studios_raw,
            [SCHEMA["studios"]["movie_id"], SCHEMA["studios"]["studio"]],
        ),
        audit_frame(
            "genres.csv",
            genres_raw,
            [SCHEMA["genres"]["movie_id"], SCHEMA["genres"]["genre"]],
        ),
        audit_frame(
            "movie_data.csv",
            movie_data_raw,
            [SCHEMA["movie_data"]["movie_id"]],
        ),
        audit_frame(
            "ratings_export.csv",
            ratings_raw,
            [
                SCHEMA["ratings"]["user_id"],
                SCHEMA["ratings"]["movie_id"],
                SCHEMA["ratings"]["rating"],
            ],
        ),
    ])

    missing_rows = []
    for source_name, source_frame in [
        ("movies.csv", movies_raw),
        ("studios.csv", studios_raw),
        ("genres.csv", genres_raw),
        ("movie_data.csv", movie_data_raw),
        ("ratings_export.csv", ratings_raw),
    ]:
        missing_counts = source_frame.isna().sum()
        for source_column, count in missing_counts.items():
            if count:
                missing_rows.append({
                    "file": source_name,
                    "column": source_column,
                    "missing": int(count),
                    "missing_ratio": float(count / len(source_frame)),
                })

    missing_df = pd.DataFrame(
        missing_rows,
        columns=["file", "column", "missing", "missing_ratio"],
    )

    schema_rows = []
    for file_name, roles in SCHEMA.items():
        for role, schema_column in roles.items():
            schema_rows.append({
                "dataset": file_name,
                "role": role,
                "column": schema_column,
            })

    schema_df = pd.DataFrame(schema_rows)

    log.add(
        "01_audit",
        "Inspection des dimensions, types, clés et valeurs manquantes",
        after=len(audit_df),
        details="Audit des cinq fichiers sources",
    )

    return SCHEMA, audit_df, missing_df, schema_df


@app.cell
def _(audit_df, missing_df, schema_df):
    audit_df
    return


@app.cell
def _(missing_df):
    missing_df
    return


@app.cell
def _(schema_df):
    schema_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Normalization and numeric extraction

    Original texts are preserved. Normalized texts are used only for matching
    and deduplication.
    """)
    return


@app.cell
def _(
    SCHEMA,
    log,
    movie_data_raw,
    movies_raw,
    np,
    pd,
    ratings_raw,
    re,
    unicodedata,
):
    def normalize_text(value):
        if pd.isna(value):
            return pd.NA
        text = unicodedata.normalize("NFKD", str(value))
        text = "".join(
            character
            for character in text
            if not unicodedata.combining(character)
        )
        text = text.casefold()
        text = re.sub(r"\s+", " ", text).strip()
        return text if text else pd.NA

    def parse_year(value):
        if pd.isna(value):
            return pd.NA
        match = re.search(r"(18|19|20)\d{2}", str(value))
        return int(match.group(0)) if match else pd.NA

    def parse_duration(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        match = re.search(r"\d+(?:[.,]\d+)?", str(value))
        if not match:
            return pd.NA
        return float(match.group(0).replace(",", "."))

    def to_identifier(series):
        return (
            series.astype("string")
            .str.strip()
            .replace({"": pd.NA})
        )

    movies = pd.DataFrame({
        "_row_id": np.arange(len(movies_raw)),
        "source_raw_id": to_identifier(
            movies_raw[SCHEMA["movies"]["id"]]
        ),
        "title_raw": movies_raw[SCHEMA["movies"]["title"]],
        "description_raw": movies_raw[SCHEMA["movies"]["description"]],
        "date_raw": movies_raw[SCHEMA["movies"]["year"]],
        "duration_raw": movies_raw[SCHEMA["movies"]["duration"]],
    })
    movies["source_id"] = (
        "gsimonx37:" + movies["source_raw_id"].astype("string")
    )
    movies["title_norm"] = movies["title_raw"].map(normalize_text)
    movies["description_norm"] = movies["description_raw"].map(
        normalize_text
    )
    movies["year"] = pd.to_numeric(
        movies["date_raw"].map(parse_year),
        errors="coerce",
    )
    movies["duration_min"] = pd.to_numeric(
        movies["duration_raw"].map(parse_duration),
        errors="coerce",
    )

    if movies["source_raw_id"].isna().any():
        missing_identifiers = int(
            movies["source_raw_id"].isna().sum()
        )
        raise ValueError(
            f"movies.csv contains {missing_identifiers} missing source ids."
        )

    if movies["source_raw_id"].duplicated().any():
        duplicates = int(movies["source_raw_id"].duplicated().sum())
        raise ValueError(
            f"movies.csv contains {duplicates} duplicated source ids."
        )

    movie_data = pd.DataFrame({
        "movie_data_row_id": np.arange(len(movie_data_raw)),
        "source_movie_id": to_identifier(
            movie_data_raw[SCHEMA["movie_data"]["movie_id"]]
        ),
        "title_raw": movie_data_raw[SCHEMA["movie_data"]["title"]],
        "overview_raw": movie_data_raw[
            SCHEMA["movie_data"]["overview"]
        ],
        "release_raw": movie_data_raw[SCHEMA["movie_data"]["year"]],
    })

    if SCHEMA["movie_data"]["duration"] is not None:
        movie_data["duration_raw"] = movie_data_raw[
            SCHEMA["movie_data"]["duration"]
        ]
    else:
        movie_data["duration_raw"] = pd.NA

    if SCHEMA["movie_data"]["genres"] is not None:
        movie_data["genres_raw"] = movie_data_raw[
            SCHEMA["movie_data"]["genres"]
        ]
    else:
        movie_data["genres_raw"] = pd.NA

    movie_data["title_norm"] = movie_data["title_raw"].map(normalize_text)
    movie_data["overview_norm"] = movie_data["overview_raw"].map(
        normalize_text
    )
    movie_data["year"] = pd.to_numeric(
        movie_data["release_raw"].map(parse_year),
        errors="coerce",
    )
    movie_data["duration_min"] = pd.to_numeric(
        movie_data["duration_raw"].map(parse_duration),
        errors="coerce",
    )

    movie_data_duplicates = movie_data[
        movie_data.duplicated(subset=["source_movie_id"], keep=False)
    ].copy()
    movie_data = (
        movie_data
        .sort_values(
            ["source_movie_id", "title_norm", "year", "movie_data_row_id"],
            na_position="last",
        )
        .drop_duplicates(subset=["source_movie_id"], keep="first")
        .reset_index(drop=True)
    )

    ratings = pd.DataFrame({
        "source_user_id": to_identifier(
            ratings_raw[SCHEMA["ratings"]["user_id"]]
        ),
        "source_movie_id": to_identifier(
            ratings_raw[SCHEMA["ratings"]["movie_id"]]
        ),
        "rating_raw": ratings_raw[SCHEMA["ratings"]["rating"]],
        "rating": pd.to_numeric(
            ratings_raw[SCHEMA["ratings"]["rating"]],
            errors="coerce",
        ),
    })

    if SCHEMA["ratings"]["timestamp"] is not None:
        ratings["timestamp"] = pd.to_datetime(
            ratings_raw[SCHEMA["ratings"]["timestamp"]],
            errors="coerce",
        )
    else:
        ratings["timestamp"] = pd.NaT

    log.add(
        "02_normalization",
        "Normalisation Unicode, casse, espaces, années et durées",
        before=len(movies_raw),
        after=len(movies),
        details=(
            f"movie_data dupliqué sur movie_id: "
            f"{len(movie_data_duplicates)} lignes"
        ),
    )

    return movie_data, movie_data_duplicates, movies, ratings


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Deduplication of `movies.csv`

    Candidate pairs are generated only inside description groups or title
    groups. Missing values never count as equal. A component is merged only
    when every pair inside it is supported by a valid rule (a coherent
    clique). Ambiguous components are kept unmerged and displayed later.
    """)
    return


@app.cell
def _(CONFIG, combinations, log, movies, np, pd):
    def add_edge(
        edges,
        first,
        second,
        rule,
        description_norm=None,
        title_norm=None,
        year_diff=None,
        duration_diff=None,
    ):
        key = (
            (first, second)
            if first < second
            else (second, first)
        )
        record = edges.get(key)
        if record is None:
            record = {
                "row_a": key[0],
                "row_b": key[1],
                "rules": set(),
                "description_norms": set(),
                "title_norms": set(),
                "year_diffs": set(),
                "duration_diffs": set(),
            }
            edges[key] = record

        record["rules"].add(rule)
        if description_norm is not None:
            record["description_norms"].add(description_norm)
        if title_norm is not None:
            record["title_norms"].add(title_norm)
        if year_diff is not None:
            record["year_diffs"].add(int(year_diff))
        if duration_diff is not None:
            record["duration_diffs"].add(float(duration_diff))

    edges = {}

    description_source = movies.loc[
        movies["description_norm"].notna(),
        ["_row_id", "description_norm"],
    ]
    for description, group in description_source.groupby(
        "description_norm",
        sort=False,
    ):
        identifiers = group["_row_id"].tolist()
        if len(identifiers) < 2:
            continue
        for first, second in combinations(identifiers, 2):
            add_edge(
                edges,
                first,
                second,
                "description",
                description_norm=description,
            )

    title_source = movies.loc[
        movies["title_norm"].notna()
        & movies["year"].notna()
        & movies["duration_min"].notna(),
        ["_row_id", "title_norm", "year", "duration_min"],
    ]
    for title, group in title_source.groupby("title_norm", sort=False):
        if len(group) < 2:
            continue
        ordered = group.sort_values(
            ["year", "duration_min", "_row_id"]
        )
        identifiers = ordered["_row_id"].to_numpy()
        years = ordered["year"].to_numpy()
        durations = ordered["duration_min"].to_numpy()

        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                year_diff = years[right] - years[left]
                if year_diff >= CONFIG["year_tolerance_years"]:
                    break
                duration_diff = abs(durations[right] - durations[left])
                if duration_diff < CONFIG["duration_tolerance_minutes"]:
                    add_edge(
                        edges,
                        int(identifiers[left]),
                        int(identifiers[right]),
                        "title_year_duration",
                        title_norm=title,
                        year_diff=year_diff,
                        duration_diff=duration_diff,
                    )

    edges_df = pd.DataFrame(
        [
            {
                "row_a": record["row_a"],
                "row_b": record["row_b"],
                "rules": "|".join(sorted(record["rules"])),
                "description_norms": "|".join(
                    sorted(record["description_norms"])
                ),
                "title_norms": "|".join(sorted(record["title_norms"])),
                "year_diffs": "|".join(
                    sorted(map(str, record["year_diffs"]))
                ),
                "duration_diffs": "|".join(
                    sorted(map(str, record["duration_diffs"]))
                ),
            }
            for record in edges.values()
        ],
        columns=[
            "row_a",
            "row_b",
            "rules",
            "description_norms",
            "title_norms",
            "year_diffs",
            "duration_diffs",
        ],
    )

    parent = np.arange(len(movies))

    def find(node):
        current = node
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    def union(first, second):
        root_first = find(first)
        root_second = find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    for _edge_record in edges.values():
        union(_edge_record["row_a"], _edge_record["row_b"])

    components_df = pd.DataFrame({
        "_row_id": np.arange(len(movies)),
        "component_root": [find(index) for index in range(len(movies))],
    })

    log.add(
        "03_dedup_candidates",
        "Descriptions identiques ou titre+année+durée compatibles",
        after=len(edges_df),
        details=f"{len(edges_df)} paires candidates",
    )

    return components_df, edges_df


@app.cell
def _(components_df, edges_df, log, movies, np, pd):
    edge_map = {}
    for edge_record in edges_df.itertuples(index=False):
        edge_map[(edge_record.row_a, edge_record.row_b)] = edge_record.rules

    movie_lookup = movies.set_index("_row_id", drop=False)
    completeness_fields = [
        "title_raw",
        "description_raw",
        "year",
        "duration_min",
    ]

    def completeness(row):
        return sum(
            pd.notna(row[field]) and str(row[field]).strip() != ""
            for field in completeness_fields
        )

    def choose_representative(members):
        order = sorted(
            members,
            key=lambda member: (
                -completeness(movie_lookup.loc[member]),
                str(movie_lookup.loc[member, "source_raw_id"]),
                int(member),
            ),
        )
        return int(order[0])

    def first_non_null(members, field):
        for member in members:
            value = movie_lookup.loc[member, field]
            if pd.notna(value) and str(value).strip() != "":
                return value
        return pd.NA

    def joined_values(members, field):
        values = {
            str(movie_lookup.loc[member, field]).strip()
            for member in members
            if pd.notna(movie_lookup.loc[member, field])
            and str(movie_lookup.loc[member, field]).strip() != ""
        }
        return sorted(values)

    component_groups = list(
        components_df.groupby("component_root")["_row_id"].apply(list)
    )

    mapping_rows = []
    group_rows = []
    ambiguous_rows = []
    canonical_rows = []

    for group_number, members in enumerate(component_groups, start=1):
        members = sorted(int(member) for member in members)
        size = len(members)
        group_id = f"group_{group_number:06d}"

        pair_count = size * (size - 1) // 2
        present_pairs = 0
        for first_index in range(size):
            for second_index in range(first_index + 1, size):
                pair = (
                    members[first_index],
                    members[second_index],
                )
                if pair in edge_map:
                    present_pairs += 1

        is_clique = pair_count == present_pairs
        representative = choose_representative(members)
        representative_row = movie_lookup.loc[representative]

        if size == 1:
            dedup_status = "singleton"
        elif is_clique:
            dedup_status = "merged"
        else:
            dedup_status = "ambiguous"

        component_rules = sorted({
            rule
            for pair, rules in edge_map.items()
            if pair[0] in members and pair[1] in members
            for rule in rules.split("|")
        })

        group_rows.append({
            "dedup_group_id": group_id,
            "group_size": size,
            "status": dedup_status,
            "representative_id": (
                representative_row["source_id"]
                if dedup_status != "ambiguous"
                else pd.NA
            ),
            "suggested_representative_id": representative_row["source_id"],
            "year": representative_row["year"],
            "merges": size - 1 if dedup_status == "merged" else 0,
            "rules": "|".join(component_rules),
        })

        if dedup_status == "ambiguous":
            ambiguous_group_id = f"ambiguous_{group_number:06d}"

            for member in members:
                member_row = movie_lookup.loc[member]
                partner_ids = []
                partner_titles = []
                member_rules = set()

                for other in members:
                    if other == member:
                        continue
                    pair = (
                        (member, other)
                        if member < other
                        else (other, member)
                    )
                    rules = edge_map.get(pair)
                    if rules is None:
                        continue
                    member_rules.update(rules.split("|"))
                    other_row = movie_lookup.loc[other]
                    partner_ids.append(other_row["source_id"])
                    partner_titles.append(other_row["title_raw"])

                ambiguous_rows.append({
                    "ambiguous_group_id": ambiguous_group_id,
                    "dedup_group_id": group_id,
                    "movie_id": member_row["source_id"],
                    "title": member_row["title_raw"],
                    "year": member_row["year"],
                    "duration": member_row["duration_min"],
                    "description": member_row["description_raw"],
                    "candidate_ids": sorted(partner_ids),
                    "candidate_titles": sorted(partner_titles),
                    "applicable_rules": sorted(member_rules),
                    "conflict_reason": (
                        "composante non clique: fusion transitive "
                        "potentielle"
                    ),
                    "suggested_representative": (
                        representative_row["source_id"]
                    ),
                })

            group_canonical_id = None
        else:
            ambiguous_group_id = pd.NA
            group_canonical_id = representative_row["source_id"]

        if dedup_status == "ambiguous":
            for member in members:
                member_row = movie_lookup.loc[member]
                canonical_rows.append({
                    "canonical_movie_id": member_row["source_id"],
                    "source_raw_id": member_row["source_raw_id"],
                    "title": member_row["title_raw"],
                    "title_norm": member_row["title_norm"],
                    "year": member_row["year"],
                    "description": member_row["description_raw"],
                    "description_norm": member_row["description_norm"],
                    "duration_min": member_row["duration_min"],
                    "source_ids": [member_row["source_id"]],
                    "variant_titles": [member_row["title_raw"]],
                    "variant_descriptions": [
                        member_row["description_raw"]
                    ],
                    "n_variants": 1,
                    "merge_status": "ambiguous_member",
                    "ambiguous_group_id": ambiguous_group_id,
                })

                mapping_rows.append({
                    "_row_id": member,
                    "source_raw_id": member_row["source_raw_id"],
                    "source_id": member_row["source_id"],
                    "canonical_movie_id": member_row["source_id"],
                    "is_representative": False,
                    "merge_status": dedup_status,
                    "dedup_group_id": group_id,
                    "ambiguous_group_id": ambiguous_group_id,
                    "merge_rule": "|".join(component_rules),
                })
        else:
            canonical_rows.append({
                "canonical_movie_id": group_canonical_id,
                "source_raw_id": representative_row["source_raw_id"],
                "title": first_non_null(members, "title_raw"),
                "title_norm": representative_row["title_norm"],
                "year": first_non_null(members, "year"),
                "description": first_non_null(
                    members,
                    "description_raw",
                ),
                "description_norm": representative_row[
                    "description_norm"
                ],
                "duration_min": first_non_null(
                    members,
                    "duration_min",
                ),
                "source_ids": sorted(
                    movie_lookup.loc[member, "source_id"]
                    for member in members
                ),
                "variant_titles": joined_values(members, "title_raw"),
                "variant_descriptions": joined_values(
                    members,
                    "description_raw",
                ),
                "n_variants": size,
                "merge_status": dedup_status,
                "ambiguous_group_id": pd.NA,
            })

            for member in members:
                member_row = movie_lookup.loc[member]
                mapping_rows.append({
                    "_row_id": member,
                    "source_raw_id": member_row["source_raw_id"],
                    "source_id": member_row["source_id"],
                    "canonical_movie_id": group_canonical_id,
                    "is_representative": member == representative,
                    "merge_status": dedup_status,
                    "dedup_group_id": group_id,
                    "ambiguous_group_id": pd.NA,
                    "merge_rule": "|".join(component_rules),
                })

    movie_id_mapping = pd.DataFrame(mapping_rows)
    catalog_meta = pd.DataFrame(canonical_rows)
    dedup_groups = pd.DataFrame(group_rows)
    ambiguous_cases = pd.DataFrame(
        ambiguous_rows,
        columns=[
            "ambiguous_group_id",
            "dedup_group_id",
            "movie_id",
            "title",
            "year",
            "duration",
            "description",
            "candidate_ids",
            "candidate_titles",
            "applicable_rules",
            "conflict_reason",
            "suggested_representative",
        ],
    )

    merged_members = int(
        (movie_id_mapping["merge_status"] == "merged").sum()
    )
    log.add(
        "04_dedup_resolution",
        "Fusion uniquement des composantes cliques cohérentes",
        before=len(movies),
        after=len(catalog_meta),
        removed=merged_members - int(
            (dedup_groups["status"] == "merged").sum()
        ),
        details=(
            f"{int((dedup_groups['status'] == 'merged').sum())} "
            f"groupes fusionnés; "
            f"{int((dedup_groups['status'] == 'ambiguous').sum())} "
            f"groupes ambigus non fusionnés"
        ),
    )

    return (
        ambiguous_cases,
        catalog_meta,
        dedup_groups,
        movie_id_mapping,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Ambiguous deduplication groups

    Ambiguous groups are never merged automatically. Every member remains a
    separate canonical film and can be inspected below.
    """)
    return


@app.cell
def _(ambiguous_cases, mo):
    if ambiguous_cases.empty:
        ambiguous_view = mo.md(
            "Aucun groupe de déduplication ambigu détecté."
        )
    else:
        display_frame = ambiguous_cases.copy()
        for display_column in ["candidate_ids", "candidate_titles", "applicable_rules"]:
            display_frame[display_column] = display_frame[display_column].map(
                lambda value: ", ".join(map(str, value))
                if isinstance(value, (list, tuple))
                else value
            )
        ambiguous_view = mo.vstack([
            mo.md(
                f"**{len(ambiguous_cases)} films** appartiennent à "
                "des groupes ambigus. Aucun n'a été fusionné."
            ),
            mo.ui.table(display_frame, pagination=True),
        ])

    ambiguous_view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Studios and genres

    Studios and genres are attached through the source→canonical mapping.
    Values are aggregated as sorted distinct lists. A missing relationship is
    stored as `None`; an existing relationship with no usable value is stored
    as an empty list.
    """)
    return


@app.cell
def _(
    SCHEMA,
    catalog_meta,
    genres_raw,
    log,
    movie_id_mapping,
    pd,
    studios_raw,
):
    mapping_lookup = (
        movie_id_mapping[["source_raw_id", "canonical_movie_id"]]
        .drop_duplicates(subset=["source_raw_id"])
    )

    def attach_list(raw_frame, id_column, value_column, output_column):
        work = pd.DataFrame({
            "source_raw_id": (
                raw_frame[id_column]
                .astype("string")
                .str.strip()
                .replace({"": pd.NA})
            ),
            "value": raw_frame[value_column],
        })
        work = work.dropna(subset=["source_raw_id"])
        work = work.merge(
            mapping_lookup,
            on="source_raw_id",
            how="left",
            validate="many_to_one",
        )

        unmatched = int(work["canonical_movie_id"].isna().sum())
        work = work.dropna(subset=["canonical_movie_id"])

        aggregated = (
            work
            .groupby("canonical_movie_id")["value"]
            .apply(
                lambda values: sorted({
                    str(value).strip()
                    for value in values
                    if pd.notna(value) and str(value).strip() != ""
                })
            )
            .rename(output_column)
        )
        return aggregated, unmatched

    studios_aggregated, studios_unmatched = attach_list(
        studios_raw,
        SCHEMA["studios"]["movie_id"],
        SCHEMA["studios"]["studio"],
        "studios",
    )
    genres_aggregated, genres_unmatched = attach_list(
        genres_raw,
        SCHEMA["genres"]["movie_id"],
        SCHEMA["genres"]["genre"],
        "genres",
    )

    catalog_context = catalog_meta.merge(
        studios_aggregated,
        left_on="canonical_movie_id",
        right_index=True,
        how="left",
    ).merge(
        genres_aggregated,
        left_on="canonical_movie_id",
        right_index=True,
        how="left",
    )

    log.add(
        "05_context_lists",
        "Rattachement des studios et genres aux films canoniques",
        after=len(catalog_context),
        details=(
            f"studios non rattachés: {studios_unmatched}; "
            f"genres non rattachés: {genres_unmatched}"
        ),
    )

    return catalog_context, genres_unmatched, studios_unmatched


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Matching `movie_data.csv` to canonical films

    The description rule is applied first. The title rule is applied only when
    no description candidate exists. Ambiguous matches are preserved for
    inspection and excluded from the final matrix.
    """)
    return


@app.cell
def _(CONFIG, catalog_context, log, movie_data, movie_data_duplicates, pd):
    metadata = {}
    descriptor_map = {}
    title_map = {}

    for catalog_row in catalog_context.itertuples(index=False):
        metadata[catalog_row.canonical_movie_id] = {
            "year": catalog_row.year,
            "duration_min": catalog_row.duration_min,
            "title": catalog_row.title,
        }
        if pd.notna(catalog_row.description_norm):
            descriptor_map.setdefault(
                catalog_row.description_norm,
                [],
            ).append(catalog_row.canonical_movie_id)
        if pd.notna(catalog_row.title_norm):
            title_map.setdefault(
                catalog_row.title_norm,
                [],
            ).append(catalog_row.canonical_movie_id)

    match_rows = []

    for movie_row in movie_data.itertuples(index=False):
        overview_key = (
            None
            if pd.isna(movie_row.overview_norm)
            else str(movie_row.overview_norm)
        )
        title_key = (
            None
            if pd.isna(movie_row.title_norm)
            else str(movie_row.title_norm)
        )
        year = movie_row.year
        duration = movie_row.duration_min

        match_status = "no_match"
        method = None
        certainty = None
        canonical_match = pd.NA
        candidates = []
        conflict = None

        description_candidates = descriptor_map.get(overview_key, [])
        if description_candidates:
            candidates = list(description_candidates)
            if len(description_candidates) != 1:
                match_status = "ambiguous"
                method = "overview"
                conflict = (
                    f"{len(description_candidates)} descriptions "
                    "identiques"
                )
            else:
                candidate = description_candidates[0]
                candidate_year = metadata[candidate]["year"]
                if (
                    pd.notna(year)
                    and pd.notna(candidate_year)
                    and abs(year - candidate_year)
                    >= CONFIG["year_tolerance_years"]
                ):
                    match_status = "ambiguous"
                    method = "overview"
                    conflict = "contradiction sur l'année"
                else:
                    match_status = "accepted"
                    method = "overview"
                    certainty = "high"
                    canonical_match = candidate
        else:
            title_candidates = title_map.get(title_key, [])
            candidates = list(title_candidates)
            if not title_candidates:
                match_status = "no_match"
            elif len(title_candidates) != 1:
                match_status = "ambiguous"
                method = "title"
                conflict = (
                    f"{len(title_candidates)} titres identiques"
                )
            else:
                candidate = title_candidates[0]
                candidate_year = metadata[candidate]["year"]
                candidate_duration = metadata[candidate]["duration_min"]

                if pd.isna(year) or pd.isna(candidate_year):
                    match_status = "ambiguous"
                    method = "title"
                    conflict = "année manquante"
                elif abs(year - candidate_year) >= CONFIG[
                    "year_tolerance_years"
                ]:
                    match_status = "ambiguous"
                    method = "title"
                    conflict = "contradiction sur l'année"
                elif (
                    pd.notna(duration)
                    and pd.notna(candidate_duration)
                    and abs(duration - candidate_duration)
                    >= CONFIG["duration_tolerance_minutes"]
                ):
                    match_status = "ambiguous"
                    method = "title"
                    conflict = "contradiction sur la durée"
                elif (
                    pd.notna(duration)
                    and pd.notna(candidate_duration)
                ):
                    match_status = "accepted"
                    method = "title_year_duration"
                    certainty = "medium"
                    canonical_match = candidate
                else:
                    match_status = "accepted"
                    method = "title_year"
                    certainty = "low"
                    canonical_match = candidate

        match_rows.append({
            "movie_data_row_id": movie_row.movie_data_row_id,
            "source_movie_id": movie_row.source_movie_id,
            "movie_title": movie_row.title_raw,
            "movie_year": year,
            "canonical_movie_id": canonical_match,
            "method": method,
            "status": match_status,
            "certainty": certainty,
            "candidates": candidates,
            "conflict": conflict,
        })

    movie_data_mapping = pd.DataFrame(match_rows)
    excluded_movies = movie_data_mapping.loc[
        movie_data_mapping["status"] != "accepted"
    ].copy()
    excluded_movies["exclusion_reason"] = excluded_movies.apply(
        lambda row: row["conflict"]
        if pd.notna(row["conflict"])
        else "aucune correspondance",
        axis=1,
    )

    log.add(
        "06_movie_matching",
        "Rapprochement overview puis titre+année+durée",
        before=len(movie_data),
        after=int((movie_data_mapping["status"] == "accepted").sum()),
        removed=len(excluded_movies),
        details=(
            f"movie_data dupliqués avant matching: "
            f"{len(movie_data_duplicates)}; "
            f"acceptés: {int((movie_data_mapping['status'] == 'accepted').sum())}; "
            f"ambigus: {int((movie_data_mapping['status'] == 'ambiguous').sum())}; "
            f"sans correspondance: {int((movie_data_mapping['status'] == 'no_match').sum())}"
        ),
    )

    return excluded_movies, movie_data_mapping


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Cleaning and attaching ratings

    Ratings are joined to accepted film matches only. Missing, non-numeric and
    out-of-scale ratings are excluded. Multiple ratings for the same
    user–canonical film pair are resolved with the latest reliable timestamp
    when available, otherwise by the mean of distinct ratings.
    """)
    return


@app.cell
def _(CONFIG, log, movie_data_mapping, pd, ratings):
    def excluded_frame(source, reason):
        result = source.copy()
        result["exclusion_reason"] = reason
        return result

    initial_ratings = len(ratings)
    work = ratings.copy()

    missing_user = work["source_user_id"].isna()
    missing_movie = work["source_movie_id"].isna()
    non_numeric = work["rating"].isna() & work["rating_raw"].notna()
    out_of_scale = (
        work["rating"].notna()
        & (
            (work["rating"] < CONFIG["rating_min"])
            | (work["rating"] > CONFIG["rating_max"])
        )
    )

    reason = pd.Series(pd.NA, index=work.index, dtype="string")
    reason[missing_user] = "utilisateur manquant"
    reason[missing_movie] = "film manquant"
    reason[non_numeric] = "note non numérique"
    reason[out_of_scale] = (
        f"note hors échelle [{CONFIG['rating_min']}, "
        f"{CONFIG['rating_max']}]"
    )

    invalid_mask = reason.notna()
    excluded_parts = [excluded_frame(work[invalid_mask], reason[invalid_mask])]
    work = work.loc[~invalid_mask].copy()

    accepted_mapping = movie_data_mapping.loc[
        movie_data_mapping["status"] == "accepted",
        ["source_movie_id", "canonical_movie_id"],
    ].drop_duplicates(subset=["source_movie_id"])

    work = work.merge(
        accepted_mapping,
        on="source_movie_id",
        how="left",
        validate="many_to_one",
    )

    unmatched_mask = work["canonical_movie_id"].isna()
    excluded_parts.append(
        excluded_frame(
            work[unmatched_mask],
            "film sans correspondance canonique acceptée",
        )
    )
    work = work.loc[~unmatched_mask].copy()

    exact_before = len(work)
    dedup_columns = [
        "source_user_id",
        "canonical_movie_id",
        "rating",
    ]
    if work["timestamp"].notna().any():
        dedup_columns.append("timestamp")
    work = work.drop_duplicates(subset=dedup_columns)
    exact_removed = exact_before - len(work)

    timestamp_coverage = float(work["timestamp"].notna().mean())
    use_timestamp = (
        timestamp_coverage >= CONFIG["timestamp_coverage_threshold"]
    )

    group_columns = ["source_user_id", "canonical_movie_id"]
    grouped_counts = (
        work.groupby(group_columns)
        .size()
        .rename("n_source_ratings")
        .reset_index()
    )

    if use_timestamp:
        work = (
            work.sort_values(["timestamp", "source_movie_id"])
            .drop_duplicates(subset=group_columns, keep="last")
            .merge(
                grouped_counts,
                on=group_columns,
                how="left",
                validate="one_to_one",
            )
        )
        aggregation = "timestamp_le_plus_recent"
    else:
        work = (
            work.drop_duplicates(subset=group_columns + ["rating"])
            .groupby(group_columns, as_index=False)
            .agg(
                rating=("rating", "mean"),
                n_source_ratings=("rating", "size"),
                source_movie_id=("source_movie_id", "first"),
                timestamp=("timestamp", "max"),
            )
        )
        aggregation = "moyenne_des_notes_distinctes"

    work["user_id"] = (
        "samlearner:" + work["source_user_id"].astype("string")
    )

    ratings_clean = work[
        [
            "user_id",
            "source_user_id",
            "source_movie_id",
            "canonical_movie_id",
            "rating",
            "n_source_ratings",
            "timestamp",
        ]
    ].sort_values(["user_id", "canonical_movie_id"]).reset_index(
        drop=True
    )

    excluded_ratings = pd.concat(
        excluded_parts,
        ignore_index=True,
    )

    observed_ratings = ratings["rating"].dropna()
    ratings_cleaning_report = {
        "initial_rows": initial_ratings,
        "observed_rating_min": (
            float(observed_ratings.min())
            if len(observed_ratings)
            else None
        ),
        "observed_rating_max": (
            float(observed_ratings.max())
            if len(observed_ratings)
            else None
        ),
        "observed_rating_nunique": int(observed_ratings.nunique()),
        "invalid_rows": int(invalid_mask.sum()),
        "unmatched_rows": int(unmatched_mask.sum()),
        "exact_duplicates_removed": int(exact_removed),
        "timestamp_coverage": timestamp_coverage,
        "aggregation": aggregation,
        "final_rows": len(ratings_clean),
        "final_users": int(ratings_clean["user_id"].nunique()),
        "final_movies": int(
            ratings_clean["canonical_movie_id"].nunique()
        ),
    }

    log.add(
        "07_ratings_cleaning",
        "Exclusion des notes invalides, jointure canonique et agrégation",
        before=initial_ratings,
        after=len(ratings_clean),
        removed=initial_ratings - len(ratings_clean),
        details=(
            f"agrégation: {aggregation}; "
            f"couverture timestamp: {timestamp_coverage:.2%}"
        ),
    )

    return excluded_ratings, ratings_clean, ratings_cleaning_report


@app.cell
def _(pd, ratings_cleaning_report):
    pd.DataFrame([ratings_cleaning_report])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Interactive activity thresholds

    The thresholds are applied iteratively: films below the film threshold are
    removed, then users below the user threshold are removed, and the process
    repeats until no further row is removed.
    """)
    return


@app.cell
def _(CONFIG, mo):
    min_ratings_per_movie = mo.ui.number(
        value=CONFIG["min_ratings_per_movie"],
        start=1,
        step=1,
        label="Nombre minimal de notes par film",
    )
    min_ratings_per_user = mo.ui.number(
        value=CONFIG["min_ratings_per_user"],
        start=1,
        step=1,
        label="Nombre minimal de notes par utilisateur",
    )

    mo.hstack(
        [min_ratings_per_movie, min_ratings_per_user],
        justify="start",
        gap=2,
    )
    return min_ratings_per_movie, min_ratings_per_user


@app.cell
def _(
    CONFIG,
    log,
    min_ratings_per_movie,
    min_ratings_per_user,
    pd,
    ratings_clean,
):
    _min_movie = int(min_ratings_per_movie.value)
    _min_user = int(min_ratings_per_user.value)

    current = ratings_clean.copy()
    history_rows = []
    stable = False
    empty_result = current.empty

    for iteration in range(1, CONFIG["max_filter_iterations"] + 1):
        rows_before = len(current)
        users_before = int(current["user_id"].nunique())
        movies_before = int(current["canonical_movie_id"].nunique())

        if current.empty:
            empty_result = True
            history_rows.append({
                "iteration": iteration,
                "rows": 0,
                "users": 0,
                "movies": 0,
                "removed_rows": 0,
                "removed_users": 0,
                "removed_movies": 0,
            })
            break

        _movie_counts = (
            current.groupby("canonical_movie_id")
            .size()
        )
        kept_movies = _movie_counts[
            _movie_counts >= _min_movie
        ].index
        removed_movies = int(
            len(_movie_counts) - len(kept_movies)
        )
        after_movies = current[
            current["canonical_movie_id"].isin(kept_movies)
        ]

        _user_counts = after_movies.groupby("user_id").size()
        kept_users = _user_counts[
            _user_counts >= _min_user
        ].index
        removed_users = int(
            len(_user_counts) - len(kept_users)
        )
        after_users = after_movies[
            after_movies["user_id"].isin(kept_users)
        ]

        history_rows.append({
            "iteration": iteration,
            "rows": len(after_users),
            "users": int(after_users["user_id"].nunique()),
            "movies": int(
                after_users["canonical_movie_id"].nunique()
            ),
            "removed_rows": rows_before - len(after_users),
            "removed_users": removed_users,
            "removed_movies": removed_movies,
            "users_before": users_before,
            "movies_before": movies_before,
        })

        current = after_users
        empty_result = current.empty

        if removed_movies == 0 and removed_users == 0:
            stable = True
            break

        if current.empty:
            break

    ratings_filtered = current.copy()
    filter_history = pd.DataFrame(history_rows)

    if empty_result:
        filter_summary = (
            f"Aucune note conservée avec les seuils "
            f"{_min_movie} (film) et {_min_user} (utilisateur)."
        )
    else:
        filter_summary = (
            f"{len(ratings_filtered):,} notes conservées, "
            f"{ratings_filtered['user_id'].nunique():,} utilisateurs, "
            f"{ratings_filtered['canonical_movie_id'].nunique():,} films; "
            f"stabilisation: {stable}, "
            f"itérations: {len(filter_history)}."
        )

    log.add(
        "08_activity_filtering",
        "Filtrage itératif films puis utilisateurs jusqu'à stabilisation",
        before=len(ratings_clean),
        after=len(ratings_filtered),
        removed=len(ratings_clean) - len(ratings_filtered),
        details=filter_summary,
    )

    return filter_history, filter_summary, ratings_filtered


@app.cell
def _(filter_history, filter_summary, mo):
    mo.vstack([
        mo.md(filter_summary),
        mo.ui.table(filter_history, pagination=True),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 11. Diagnostic visualizations

    Activity plots compare the cleaned ratings before and after threshold
    filtering. Films with an unknown release year are signalled separately.
    """)
    return


@app.cell
def _(catalog_context, ratings_clean, ratings_filtered, pd):
    all_film_counts = (
        ratings_clean.groupby("canonical_movie_id")
        .size()
        .rename("count_all")
    )
    filtered_film_counts = (
        ratings_filtered.groupby("canonical_movie_id")
        .size()
        .rename("count_filtered")
    )
    all_user_counts = (
        ratings_clean.groupby("user_id")
        .size()
        .rename("count_all")
    )
    filtered_user_counts = (
        ratings_filtered.groupby("user_id")
        .size()
        .rename("count_filtered")
    )

    film_activity = (
        catalog_context[["canonical_movie_id", "year"]]
        .drop_duplicates(subset=["canonical_movie_id"])
        .set_index("canonical_movie_id")
        .join(all_film_counts, how="left")
        .join(filtered_film_counts, how="left")
        .fillna({"count_all": 0, "count_filtered": 0})
        .reset_index()
    )
    film_activity["year_sort"] = film_activity["year"].fillna(9999)
    film_activity = film_activity.sort_values(
        ["year_sort", "canonical_movie_id"]
    ).reset_index(drop=True)

    user_activity = (
        pd.concat(
            [all_user_counts, filtered_user_counts],
            axis=1,
        )
        .fillna(0)
        .sort_values("count_all", ascending=False)
        .reset_index()
    )

    return film_activity, user_activity


@app.cell
def _(catalog_context, movies, plt):
    before_years = movies["year"].dropna()
    after_years = catalog_context["year"].dropna()
    unknown_before = int(movies["year"].isna().sum())
    unknown_after = int(catalog_context["year"].isna().sum())

    _figure, _axis = plt.subplots(figsize=(12, 6))
    _bins = range(1880, 2035, 5)
    _axis.hist(
        [before_years, after_years],
        bins=_bins,
        label=[
            f"Avant déduplication ({unknown_before:,} années inconnues)",
            f"Après déduplication ({unknown_after:,} années inconnues)",
        ],
        alpha=0.6,
    )
    _axis.set_title("Nombre de films par année de sortie")
    _axis.set_xlabel("Année de sortie")
    _axis.set_ylabel("Nombre de films")
    _axis.grid(True, alpha=0.3)
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(dedup_groups, plt):
    known_groups = dedup_groups.dropna(subset=["year"])
    average_merges = (
        known_groups.groupby("year")["merges"]
        .mean()
        .sort_index()
    )

    _figure, _axis = plt.subplots(figsize=(12, 6))
    _axis.plot(
        average_merges.index,
        average_merges.values,
        marker="o",
        linewidth=1.5,
    )
    _axis.set_title(
        "Nombre moyen d'enregistrements supplémentaires fusionnés "
        "par film canonique et par année"
    )
    _axis.set_xlabel("Année de sortie du film canonique")
    _axis.set_ylabel("Fusions moyennes par film (g - 1)")
    _axis.grid(True, alpha=0.3)
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(film_activity, plt):
    _figure, _axis = plt.subplots(figsize=(14, 6))
    _ranks = range(len(film_activity))
    _axis.plot(
        _ranks,
        film_activity["count_all"],
        linewidth=0.8,
        label="Avant filtrage",
    )
    _axis.plot(
        _ranks,
        film_activity["count_filtered"],
        linewidth=0.8,
        label="Après filtrage",
    )
    _axis.set_yscale("log")
    _axis.set_title(
        "Nombre de notes par film, films triés par année puis identifiant"
    )
    _axis.set_xlabel("Rang du film (année croissante puis identifiant)")
    _axis.set_ylabel("Nombre de notes (échelle log)")
    _axis.grid(True, which="both", alpha=0.3)
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(plt, user_activity):
    _figure, _axis = plt.subplots(figsize=(14, 6))
    _ranks = range(len(user_activity))
    _axis.plot(
        _ranks,
        user_activity["count_all"],
        linewidth=0.9,
        label="Avant filtrage",
    )
    _axis.plot(
        _ranks,
        user_activity["count_filtered"],
        linewidth=0.9,
        label="Après filtrage",
    )
    _axis.set_yscale("log")
    _axis.set_title(
        "Nombre de notes par utilisateur, triés par activité décroissante"
    )
    _axis.set_xlabel("Rang de l'utilisateur (activité décroissante)")
    _axis.set_ylabel("Nombre de notes (échelle log)")
    _axis.grid(True, which="both", alpha=0.3)
    _axis.legend()
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(np, plt, ratings_clean, ratings_filtered):
    def _film_counts_series(frame):
        return frame.groupby("canonical_movie_id").size()

    def _user_counts_series(frame):
        return frame.groupby("user_id").size()

    film_before = _film_counts_series(ratings_clean)
    film_after = _film_counts_series(ratings_filtered)
    user_before = _user_counts_series(ratings_clean)
    user_after = _user_counts_series(ratings_filtered)

    _figure, _axes = plt.subplots(1, 2, figsize=(14, 5))

    for _axis, before, after, entity in [
        (_axes[0], film_before, film_after, "films"),
        (_axes[1], user_before, user_after, "utilisateurs"),
    ]:
        positive = before[before > 0]
        positive_after = after[after > 0]
        if positive_after.empty:
            minimum = positive.min()
            maximum = positive.max()
        else:
            minimum = min(positive.min(), positive_after.min())
            maximum = max(positive.max(), positive_after.max())
        minimum = max(minimum, 1)
        maximum = max(maximum, 2)
        if maximum <= minimum:
            maximum = minimum * 1.5
        _bins = np.logspace(
            np.log10(minimum),
            np.log10(maximum),
            50,
        )
        _axis.hist(
            positive,
            bins=_bins,
            density=True,
            alpha=0.55,
            label="Avant filtrage",
        )
        if not positive_after.empty:
            _axis.hist(
                positive_after,
                bins=_bins,
                density=True,
                alpha=0.55,
                label="Après filtrage",
            )
        _axis.set_xscale("log")
        _axis.set_yscale("log")
        _axis.set_title(f"Distribution du nombre de notes par {entity}")
        _axis.set_xlabel(f"Nombre de notes par {entity} (log)")
        _axis.set_ylabel("Densité (log)")
        _axis.grid(True, which="both", alpha=0.3)
        _axis.legend()

    _figure.tight_layout()
    _figure
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 12. CSR matrix and aligned metadata

    The matrix is built directly from the long ratings table. No
    densification is performed, and valid zero ratings remain explicitly
    stored.
    """)
    return


@app.cell
def _(
    catalog_context,
    csr_matrix,
    log,
    mo,
    np,
    pd,
    ratings_filtered,
):
    if ratings_filtered.empty:
        mo.stop(
            True,
            mo.md(
                "La matrice ne peut pas être construite : le filtrage "
                "a supprimé toutes les notes."
            ),
        )

    user_ids = (
        pd.Index(ratings_filtered["user_id"].dropna().unique())
        .sort_values()
    )
    movie_ids = (
        pd.Index(
            ratings_filtered["canonical_movie_id"].dropna().unique()
        )
        .sort_values()
    )

    user_codes = pd.Categorical(
        ratings_filtered["user_id"],
        categories=user_ids,
    ).codes
    movie_codes = pd.Categorical(
        ratings_filtered["canonical_movie_id"],
        categories=movie_ids,
    ).codes

    if (user_codes < 0).any() or (movie_codes < 0).any():
        raise ValueError(
            "Some ratings could not be mapped to matrix indices."
        )

    ratings_matrix = csr_matrix(
        (
            ratings_filtered["rating"].to_numpy(dtype=np.float32),
            (user_codes, movie_codes),
        ),
        shape=(len(user_ids), len(movie_ids)),
    )

    user_mapping = pd.DataFrame({
        "user_id": user_ids,
        "row_index": np.arange(len(user_ids), dtype=np.int64),
    })

    movie_mapping = pd.DataFrame({
        "canonical_movie_id": movie_ids,
        "col_index": np.arange(len(movie_ids), dtype=np.int64),
    })

    context_columns = [
        "canonical_movie_id",
        "source_raw_id",
        "title",
        "year",
        "description",
        "duration_min",
        "genres",
        "studios",
        "source_ids",
        "n_variants",
        "merge_status",
    ]
    available_context_columns = [
        column
        for column in context_columns
        if column in catalog_context.columns
    ]
    movie_context = (
        movie_mapping
        .merge(
            catalog_context[available_context_columns],
            on="canonical_movie_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("col_index")
        .reset_index(drop=True)
    )

    log.add(
        "09_matrix",
        "Construction CSR user × film et alignement du contexte",
        after=len(ratings_filtered),
        details=(
            f"shape={ratings_matrix.shape}; "
            f"nnz={ratings_matrix.nnz}"
        ),
    )

    return movie_context, movie_mapping, ratings_matrix, user_mapping


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Exports

    All files are written to the configured output directory. Source files are
    only read.
    """)
    return


@app.cell
def _(
    CONFIG,
    DATASETS,
    SCHEMA,
    SOURCE_PATHS,
    Path,
    ambiguous_cases,
    audit_df,
    catalog_context,
    datetime,
    dedup_groups,
    edges_df,
    excluded_movies,
    excluded_ratings,
    filter_history,
    json,
    log,
    movie_context,
    movie_data_duplicates,
    movie_data_mapping,
    movie_id_mapping,
    movie_mapping,
    pyarrow,
    ratings_clean,
    ratings_cleaning_report,
    ratings_filtered,
    ratings_matrix,
    save_npz,
    timezone,
    user_mapping,
):
    if pyarrow is None:
        raise RuntimeError(
            "pyarrow is required for Parquet exports. "
            "Install it with `pip install pyarrow`."
        )

    output_dir = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(output_dir / "ratings_matrix.npz", ratings_matrix)

    exports = {
        "user_mapping.parquet": user_mapping,
        "movie_mapping.parquet": movie_mapping,
        "movie_context.parquet": movie_context,
        "ratings_clean.parquet": ratings_filtered,
        "movie_id_mapping.parquet": movie_id_mapping,
        "movie_data_mapping.parquet": movie_data_mapping,
        "dedup_groups.parquet": dedup_groups,
        "dedup_edges.parquet": edges_df,
        "ambiguous_cases.parquet": ambiguous_cases,
        "excluded_movies.parquet": excluded_movies,
        "excluded_ratings.parquet": excluded_ratings,
        "movie_data_duplicates.parquet": movie_data_duplicates,
        "catalog_context.parquet": catalog_context,
        "filter_history.parquet": filter_history,
        "audit_sources.parquet": audit_df,
        "ratings_clean_all.parquet": ratings_clean,
    }

    for export_filename, export_frame in exports.items():
        export_frame.to_parquet(
            output_dir / export_filename,
            index=False,
        )

    processing_report = log.frame()
    processing_report.to_csv(
        output_dir / "processing_report.csv",
        index=False,
    )

    config_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in CONFIG.items()
        },
        "datasets": DATASETS,
        "source_paths": {
            key: str(value) for key, value in SOURCE_PATHS.items()
        },
        "schema": SCHEMA,
        "ratings_cleaning_report": ratings_cleaning_report,
        "matrix": {
            "shape": list(ratings_matrix.shape),
            "nnz": int(ratings_matrix.nnz),
            "dtype": str(ratings_matrix.dtype),
        },
    }

    with open(output_dir / "processing_config.json", "w") as handle:
        json.dump(
            config_payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    output_dir
    return (processing_report,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Consistency checks
    """)
    return


@app.cell
def _(
    catalog_context,
    min_ratings_per_movie,
    min_ratings_per_user,
    movie_context,
    movie_mapping,
    pd,
    ratings_filtered,
    ratings_matrix,
    user_mapping,
):
    _min_movie_check = int(min_ratings_per_movie.value)
    _min_user_check = int(min_ratings_per_user.value)

    _movie_counts_check = ratings_filtered.groupby(
        "canonical_movie_id"
    ).size()
    _user_counts_check = ratings_filtered.groupby("user_id").size()

    checks = []

    checks.append({
        "check": "unicité des identifiants canoniques",
        "passed": bool(
            movie_mapping["canonical_movie_id"].is_unique
        ),
        "value": int(movie_mapping["canonical_movie_id"].nunique()),
    })
    checks.append({
        "check": "unicité finale des couples user-film",
        "passed": bool(
            not ratings_filtered.duplicated(
                subset=["user_id", "canonical_movie_id"]
            ).any()
        ),
        "value": len(ratings_filtered),
    })
    checks.append({
        "check": "seuil minimal par film respecté",
        "passed": bool(
            _movie_counts_check.empty or (_movie_counts_check >= _min_movie_check).all()
        ),
        "value": int(_movie_counts_check.min()) if len(_movie_counts_check) else 0,
    })
    checks.append({
        "check": "seuil minimal par utilisateur respecté",
        "passed": bool(
            _user_counts_check.empty or (_user_counts_check >= _min_user_check).all()
        ),
        "value": int(_user_counts_check.min()) if len(_user_counts_check) else 0,
    })
    checks.append({
        "check": "dimensions matrice et mappings",
        "passed": bool(
            ratings_matrix.shape
            == (len(user_mapping), len(movie_mapping))
        ),
        "value": f"{ratings_matrix.shape} / "
        f"{(len(user_mapping), len(movie_mapping))}",
    })
    checks.append({
        "check": "alignement du contexte sur les colonnes",
        "passed": bool(
            movie_context["canonical_movie_id"].tolist()
            == movie_mapping["canonical_movie_id"].tolist()
        ),
        "value": len(movie_context),
    })
    checks.append({
        "check": "nnz égal au nombre de notes finales",
        "passed": bool(ratings_matrix.nnz == len(ratings_filtered)),
        "value": f"{ratings_matrix.nnz} / {len(ratings_filtered)}",
    })
    checks.append({
        "check": "contexte aligné sur le catalogue",
        "passed": bool(
            movie_context["canonical_movie_id"].isin(
                catalog_context["canonical_movie_id"]
            ).all()
        ),
        "value": int(
            movie_context["canonical_movie_id"].nunique()
        ),
    })

    checks_df = pd.DataFrame(checks)
    if not checks_df["passed"].all():
        failed = checks_df.loc[~checks_df["passed"], "check"].tolist()
        raise AssertionError(f"Contrôles échoués: {failed}")

    checks_df
    return (checks_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 15. Processing flow and final report
    """)
    return


@app.cell
def _(checks_df, log, mo):
    report = log.frame()
    steps = report.to_dict("records")

    lines = ["flowchart LR"]
    for index, step in enumerate(steps):
        node_id = f"S{index}"
        label = (
            f"{step['step']}<br/>"
            f"{step['after'] if step['after'] is not None else ''}"
        )
        lines.append(f'    {node_id}["{label}"]')
        if index > 0:
            lines.append(f"    S{index - 1} --> {node_id}")

    mo.vstack([
        mo.md(
            "Toutes les vérifications de cohérence sont passées."
            if checks_df["passed"].all()
            else "Certaines vérifications ont échoué."
        ),
        mo.mermaid("\n".join(lines)),
        mo.ui.table(report, pagination=True),
    ])
    return


@app.cell
def _(processing_report):
    processing_report
    return


if __name__ == "__main__":
    app.run()
