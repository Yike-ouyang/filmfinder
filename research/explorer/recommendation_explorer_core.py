from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEVELS = (0, 1, 2, 5, 10, 20, 50, 100)
REQUIRED_CLEANING = (
    "ratings_matrix.npz",
    "user_ids.npy",
    "movie_ids.npy",
    "movie_context.json",
)
REQUIRED_BASELINE = (
    "config.json",
    "models/M5.npz",
    "splits/cold_cohorts.csv",
    "splits/train_visible_indices.npy",
    "splits/warm_cohorts.csv",
    "splits/warm_hidden_indices.npy",
)


class ArtifactError(RuntimeError):
    pass


def load_module(path, name):
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"Module introuvable : {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ArtifactError(f"Module illisible : {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _resolve_candidate(value, bases):
    if value is None:
        return None
    value = Path(str(value)).expanduser()
    if value.is_absolute():
        return value
    for base in bases:
        candidate = Path(base) / value
        if candidate.exists():
            return candidate
    return Path(bases[0]) / value


def _first_existing(candidates):
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            return Path(candidate)
    return None


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def _as_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                import ast

                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return [text] if text else []
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [str(parsed).strip()] if str(parsed).strip() else []
        return [item.strip() for item in text.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value).strip()] if str(value).strip() else []


@dataclass
class ExplorerConfig:
    label: str
    approach: str
    method: str | None
    n_neighbors: int
    top_k: int
    pool_size: int | None
    source: str

    @property
    def uses_knn(self):
        return self.approach in ("knn_m5", "knn_popularity", "knn_pop")


def _config_from_cold_global(payload, source):
    method = payload.get("method")
    if method in ("fallback", None):
        approach = "fallback"
    else:
        approach = "knn_m5"
    return ExplorerConfig(
        label=(
            f"{method or 'fallback'}|N={int(payload.get('n_neighbors') or 0)}"
            f"|k={int(payload.get('top_k') or 0)}"
            f"|pool={payload.get('pool_size')}"
        ),
        approach=approach,
        method=method,
        n_neighbors=int(payload.get("n_neighbors") or 0),
        top_k=int(payload.get("top_k") or 0),
        pool_size=payload.get("pool_size"),
        source=source,
    )


def _config_from_warm(payload, source):
    approach = str(payload.get("approach", "knn_m5"))
    method = payload.get("similarity")
    if method is None and approach in ("knn_m5", "knn_popularity", "knn_pop"):
        method = "centered"
    if approach not in ("knn_m5", "knn_popularity", "knn_pop", "m5_direct", "popularity"):
        raise ArtifactError(f"Approche warm non supportée : {approach}")
    n_neighbors = int(payload.get("n_neighbors") or 0)
    top_k = int(payload.get("top_k") or 0)
    m = payload.get("m")
    label = approach
    if approach.startswith("knn"):
        label = f"{approach}|{method}|N={n_neighbors}|k={top_k}"
        if approach == "knn_popularity":
            label = f"{label}|m={m}"
    return ExplorerConfig(
        label=label,
        approach=approach,
        method=method,
        n_neighbors=n_neighbors,
        top_k=top_k,
        pool_size=None,
        source=source,
    )


@dataclass
class ExplorerArtifacts:
    cleaning_dir: Path
    baseline_dir: Path
    knn_dir: Path
    scripts_dir: Path
    rec_scripts_dir: Path
    model_output_dir: Path
    m5_path: Path
    warnings: list[str] = field(default_factory=list)


@dataclass
class UserRecord:
    user_row: int
    user_id: str
    scenario: str
    cohort: str
    n_pool: int | None
    n_eval: int | None
    n_visible: int | None
    n_hidden: int | None
    levels: tuple[int, ...]
    avg_visible_all: float
    n_relevant_hidden: int

    @property
    def label(self):
        if self.scenario == "cold":
            return (
                f"{self.user_id} — cold — pool {self.n_pool}, "
                f"hidden {self.n_eval}"
            )
        return (
            f"{self.user_id} — warm — visibles {self.n_visible}, "
            f"hidden {self.n_hidden}"
        )


@dataclass
class ExplorerContext:
    artifacts: ExplorerArtifacts
    fm: object
    evalm5: object
    eval8: object
    knn9: object
    baseline: dict
    context: dict
    data: dict
    users: dict[int, UserRecord]
    user_frame: pd.DataFrame
    movie_meta: pd.DataFrame
    train_visible: np.ndarray
    warm_hidden: np.ndarray
    config_cold: ExplorerConfig
    config_warm: ExplorerConfig
    warning_messages: list[str] = field(default_factory=list)

    @property
    def threshold(self):
        return float(self.context["threshold"])

    @property
    def factor(self):
        return float(self.context["factor"])

    def user_options(self):
        return {
            record.label: int(record.user_row)
            for record in sorted(
                self.users.values(),
                key=lambda record: record.label,
            )
        }

    def movie_label(self, movie_col):
        row = self.movie_meta.iloc[int(movie_col)]
        year = row.get("year")
        year_text = "" if pd.isna(year) else f" ({int(year)})"
        return f"{row.get('title', movie_col)}{year_text}"

    def movie_options(self, movie_cols):
        options = {}
        for movie in np.asarray(movie_cols, dtype=np.int64):
            label = self.movie_label(int(movie))
            if label in options:
                label = f"{label} [{int(movie)}]"
            options[label] = int(movie)
        return options


def resolve_artifacts(
    cleaning_dir=None,
    baseline_dir=None,
    knn_dir=None,
    scripts_dir=None,
):
    bases = (REPO_ROOT, Path.cwd())
    knn_candidates = []
    env_knn = _env_path("FILMFINDER_KNN_M5_DIR")
    if env_knn is not None:
        knn_candidates.append(env_knn)
    if knn_dir is not None:
        knn_candidates.append(Path(knn_dir).expanduser())
    knn_candidates.extend([
        REPO_ROOT / "research" / "results" / "knn_m5_optimization",
        Path.cwd() / "research" / "results" / "knn_m5_optimization",
    ])
    resolved_knn = _first_existing(knn_candidates)
    if resolved_knn is None:
        raise ArtifactError(
            "Dossier knn_m5_optimization introuvable. Définir "
            "FILMFINDER_KNN_M5_DIR ou relancer "
            "9.KNNM5Optimization.py."
        )
    knn_config_path = resolved_knn / "config.json"
    knn_config = _read_json(knn_config_path) if knn_config_path.exists() else {}
    knn_paths = knn_config.get("paths", {})

    baseline_candidates = []
    env_baseline = _env_path("FILMFINDER_RECOMMENDATION_DIR")
    if env_baseline is not None:
        baseline_candidates.append(env_baseline)
    if baseline_dir is not None:
        baseline_candidates.append(Path(baseline_dir).expanduser())
    config_baseline = _resolve_candidate(knn_paths.get("baseline_dir"), bases)
    if config_baseline is not None:
        baseline_candidates.append(config_baseline)
    baseline_candidates.extend([
        REPO_ROOT / "research" / "results" / "recommendation_system",
        Path.cwd() / "research" / "results" / "recommendation_system",
    ])
    resolved_baseline = _first_existing(baseline_candidates)
    if resolved_baseline is None:
        raise ArtifactError(
            "Dossier recommendation_system introuvable. Définir "
            "FILMFINDER_RECOMMENDATION_DIR ou relancer "
            "8.RecommendationSystem.py."
        )
    baseline_config_path = resolved_baseline / "config.json"
    baseline_config = (
        _read_json(baseline_config_path)
        if baseline_config_path.exists()
        else {}
    )
    baseline_paths = baseline_config.get("paths", {})

    cleaning_candidates = []
    env_cleaning = _env_path("FILMFINDER_CLEANING_DIR")
    if env_cleaning is not None:
        cleaning_candidates.append(env_cleaning)
    if cleaning_dir is not None:
        cleaning_candidates.append(Path(cleaning_dir).expanduser())
    for value in (
        knn_paths.get("cleaning_dir"),
        baseline_paths.get("cleaning_dir"),
        baseline_config.get("args", {}).get("cleaning_dir"),
    ):
        resolved = _resolve_candidate(value, bases)
        if resolved is not None:
            cleaning_candidates.append(resolved)
    cleaning_candidates.extend([REPO_ROOT, Path.cwd()])
    resolved_cleaning = _first_existing(cleaning_candidates)
    if resolved_cleaning is None:
        raise ArtifactError(
            "Dossier cleaning introuvable. Définir FILMFINDER_CLEANING_DIR."
        )

    m5_candidates = []
    env_m5 = _env_path("FILMFINDER_M5_PATH")
    if env_m5 is not None:
        m5_candidates.append(env_m5)
    m5_candidates.extend([
        resolved_baseline / "models" / "M5.npz",
        _resolve_candidate(knn_paths.get("m5_path"), bases),
    ])
    resolved_m5 = _first_existing(m5_candidates)
    if resolved_m5 is None:
        raise ArtifactError(
            "Modèle M5 introuvable. Relancer 8.RecommendationSystem.py "
            "ou définir FILMFINDER_M5_PATH."
        )

    if scripts_dir is not None:
        resolved_scripts = Path(scripts_dir).expanduser()
    else:
        resolved_scripts = _env_path("FILMFINDER_SCRIPTS_DIR")
        if resolved_scripts is None:
            resolved_scripts = REPO_ROOT / "research" / "modeling"

    model_candidates = []
    env_model = _env_path("FILMFINDER_MODEL_OUTPUT_DIR")
    if env_model is not None:
        model_candidates.append(env_model)
    model_candidates.extend([
        _resolve_candidate(knn_paths.get("model_output_dir"), bases),
        _resolve_candidate(baseline_paths.get("model_output_dir"), bases),
        resolved_cleaning / "modeling",
    ])
    resolved_model = _first_existing(model_candidates)
    if resolved_model is None:
        resolved_model = resolved_cleaning / "modeling"

    warnings = []
    missing = [
        name
        for name in REQUIRED_CLEANING
        if not (resolved_cleaning / name).exists()
    ]
    if missing:
        raise ArtifactError(
            f"Fichiers cleaning manquants dans {resolved_cleaning}: {missing}. "
            "Relancer research/preprocessing/data_cleaning.py."
        )
    missing = [
        name
        for name in REQUIRED_BASELINE
        if not (resolved_baseline / name).exists()
    ]
    if missing:
        raise ArtifactError(
            f"Fichiers baseline manquants dans {resolved_baseline}: {missing}. "
            "Relancer research/recommendation/8.RecommendationSystem.py."
        )
    selected_cold = resolved_knn / "selection" / "selected_configs.json"
    if not selected_cold.exists():
        raise ArtifactError(
            f"Config globale manquante : {selected_cold}. "
            "Relancer research/recommendation/9.KNNM5Optimization.py."
        )
    warm_selected = resolved_baseline / "warm" / "selected_configs.json"
    if not warm_selected.exists():
        warnings.append(
            "warm/selected_configs.json absent : la configuration cold "
            "globale sera utilisée pour les utilisateurs warm."
        )

    return ExplorerArtifacts(
        cleaning_dir=resolved_cleaning,
        baseline_dir=resolved_baseline,
        knn_dir=resolved_knn,
        scripts_dir=resolved_scripts,
        rec_scripts_dir=(
            Path(__file__).resolve().parent.parent / "recommendation"
        ),
        model_output_dir=resolved_model,
        m5_path=resolved_m5,
        warnings=warnings,
    )


def build_movie_meta(data):
    movie_context = data["movie_context"].copy()
    movie_mapping = data["movie_mapping"].copy()
    n_movies = len(movie_mapping)
    meta = pd.DataFrame({
        "movie_col": np.arange(n_movies, dtype=np.int64),
        "movie_id": movie_mapping["canonical_movie_id"].to_numpy(),
    })
    if "name" in movie_context.columns:
        meta["title"] = movie_context["name"].to_numpy()
    else:
        meta["title"] = meta["movie_id"]
    if "year" in movie_context.columns:
        meta["year"] = pd.to_numeric(
            movie_context["year"], errors="coerce"
        ).to_numpy()
    else:
        meta["year"] = np.nan
    genres_column = "genres" if "genres" in movie_context.columns else None
    if genres_column is not None:
        meta["genres"] = movie_context[genres_column].map(_as_list)
    else:
        meta["genres"] = [[] for _ in range(n_movies)]
    studios_column = None
    for candidate in ("studios", "studio"):
        if candidate in movie_context.columns:
            studios_column = candidate
            break
    if studios_column is not None:
        meta["studios"] = movie_context[studios_column].map(_as_list)
    else:
        meta["studios"] = [[] for _ in range(n_movies)]
    if "description" in movie_context.columns:
        meta["description"] = movie_context["description"].to_numpy()
    else:
        meta["description"] = ""
    meta["title"] = meta["title"].fillna(
        pd.Series(meta["movie_id"], index=meta.index)
    ).astype(str)
    meta["genres_text"] = meta["genres"].map(
        lambda values: ", ".join(values) if values else ""
    )
    meta["studios_text"] = meta["studios"].map(
        lambda values: ", ".join(values) if values else ""
    )
    meta["has_poster"] = any(
        "poster" in str(column).casefold() or "image" in str(column).casefold()
        for column in movie_context.columns
    )
    return meta


def _positions_for_user(indices, user_rows, user):
    indices = np.asarray(indices, dtype=np.int64)
    mask = user_rows[indices] == int(user)
    return indices[mask]


def build_user_records(context, artifacts):
    baseline = context["baseline"]
    data = context["data"]
    observations = baseline["observations"]
    movie_cols = baseline["movie_cols"]
    y_all = baseline["y_all"]
    user_rows = baseline["user_rows"]
    split = baseline["split"]
    threshold = float(context["context"]["threshold"])
    user_mapping = data["user_mapping"].set_index("row_index")["user_id"]
    cold_cohorts = baseline["cold_cohorts"]
    warm_frame = pd.read_csv(
        artifacts.baseline_dir / "splits" / "warm_cohorts.csv"
    )
    train_visible = baseline["train_visible_indices"]
    warm_hidden = np.load(
        artifacts.baseline_dir / "splits" / "warm_hidden_indices.npy"
    )
    records = {}
    rows = []
    for _, row in cold_cohorts.iterrows():
        if row["cohort"] != "cold_test":
            continue
        user = int(row["user_row"])
        pool = np.asarray(
            split["pool_order_by_user"][user], dtype=np.int64
        )
        hidden = np.asarray(
            split["eval_rows_by_user"][user], dtype=np.int64
        )
        if pool.size != int(row["n_pool"]) or hidden.size != int(row["n_eval"]):
            raise ArtifactError(
                f"Split cold incohérent pour l'utilisateur {user}."
            )
        levels = tuple(
            int(level) for level in DEFAULT_LEVELS if int(level) <= pool.size
        )
        record = UserRecord(
            user_row=user,
            user_id=str(user_mapping.get(user, user)),
            scenario="cold",
            cohort="cold_test",
            n_pool=int(pool.size),
            n_eval=int(hidden.size),
            n_visible=None,
            n_hidden=None,
            levels=levels,
            avg_visible_all=float(y_all[pool].mean()) if pool.size else np.nan,
            n_relevant_hidden=int(
                (y_all[hidden] > threshold).sum()
            ),
        )
        records[user] = record
        rows.append(record)
    for _, row in warm_frame.iterrows():
        if row["cohort"] != "warm_test":
            continue
        user = int(row["user_row"])
        visible_positions = _positions_for_user(
            train_visible, user_rows, user
        )
        hidden_positions = _positions_for_user(
            warm_hidden, user_rows, user
        )
        if (
            visible_positions.size != int(row["n_visible"])
            or hidden_positions.size != int(row["n_hidden"])
        ):
            raise ArtifactError(
                f"Split warm incohérent pour l'utilisateur {user}. "
                "Relancer 8.RecommendationSystem.py."
            )
        record = UserRecord(
            user_row=user,
            user_id=str(user_mapping.get(user, user)),
            scenario="warm",
            cohort="warm_test",
            n_pool=None,
            n_eval=None,
            n_visible=int(visible_positions.size),
            n_hidden=int(hidden_positions.size),
            levels=(0,),
            avg_visible_all=float(
                y_all[visible_positions].mean()
            ) if visible_positions.size else np.nan,
            n_relevant_hidden=int(
                (y_all[hidden_positions] > threshold).sum()
            ),
        )
        records[user] = record
        rows.append(record)
    if set(
        record.user_row for record in records.values()
        if record.scenario == "cold"
    ) & set(
        record.user_row for record in records.values()
        if record.scenario == "warm"
    ):
        raise ArtifactError("Des utilisateurs sont à la fois cold et warm.")
    frame = pd.DataFrame([
        {
            "user_row": record.user_row,
            "user_id": record.user_id,
            "scenario": record.scenario,
            "cohort": record.cohort,
            "n_pool": record.n_pool,
            "n_eval": record.n_eval,
            "n_visible": record.n_visible,
            "n_hidden": record.n_hidden,
            "avg_visible_all": record.avg_visible_all,
            "n_relevant_hidden": record.n_relevant_hidden,
        }
        for record in records.values()
    ])
    return records, frame, train_visible, warm_hidden


def load_explorer(
    cleaning_dir=None,
    baseline_dir=None,
    knn_dir=None,
    scripts_dir=None,
):
    artifacts = resolve_artifacts(
        cleaning_dir=cleaning_dir,
        baseline_dir=baseline_dir,
        knn_dir=knn_dir,
        scripts_dir=scripts_dir,
    )
    fm = load_module(
        artifacts.scripts_dir / "6.IncrementalVsRetraining.py",
        "_explorer_fm",
    )
    evalm5 = load_module(
        artifacts.scripts_dir / "7.EvaluateM5.py",
        "_explorer_evalm5",
    )
    eval8 = load_module(
        artifacts.rec_scripts_dir / "8.RecommendationSystem.py",
        "_explorer_eval8",
    )
    knn9 = load_module(
        artifacts.rec_scripts_dir / "9.KNNM5Optimization.py",
        "_explorer_knn9",
    )
    args = SimpleNamespace(
        baseline_dir=artifacts.baseline_dir,
        cleaning_dir=artifacts.cleaning_dir,
        model_output_dir=artifacts.model_output_dir,
        seed=None,
        similarity_shrink=None,
        m5_path=artifacts.m5_path,
    )
    baseline = knn9.load_baseline(eval8, fm, args)

    selected_cold_payload = _read_json(
        artifacts.knn_dir / "selection" / "selected_configs.json"
    )
    global_payload = selected_cold_payload.get("global")
    if not global_payload:
        raise ArtifactError(
            "Aucune configuration globale dans selected_configs.json."
        )
    config_cold = _config_from_cold_global(
        global_payload,
        str(artifacts.knn_dir / "selection" / "selected_configs.json"),
    )
    warning_messages = list(artifacts.warnings)
    warm_selected_path = (
        artifacts.baseline_dir / "warm" / "selected_configs.json"
    )
    if warm_selected_path.exists():
        warm_payload = _read_json(warm_selected_path).get("0|knn_m5")
        if warm_payload is None:
            warning_messages.append(
                "0|knn_m5 absent de warm/selected_configs.json."
            )
            config_warm = config_cold
        else:
            config_warm = _config_from_warm(
                warm_payload, str(warm_selected_path)
            )
    else:
        config_warm = config_cold
    topks = sorted({
        int(config.top_k)
        for config in (config_cold, config_warm)
        if int(config.top_k) > 0
    })
    if not topks:
        raise ArtifactError("Aucune valeur top_k dans les configurations.")
    context = knn9.build_context(eval8, fm, baseline, args, topks)
    baseline["item_scores_ref"] = context["item_scores"]
    data = baseline["data"]
    movie_meta = build_movie_meta(data)
    records, user_frame, train_visible, warm_hidden = build_user_records(
        {
            "baseline": baseline,
            "data": data,
            "context": context,
        },
        artifacts,
    )
    return ExplorerContext(
        artifacts=artifacts,
        fm=fm,
        evalm5=evalm5,
        eval8=eval8,
        knn9=knn9,
        baseline=baseline,
        context=context,
        data=data,
        users=records,
        user_frame=user_frame,
        movie_meta=movie_meta,
        train_visible=train_visible,
        warm_hidden=warm_hidden,
        config_cold=config_cold,
        config_warm=config_warm,
        warning_messages=warning_messages,
    )


@dataclass
class PipelineResult:
    user: UserRecord
    scenario: str
    level: int
    config: ExplorerConfig
    source: str
    visible_cols: np.ndarray
    visible_ratings: np.ndarray
    hidden_movies: np.ndarray
    hidden_ratings: np.ndarray
    neighbor_order: np.ndarray
    similarity: np.ndarray | None = None
    neighbors: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    removed_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    scored_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    top20_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    tracking_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    genre_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    missed_relevant: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)
    stage_counts: dict = field(default_factory=dict)
    item_scores: np.ndarray | None = None
    full_ranked: np.ndarray | None = None
    top20: np.ndarray | None = None
    contributions: dict = field(default_factory=dict)
    similarity_method: str | None = None


def _metadata_frame(ctx, movies):
    movies = np.asarray(movies, dtype=np.int64)
    if movies.size == 0:
        return pd.DataFrame(columns=[
            "movie_col", "movie_id", "title", "year", "genres_text",
            "studios_text", "genres",
        ])
    frame = ctx.movie_meta.set_index("movie_col").reindex(movies)
    frame = frame.reset_index()
    frame["genres_text"] = frame["genres_text"].fillna("")
    frame["title"] = frame["title"].fillna(
        pd.Series(frame["movie_id"].astype(str), index=frame.index)
    )
    return frame


def _rating_map(cols, ratings):
    return {
        int(movie): float(rating)
        for movie, rating in zip(
            np.asarray(cols, dtype=np.int64),
            np.asarray(ratings, dtype=np.float64),
        )
    }


def _popularity_scores(ctx, m):
    popularity = ctx.context["popularity"]
    key = float(m)
    if key in popularity["scores"]:
        return popularity["scores"][key]
    counts = popularity["counts"]
    means = popularity["means"]
    global_mean = float(popularity["global_mean"])
    return (
        counts / (counts + key) * means
        + key / (counts + key) * global_mean
    )


def _profile_tables(ctx, visible_cols, visible_ratings):
    frame = _metadata_frame(ctx, visible_cols)
    if frame.empty:
        return (
            frame.assign(user_rating=pd.Series(dtype=float), is_top=pd.Series(dtype=bool)),
            frame,
            pd.DataFrame(columns=["genre", "count", "mean_rating"]),
        )
    frame["user_rating"] = np.asarray(visible_ratings, dtype=np.float64)
    order = np.argsort(-frame["user_rating"].to_numpy())
    top_positions = set(order[:min(5, order.size)].tolist())
    frame["is_top"] = [
        index in top_positions for index in range(frame.shape[0])
    ]
    frame = frame.sort_values(
        "user_rating", ascending=False
    ).reset_index(drop=True)
    top_profile = frame.head(5).copy()
    genre_rows = []
    for _, row in frame.iterrows():
        for genre in row["genres"]:
            genre_rows.append((genre, float(row["user_rating"])))
    if genre_rows:
        genre_frame = pd.DataFrame(
            genre_rows, columns=["genre", "rating"]
        )
        genre_summary = (
            genre_frame.groupby("genre")["rating"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"mean": "mean_rating"})
            .sort_values(["count", "mean_rating"], ascending=False)
            .reset_index(drop=True)
        )
    else:
        genre_summary = pd.DataFrame(
            columns=["genre", "count", "mean_rating"]
        )
    return (
        frame,
        top_profile,
        genre_summary,
    )


def _similarity_for(ctx, prepared, method, user):
    if method == "centered":
        return ctx.eval8.centered_similarity(
            prepared["centered_target"],
            prepared["binary_target"],
            ctx.context["similarity_structures"]["centered"],
            ctx.context["similarity_structures"]["binary"],
            ctx.context["similarity_structures"]["norms"],
            float(ctx.context["shrink"]),
        )
    if method == "latent":
        return ctx.eval8.latent_similarity(
            prepared["latent_target"],
            ctx.context["latent_matrix"],
            ctx.context["latent_norms"],
        )
    return None


def _neighbor_contributions(ctx, neighbor_order, top_k):
    if neighbor_order is None or len(neighbor_order) == 0:
        return {}, 0, np.zeros(0, dtype=np.int64)
    cache = ctx.context["topk_cache"]
    if int(top_k) not in cache:
        raise ArtifactError(
            f"top_k={top_k} absent du cache construit par build_context."
        )
    lists = cache[int(top_k)]
    matrix = ctx.context["visible_matrix"]
    contributions = {}
    occurrences = 0
    for neighbor in np.asarray(neighbor_order, dtype=np.int64):
        cols = np.asarray(lists[int(neighbor)], dtype=np.int64)
        if cols.size == 0:
            continue
        row = matrix.getrow(int(neighbor))
        lookup = {
            int(movie): float(value)
            for movie, value in zip(row.indices, row.data)
        }
        occurrences += int(cols.size)
        for movie in cols:
            value = lookup.get(int(movie), np.nan)
            contributions.setdefault(int(movie), []).append(
                (int(neighbor), float(value))
            )
    return contributions, occurrences, np.asarray(
        sorted(contributions), dtype=np.int64
    )


def _neighbor_table(ctx, neighbor_order, similarity):
    if neighbor_order is None or len(neighbor_order) == 0:
        return pd.DataFrame(columns=[
            "neighbor_row", "neighbor_user_id", "similarity",
            "common_rated_movies", "n_ratings",
        ])
    user_rows = ctx.baseline["user_rows"]
    counts = ctx.context["similarity_structures"]["counts"]
    user_mapping = ctx.data["user_mapping"].set_index("row_index")["user_id"]
    rows = []
    for neighbor in np.asarray(neighbor_order, dtype=np.int64):
        rows.append({
            "neighbor_row": int(neighbor),
            "neighbor_user_id": str(
                user_mapping.get(int(neighbor), int(neighbor))
            ),
            "similarity": float(similarity[int(neighbor)])
            if similarity is not None
            else np.nan,
            "common_rated_movies": int(0),
            "n_ratings": int(counts[int(neighbor)]),
        })
    return pd.DataFrame(rows)


def _add_neighbor_overlap(ctx, frame, target_cols):
    if frame.empty:
        return frame
    target = set(
        int(movie) for movie in np.asarray(target_cols, dtype=np.int64)
    )
    matrix = ctx.context["visible_matrix"]
    values = []
    for neighbor in frame["neighbor_row"].to_numpy():
        row = matrix.getrow(int(neighbor))
        values.append(len(target.intersection(row.indices.tolist())))
    frame = frame.copy()
    frame["common_rated_movies"] = values
    return frame


def _popularity_scores_for_config(ctx, config):
    if config.approach in ("knn_popularity", "popularity"):
        return _popularity_scores(
            ctx, ctx.baseline["fallback_m"]
        )
    return None


def _score_array_for(ctx, config, item_scores):
    if config.approach in ("knn_popularity", "popularity"):
        return _popularity_scores_for_config(ctx, config)
    return item_scores


def _exclusion_reason(
    movie,
    catalogue_mask,
    visible_set,
    in_pool,
    in_top20,
    source,
    raw_movies,
):
    movie = int(movie)
    if movie in visible_set:
        return "déjà noté (profil visible)"
    if not bool(catalogue_mask[movie]):
        return "hors catalogue d'entraînement"
    if not in_pool:
        if source == "knn":
            return "éliminé par le plafond du pool"
        if source == "fallback":
            return "hors du pool de popularité"
        return "non retenu dans le pool"
    if not in_top20:
        return "dans le pool mais hors du top 20"
    return "recommandé dans le top 20"


def compute_pipeline(ctx, user_row, scenario=None, level=None):
    user_row = int(user_row)
    record = ctx.users.get(user_row)
    if record is None:
        raise ArtifactError(f"Utilisateur inconnu : {user_row}")
    scenario = scenario or record.scenario
    if scenario != record.scenario:
        raise ArtifactError(
            f"Scénario {scenario} incompatible avec l'utilisateur {user_row}."
        )
    if scenario == "cold":
        level = int(level if level is not None else 0)
        if level not in record.levels:
            raise ArtifactError(
                f"Niveau {level} indisponible pour l'utilisateur {user_row}."
            )
        profile = ctx.knn9.prepare_profile(
            ctx.eval8, ctx.evalm5, ctx.fm, ctx.baseline, user_row
        )
        prepared = ctx.knn9.adapt_profile(
            ctx.eval8, ctx.evalm5, ctx.baseline, profile, level
        )
        visible_cols = np.asarray(
            prepared["visible_cols"], dtype=np.int64
        )
        visible_ratings = np.asarray(
            profile["pool_ratings"][:visible_cols.size],
            dtype=np.float64,
        )
        hidden_movies = np.asarray(
            profile["hidden_movies"], dtype=np.int64
        )
        hidden_ratings = np.asarray(
            profile["hidden_ratings"], dtype=np.float64
        )
        item_scores = np.asarray(
            prepared["item_scores"], dtype=np.float64
        )
        centered_target = prepared["centered_target"]
        binary_target = prepared["binary_target"]
        latent_target = prepared["latent_target"]
        config = ctx.config_cold
        if level == 0:
            config = ExplorerConfig(
                label=f"fallback|pool={ctx.config_cold.pool_size}",
                approach="fallback",
                method=None,
                n_neighbors=0,
                top_k=0,
                pool_size=ctx.config_cold.pool_size,
                source=ctx.config_cold.source,
            )
    else:
        visible_positions = _positions_for_user(
            ctx.train_visible, ctx.baseline["user_rows"], user_row
        )
        hidden_positions = _positions_for_user(
            ctx.warm_hidden, ctx.baseline["user_rows"], user_row
        )
        movie_cols = ctx.baseline["movie_cols"]
        y_all = ctx.baseline["y_all"]
        visible_cols = np.asarray(
            movie_cols[visible_positions], dtype=np.int64
        )
        visible_ratings = np.asarray(
            y_all[visible_positions], dtype=np.float64
        )
        hidden_movies = np.asarray(
            movie_cols[hidden_positions], dtype=np.int64
        )
        hidden_ratings = np.asarray(
            y_all[hidden_positions], dtype=np.float64
        )
        visible_row = ctx.context["visible_matrix"].getrow(user_row)
        binary_target = visible_row.copy()
        binary_target.data[:] = 1.0
        centered_target = (
            ctx.context["similarity_structures"]["centered"]
            .getrow(user_row)
        )
        latent_target = ctx.eval8.normalize_vector(
            ctx.context["latent_matrix"][user_row]
        )
        item_scores = ctx.eval8.warm_item_scores(
            ctx.baseline["engine"],
            ctx.context["item_scores"],
            ctx.baseline["features"],
            user_row,
        )
        config = ctx.config_warm
        level = 0

    visible_set = set(int(movie) for movie in visible_cols)
    similarity = None
    neighbor_order = np.zeros(0, dtype=np.int64)
    contributions = {}
    raw_occurrences = 0
    raw_movies = np.zeros(0, dtype=np.int64)
    source = "fallback"
    eligible = np.zeros(0, dtype=np.int64)
    support_map = {}
    pool_rank_map = {}
    order = None

    if config.uses_knn and not (
        scenario == "cold" and int(level) == 0
    ):
        similarity = _similarity_for(
            ctx,
            {
                "centered_target": centered_target,
                "binary_target": binary_target,
                "latent_target": latent_target,
            },
            config.method,
            user_row,
        )
        neighbor_order = ctx.eval8.neighbor_order(
            similarity, user_row, int(config.n_neighbors)
        )
        contributions, raw_occurrences, raw_movies = (
            _neighbor_contributions(
                ctx, neighbor_order, int(config.top_k)
            )
        )
        union = ctx.knn9.build_knn_union(
            neighbor_order,
            int(config.n_neighbors),
            int(config.top_k),
            ctx.context["topk_cache"],
            ctx.context["catalogue_mask"],
            visible_cols,
            ctx.context["popularity"]["counts"],
        )
        union_candidates, union_support, union_order = union
        if union_candidates is None:
            source = "fallback"
            eligible = ctx.knn9.build_fallback_candidates(
                ctx.context["fallback_order"], visible_cols
            )
            order = None
        else:
            source = "knn"
            eligible = np.asarray(union_candidates, dtype=np.int64)
            order = np.asarray(union_order, dtype=np.int64)
            support_map = {
                int(movie): int(value)
                for movie, value in zip(
                    union_candidates, union_support
                )
            }
    elif config.uses_knn and scenario == "cold" and int(level) == 0:
        source = "fallback"
        eligible = ctx.knn9.build_fallback_candidates(
            ctx.context["fallback_order"], visible_cols
        )
    elif config.approach in ("m5_direct", "popularity"):
        source = "all"
        eligible = ctx.context["catalogue_cols"][
            ~np.isin(ctx.context["catalogue_cols"], visible_cols)
        ]
    else:
        source = "fallback"
        eligible = ctx.knn9.build_fallback_candidates(
            ctx.context["fallback_order"], visible_cols
        )

    if source == "knn":
        candidates = ctx.knn9.apply_pool_cap(
            eligible,
            order,
            config.pool_size,
            ctx.context["fallback_size"],
            "knn",
        )
        if order is not None:
            capped = order[:int(config.pool_size)] if config.pool_size is not None else order
            pool_rank_map = {
                int(eligible[int(position)]): rank + 1
                for rank, position in enumerate(capped)
            }
    elif source == "fallback":
        candidates = ctx.knn9.apply_pool_cap(
            eligible,
            np.arange(eligible.size),
            config.pool_size,
            ctx.context["fallback_size"],
            "fallback",
        )
        pool_rank_map = {
            int(movie): rank + 1
            for rank, movie in enumerate(candidates)
        }
    else:
        candidates = eligible
        pool_rank_map = {
            int(movie): rank + 1
            for rank, movie in enumerate(candidates)
        }
    candidates = np.asarray(candidates, dtype=np.int64)
    pool_set = set(int(movie) for movie in candidates)
    eligible_set = set(int(movie) for movie in eligible)
    hidden_set = set(int(movie) for movie in hidden_movies)
    truth_map = {
        int(movie): float(rating)
        for movie, rating in zip(hidden_movies, hidden_ratings)
    }
    relevant = hidden_movies[hidden_ratings > ctx.threshold]
    relevant_set = set(int(movie) for movie in relevant)
    ideal = ctx.eval8.ideal_dcg(
        hidden_ratings, ctx.factor, 20
    ) if hidden_movies.size else 0.0

    score_array = _score_array_for(ctx, config, item_scores)
    if score_array is None:
        raise ArtifactError(
            f"Aucun score disponible pour l'approche {config.approach}."
        )
    if candidates.size:
        scores = np.asarray(
            score_array[candidates], dtype=np.float64
        )
        top20 = ctx.eval8.top_k_columns(
            candidates, scores, 20
        )
        full_ranked = ctx.eval8.top_k_columns(
            candidates, scores, candidates.size
        )
    else:
        scores = np.zeros(0, dtype=np.float64)
        top20 = np.zeros(0, dtype=np.int64)
        full_ranked = np.zeros(0, dtype=np.int64)
    top20_set = set(int(movie) for movie in top20)
    rank_map = {
        int(movie): rank + 1
        for rank, movie in enumerate(full_ranked)
    }
    score_map = {
        int(movie): float(value)
        for movie, value in zip(candidates, scores)
    }

    catalogue_mask = ctx.context["catalogue_mask"]
    raw_set = set(int(movie) for movie in raw_movies)
    base_movies = (
        raw_movies if source == "knn" else (
            candidates if source == "fallback" else eligible
        )
    )
    candidate_rows = []
    if len(base_movies):
        meta = _metadata_frame(ctx, base_movies).set_index("movie_col")
        popularity_means = ctx.context["popularity"]["means"]
        for movie in np.asarray(base_movies, dtype=np.int64):
            movie_int = int(movie)
            values = contributions.get(movie_int, [])
            ratings = [
                value for _, value in values
                if not np.isnan(value)
            ]
            support = support_map.get(
                movie_int,
                len(values) if values else 0,
            )
            candidate_rows.append({
                "movie_col": movie_int,
                "movie_id": meta.at[movie_int, "movie_id"],
                "title": meta.at[movie_int, "title"],
                "year": meta.at[movie_int, "year"],
                "genres_text": meta.at[movie_int, "genres_text"],
                "support": int(support) if values else np.nan,
                "contributors": ", ".join(
                    str(neighbor) for neighbor, _ in values
                ),
                "neighbor_ratings": ", ".join(
                    f"{neighbor}:{rating:g}"
                    for neighbor, rating in values
                ),
                "neighbor_avg_rating": float(np.mean(ratings))
                if ratings else np.nan,
                "neighbor_max_rating": float(np.max(ratings))
                if ratings else np.nan,
                "global_avg_rating": float(
                    popularity_means[movie_int]
                ) if np.isfinite(popularity_means[movie_int]) else np.nan,
                "in_pool": movie_int in pool_set,
                "pool_rank": pool_rank_map.get(movie_int),
                "predicted_score": score_map.get(movie_int),
                "rank": rank_map.get(movie_int),
                "in_top20": movie_int in top20_set,
                "hidden_rating": truth_map.get(movie_int),
                "known_in_test": movie_int in hidden_set,
                "relevant": (
                    movie_int in relevant_set
                    if movie_int in hidden_set
                    else None
                ),
                "exclusion_reason": _exclusion_reason(
                    movie_int,
                    catalogue_mask,
                    visible_set,
                    movie_int in pool_set,
                    movie_int in top20_set,
                    source,
                    raw_set,
                ),
            })
    candidate_table = pd.DataFrame(candidate_rows)
    if candidate_table.empty:
        candidate_table = pd.DataFrame(columns=[
            "movie_col", "movie_id", "title", "year", "genres_text",
            "support", "contributors", "neighbor_ratings",
            "neighbor_avg_rating", "neighbor_max_rating",
            "global_avg_rating", "in_pool", "pool_rank",
            "predicted_score", "rank", "in_top20", "hidden_rating",
            "known_in_test", "relevant", "exclusion_reason",
        ])
    scored_table = candidate_table[
        candidate_table["in_pool"].astype(bool)
    ].copy() if not candidate_table.empty else candidate_table.copy()
    if not scored_table.empty:
        scored_table = scored_table.sort_values(
            "rank", na_position="last"
        ).reset_index(drop=True)

    if not candidate_table.empty:
        removed = candidate_table[
            ~candidate_table["in_pool"].astype(bool)
        ][[
            "movie_col", "movie_id", "title", "year", "genres_text",
            "support", "neighbor_avg_rating", "exclusion_reason",
        ]].copy()
    else:
        removed = pd.DataFrame(columns=[
            "movie_col", "movie_id", "title", "year", "genres_text",
            "support", "neighbor_avg_rating", "exclusion_reason",
        ])

    top_rows = []
    if top20.size:
        meta_top = _metadata_frame(ctx, top20).set_index("movie_col")
        for rank, movie in enumerate(top20, start=1):
            movie_int = int(movie)
            true_rating = truth_map.get(movie_int)
            top_rows.append({
                "rank": rank,
                "movie_col": movie_int,
                "movie_id": meta_top.at[movie_int, "movie_id"],
                "title": meta_top.at[movie_int, "title"],
                "year": meta_top.at[movie_int, "year"],
                "genres_text": meta_top.at[movie_int, "genres_text"],
                "predicted_score": float(score_map[movie_int]),
                "known_in_test": movie_int in hidden_set,
                "true_rating": true_rating,
                "relevant": (
                    bool(true_rating > ctx.threshold)
                    if true_rating is not None
                    else None
                ),
                "status": (
                    "pertinent"
                    if true_rating is not None and true_rating > ctx.threshold
                    else (
                        "non pertinent"
                        if true_rating is not None
                        else "inconnu"
                    )
                ),
                "support": (
                    int(support_map.get(movie_int, 0))
                    if movie_int in support_map
                    else (
                        int(len(contributions.get(movie_int, [])))
                        if contributions.get(movie_int)
                        else np.nan
                    )
                ),
            })
    top20_table = pd.DataFrame(top_rows)
    if top20_table.empty:
        top20_table = pd.DataFrame(columns=[
            "rank", "movie_col", "movie_id", "title", "year",
            "genres_text", "predicted_score", "known_in_test",
            "true_rating", "relevant", "status", "support",
        ])

    candidate_recall = ctx.eval8.candidate_recall(
        candidates, relevant
    )
    candidate_recall_before_pool = ctx.eval8.candidate_recall(
        eligible, relevant
    )
    recall20 = ctx.eval8.recall_at_k(top20, relevant, 20)
    ndcg_known, known_count = ctx.eval8.ndcg_known(
        top20,
        truth_map,
        ideal,
        ctx.factor,
        20,
    )
    mean_true, mean_true_n = ctx.eval8.mean_true_rating(
        top20, truth_map
    )
    mean_predicted = (
        float(np.mean([score_map[int(movie)] for movie in top20]))
        if top20.size else None
    )
    if hidden_movies.size:
        heldout_ranked = ctx.eval8.top_k_columns(
            hidden_movies,
            score_array[hidden_movies],
            20,
        )
        ndcg_heldout = ctx.eval8.ndcg_from_ranked(
            heldout_ranked,
            truth_map,
            ideal,
            ctx.factor,
            20,
        )
    else:
        ndcg_heldout = None
    missed_relevant = _metadata_frame(
        ctx,
        np.asarray(
            [movie for movie in relevant if int(movie) not in pool_set],
            dtype=np.int64,
        ),
    )
    if not missed_relevant.empty:
        missed_relevant["true_rating"] = missed_relevant["movie_col"].map(
            truth_map
        )
        missed_relevant["relevant"] = True
    metrics = {
        "candidate_recall": candidate_recall,
        "candidate_recall_before_pool": candidate_recall_before_pool,
        "recall20": recall20,
        "ndcg_known": ndcg_known,
        "ndcg_heldout": ndcg_heldout,
        "known_count": int(known_count),
        "mean_true_rating": mean_true,
        "mean_true_n": int(mean_true_n),
        "mean_predicted_rating": mean_predicted,
        "relevant_hidden": int(relevant.size),
        "hidden_count": int(hidden_movies.size),
        "ideal_dcg20": float(ideal),
        "candidate_size": int(candidates.size),
        "eligible_before_pool": int(eligible.size),
    }
    profile_table, top_profile, genre_summary = _profile_tables(
        ctx, visible_cols, visible_ratings
    )
    neighbors = _neighbor_table(
        ctx, neighbor_order, similarity
    )
    neighbors = _add_neighbor_overlap(
        ctx, neighbors, visible_cols
    )
    stage_counts = {
        "visible": int(visible_cols.size),
        "hidden": int(hidden_movies.size),
        "neighbors": int(neighbor_order.size),
        "raw_occurrences": int(
            raw_occurrences if source == "knn" else eligible.size
        ),
        "unique_candidates": int(
            raw_movies.size if source == "knn" else eligible.size
        ),
        "duplicates_removed": int(
            max(0, raw_occurrences - raw_movies.size)
            if source == "knn" else 0
        ),
        "after_catalogue": int(
            sum(
                1 for movie in raw_movies
                if bool(catalogue_mask[int(movie)])
            ) if source == "knn" else eligible.size
        ),
        "after_seen": int(
            sum(
                1 for movie in raw_movies
                if bool(catalogue_mask[int(movie)])
                and int(movie) not in visible_set
            ) if source == "knn" else eligible.size
        ),
        "eligible_before_pool": int(eligible.size),
        "pool_cap": int(candidates.size),
        "scored": int(candidates.size),
        "top20": int(top20.size),
        "removed_by_pool": int(
            max(0, eligible.size - candidates.size)
        ),
        "hidden_removed_as_seen": int(
            len(hidden_set.intersection(visible_set))
        ),
    }
    tracking_movies = np.asarray(
        sorted(
            visible_set
            | raw_set
            | pool_set
            | top20_set
            | hidden_set
        ),
        dtype=np.int64,
    )
    tracking_meta = _metadata_frame(ctx, tracking_movies)
    tracking_rows = []
    if not tracking_meta.empty:
        meta_indexed = tracking_meta.set_index("movie_col")
        visible_map = _rating_map(visible_cols, visible_ratings)
        for movie in tracking_movies:
            movie_int = int(movie)
            values = contributions.get(movie_int, [])
            in_pool = movie_int in pool_set
            in_top = movie_int in top20_set
            tracking_rows.append({
                "movie_col": movie_int,
                "movie_id": meta_indexed.at[movie_int, "movie_id"],
                "title": meta_indexed.at[movie_int, "title"],
                "year": meta_indexed.at[movie_int, "year"],
                "genres_text": meta_indexed.at[movie_int, "genres_text"],
                "in_visible": movie_int in visible_set,
                "visible_rating": visible_map.get(movie_int),
                "in_raw_neighbor_lists": movie_int in raw_set,
                "raw_occurrences": len(values),
                "contributing_neighbors": ", ".join(
                    str(neighbor) for neighbor, _ in values
                ),
                "neighbor_ratings": ", ".join(
                    f"{neighbor}:{rating:g}"
                    for neighbor, rating in values
                ),
                "in_eligible": movie_int in eligible_set,
                "in_pool": in_pool,
                "pool_rank": pool_rank_map.get(movie_int),
                "predicted_score": score_map.get(movie_int),
                "final_rank": rank_map.get(movie_int),
                "in_top20": in_top,
                "hidden_rating": truth_map.get(movie_int),
                "known_in_test": movie_int in hidden_set,
                "relevant": (
                    movie_int in relevant_set
                    if movie_int in hidden_set
                    else None
                ),
                "exclusion_reason": _exclusion_reason(
                    movie_int,
                    catalogue_mask,
                    visible_set,
                    in_pool,
                    in_top,
                    source,
                    raw_set,
                ),
            })
    tracking_table = pd.DataFrame(tracking_rows)
    return PipelineResult(
        user=record,
        scenario=scenario,
        level=int(level),
        config=config,
        source=source,
        visible_cols=visible_cols,
        visible_ratings=visible_ratings,
        hidden_movies=hidden_movies,
        hidden_ratings=hidden_ratings,
        neighbor_order=neighbor_order,
        similarity=similarity,
        neighbors=neighbors,
        candidate_table=candidate_table,
        removed_table=removed,
        scored_table=scored_table,
        top20_table=top20_table,
        tracking_table=tracking_table,
        profile_table=profile_table,
        top_profile=top_profile,
        genre_summary=genre_summary,
        missed_relevant=missed_relevant,
        metrics=metrics,
        stage_counts=stage_counts,
        item_scores=item_scores,
        full_ranked=full_ranked,
        top20=top20,
        contributions=contributions,
        similarity_method=config.method,
    )


def neighbor_detail(ctx, pipeline, neighbor):
    neighbor = int(neighbor)
    matrix = ctx.context["visible_matrix"]
    row = matrix.getrow(neighbor)
    neighbor_map = {
        int(movie): float(value)
        for movie, value in zip(row.indices, row.data)
    }
    target_map = _rating_map(
        pipeline.visible_cols, pipeline.visible_ratings
    )
    movies = np.asarray(sorted(neighbor_map), dtype=np.int64)
    frame = _metadata_frame(ctx, movies)
    if not frame.empty:
        frame["neighbor_rating"] = frame["movie_col"].map(neighbor_map)
        frame["target_rating"] = frame["movie_col"].map(target_map)
        frame["common"] = frame["target_rating"].notna()
        frame = frame.sort_values(
            "neighbor_rating", ascending=False
        ).reset_index(drop=True)
    top_neighbor = (
        frame.sort_values(
            "neighbor_rating", ascending=False
        ).head(10).copy()
        if not frame.empty else frame.copy()
    )
    common = (
        frame[frame["common"]].sort_values(
            "neighbor_rating", ascending=False
        ).copy()
        if not frame.empty else frame.copy()
    )
    unseen = (
        frame[~frame["common"]].sort_values(
            "neighbor_rating", ascending=False
        ).head(20).copy()
        if not frame.empty else frame.copy()
    )
    return {
        "all": frame,
        "top": top_neighbor,
        "common": common,
        "unseen": unseen,
        "similarity": (
            float(pipeline.similarity[neighbor])
            if pipeline.similarity is not None
            else np.nan
        ),
        "n_ratings": int(len(neighbor_map)),
    }


def track_movie(ctx, pipeline, movie_col):
    movie_col = int(movie_col)
    frame = pipeline.tracking_table
    matches = frame[frame["movie_col"] == movie_col]
    if matches.empty:
        meta = _metadata_frame(ctx, np.array([movie_col]))
        row = meta.iloc[0].to_dict()
        row["exclusion_reason"] = "absent des structures du pipeline"
        return row
    return matches.iloc[0].to_dict()


def candidate_movie_options(pipeline):
    frame = pipeline.tracking_table
    if frame.empty:
        return {}
    labels = {}
    for _, row in frame.iterrows():
        label = f"{row['title']}"
        if not pd.isna(row.get("year")):
            label += f" ({int(row['year'])})"
        if label in labels:
            label += f" [{int(row['movie_col'])}]"
        labels[label] = int(row["movie_col"])
    return labels
