import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # FilmFinder — Explorateur de recommandations

    Ce notebook inspecte, pour un utilisateur du jeu de test, chaque étape du
    pipeline KNN + M5 :

    profil visible -> voisins -> candidats -> filtrage -> pool -> scoring M5 ->
    top 20 -> évaluation.

    Les notes cachées ne servent **jamais** à construire le profil, les
    similarités, la popularité ou les candidats. Elles ne sont utilisées
    qu'après le classement, pour l'évaluation et l'explication.

    La configuration cold globale est lue dans
    `knn_m5_optimization/selection/selected_configs.json`. La configuration
    warm est la meilleure configuration warm de `8.RecommendationSystem.py`.
    Les deux configurations réelles sont affichées dans la section
    « Configuration et intégrité ». M5 n'est jamais réentraîné.
    """)
    return


@app.cell
def _():
    import importlib.util
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    CORE_PATH = Path(__file__).with_name("recommendation_explorer_core.py")
    _spec = importlib.util.spec_from_file_location(
        "recommendation_explorer_core", CORE_PATH
    )
    core = importlib.util.module_from_spec(_spec)
    sys.modules["recommendation_explorer_core"] = core
    _spec.loader.exec_module(core)
    return CORE_PATH, core, json, mo, np, os, pd, plt


@app.cell
def _(core):
    CONTEXT = None
    LOAD_ERROR = None
    PIPELINE_CACHE = {}
    try:
        CONTEXT = core.load_explorer()
    except Exception as _error:
        LOAD_ERROR = f"{type(_error).__name__} : {_error}"
    return CONTEXT, LOAD_ERROR, PIPELINE_CACHE


@app.cell
def _(CORE_PATH, CONTEXT, LOAD_ERROR, core, mo):
    if CONTEXT is None:
        mo.stop(
            True,
            mo.md(
                "## Artefacts indisponibles\n\n"
                f"```text\n{LOAD_ERROR}\n```\n\n"
                "Vérifier les variables d'environnement "
                "`FILMFINDER_CLEANING_DIR`, "
                "`FILMFINDER_RECOMMENDATION_DIR`, "
                "`FILMFINDER_KNN_M5_DIR`, puis régénérer les artefacts :\n\n"
                "```bash\n"
                "python research/preprocessing/data_cleaning.py\n"
                "python research/recommendation/8.RecommendationSystem.py \\\n"
                "  --cleaning-dir \"$FILMFINDER_CLEANING_DIR\" \\\n"
                "  --output-dir research/results/recommendation_system\n"
                "python research/recommendation/9.KNNM5Optimization.py \\\n"
                "  --baseline-dir research/results/recommendation_system \\\n"
                "  --output-dir research/results/knn_m5_optimization\n"
                "```"
            ),
        )
    _artifacts = CONTEXT.artifacts
    _cold = CONTEXT.config_cold
    _warm = CONTEXT.config_warm
    _table = pd.DataFrame([
        {"élément": "core", "chemin": str(CORE_PATH)},
        {"élément": "cleaning", "chemin": str(_artifacts.cleaning_dir)},
        {"élément": "baseline", "chemin": str(_artifacts.baseline_dir)},
        {"élément": "knn_m5", "chemin": str(_artifacts.knn_dir)},
        {"élément": "M5", "chemin": str(_artifacts.m5_path)},
        {"élément": "config cold", "chemin": _cold.label},
        {"élément": "config warm", "chemin": _warm.label},
    ])
    _warnings = CONTEXT.warning_messages
    mo.vstack([
        mo.md("## Artefacts et configuration"),
        mo.ui.table(_table, pagination=False),
        mo.md(
            "\n".join(f"- {warning}" for warning in _warnings)
            if _warnings else ""
        ),
    ])
    return


@app.cell
def _(CONTEXT, mo):
    if CONTEXT is None:
        mo.stop(True, mo.md("Contexte non chargé."))
    _options = CONTEXT.user_options()
    _first = next(
        (
            label for label in _options
            if "cold" in label
        ),
        next(iter(_options)),
    )
    user_widget = mo.ui.dropdown(
        options=_options,
        value=_first,
        searchable=True,
        label="Utilisateur test",
    )
    user_widget
    return (user_widget,)


@app.cell
def _(CONTEXT, mo, user_widget):
    if CONTEXT is None:
        mo.stop(True, mo.md("Contexte non chargé."))
    _record = CONTEXT.users[int(user_widget.value)]
    if _record.scenario == "cold":
        _options = {
            f"{level} notes révélées": int(level)
            for level in _record.levels
        }
        _default_level = 20 if 20 in _record.levels else _record.levels[-1]
        level_widget = mo.ui.dropdown(
            options=_options,
            value=next(
                label for label, value in _options.items()
                if value == _default_level
            ),
            label="Niveau de révélation",
        )
        _display = level_widget
    else:
        level_widget = None
        _display = mo.md(
            "Profil **warm** : toutes les notes visibles de l'utilisateur "
            "sont utilisées, sans niveau de révélation."
        )
    mo.vstack([_display])
    return (level_widget,)


@app.cell
def _(CONTEXT, PIPELINE_CACHE, core, level_widget, mo, user_widget):
    if CONTEXT is None:
        mo.stop(True, mo.md("Contexte non chargé."))
    _user = CONTEXT.users[int(user_widget.value)]
    _level = (
        int(level_widget.value)
        if level_widget is not None and level_widget.value is not None
        else 0
    )
    _key = (int(_user.user_row), _level)
    if _key not in PIPELINE_CACHE:
        try:
            PIPELINE_CACHE[_key] = core.compute_pipeline(
                CONTEXT, _user.user_row, level=_level
            )
        except Exception as _error:
            mo.stop(
                True,
                mo.md(
                    "## Erreur de calcul du pipeline\n\n"
                    f"```text\n{type(_error).__name__} : {_error}\n```"
                ),
            )
        while len(PIPELINE_CACHE) > 32:
            PIPELINE_CACHE.pop(next(iter(PIPELINE_CACHE)))
    pipeline = PIPELINE_CACHE[_key]
    return (pipeline,)


@app.cell
def _(CONTEXT, mo, np, pd, pipeline):
    _user = pipeline.user
    _visible_mean = (
        float(pipeline.visible_ratings.mean())
        if pipeline.visible_ratings.size else np.nan
    )
    _fixed = (
        _user.scenario == "cold"
        and int(_user.n_pool or 0) >= 100
    )
    _info = pd.DataFrame([
        {"champ": "user_id", "valeur": _user.user_id},
        {"champ": "user_row", "valeur": _user.user_row},
        {"champ": "scénario", "valeur": _user.scenario},
        {"champ": "cohorte", "valeur": _user.cohort},
        {"champ": "notes visibles utilisées", "valeur": int(pipeline.visible_cols.size)},
        {"champ": "notes hidden", "valeur": int(pipeline.hidden_movies.size)},
        {"champ": "moyenne visible", "valeur": round(_visible_mean, 3) if np.isfinite(_visible_mean) else "N/A"},
        {"champ": "hidden pertinents (> 8)", "valeur": int(_user.n_relevant_hidden)},
        {"champ": "cohorte fixe pool >= 100", "valeur": bool(_fixed)},
        {"champ": "configuration", "valeur": pipeline.config.label},
        {"champ": "source candidats", "valeur": pipeline.source},
    ])
    _warning = (
        "Cet utilisateur cold est hors de la cohorte fixe `pool >= 100` : "
        "ses métriques ne sont pas directement comparables aux tableaux "
        "agrégés de l'expérience."
        if _user.scenario == "cold" and not _fixed
        else ""
    )
    mo.vstack([
        mo.md("## Utilisateur et profil"),
        mo.ui.table(_info, pagination=False),
        mo.md(_warning) if _warning else mo.md(""),
    ])
    return


@app.cell
def _(mo, pipeline, plt):
    _profile = pipeline.profile_table
    _figure, _axes = plt.subplots(1, 2, figsize=(14, 4))
    if not _profile.empty:
        _axes[0].hist(
            _profile["user_rating"].to_numpy(dtype=float),
            bins=min(20, max(5, _profile.shape[0] // 3 + 1)),
            color="#4c72b0",
            alpha=0.85,
        )
        _axes[0].set_title("Distribution des notes visibles")
        _axes[0].set_xlabel("Note")
        _axes[0].set_ylabel("Nombre de films")
        _axes[0].grid(True, alpha=0.3)
        _genres = pipeline.genre_summary.head(12)
        if not _genres.empty:
            _axes[1].barh(
                _genres["genre"][::-1],
                _genres["count"][::-1],
                color="#55a868",
            )
            _axes[1].set_title("Genres les plus présents")
            _axes[1].set_xlabel("Nombre de films visibles")
            _axes[1].grid(True, alpha=0.3)
    _figure.tight_layout()
    _profile_columns = [
        "movie_id", "title", "year", "genres_text",
        "studios_text", "user_rating", "is_top",
    ]
    _display = [
        mo.md("### Profil visible"),
        mo.ui.table(
            _profile[_profile_columns] if not _profile.empty else _profile,
            pagination=True,
        ),
        mo.md("### Films les mieux notés"),
        mo.ui.table(
            pipeline.top_profile[_profile_columns]
            if not pipeline.top_profile.empty
            else pipeline.top_profile,
            pagination=False,
        ),
        _figure,
    ]
    mo.vstack(_display)
    return


@app.cell
def _(mo, pipeline):
    _neighbors = pipeline.neighbors
    if pipeline.source == "fallback" and pipeline.neighbor_order.size == 0:
        _display = mo.md(
            "### Voisins\n\n"
            "Aucun voisin : le pipeline utilise le fallback popularité "
            "pour ce profil."
        )
    else:
        _display = mo.vstack([
            mo.md(
                "### Voisins\n\n"
                f"Similarité : **{pipeline.config.method or 'N/A'}**, "
                f"N = **{pipeline.config.n_neighbors}**, "
                f"top_k par voisin = **{pipeline.config.top_k}**."
            ),
            mo.ui.table(_neighbors, pagination=True),
        ])
    _display
    return


@app.cell
def _(mo, pipeline):
    _neighbors = pipeline.neighbors
    if _neighbors.empty:
        neighbor_widget = None
        _display = mo.md("Pas de voisin à inspecter.")
    else:
        _options = {
            f"{_row['neighbor_user_id']} — similarité {_row['similarity']:.4f}": int(_row["neighbor_row"])
            for _, _row in _neighbors.iterrows()
        }
        neighbor_widget = mo.ui.dropdown(
            options=_options,
            value=next(iter(_options)),
            searchable=True,
            label="Inspecter un voisin",
        )
        _display = neighbor_widget
    _display
    return (neighbor_widget,)


@app.cell
def _(CONTEXT, core, mo, neighbor_widget, pipeline):
    if neighbor_widget is None or neighbor_widget.value is None:
        mo.stop(True, mo.md("Sélectionner un voisin."))
    _detail = core.neighbor_detail(CONTEXT, pipeline, int(neighbor_widget.value))
    mo.vstack([
        mo.md(
            f"### Détail du voisin\n\n"
            f"Similarité : **{_detail['similarity']:.4f}** — "
            f"{_detail['n_ratings']} notes visibles."
        ),
        mo.md("#### Films les mieux notés par le voisin"),
        mo.ui.table(
            _detail["top"][[
                "title", "year", "genres_text",
                "neighbor_rating", "target_rating", "common",
            ]],
            pagination=False,
        ),
        mo.md("#### Films notés en commun"),
        mo.ui.table(
            _detail["common"][[
                "title", "year", "neighbor_rating", "target_rating",
            ]],
            pagination=True,
        ),
        mo.md("#### Films bien notés par le voisin, non notés par la cible"),
        mo.ui.table(
            _detail["unseen"][[
                "title", "year", "genres_text", "neighbor_rating",
            ]],
            pagination=True,
        ),
    ])
    return


@app.cell
def _(mo, np, pd, pipeline):
    _counts = pipeline.stage_counts
    _stages = pd.DataFrame([
        {"étape": "notes visibles", "nombre": _counts["visible"]},
        {"étape": "notes hidden", "nombre": _counts["hidden"]},
        {"étape": "voisins retenus", "nombre": _counts["neighbors"]},
        {"étape": "occurrences brutes", "nombre": _counts["raw_occurrences"]},
        {"étape": "films uniques", "nombre": _counts["unique_candidates"]},
        {"étape": "doublons retirés", "nombre": _counts["duplicates_removed"]},
        {"étape": "après catalogue", "nombre": _counts["after_catalogue"]},
        {"étape": "après films déjà vus", "nombre": _counts["after_seen"]},
        {"étape": "pool final", "nombre": _counts["pool_cap"]},
        {"étape": "films scorés", "nombre": _counts["scored"]},
        {"étape": "top 20", "nombre": _counts["top20"]},
    ])
    _candidates = pipeline.candidate_table
    _genres = sorted({
        genre.strip()
        for text in _candidates.get("genres_text", pd.Series(dtype=str)).fillna("")
        for genre in str(text).split(",")
        if genre.strip()
    })
    genre_widget = mo.ui.dropdown(
        options={"Tous les genres": "", **{genre: genre for genre in _genres}},
        value="Tous les genres",
        label="Filtrer par genre",
    )
    candidate_view_widget = mo.ui.dropdown(
        options={"Tout le pool candidat": "all", "Pool final uniquement": "pool"},
        value="Tout le pool candidat",
        label="Affichage des candidats",
    )
    if not _candidates.empty:
        _scores = _candidates["predicted_score"].dropna()
        _score_start = float(min(0.0, _scores.min())) if not _scores.empty else 0.0
        _score_stop = float(_scores.max()) if not _scores.empty else 1.0
        if _score_stop <= _score_start:
            _score_stop = _score_start + 1.0
        _years = _candidates["year"].dropna()
        _year_start = int(_years.min()) if not _years.empty else 1900
        _year_stop = int(_years.max()) if not _years.empty else 2030
        if _year_stop <= _year_start:
            _year_stop = _year_start + 1
        _support_max = int(
            _candidates["support"].fillna(0).max()
        ) if _candidates["support"].notna().any() else 1
    else:
        _score_start, _score_stop = 0.0, 1.0
        _year_start, _year_stop = 1900, 2030
        _support_max = 1
    score_min_widget = mo.ui.number(
        start=_score_start,
        stop=_score_stop,
        step=0.01,
        value=_score_start,
        label="Score prédit minimum",
    )
    score_max_widget = mo.ui.number(
        start=_score_start,
        stop=_score_stop,
        step=0.01,
        value=_score_stop,
        label="Score prédit maximum",
    )
    year_min_widget = mo.ui.number(
        start=_year_start,
        stop=_year_stop,
        step=1,
        value=_year_start,
        label="Année minimum",
    )
    year_max_widget = mo.ui.number(
        start=_year_start,
        stop=_year_stop,
        step=1,
        value=_year_stop,
        label="Année maximum",
    )
    support_widget = mo.ui.slider(
        start=0,
        stop=max(1, _support_max),
        value=0,
        label="Support minimum (voisins)",
    )
    mo.vstack([
        mo.md("## Génération des candidats"),
        mo.ui.table(_stages, pagination=False),
        mo.vstack([
            genre_widget,
            candidate_view_widget,
            score_min_widget,
            score_max_widget,
            year_min_widget,
            year_max_widget,
            support_widget,
        ]),
    ])
    return (
        candidate_view_widget,
        genre_widget,
        score_max_widget,
        score_min_widget,
        support_widget,
        year_max_widget,
        year_min_widget,
    )


@app.cell
def _(
    candidate_view_widget,
    genre_widget,
    mo,
    pd,
    pipeline,
    score_max_widget,
    score_min_widget,
    support_widget,
    year_max_widget,
    year_min_widget,
):
    _frame = pipeline.candidate_table.copy()
    if _frame.empty:
        filtered_candidates = _frame
    else:
        if candidate_view_widget.value == "pool":
            _frame = _frame[_frame["in_pool"].astype(bool)].copy()
        if genre_widget.value:
            _selected = str(genre_widget.value)
            _frame = _frame[
                _frame["genres_text"].fillna("").map(
                    lambda text: _selected in {
                        item.strip()
                        for item in str(text).split(",")
                        if item.strip()
                    }
                )
            ].copy()
        if not _frame["predicted_score"].dropna().empty:
            _low = float(score_min_widget.value)
            _high = float(score_max_widget.value)
            _frame = _frame[
                _frame["predicted_score"].isna()
                | (
                    (_frame["predicted_score"] >= _low)
                    & (_frame["predicted_score"] <= _high)
                )
            ].copy()
        _low_year = float(year_min_widget.value)
        _high_year = float(year_max_widget.value)
        _frame = _frame[
            _frame["year"].isna()
            | (
                (_frame["year"] >= _low_year)
                & (_frame["year"] <= _high_year)
            )
        ].copy()
        if support_widget.value > 0:
            _frame = _frame[
                _frame["support"].fillna(0) >= support_widget.value
            ].copy()
        filtered_candidates = _frame
    _columns = [
        "movie_id", "title", "year", "genres_text", "support",
        "neighbor_avg_rating", "global_avg_rating", "in_pool",
        "pool_rank", "predicted_score", "rank", "in_top20",
        "hidden_rating", "relevant", "exclusion_reason",
    ]
    _metric = pipeline.metrics
    _gt = pd.DataFrame([
        {"indicateur": "films hidden", "valeur": _metric["hidden_count"]},
        {"indicateur": "hidden pertinents (> 8)", "valeur": _metric["relevant_hidden"]},
        {"indicateur": "candidats dans le test", "valeur": int(pipeline.candidate_table["known_in_test"].fillna(False).sum()) if not pipeline.candidate_table.empty else 0},
        {"indicateur": "candidats pertinents", "valeur": int(pipeline.candidate_table["relevant"].fillna(False).sum()) if not pipeline.candidate_table.empty else 0},
        {"indicateur": "candidate recall final", "valeur": _metric["candidate_recall"] if _metric["candidate_recall"] is not None else "N/A"},
        {"indicateur": "candidate recall avant pool", "valeur": _metric["candidate_recall_before_pool"] if _metric["candidate_recall_before_pool"] is not None else "N/A"},
    ])
    mo.vstack([
        mo.ui.table(_frame[_columns], pagination=True),
        mo.md("### Analyse ground truth des candidats"),
        mo.ui.table(_gt, pagination=False),
        mo.md("### Films pertinents non récupérés dans le pool"),
        mo.ui.table(
            pipeline.missed_relevant[[
                "title", "year", "genres_text", "true_rating",
            ]] if not pipeline.missed_relevant.empty else pipeline.missed_relevant,
            pagination=True,
        ),
    ])
    return (filtered_candidates,)


@app.cell
def _(mo, pipeline):
    _counts = pipeline.stage_counts
    mo.vstack([
        mo.md("## Filtrage des candidats"),
        mo.md(
            f"- Occurrences brutes : **{_counts['raw_occurrences']}**\n"
            f"- Films uniques : **{_counts['unique_candidates']}**\n"
            f"- Doublons retirés : **{_counts['duplicates_removed']}**\n"
            f"- Retirés hors catalogue : "
            f"**{_counts['unique_candidates'] - _counts['after_catalogue']}**\n"
            f"- Retirés car déjà notés : "
            f"**{_counts['after_catalogue'] - _counts['after_seen']}**\n"
            f"- Retirés par le pool : **{_counts['removed_by_pool']}**\n"
            f"- Restants dans le pool : **{_counts['pool_cap']}**\n"
            f"- Hidden retirés comme déjà vus : "
            f"**{_counts['hidden_removed_as_seen']}**"
        ),
        mo.ui.table(pipeline.removed_table, pagination=True),
    ])
    return


@app.cell
def _(filtered_candidates, mo, pipeline, plt):
    _scored = filtered_candidates[
        filtered_candidates["in_pool"].astype(bool)
    ].copy() if not filtered_candidates.empty else filtered_candidates.copy()
    if not _scored.empty:
        _scored = _scored.sort_values("rank", na_position="last")
    _figure, _axis = plt.subplots(figsize=(9, 4))
    if not _scored.empty and _scored["predicted_score"].notna().any():
        _axis.hist(
            _scored["predicted_score"].dropna().to_numpy(dtype=float),
            bins=min(30, max(5, _scored.shape[0] // 3)),
            color="#c44e52",
            alpha=0.8,
        )
        _axis.set_title("Distribution des scores M5 des candidats du pool")
        _axis.set_xlabel("Score prédit")
        _axis.set_ylabel("Nombre de films")
        _axis.grid(True, alpha=0.3)
    _known = _scored[_scored["known_in_test"].fillna(False)] if not _scored.empty else _scored
    _comparison = _known[[
        "title", "year", "predicted_score", "hidden_rating",
    ]].copy() if not _known.empty else _known
    if not _comparison.empty:
        _comparison["erreur"] = (
            _comparison["predicted_score"] - _comparison["hidden_rating"]
        )
    mo.vstack([
        mo.md("## Scoring et classement"),
        mo.md(
            "Score utilisé : **M5** (prédiction, pas la note réelle). "
            "La colonne `hidden_rating` est la ground truth et n'a pas "
            "servi au classement."
        ),
        _figure,
        mo.ui.table(_scored[[
            "movie_id", "title", "year", "genres_text", "support",
            "neighbor_avg_rating", "predicted_score", "rank",
            "hidden_rating", "relevant",
        ]], pagination=True),
        mo.md("### Comparaison prédit vs réel pour les candidats hidden"),
        mo.ui.table(_comparison, pagination=True),
    ])
    return


@app.cell
def _(CONTEXT, mo, np, pipeline):
    _metrics = pipeline.metrics
    _rows = [
        {"métrique": "Candidate Recall", "valeur": _metrics["candidate_recall"]},
        {"métrique": "Recall@20", "valeur": _metrics["recall20"]},
        {"métrique": "Known NDCG@20", "valeur": _metrics["ndcg_known"]},
        {"métrique": "NDCG held-out (diagnostic)", "valeur": _metrics["ndcg_heldout"]},
        {"métrique": "recommandations avec ground truth", "valeur": _metrics["known_count"]},
        {"métrique": "note réelle moyenne des recommandations évaluables", "valeur": _metrics["mean_true_rating"]},
        {"métrique": "score prédit moyen (top 20)", "valeur": _metrics["mean_predicted_rating"]},
        {"métrique": "hidden count", "valeur": _metrics["hidden_count"]},
        {"métrique": "hidden pertinents", "valeur": _metrics["relevant_hidden"]},
    ]
    _display = []
    for _metric_row in _rows:
        value = _metric_row["valeur"]
        _metric_row["valeur"] = "N/A" if value is None or (
            isinstance(value, float) and not np.isfinite(value)
        ) else (
            round(value, 4) if isinstance(value, float) else value
        )
        _display.append(_metric_row)
    mo.vstack([
        mo.md("## Top 20 et évaluation de l'utilisateur"),
        mo.ui.table(pipeline.top20_table, pagination=False),
        mo.md(
            "Statuts : **pertinent** = note réelle > 8 ; "
            "**non pertinent** = note réelle <= 8 ; "
            "**inconnu** = aucune ground truth. "
            "Les inconnus ne sont pas des retours négatifs.\n\n"
            "Known NDCG@20 est conditionnel : il ne regarde que les films "
            "hidden présents dans le top 20 et conserve leurs positions "
            "d'origine. Il ne doit pas être comparé directement au NDCG "
            "held-out."
        ),
        mo.ui.table(_display, pagination=False),
        mo.md(
            "Aucun poster/image n'est présent dans movie_context.json : "
            "les recommandations sont affichées sous forme de tableau."
            if not CONTEXT.movie_meta["has_poster"].any()
            else ""
        ),
    ])
    return


@app.cell
def _(mo, pipeline, plt):
    _counts = pipeline.stage_counts
    _labels = [
        "visibles", "voisins", "occurrences", "uniques",
        "après filtres", "pool", "top 20",
    ]
    _values = [
        _counts["visible"], _counts["neighbors"], _counts["raw_occurrences"],
        _counts["unique_candidates"], _counts["eligible_before_pool"],
        _counts["pool_cap"], _counts["top20"],
    ]
    _figure, _axis = plt.subplots(figsize=(10, 4))
    _axis.bar(_labels, _values, color="#4c72b0")
    _axis.set_title("Entonnoir du pipeline de recommandation")
    _axis.set_ylabel("Nombre de films")
    _axis.tick_params(axis="x", rotation=15)
    _axis.grid(True, axis="y", alpha=0.3)
    _figure.tight_layout()
    mo.vstack([
        mo.md("## Résumé du pipeline"),
        _figure,
    ])
    return


@app.cell
def _(mo, pd, pipeline):
    _labels = {}
    for _, _track_row in pipeline.tracking_table.iterrows():
        _label = str(_track_row["title"])
        if not pd.isna(_track_row["year"]):
            _label += f" ({int(_track_row['year'])})"
        if _label in _labels:
            _label += f" [{int(_track_row['movie_col'])}]"
        _labels[_label] = int(_track_row["movie_col"])
    movie_widget = mo.ui.dropdown(
        options=_labels,
        value=next(iter(_labels)) if _labels else None,
        searchable=True,
        label="Suivre un film dans le pipeline",
    )
    movie_widget
    return (movie_widget,)


@app.cell
def _(CONTEXT, core, mo, movie_widget, pd, pipeline):
    if movie_widget is None or movie_widget.value is None:
        mo.stop(True, mo.md("Aucun film à suivre."))
    _movie = int(movie_widget.value)
    _row = core.track_movie(CONTEXT, pipeline, _movie)
    _contributions = pipeline.contributions.get(_movie, [])
    _neighbor_rows = pd.DataFrame([
        {
            "voisin": neighbor,
            "note donnée au film": rating,
        }
        for neighbor, rating in _contributions
    ])
    _summary = pd.DataFrame([
        {"étape": "présent dans le profil visible", "valeur": bool(_row.get("in_visible", False))},
        {"étape": "récupéré par les voisins", "valeur": int(_row.get("raw_occurrences", 0) or 0)},
        {"étape": "présent après dédoublonnage", "valeur": bool(_row.get("in_raw_neighbor_lists", False))},
        {"étape": "présent dans le pool", "valeur": bool(_row.get("in_pool", False))},
        {"étape": "rang dans le pool", "valeur": _row.get("pool_rank")},
        {"étape": "score M5", "valeur": _row.get("predicted_score")},
        {"étape": "rang final", "valeur": _row.get("final_rank")},
        {"étape": "présent dans le top 20", "valeur": bool(_row.get("in_top20", False))},
        {"étape": "note réelle hidden", "valeur": _row.get("hidden_rating")},
        {"étape": "pertinent", "valeur": _row.get("relevant")},
        {"étape": "raison d'exclusion", "valeur": _row.get("exclusion_reason")},
    ])
    mo.vstack([
        mo.md("## Suivi d'un film"),
        mo.ui.table(_summary, pagination=False),
        mo.md("### Voisins ayant noté ce film"),
        mo.ui.table(_neighbor_rows, pagination=True),
    ])
    return


@app.cell
def _(CONTEXT, mo):
    if CONTEXT is None:
        mo.stop(True, mo.md("Contexte non chargé."))
    _cold = CONTEXT.config_cold
    _warm = CONTEXT.config_warm
    mo.md(
        "## Configuration et intégrité\n\n"
        f"- Cold : `{_cold.label}` — source `{_cold.source}`\n"
        f"- Warm : `{_warm.label}` — source `{_warm.source}`\n"
        f"- M5 : `{CONTEXT.artifacts.m5_path}` (non réentraîné)\n"
        f"- cleaning : `{CONTEXT.artifacts.cleaning_dir}`\n"
        f"- baseline : `{CONTEXT.artifacts.baseline_dir}`\n"
        f"- knn_m5 : `{CONTEXT.artifacts.knn_dir}`\n\n"
        "Les similarités et la popularité utilisent uniquement la matrice "
        "d'entraînement visible. Les notes cachées ne sont exploitées "
        "qu'après le classement pour l'affichage et l'évaluation."
    )
    return


if __name__ == "__main__":
    app.run()
