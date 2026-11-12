import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import unicodedata
    from collections import defaultdict
    from scipy.sparse import csr_matrix, save_npz

    return csr_matrix, mo, np, pd, save_npz, unicodedata


@app.cell
def _(mo):
    mo.md(r"""
    # Load the dataset
    """)
    return


@app.cell
def _(pd):
    studios = pd.read_csv("data/studios.csv")
    movies=pd.read_csv("data/movies.csv")
    movies_users = pd.read_csv(
            "data/movie_data.csv",
            engine="python"
        )
    rating_export=pd.read_csv(
            "data/ratings_export.csv",
            engine="python"
        )
    return movies, movies_users, rating_export, studios


@app.cell
def _(mo):
    mo.md(r"""
    # Uniformize the name of the columns
    """)
    return


@app.cell
def _(movies):
    movies.columns
    return


@app.cell
def _(movies_users):
    movies_users.columns
    return


@app.cell
def _(movies, studios):
    studios_agg = (
        studios.groupby("id")["studio"]
        .agg(lambda x: sorted(set(x.dropna())))
        .reset_index()
    )

    movies_studios = movies.merge(
        studios_agg,
        on="id",
        how="left",
        validate="one_to_one"
    )
    return (movies_studios,)


@app.cell
def _(movies_studios, movies_users):
    # Nettoyage du catalogue des films
    movies_clean = (
        movies_studios[["id", "name", "date", "description", "minute","studio"]]
        .rename(columns={"id": "1_id"})
        .copy()
    )

    # Nettoyage du catalogue des films utilisateurs
    movies_users_clean = (
        movies_users[
            [
                "_id",
                "genres",
                "movie_id",
                "year_released",
                "runtime",
                "movie_title",
                "overview",
            ]
        ]
        .rename(columns={
            "year_released": "date",
            "runtime": "minute",
            "movie_title": "name",
            "overview": "description",
        })
        .copy()
    )
    return movies_clean, movies_users_clean


@app.cell
def _(movies_clean):
    movies_clean # 941 597 rows
    return


@app.cell
def _(movies_users_clean):
    movies_users_clean # 285 963 rows
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Merge the duplicates in movies
    """)
    return


@app.cell
def _(movies_clean, np, pd, unicodedata):



    def _deduplicate_movies(_movies):

        _df = _movies.copy().reset_index(drop=True)
        _n = len(_df)

        # 1. Normalisation
        def _normalize(_value):
            if pd.isna(_value):
                return None
            _text = unicodedata.normalize("NFKD", str(_value))
            _text = "".join(
                _c for _c in _text
                if not unicodedata.combining(_c)
            )
            _text = " ".join(_text.casefold().split())
            return _text or None

        _df["_title_key"] = _df["name"].map(_normalize)
        _df["_desc_key"] = _df["description"].map(_normalize)
        _df["_year"] = pd.to_numeric(_df["date"], errors="coerce")
        _df["_duration"] = pd.to_numeric(_df["minute"], errors="coerce")

        # 2. Union-Find : regroupement rapide
        _parent = np.arange(_n)
        _rank = np.zeros(_n, dtype=np.int8)

        def _find(_x):
            while _parent[_x] != _x:
                _parent[_x] = _parent[_parent[_x]]
                _x = _parent[_x]
            return _x

        def _union(_a, _b):
            _ra, _rb = _find(_a), _find(_b)
            if _ra == _rb:
                return
            if _rank[_ra] < _rank[_rb]:
                _ra, _rb = _rb, _ra
            _parent[_rb] = _ra
            if _rank[_ra] == _rank[_rb]:
                _rank[_ra] += 1

        # 3. Même description
        _valid_desc = _df["_desc_key"].notna()

        for _, _group in _df[_valid_desc].groupby(
            "_desc_key", sort=False
        ):
            _ids = _group.index.to_numpy()
            for _idx in _ids[1:]:
                _union(int(_ids[0]), int(_idx))

        # 4. Même titre, même durée, année +/- 1
        _valid = (
            _df["_title_key"].notna()
            & _df["_year"].notna()
            & _df["_duration"].notna()
        )

        for _, _group in _df[_valid].groupby(
            ["_title_key", "_duration"], sort=False
        ):
            _ordered = _group.sort_values("_year")
            _ids = _ordered.index.to_numpy()
            _years = _ordered["_year"].to_numpy()

            for _i in range(1, len(_ids)):
                if _years[_i] - _years[_i - 1] <= 1:
                    _union(int(_ids[_i - 1]), int(_ids[_i]))

        # 5. Identification des groupes
        _df["_group"] = [_find(_i) for _i in range(_n)]

        # 6. Fusion des studios
        def _merge_studios(_series):
            _values = set()
            for _item in _series:
                if isinstance(_item, (list, tuple, set)):
                    _values.update(
                        str(_s).strip()
                        for _s in _item
                        if pd.notna(_s) and str(_s).strip()
                    )
                elif pd.notna(_item) and str(_item).strip():
                    _values.add(str(_item).strip())
            return sorted(_values)

        # 7. Une ligne canonique par groupe
        _representatives = (
            _df.drop_duplicates("_group", keep="first")
            .set_index("_group")
        )

        _studios = _df.groupby("_group")["studio"].agg(
            _merge_studios
        )

        _result = _representatives[
            ["1_id", "name", "date", "description", "minute"]
        ].join(_studios)

        _result = _result.reset_index(drop=True)

        print(f"Films initiaux : {_n:,}")
        print(f"Films finaux   : {len(_result):,}")
        print(f"Doublons fusionnés : {_n - len(_result):,}")

        return _result


    movies_dedup = _deduplicate_movies(movies_clean)
    return (movies_dedup,)


@app.cell
def _(movies_dedup):
    movies_dedup # 912 745 rows
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Merge movies and movies_use
    """)
    return


@app.cell
def _(movies_dedup, movies_users_clean):
    def _merge_movie_catalogs(_movies, _users):
        import pandas as pd
        import unicodedata
        from collections import defaultdict

        def _normalize(value):
            if pd.isna(value):
                return None
            text = unicodedata.normalize("NFKD", str(value))
            text = "".join(
                c for c in text if not unicodedata.combining(c)
            )
            return " ".join(text.casefold().split()) or None

        def _valid(value):
            return (
                pd.notna(value)
                and str(value).strip() != ""
            )

        # Préparation des clés
        a = _movies.copy().reset_index(drop=True)
        b = _users.copy().reset_index(drop=True)

        for df in (a, b):
            df["_title"] = df["name"].map(_normalize)
            df["_desc"] = df["description"].map(_normalize)
            df["_year"] = pd.to_numeric(df["date"], errors="coerce")
            df["_minute"] = pd.to_numeric(df["minute"], errors="coerce")

        # Indexation du catalogue
        desc_index = defaultdict(list)
        title_index = defaultdict(list)

        for i, row in enumerate(a.itertuples(index=False)):
            desc = a.at[i, "_desc"]
            title = a.at[i, "_title"]
            year = a.at[i, "_year"]
            minute = a.at[i, "_minute"]

            if desc is not None:
                desc_index[desc].append(i)

            if title is not None and pd.notna(year) and pd.notna(minute):
                title_index[(title, minute)].append((i, year))

        # Recherche des correspondances uniques
        matches = []
        ambiguous = 0

        for j in range(len(b)):
            candidates = set()

            desc = b.at[j, "_desc"]
            title = b.at[j, "_title"]
            year = b.at[j, "_year"]
            minute = b.at[j, "_minute"]

            if desc is not None:
                candidates.update(desc_index.get(desc, []))

            if title is not None and pd.notna(year) and pd.notna(minute):
                for i, candidate_year in title_index.get(
                    (title, minute), []
                ):
                    if abs(year - candidate_year) <= 1:
                        candidates.add(i)

            if len(candidates) == 1:
                matches.append((j, next(iter(candidates))))
            elif len(candidates) > 1:
                ambiguous += 1

        # Construction du résultat : correspondances uniquement
        rows = []

        for j, i in matches:
            user = b.iloc[j]
            movie = a.iloc[i]

            years = [
                y for y in (user["_year"], movie["_year"])
                if pd.notna(y)
            ]

            rows.append({
                "_id": user["_id"],
                "movie_id": user["movie_id"],
                "1_id": movie["1_id"],
                "name": (
                    user["name"]
                    if _valid(user["name"])
                    else movie["name"]
                ),
                "date": int(min(years)) if years else pd.NA,
                "minute": (
                    user["minute"]
                    if _valid(user["minute"])
                    else movie["minute"]
                ),
                "description": (
                    user["description"]
                    if _valid(user["description"])
                    else movie["description"]
                ),
                "genres": user["genres"],
                "studio": movie["studio"],
            })

        result = pd.DataFrame(
            rows,
            columns=[
                "_id", "movie_id", "1_id", "name",
                "date", "minute", "description",
                "genres", "studio"
            ]
        )

        print(f"Films movies_dedup : {len(a):,}")
        print(f"Films movies_users : {len(b):,}")
        print(f"Correspondances    : {len(result):,}")
        print(f"Cas ambigus exclus : {ambiguous:,}")

        return result


    movies_final = _merge_movie_catalogs(
        movies_dedup,
        movies_users_clean
    )
    return (movies_final,)


@app.cell
def _(movies_final):
    movies_final # 253 223 rows
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Keep only movies existing in the final dataset
    """)
    return


@app.cell
def _(rating_export):
    rating_export # 11 078 167
    return


@app.cell
def _(movies_final, rating_export):
    scores_clean = rating_export[
        rating_export["movie_id"].isin(movies_final["movie_id"].dropna())
    ].copy()
    return (scores_clean,)


@app.cell
def _(scores_clean):
    scores_clean # 10 552 049
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Remove movies with less then 20 reviews and user with less then 10
    """)
    return


@app.cell
def _(scores_clean):
    def _filter_scores(_scores, min_movie=20, min_user=10):
        _df = _scores.copy()

        while True:
            _n_before = len(_df)

            # Films avec au moins 20 notes
            _movie_counts = _df.groupby("movie_id")["movie_id"].transform("size")
            _df = _df[_movie_counts >= min_movie]

            # Utilisateurs avec au moins 10 notes
            _user_counts = _df.groupby("user_id")["user_id"].transform("size")
            _df = _df[_user_counts >= min_user]

            if len(_df) == _n_before:
                break

        print(f"Notes conservées : {len(_df):,}")
        print(f"Films conservés : {_df['movie_id'].nunique():,}")
        print(f"Utilisateurs conservés : {_df['user_id'].nunique():,}")

        return _df


    scores_filtered = _filter_scores(
        scores_clean,
        min_movie=20,
        min_user=10
    )
    return (scores_filtered,)


@app.cell
def _(scores_filtered):
    scores_filtered # 9 728 981 rows, 40,981 films, 7,369 users
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Save a sparse matrix
    """)
    return


@app.cell
def _(csr_matrix, np, pd, save_npz, scores_filtered):
    def _save_sparse_matrix(_scores):


        # 1. Conserver uniquement les colonnes nécessaires
        _df = _scores[
            ["user_id", "movie_id", "rating_val"]
        ].copy()

        # 2. Nettoyer les notes
        _df["rating_val"] = pd.to_numeric(
            _df["rating_val"], errors="coerce"
        )

        _df = _df.dropna(
            subset=["user_id", "movie_id", "rating_val"]
        )

        # 3. Une seule note par couple utilisateur-film
        _df = (
            _df.groupby(
                ["user_id", "movie_id"],
                as_index=False,
                sort=False
            )["rating_val"]
            .mean()
        )

        # 4. Encoder les identifiants
        _user_codes, _user_ids = pd.factorize(
            _df["user_id"], sort=True
        )

        _movie_codes, _movie_ids = pd.factorize(
            _df["movie_id"], sort=True
        )

        # 5. Construire la matrice sparse CSR
        _matrix = csr_matrix(
            (
                _df["rating_val"].to_numpy(dtype=np.float32),
                (_user_codes, _movie_codes)
            ),
            shape=(len(_user_ids), len(_movie_ids)),
            dtype=np.float32
        )

        # 6. Sauvegarder
        save_npz("ratings_matrix.npz", _matrix)

        np.save(
            "user_ids.npy",
            _user_ids.to_numpy()
        )

        np.save(
            "movie_ids.npy",
            _movie_ids.to_numpy()
        )

        # 7. Résumé
        print(f"Utilisateurs : {len(_user_ids):,}")
        print(f"Films        : {len(_movie_ids):,}")
        print(f"Notes        : {_matrix.nnz:,}")
        print(f"Dimensions   : {_matrix.shape}")
        print(f"Sparsité     : {1 - _matrix.nnz / (_matrix.shape[0] * _matrix.shape[1]):.4%}")

        return _matrix


    ratings_matrix = _save_sparse_matrix(scores_filtered)
    return


@app.cell
def _(movies_final):
    def _save_movie_context(_movies):
        import pandas as pd
        import numpy as np

        # 1. Charger les identifiants de la matrice sparse
        _movie_ids = np.load(
            "movie_ids.npy",
            allow_pickle=True
        )

        # 2. Conserver uniquement les films de la matrice
        _context = (
            _movies
            .drop_duplicates(subset="movie_id")
            .set_index("movie_id")
            .reindex(_movie_ids)
            .reset_index()
        )

        # 3. Vérifier l'alignement
        assert _context["movie_id"].notna().all(), (
            "Certains films sont absents du catalogue."
        )

        assert _context["movie_id"].tolist() == list(_movie_ids), (
            "Les identifiants ne sont pas alignés."
        )

        # 4. Sauvegarder en JSON
        _context.to_json(
            "movie_context.json",
            orient="records",
            force_ascii=False,
            indent=2
        )

        print(f"Films sauvegardés : {len(_context):,}")
        print("Fichier créé : movie_context.json")

        return _context


    movie_context = _save_movie_context(movies_final)
    return


if __name__ == "__main__":
    app.run()
