from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class KNNM5Recommender:
    def __init__(self, store, model):
        self.store = store
        self.model = model
        self.config = store.recommendation_config
        self.ratings = store.ratings.tocsr()
        self.n_users, self.n_movies = self.ratings.shape
        self.counts = np.asarray(
            self.ratings.getnnz(axis=0), dtype=np.float64
        )
        sums = np.asarray(self.ratings.sum(axis=0)).ravel()
        self.means = np.divide(
            sums,
            self.counts,
            out=np.zeros_like(sums),
            where=self.counts > 0,
        )
        observed = sums[self.counts > 0]
        global_mean = (
            float(np.asarray(self.ratings.sum()).ravel()[0]
                  / self.ratings.nnz)
            if self.ratings.nnz else 0.0
        )
        self.global_mean = global_mean
        self.centered = self.ratings.copy()
        row_counts = np.diff(self.ratings.indptr)
        row_sums = np.asarray(self.ratings.sum(axis=1)).ravel()
        row_means = np.divide(
            row_sums,
            np.maximum(row_counts, 1),
            out=np.zeros_like(row_sums),
            where=row_counts > 0,
        )
        self.centered.data -= np.repeat(row_means, row_counts)
        self.binary = self.ratings.copy()
        self.binary.data[:] = 1.0
        self.norms = np.sqrt(
            np.asarray(self.centered.multiply(self.centered).sum(axis=1))
            .ravel()
        )
        m = float(self.config.get("fallback_popularity_m", 100.0))
        self.popularity_scores = (
            self.counts / (self.counts + m) * self.means
            + m / (self.counts + m) * global_mean
        )
        self.catalogue_mask = self.counts > 0
        self.catalogue_cols = np.flatnonzero(
            self.catalogue_mask
        ).astype(np.int64)
        self.fallback_order = self.catalogue_cols[
            np.argsort(-self.popularity_scores[self.catalogue_cols])
        ].astype(np.int64)
        self.top_k_per_neighbor = int(
            self.config.get("top_k_per_neighbor", 20)
        )
        self.topk_cache = self._build_topk_cache(
            self.top_k_per_neighbor
        )
        self.shrink = float(self.config.get("similarity_shrink", 10.0))
        self.n_neighbors = int(self.config.get("n_neighbors", 50))
        self.pool_size = int(self.config.get("candidate_pool_size", 100))
        self.fallback_candidates = int(
            self.config.get("fallback_candidates", 500)
        )
        self.default_top_n = int(self.config.get("default_top_n", 20))
        ranking = str(self.config.get("ranking", "m5")).casefold()
        self.score_type = (
            "m5_predicted_rating"
            if ranking == "m5"
            else "bayesian_popularity"
        )

    def _build_topk_cache(self, top_k):
        cache = [None] * self.n_users
        for user in range(self.n_users):
            start = self.ratings.indptr[user]
            end = self.ratings.indptr[user + 1]
            cols = self.ratings.indices[start:end]
            values = self.ratings.data[start:end]
            if cols.size == 0:
                cache[user] = np.zeros(0, dtype=np.int64)
                continue
            order = np.lexsort(
                (cols, -self.counts[cols], -values)
            )[:int(top_k)]
            cache[user] = cols[order].astype(np.int64)
        return cache

    def _target_centered(self, cols, ratings):
        cols = np.asarray(cols, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float64)
        if cols.size == 0:
            return None, None
        mean = float(ratings.mean())
        centered = csr_matrix(
            (ratings - mean, (np.zeros(cols.size, dtype=np.int64), cols)),
            shape=(1, self.n_movies),
            dtype=np.float64,
        )
        binary = csr_matrix(
            (np.ones(cols.size), (np.zeros(cols.size, dtype=np.int64), cols)),
            shape=(1, self.n_movies),
            dtype=np.float64,
        )
        return centered, binary

    def _similarity(self, centered_target, binary_target, exclude_user=None):
        if centered_target is None:
            return None
        target_norm = float(
            np.sqrt(centered_target.multiply(centered_target).sum())
        )
        if target_norm <= 0:
            return None
        dots = np.asarray(
            (self.centered @ centered_target.T).toarray()
        ).ravel().astype(np.float64)
        overlap = np.asarray(
            (self.binary @ binary_target.T).toarray()
        ).ravel().astype(np.float64)
        similarity = dots / (self.norms * target_norm + 1e-12)
        similarity *= overlap / (overlap + self.shrink)
        similarity[overlap <= 0] = -np.inf
        similarity[self.norms <= 0] = -np.inf
        if exclude_user is not None:
            user = int(exclude_user)
            if 0 <= user < similarity.size:
                similarity[user] = -np.inf
        return similarity

    def _neighbor_order(self, similarity):
        if similarity is None:
            return np.zeros(0, dtype=np.int64)
        similarity = np.asarray(similarity, dtype=np.float64).copy()
        valid = np.isfinite(similarity)
        if not valid.any():
            return np.zeros(0, dtype=np.int64)
        order = np.argsort(-similarity)
        order = order[valid[order]]
        return order[:self.n_neighbors].astype(np.int64)

    def _fallback_candidates(self, seen):
        candidates = self.fallback_order
        if seen.size:
            candidates = candidates[~np.isin(candidates, seen)]
        eligible = int(candidates.size)
        limit = (
            self.fallback_candidates
            if self.pool_size is None
            else min(self.pool_size, self.fallback_candidates)
        )
        candidates = np.asarray(candidates[:limit], dtype=np.int64)
        return candidates, "fallback", {
            "candidate_source": "fallback",
            "neighbors": 0,
            "raw_occurrences": eligible,
            "unique_candidates": eligible,
            "after_seen": eligible,
            "eligible_before_pool": eligible,
            "pool_candidates": int(candidates.size),
        }

    def _candidate_pool(self, similarity, seen):
        neighbor_order = self._neighbor_order(similarity)
        arrays = [
            self.topk_cache[int(neighbor)]
            for neighbor in neighbor_order
            if self.topk_cache[int(neighbor)].size > 0
        ]
        neighbors = int(neighbor_order.size)
        if not arrays:
            candidates, source, stats = self._fallback_candidates(seen)
            stats["neighbors"] = neighbors
            return candidates, source, stats
        raw_occurrences = int(sum(array.size for array in arrays))
        concatenated = np.concatenate(arrays)
        support = np.bincount(
            concatenated, minlength=self.n_movies
        ).astype(np.float64)
        candidates = np.flatnonzero(support > 0).astype(np.int64)
        unique_candidates = int(candidates.size)
        candidates = candidates[self.catalogue_mask[candidates]]
        if seen.size:
            candidates = candidates[~np.isin(candidates, seen)]
        after_seen = int(candidates.size)
        if candidates.size == 0:
            candidates, source, stats = self._fallback_candidates(seen)
            stats.update({
                "neighbors": neighbors,
                "raw_occurrences": raw_occurrences,
                "unique_candidates": unique_candidates,
                "after_seen": 0,
            })
            return candidates, source, stats
        support = support[candidates]
        order = np.lexsort(
            (
                candidates,
                -self.counts[candidates],
                -support,
            )
        )
        if self.pool_size is not None:
            candidates = candidates[order[:self.pool_size]]
        else:
            candidates = candidates[order]
        candidates = np.asarray(candidates, dtype=np.int64)
        return candidates, "knn", {
            "candidate_source": "knn",
            "neighbors": neighbors,
            "raw_occurrences": raw_occurrences,
            "unique_candidates": unique_candidates,
            "after_seen": after_seen,
            "eligible_before_pool": after_seen,
            "pool_candidates": int(candidates.size),
        }

    @staticmethod
    def _top_k_columns(cols, scores, k):
        cols = np.asarray(cols, dtype=np.int64)
        if cols.size == 0:
            return np.zeros(0, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if cols.size <= int(k):
            order = np.lexsort((cols, -scores))
            return cols[order]
        part = np.argpartition(-scores, int(k) - 1)[:int(k)]
        order = part[np.lexsort((cols[part], -scores[part]))]
        return cols[order]

    def _format(self, top_cols, score_map):
        rows = []
        for rank, movie_col in enumerate(top_cols, start=1):
            movie_col = int(movie_col)
            info = self.store.movie_info(movie_col)
            rows.append({
                "rank": rank,
                "movie_id": info["movie_id"],
                "internal_movie_id": movie_col,
                "title": info["title"],
                "year": info["year"],
                "genres": info["genres"],
                "score": float(score_map[movie_col]),
                "score_type": self.score_type,
                "tmdb_id": info.get("tmdb_id"),
                "poster_path": info.get("poster_path"),
                "backdrop_path": info.get("backdrop_path"),
            })
        return rows

    def _recommend_cols(self, candidates, scores, top_n):
        top_cols = self._top_k_columns(candidates, scores, int(top_n))
        score_map = {
            int(movie): float(score)
            for movie, score in zip(candidates, scores)
        }
        return self._format(top_cols, score_map)

    def recommend_existing(self, user_row, top_n):
        user_row = int(user_row)
        row = self.ratings.getrow(user_row)
        seen = row.indices.astype(np.int64)
        centered = self.centered.getrow(user_row)
        binary = self.binary.getrow(user_row)
        similarity = self._similarity(
            centered, binary, exclude_user=user_row
        )
        candidates, source, stats = self._candidate_pool(
            similarity, seen
        )
        scores = self.model.score_user(user_row, candidates)
        return self._recommend_cols(candidates, scores, top_n), stats

    def recommend_new(self, movie_cols, ratings, top_n, extra_seen=None):
        movie_cols = np.asarray(movie_cols, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float64)
        if extra_seen is None:
            extra_seen = np.zeros(0, dtype=np.int64)
        extra_seen = np.asarray(extra_seen, dtype=np.int64)
        seen = np.unique(np.concatenate([movie_cols, extra_seen]))
        if movie_cols.size:
            centered, binary = self._target_centered(
                movie_cols, ratings
            )
            similarity = self._similarity(centered, binary)
        else:
            similarity = None
        candidates, source, stats = self._candidate_pool(similarity, seen)
        stats["ratings_count"] = int(movie_cols.size)
        stats["watched_count"] = int(extra_seen.size)
        if movie_cols.size:
            base, latent = self.model.item_components(movie_cols)
            from .cold_start import fit_cold_profile

            theta = fit_cold_profile(
                base,
                latent,
                ratings,
                float(self.config.get("cold_adapt_reg_bias", 0.01)),
                float(self.config.get("cold_adapt_reg_factors", 0.1)),
            )
        else:
            theta = np.zeros(self.model.n_factors + 1, dtype=np.float64)
        scores = self.model.score_cold(theta, candidates)
        return self._recommend_cols(candidates, scores, top_n), stats
