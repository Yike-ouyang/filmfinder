from __future__ import annotations

import numpy as np


class M5Model:
    def __init__(self, store):
        weights = store.m5_weights
        self.w0 = float(weights["w0"][0])
        self.w = np.asarray(weights["w"], dtype=np.float64)
        self.V = np.asarray(weights["V"], dtype=np.float64)
        self.active_mask = np.asarray(
            weights["active_mask"], dtype=bool
        )
        self.n_features = int(weights["n_features"][0])
        self.n_factors = int(weights["n_factors"][0])
        if self.w.shape != (self.n_features,):
            raise ValueError("Invalid w shape.")
        if self.V.shape != (self.n_features, self.n_factors):
            raise ValueError("Invalid V shape.")
        self.item_features = store.item_features
        if self.item_features.shape[1] != self.n_features:
            raise ValueError(
                "Movie features and M5 feature dimensions disagree."
            )
        user_indices = store.feature_mappings.get("user_feature_indices")
        if user_indices is None:
            user_indices = np.arange(store.ratings.shape[0])
        self.user_feature_indices = np.asarray(
            user_indices, dtype=np.int64
        )
        item_latent = np.asarray(
            self.item_features @ self.V, dtype=np.float64
        )
        item_squared = self.item_features.multiply(self.item_features)
        factor_squared = np.einsum("ij,ij->i", self.V, self.V)
        self.base_scores = (
            self.w0
            + np.asarray(
                self.item_features @ self.w, dtype=np.float64
            ).ravel()
            + 0.5
            * (
                np.einsum("ij,ij->i", item_latent, item_latent)
                - np.asarray(
                    item_squared @ factor_squared, dtype=np.float64
                ).ravel()
            )
        )
        self.latent_scores = item_latent

    def user_vector(self, user_row):
        index = int(self.user_feature_indices[int(user_row)])
        return self.w[index], self.V[index]

    def score_user(self, user_row, movie_cols):
        cols = np.asarray(movie_cols, dtype=np.int64)
        user_weight, user_latent = self.user_vector(user_row)
        return (
            self.base_scores[cols]
            + user_weight
            + self.latent_scores[cols] @ user_latent
        )

    def score_user_rows(self, user_rows, movie_cols):
        users = np.asarray(user_rows, dtype=np.int64)
        cols = np.asarray(movie_cols, dtype=np.int64)
        indices = self.user_feature_indices[users]
        return (
            self.base_scores[cols]
            + self.w[indices]
            + np.einsum(
                "ij,ij->i",
                self.latent_scores[cols],
                self.V[indices],
            )
        )

    def score_cold(self, theta, movie_cols):
        cols = np.asarray(movie_cols, dtype=np.int64)
        theta = np.asarray(theta, dtype=np.float64)
        return (
            self.base_scores[cols]
            + theta[0]
            + self.latent_scores[cols] @ theta[1:]
        )

    def item_components(self, movie_cols):
        cols = np.asarray(movie_cols, dtype=np.int64)
        return self.base_scores[cols], self.latent_scores[cols]
