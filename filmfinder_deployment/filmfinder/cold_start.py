from __future__ import annotations

import numpy as np


def build_ridge_system(base, latent, ratings):
    target = np.asarray(ratings, dtype=np.float64) - np.asarray(
        base, dtype=np.float64
    )
    matrix = np.column_stack([
        np.ones(target.size, dtype=np.float64),
        np.asarray(latent, dtype=np.float64),
    ])
    return matrix, target


def solve_ridge(matrix, target, lambda_bias, lambda_factors):
    regularizer = np.full(matrix.shape[1], float(lambda_factors))
    regularizer[0] = float(lambda_bias)
    gram = matrix.T @ matrix
    gram[np.diag_indices_from(gram)] += regularizer
    try:
        return np.linalg.solve(gram, matrix.T @ target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(matrix, target, rcond=None)[0]


def fit_cold_profile(base, latent, ratings, lambda_bias, lambda_factors):
    matrix, target = build_ridge_system(base, latent, ratings)
    return solve_ridge(matrix, target, lambda_bias, lambda_factors)
