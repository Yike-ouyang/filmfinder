import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    return Path, pd


@app.cell
def _(pd):
    df=pd.read_csv("movies_final.csv")
    return (df,)


@app.cell
def _(Path, pd):
    data_path = Path("/users/eleves-b/2023/yike.ouyang/.cache/kagglehub/datasets/samlearner/letterboxd-movie-ratings-data/versions/6")


    ratings_export_user = pd.read_csv(data_path / "ratings_export.csv", engine="python")

    users_export_user = pd.read_csv(data_path / "users_export.csv", engine="python")
    return (ratings_export_user,)


@app.cell
def _(df, ratings_export_user):
    ratings_export_user_filtered = ratings_export_user[
        ratings_export_user["movie_id"].isin(df["movie_id"].explode())
    ]
    return (ratings_export_user_filtered,)


@app.cell
def _(ratings_export_user_filtered):
    ratings_matrix = ratings_export_user_filtered.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating_val",
        aggfunc="mean"
    )

    ratings_matrix = ratings_matrix.astype("Float64")
    return


if __name__ == "__main__":
    app.run()
