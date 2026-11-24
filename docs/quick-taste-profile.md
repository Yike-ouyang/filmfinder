# Quick Taste Profile selection report

Generated on 2026-11-24 from the production artifacts.

## Criteria

- Eligible pool: at least 1000 ratings, a TMDB poster and non-empty genres.
- Released in 2019 or earlier so every film belongs to the FilmFinder library.
- Popular anchors: at least 2000 ratings and a mean rating of at least 7.8.
- Discriminative films: at least 1000 ratings, mean between 4.5 and 8.5, variance of at least 3.5.
- Diversity uses M5 item latent cosine, genre Jaccard and decades.
- The final 20 films are frozen in `quick_taste_movies.json` after manual review of the algorithmic proposals.

## Selection

| # | Film | Year | Pool | Ratings | Mean | Variance |
|---|---|---|---|---|---|---|
| 1 | Alien | 1979 | popular | 4538 | 8.82 | 1.78 |
| 2 | Amélie | 2001 | popular | 2784 | 7.92 | 3.12 |
| 3 | Back to the Future | 1985 | popular | 4337 | 8.54 | 2.13 |
| 4 | Parasite | 2019 | popular | 5989 | 9.16 | 1.42 |
| 5 | Spider-Man: Into the Spider-Verse | 2018 | popular | 5287 | 8.67 | 1.80 |
| 6 | Spirited Away | 2001 | popular | 4192 | 8.79 | 1.98 |
| 7 | The Dark Knight | 2008 | popular | 5054 | 8.55 | 2.55 |
| 8 | The Godfather | 1972 | popular | 4278 | 9.05 | 1.83 |
| 9 | The Good, the Bad and the Ugly | 1966 | popular | 2552 | 8.89 | 1.82 |
| 10 | The Matrix | 1999 | popular | 4899 | 8.33 | 2.44 |
| 11 | American Beauty | 1999 | discriminative | 3416 | 7.72 | 4.22 |
| 12 | Bohemian Rhapsody | 2018 | discriminative | 3722 | 5.14 | 4.88 |
| 13 | Forrest Gump | 1994 | discriminative | 3903 | 7.24 | 4.94 |
| 14 | Joker | 2019 | discriminative | 5461 | 6.60 | 4.64 |
| 15 | La La Land | 2016 | discriminative | 5054 | 8.19 | 3.64 |
| 16 | Mamma Mia! | 2008 | discriminative | 2421 | 6.37 | 5.46 |
| 17 | Midsommar | 2019 | discriminative | 4974 | 7.47 | 3.84 |
| 18 | Star Wars: The Last Jedi | 2017 | discriminative | 4748 | 7.00 | 4.27 |
| 19 | Titanic | 1997 | discriminative | 3554 | 7.41 | 3.58 |
| 20 | Twilight | 2008 | discriminative | 2785 | 4.80 | 5.39 |

## Diversity

- Decades: {'1960s': 1, '1970s': 2, '1980s': 1, '1990s': 4, '2000s': 5, '2010s': 7}
- Distinct genres: 16
- Genre counts: {'Action': 4, 'Adventure': 3, 'Animation': 2, 'Comedy': 7, 'Crime': 3, 'Drama': 11, 'Family': 2, 'Fantasy': 2, 'History': 1, 'Horror': 2, 'Music': 2, 'Mystery': 1, 'Romance': 6, 'Science Fiction': 5, 'Thriller': 3, 'Western': 1}
- Mean latent cosine: 0.506
- Mean genre Jaccard: 0.154
- Max latent cosine: 0.983
- Max genre Jaccard: 1.000

The M5 latent factors are learned co-occurrence representations, not interpretable genre axes. They are used only as a redundancy penalty.

## Cold-start findings

The existing engine was evaluated on real users who rated at least 12 of the 20 selected films. For each user, 3 random subsets were drawn per threshold and compared with the recommendations obtained from all of their ratings on the 20 films.

| Ratings | Candidate source | Top-20 overlap | Rank correlation |
|---|---|---|---|
| 0 | popularity fallback | - | - |
| 1 | popularity fallback | 5.1 / 20 | 0.23 |
| 2 | KNN | 6.2 / 20 | 0.43 |
| 3 | KNN | 6.3 / 20 | 0.45 |
| 5 | KNN | 7.8 / 20 | 0.54 |
| 10 | KNN | 10.5 / 20 | 0.72 |

- With 0 or 1 rating, the centered similarity vector is degenerate, so the engine falls back to popularity.
- With 2 ratings, KNN is engaged but the profile remains unstable.
- The product requires at least 3 ratings. If all ratings are identical, the centered similarity is still degenerate and the engine falls back to popularity.

## API contract

- `GET /api/v1/taste-profile/movies` returns the frozen 20-film selection, the minimum rating count, the recommended rating count, and the current user rating when authenticated.
- `POST /api/v1/taste-profile/recommendations` accepts exactly 20 answers `{internal_movie_id, rating}` where rating is an integer from 1 to 10 or `null` for Haven't Seen.
- Haven't Seen answers never become ratings and never mark a film as watched. Rated films are excluded from the returned recommendations.
- Ratings use the same 1-10 scale as normalized Letterboxd ratings and the Film library.
- At least 3 ratings are required to generate recommendations; 10 ratings are recommended.

## Regenerate

```bash
python packages/recommendation-engine/scripts/select_quick_taste_movies.py
```

The script writes `apps/backend/app/data/quick_taste_movies.json` and this report. Run it only when the catalogue artifacts change.
