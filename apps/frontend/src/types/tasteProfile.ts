export type TasteProfileRating = number;

export interface TasteProfileMovie {
  internal_movie_id: number;
  movie_id: string;
  title: string;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  poster_path: string | null;
  backdrop_path: string | null;
  user_rating: number | null;
}

export interface TasteProfileMoviesResponse {
  movies: TasteProfileMovie[];
  min_ratings: number;
  recommended_min_ratings: number;
}

export interface TasteProfileAnswer {
  internal_movie_id: number;
  rating: TasteProfileRating | null;
}
