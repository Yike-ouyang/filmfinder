export type TasteProfileRating = 3 | 5 | 8;

export type TasteProfileAction =
  | "loved"
  | "okay"
  | "disliked"
  | "no-opinion";

export interface TasteProfileMovie {
  internal_movie_id: number;
  movie_id: string;
  title: string;
  year: number | null;
  genres: string[];
  tmdb_id: number | null;
  poster_path: string | null;
  backdrop_path: string | null;
}

export interface TasteProfileRatingScale {
  loved: number;
  okay: number;
  disliked: number;
}

export interface TasteProfileMoviesResponse {
  movies: TasteProfileMovie[];
  min_ratings: number;
  rating_scale: TasteProfileRatingScale;
}

export interface TasteProfileAnswer {
  internal_movie_id: number;
  rating: TasteProfileRating | null;
}
