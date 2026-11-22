export type FilmListStatus = "watchlist" | "do_not_watch" | "seen";
export type FilmSort = "popular" | "rating" | "release_date";
export type FilmOrder = "asc" | "desc";

export interface FilmListItem {
  internal_movie_id: number;
  movie_id: string;
  title: string;
  year: number | null;
  release_date: string | null;
  genres: string[];
  tmdb_id: number | null;
  poster_path: string | null;
  backdrop_path: string | null;
  ratings_count: number;
  rating_mean: number;
  weighted_rating: number;
  user_rating: number | null;
  user_list_status: FilmListStatus | null;
}

export interface FilmDetail extends FilmListItem {
  summary: string | null;
}

export interface FilmListResponse {
  items: FilmListItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  temporal_limit: number;
}

export interface FilmFacets {
  genres: string[];
  temporal_limit: number;
  total: number;
}

export interface FilmListParams {
  page: number;
  search: string;
  sort: FilmSort;
  order: FilmOrder;
  genre: string;
}

export interface RatingResponse {
  internal_movie_id: number;
  movie_id: string;
  title: string;
  rating: number | null;
  ratings_count: number;
}

export interface ProfileSummary {
  ratings_count: number;
  watched_count: number;
  watchlist_count: number;
  do_not_watch_count: number;
  seen_count: number;
  recommended_min_ratings: number;
  can_generate: boolean;
}
