export type DecisionKind = "interested" | "passed" | "seen";

export type SwipeDirection = "left" | "right" | "down";

export interface ProfileSummary {
  ratings_found: number;
  ratings_matched: number;
  watched_movies_matched: number;
  match_rate: number;
}

export interface RecommendationInfo {
  model: string;
  algorithm: string;
  requested_count: number;
  returned_count: number;
}

export interface Recommendation {
  rank: number;
  movie_id: string | number;
  internal_movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  score: number;
  score_type: string;
  tmdb_id: number | null;
  poster_path: string | null;
  backdrop_path: string | null;
}

export interface RecommendationResponse {
  profile: ProfileSummary;
  recommendation_info: RecommendationInfo;
  recommendations: Recommendation[];
}

export interface Decision {
  key: string;
  kind: DecisionKind;
}

export interface RecommendationBatch {
  version: 1;
  batchId?: number;
  savedAt: number;
  response: RecommendationResponse;
  decisions: Decision[];
}

export type BatchLoadStatus =
  | "ready"
  | "missing"
  | "expired"
  | "invalid"
  | "unavailable";

export type RequestMoreStatus = "ready" | "no_new" | "error";

export interface RequestMoreResult {
  status: RequestMoreStatus;
  message?: string;
}
