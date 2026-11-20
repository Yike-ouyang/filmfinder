import type {
  Decision,
  Recommendation,
  RecommendationBatch,
  RecommendationResponse,
} from "@/types/recommendations";

const TITLES = ["Film One", "Film Two", "Film Three", "Film Four"];

export function makeRecommendation(
  index = 0,
  overrides: Partial<Recommendation> = {},
): Recommendation {
  const number = index + 1;
  return {
    rank: number,
    movie_id: `movie-${number}`,
    internal_movie_id: 1000 + number,
    title: TITLES[index] ?? `Film ${number}`,
    year: 1990 + number,
    genres: ["Drama", "Thriller"],
    score: 9.2 - index * 0.1,
    score_type: "m5_predicted_rating",
    tmdb_id: 5000 + number,
    poster_path: `/poster-${number}.jpg`,
    backdrop_path: `/backdrop-${number}.jpg`,
    ...overrides,
  };
}

export function makeResponse(
  count = 3,
  overrides: Partial<RecommendationResponse> = {},
): RecommendationResponse {
  return {
    profile: {
      ratings_found: 20,
      ratings_matched: 18,
      watched_movies_matched: 30,
      match_rate: 0.9,
    },
    recommendation_info: {
      model: "M5",
      algorithm: "knn_m5",
      requested_count: 100,
      returned_count: count,
    },
    recommendations: Array.from({ length: count }, (_, index) =>
      makeRecommendation(index),
    ),
    ...overrides,
  };
}

export function makeBatch(
  count = 3,
  decisions: Decision[] = [],
): RecommendationBatch {
  return {
    version: 1,
    savedAt: Date.now(),
    response: makeResponse(count),
    decisions,
  };
}
