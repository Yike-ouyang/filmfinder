import { ApiError } from "@/lib/api";
import { authenticatedRequest } from "@/lib/auth";
import type { ProfileSummary, RatingResponse } from "@/types/films";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value)
    ? value
    : null;
}

function normalizeRatingResponse(payload: unknown): RatingResponse {
  if (!isRecord(payload)) {
    throw new ApiError("invalid_response", 200);
  }
  const internal = asInteger(payload.internal_movie_id);
  const movieId = payload.movie_id;
  const title = payload.title;
  const rating = asInteger(payload.rating);
  const count = asInteger(payload.ratings_count);
  if (
    internal === null ||
    typeof movieId !== "string" ||
    typeof title !== "string" ||
    count === null
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    internal_movie_id: internal,
    movie_id: movieId,
    title,
    rating,
    ratings_count: count,
  };
}

export async function setFilmRating(
  filmId: number,
  rating: number,
): Promise<RatingResponse> {
  const payload = await authenticatedRequest(
    `/api/v1/me/ratings/${filmId}`,
    {
      method: "PUT",
      body: JSON.stringify({ rating }),
    },
  );
  return normalizeRatingResponse(payload);
}

export async function deleteFilmRating(
  filmId: number,
): Promise<RatingResponse> {
  const payload = await authenticatedRequest(
    `/api/v1/me/ratings/${filmId}`,
    { method: "DELETE" },
  );
  return normalizeRatingResponse(payload);
}

export async function fetchProfileSummary(): Promise<ProfileSummary> {
  const payload = await authenticatedRequest("/api/v1/me/profile-summary");
  if (!isRecord(payload)) {
    throw new ApiError("invalid_response", 200);
  }
  const required = [
    "ratings_count",
    "watched_count",
    "watchlist_count",
    "do_not_watch_count",
    "seen_count",
    "recommended_min_ratings",
  ] as const;
  const values = required.map((key) => asInteger(payload[key]));
  if (values.some((value) => value === null)) {
    throw new ApiError("invalid_response", 200);
  }
  if (typeof payload.can_generate !== "boolean") {
    throw new ApiError("invalid_response", 200);
  }
  return {
    ratings_count: values[0] as number,
    watched_count: values[1] as number,
    watchlist_count: values[2] as number,
    do_not_watch_count: values[3] as number,
    seen_count: values[4] as number,
    recommended_min_ratings: values[5] as number,
    can_generate: payload.can_generate,
  };
}
