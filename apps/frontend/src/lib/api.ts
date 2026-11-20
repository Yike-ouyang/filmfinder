import {
  RECOMMENDATION_LIMIT,
  UPLOAD_TIMEOUT_MS,
} from "@/lib/constants";
import type {
  Recommendation,
  RecommendationResponse,
} from "@/types/recommendations";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || DEFAULT_API_BASE_URL
).replace(/\/+$/, "");

const ERROR_MESSAGES: Record<string, string> = {
  empty_upload: "This archive is empty.",
  invalid_zip:
    "This file is not a valid ZIP archive. Export your data again from Letterboxd.",
  missing_ratings:
    "ratings.csv is missing. Export your data again from Letterboxd.",
  corrupted_zip: "This archive appears to be damaged. Try exporting it again.",
  unsafe_zip_path: "This archive could not be read safely.",
  encrypted_zip: "Encrypted archives are not supported.",
  duplicate_zip_entry: "This archive contains duplicate files.",
  invalid_letterboxd_archive:
    "We could not read this Letterboxd archive. Make sure it is the original export.",
  upload_too_large:
    "This archive is larger than the upload limit. Try a smaller export.",
  too_many_entries: "This archive contains too many files.",
  uncompressed_too_large:
    "This archive is too large once extracted. Try a smaller export.",
  no_matched_ratings:
    "We could not match enough of your ratings to the FilmFinder catalogue. Try a profile with more rated films.",
  no_recommendations:
    "No recommendations are available for this profile right now.",
  invalid_limit: "The recommendation settings are invalid.",
  recommendation_input_error:
    "This profile could not be used to generate recommendations.",
  inference_error:
    "The recommendation engine could not complete this request. Try again in a moment.",
  invalid_score:
    "The recommendation engine returned an invalid result. Try again in a moment.",
  engine_unavailable:
    "The recommendation service is not ready yet. Start the backend model and try again.",
  validation_error: "The upload request was rejected. Try selecting the ZIP again.",
  invalid_response:
    "The recommendation service returned an unexpected response.",
  backend_unavailable:
    "Cannot reach the FilmFinder backend. Make sure it is running and try again.",
  backend_error:
    "The FilmFinder backend could not complete this request. Try again in a moment.",
  timeout:
    "The analysis took too long. Check the backend and try again.",
  cancelled: "The upload was cancelled.",
};

export class ApiError extends Error {
  readonly code: string;
  readonly status: number | null;

  constructor(code: string, status: number | null, message?: string) {
    super(message ?? ERROR_MESSAGES[code] ?? "Something went wrong.");
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

export interface RequestRecommendationsOptions {
  file: File;
  limit?: number;
  signal?: AbortSignal;
  timeoutMs?: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asInteger(value: unknown): number | null {
  const number = asNumber(value);
  return number !== null && Number.isInteger(number) ? number : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNullableString(value: unknown): string | null {
  return value === null || value === undefined ? null : asString(value);
}

function normalizeRecommendation(
  value: unknown,
  index: number,
): Recommendation {
  if (!isRecord(value)) {
    throw new ApiError(
      "invalid_response",
      200,
      "A recommendation entry is missing required fields.",
    );
  }

  const rank = asInteger(value.rank);
  const movieId = value.movie_id;
  const internalMovieId = asInteger(value.internal_movie_id);
  const title = asString(value.title);
  const score = asNumber(value.score);
  const scoreType = asString(value.score_type);

  if (
    rank === null ||
    (typeof movieId !== "string" && typeof movieId !== "number") ||
    internalMovieId === null ||
    !title ||
    score === null ||
    !scoreType
  ) {
    throw new ApiError(
      "invalid_response",
      200,
      "A recommendation entry is missing required fields.",
    );
  }

  const genres = Array.isArray(value.genres)
    ? value.genres.filter((genre): genre is string => typeof genre === "string")
    : [];

  return {
    rank: rank || index + 1,
    movie_id: movieId,
    internal_movie_id: internalMovieId,
    title,
    year: asInteger(value.year),
    genres,
    score,
    score_type: scoreType,
    tmdb_id: asInteger(value.tmdb_id),
    poster_path: asNullableString(value.poster_path),
    backdrop_path: asNullableString(value.backdrop_path),
  };
}

export function normalizeRecommendationResponse(
  payload: unknown,
): RecommendationResponse {
  if (!isRecord(payload)) {
    throw new ApiError("invalid_response", 200);
  }

  const profile = payload.profile;
  const info = payload.recommendation_info;
  const recommendations = payload.recommendations;

  if (!isRecord(profile) || !isRecord(info) || !Array.isArray(recommendations)) {
    throw new ApiError("invalid_response", 200);
  }

  const ratingsFound = asInteger(profile.ratings_found);
  const ratingsMatched = asInteger(profile.ratings_matched);
  const watchedMatched = asInteger(profile.watched_movies_matched);
  const matchRate = asNumber(profile.match_rate);
  const model = asString(info.model);
  const algorithm = asString(info.algorithm);
  const requestedCount = asInteger(info.requested_count);
  const returnedCount = asInteger(info.returned_count);

  if (
    ratingsFound === null ||
    ratingsMatched === null ||
    watchedMatched === null ||
    matchRate === null ||
    !model ||
    !algorithm ||
    requestedCount === null ||
    returnedCount === null
  ) {
    throw new ApiError(
      "invalid_response",
      200,
      "The recommendation service response is incomplete.",
    );
  }

  if (recommendations.length === 0) {
    throw new ApiError("no_recommendations", 200);
  }

  return {
    profile: {
      ratings_found: ratingsFound,
      ratings_matched: ratingsMatched,
      watched_movies_matched: watchedMatched,
      match_rate: matchRate,
    },
    recommendation_info: {
      model,
      algorithm,
      requested_count: requestedCount,
      returned_count: returnedCount,
    },
    recommendations: recommendations.map(normalizeRecommendation),
  };
}

async function readErrorCode(response: Response): Promise<string | null> {
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload) && isRecord(payload.error)) {
      return asString(payload.error.code);
    }
  } catch {
    return null;
  }
  return null;
}

function statusFallbackCode(status: number): string {
  if (status === 413) {
    return "upload_too_large";
  }
  if (status === 422) {
    return "validation_error";
  }
  if (status === 503) {
    return "engine_unavailable";
  }
  return "backend_error";
}

export async function requestRecommendations({
  file,
  limit = RECOMMENDATION_LIMIT,
  signal,
  timeoutMs = UPLOAD_TIMEOUT_MS,
}: RequestRecommendationsOptions): Promise<RecommendationResponse> {
  const formData = new FormData();
  formData.append("file", file, file.name);

  const controller = new AbortController();
  let timedOut = false;
  let externallyAborted = signal?.aborted ?? false;

  const onExternalAbort = () => {
    externallyAborted = true;
    controller.abort();
  };

  if (signal) {
    if (signal.aborted) {
      throw new ApiError("cancelled", null);
    }
    signal.addEventListener("abort", onExternalAbort, { once: true });
  }

  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const url = `${API_BASE_URL}/api/v1/recommendations/from-letterboxd?limit=${encodeURIComponent(String(limit))}`;
    const response = await fetch(url, {
      method: "POST",
      body: formData,
      headers: { Accept: "application/json" },
      credentials: "omit",
      cache: "no-store",
      signal: controller.signal,
    });

    if (!response.ok) {
      const code = (await readErrorCode(response)) ?? statusFallbackCode(response.status);
      throw new ApiError(code, response.status);
    }

    let payload: unknown;
    try {
      payload = await response.json();
    } catch {
      throw new ApiError("invalid_response", response.status);
    }

    return normalizeRecommendationResponse(payload);
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }
    if (error instanceof DOMException && error.name === "AbortError") {
      if (timedOut) {
        throw new ApiError("timeout", null);
      }
      if (externallyAborted) {
        throw new ApiError("cancelled", null);
      }
      throw new ApiError("timeout", null);
    }
    if (error instanceof TypeError) {
      throw new ApiError("backend_unavailable", null);
    }
    throw new ApiError("backend_error", null);
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", onExternalAbort);
  }
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return ERROR_MESSAGES[error.code] ?? error.message;
  }
  return "Something went wrong. Please try again.";
}
