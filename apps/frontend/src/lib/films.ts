import { API_BASE_URL, ApiError } from "@/lib/api";
import { FILMS_PAGE_SIZE } from "@/lib/constants";
import type {
  FilmDetail,
  FilmFacets,
  FilmListParams,
  FilmListItem,
  FilmListResponse,
} from "@/types/films";

const FILMS_TIMEOUT_MS = 30_000;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value)
    ? value
    : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : null;
}

function asNullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function normalizeFilm(value: unknown): FilmListItem {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  const internal = asInteger(value.internal_movie_id);
  const movieId = asNullableString(value.movie_id);
  const title = asNullableString(value.title);
  const count = asInteger(value.ratings_count);
  const mean = asNumber(value.rating_mean);
  const weighted = asNumber(value.weighted_rating);
  if (
    internal === null ||
    !movieId ||
    !title ||
    count === null ||
    mean === null ||
    weighted === null
  ) {
    throw new ApiError(
      "invalid_response",
      200,
      "A film entry is missing required fields.",
    );
  }
  const status = value.user_list_status;
  return {
    internal_movie_id: internal,
    movie_id: movieId,
    title,
    year: asInteger(value.year),
    release_date: asNullableString(value.release_date),
    genres: Array.isArray(value.genres)
      ? value.genres.filter(
          (genre): genre is string => typeof genre === "string",
        )
      : [],
    tmdb_id: asInteger(value.tmdb_id),
    poster_path: asNullableString(value.poster_path),
    backdrop_path: asNullableString(value.backdrop_path),
    ratings_count: count,
    rating_mean: mean,
    weighted_rating: weighted,
    user_rating: asInteger(value.user_rating),
    user_list_status:
      status === "watchlist" ||
      status === "do_not_watch" ||
      status === "seen"
        ? status
        : null,
  };
}

export function normalizeFilmListResponse(
  payload: unknown,
): FilmListResponse {
  if (!isRecord(payload) || !Array.isArray(payload.items)) {
    throw new ApiError("invalid_response", 200);
  }
  const page = asInteger(payload.page);
  const pageSize = asInteger(payload.page_size);
  const total = asInteger(payload.total);
  const totalPages = asInteger(payload.total_pages);
  const temporalLimit = asInteger(payload.temporal_limit);
  if (
    page === null ||
    pageSize === null ||
    total === null ||
    totalPages === null ||
    temporalLimit === null
  ) {
    throw new ApiError(
      "invalid_response",
      200,
      "The film list response is incomplete.",
    );
  }
  return {
    items: payload.items.map(normalizeFilm),
    page,
    page_size: pageSize,
    total,
    total_pages: totalPages,
    temporal_limit: temporalLimit,
  };
}

export function normalizeFilmDetail(payload: unknown): FilmDetail {
  const film = normalizeFilm(payload);
  if (!isRecord(payload)) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    ...film,
    summary: asNullableString(payload.summary),
  };
}

export function normalizeFilmFacets(payload: unknown): FilmFacets {
  if (!isRecord(payload) || !Array.isArray(payload.genres)) {
    throw new ApiError("invalid_response", 200);
  }
  const temporalLimit = asInteger(payload.temporal_limit);
  const total = asInteger(payload.total);
  if (temporalLimit === null || total === null) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    genres: payload.genres.filter(
      (genre): genre is string => typeof genre === "string",
    ),
    temporal_limit: temporalLimit,
    total,
  };
}

async function readErrorCode(response: Response): Promise<string | null> {
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload) && isRecord(payload.error)) {
      return asNullableString(payload.error.code);
    }
  } catch {
    return null;
  }
  return null;
}

async function requestJson(
  path: string,
  signal?: AbortSignal,
  timeoutMs = FILMS_TIMEOUT_MS,
): Promise<unknown> {
  const controller = new AbortController();
  let timedOut = false;
  let externallyAborted = signal?.aborted ?? false;
  const onAbort = () => {
    externallyAborted = true;
    controller.abort();
  };
  if (signal) {
    if (signal.aborted) {
      throw new ApiError("cancelled", null);
    }
    signal.addEventListener("abort", onAbort, { once: true });
  }
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { Accept: "application/json" },
      credentials: "include",
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) {
      const code = (await readErrorCode(response)) ?? "backend_error";
      throw new ApiError(code, response.status);
    }
    return response.json();
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
    window.clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

export async function fetchFilms(
  params: FilmListParams,
  signal?: AbortSignal,
): Promise<FilmListResponse> {
  const query = new URLSearchParams({
    page: String(params.page),
    page_size: String(FILMS_PAGE_SIZE),
  });
  if (params.search.trim()) {
    query.set("search", params.search.trim());
  }
  if (params.sort) {
    query.set("sort", params.sort);
  }
  if (params.order !== "desc") {
    query.set("order", params.order);
  }
  if (params.genre) {
    query.set("genre", params.genre);
  }
  const payload = await requestJson(
    `/api/v1/films?${query.toString()}`,
    signal,
  );
  return normalizeFilmListResponse(payload);
}

export async function fetchFilmFacets(
  signal?: AbortSignal,
): Promise<FilmFacets> {
  const payload = await requestJson("/api/v1/films/facets", signal);
  return normalizeFilmFacets(payload);
}

export async function fetchFilmDetail(
  filmId: number,
  signal?: AbortSignal,
): Promise<FilmDetail> {
  const payload = await requestJson(
    `/api/v1/films/${encodeURIComponent(String(filmId))}`,
    signal,
  );
  return normalizeFilmDetail(payload);
}
