import { TMDB_WEB_BASE_URL } from "@/lib/constants";

const TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p";

function normalizePath(path: string | null | undefined): string | null {
  if (!path || typeof path !== "string") {
    return null;
  }
  const trimmed = path.trim();
  if (!trimmed.startsWith("/") || trimmed.startsWith("//")) {
    return null;
  }
  return trimmed;
}

export function getTmdbPosterUrl(
  path: string | null | undefined,
): string | null {
  const normalized = normalizePath(path);
  return normalized ? `${TMDB_IMAGE_BASE_URL}/w500${normalized}` : null;
}

export function getTmdbBackdropUrl(
  path: string | null | undefined,
): string | null {
  const normalized = normalizePath(path);
  return normalized ? `${TMDB_IMAGE_BASE_URL}/w1280${normalized}` : null;
}

export function getTmdbMovieUrl(
  tmdbId: number | null | undefined,
): string | null {
  if (!tmdbId || !Number.isInteger(tmdbId) || tmdbId <= 0) {
    return null;
  }
  return `${TMDB_WEB_BASE_URL}/movie/${tmdbId}`;
}
