export const SITE_NAME = "FilmFinder";
export const SITE_TAGLINE = "Your next favorite film is waiting.";
export const SITE_DESCRIPTION =
  "Import your Letterboxd history. Discover movies made for your taste.";

export const RECOMMENDATION_LIMIT = 100;
export const SESSION_TTL_MS = 24 * 60 * 60 * 1000;
export const UPLOAD_TIMEOUT_MS = 180_000;

export const HERO_BACKDROP_PATH = "/cfT29Im5VDvjE0RpyKOSdCKZal7.jpg";

export const LOADING_MESSAGES = [
  "Reading your Letterboxd history...",
  "Finding your cinematic taste...",
  "Preparing your recommendations...",
] as const;

export const MAX_UPLOAD_SIZE_MB = (() => {
  const raw = process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB;
  const parsed = raw ? Number(raw) : 20;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 20;
})();

export const MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024;

export const STORAGE_KEY = "filmfinder.recommendation-batch.v1";

export const TMDB_WEB_BASE_URL = "https://www.themoviedb.org";
