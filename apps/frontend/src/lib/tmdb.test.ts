import { describe, expect, it } from "vitest";
import {
  getTmdbBackdropUrl,
  getTmdbMovieUrl,
  getTmdbPosterUrl,
} from "@/lib/tmdb";

describe("tmdb helpers", () => {
  it("builds poster and backdrop CDN URLs", () => {
    expect(getTmdbPosterUrl("/poster.jpg")).toBe(
      "https://image.tmdb.org/t/p/w500/poster.jpg",
    );
    expect(getTmdbBackdropUrl("/backdrop.jpg")).toBe(
      "https://image.tmdb.org/t/p/w1280/backdrop.jpg",
    );
  });

  it("returns null for missing or unsafe paths", () => {
    expect(getTmdbPosterUrl(null)).toBeNull();
    expect(getTmdbPosterUrl(undefined)).toBeNull();
    expect(getTmdbPosterUrl("")).toBeNull();
    expect(getTmdbPosterUrl("poster.jpg")).toBeNull();
    expect(getTmdbBackdropUrl("//evil.example/image.jpg")).toBeNull();
  });

  it("builds a TMDB web link only for valid ids", () => {
    expect(getTmdbMovieUrl(550)).toBe(
      "https://www.themoviedb.org/movie/550",
    );
    expect(getTmdbMovieUrl(null)).toBeNull();
    expect(getTmdbMovieUrl(0)).toBeNull();
    expect(getTmdbMovieUrl(1.5)).toBeNull();
  });
});
