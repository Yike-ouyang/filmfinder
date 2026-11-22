import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  errorMessage,
  requestTasteProfileMovies,
  requestTasteProfileRecommendations,
} from "@/lib/api";
import { makeResponse } from "@/test/fixtures";

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const moviesPayload = {
  movies: [
    {
      internal_movie_id: 1,
      movie_id: "film-one",
      title: "Film One",
      year: 2001,
      genres: ["Drama"],
      tmdb_id: 10,
      poster_path: "/poster.jpg",
      backdrop_path: "/backdrop.jpg",
    },
  ],
  min_ratings: 3,
  rating_scale: { loved: 8, okay: 5, disliked: 3 },
};

beforeEach(() => {
  document.cookie = "ff_csrf=test-csrf; path=/";
});

afterEach(() => {
  document.cookie = "ff_csrf=; Max-Age=0; path=/";
  vi.unstubAllGlobals();
});

describe("Taste Profile API", () => {
  it("loads the fixed movie selection", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(moviesPayload));
    vi.stubGlobal("fetch", fetchMock);

    const result = await requestTasteProfileMovies();

    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://localhost:8000/api/v1/taste-profile/movies",
    );
    expect(result.movies).toHaveLength(1);
    expect(result.min_ratings).toBe(3);
    expect(result.movies[0].title).toBe("Film One");
  });

  it("posts all answers with CSRF credentials and normalizes recommendations", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(makeResponse(2)));
    vi.stubGlobal("fetch", fetchMock);

    const answers = [
      { internal_movie_id: 1, rating: 8 as const },
      { internal_movie_id: 2, rating: null },
    ];
    const result = await requestTasteProfileRecommendations({
      answers,
      limit: 20,
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain("/api/v1/taste-profile/recommendations");
    expect(url).toContain("limit=20");
    expect(init?.method).toBe("POST");
    expect(init?.credentials).toBe("include");
    expect(new Headers(init?.headers).get("X-CSRF-Token")).toBe(
      "test-csrf",
    );
    expect(JSON.parse(String(init?.body))).toEqual({ answers });
    expect(result.recommendations).toHaveLength(2);
  });

  it("maps insufficient rating errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "insufficient_taste_ratings",
              message: "Backend detail",
            },
          },
          422,
        ),
      ),
    );

    const error = await requestTasteProfileRecommendations({
      answers: [{ internal_movie_id: 1, rating: null }],
    }).catch((caught: unknown) => caught);

    expect(error).toMatchObject({
      code: "insufficient_taste_ratings",
      status: 422,
    });
    expect(error).toBeInstanceOf(ApiError);
    expect(errorMessage(error)).toContain("at least 3");
  });
});
