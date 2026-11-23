import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MovieDetails } from "@/components/discover/MovieDetails";
import { makeRecommendation } from "@/test/fixtures";

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("MovieDetails", () => {
  it("loads the catalogue summary and release date for a recommendation", async () => {
    const recommendation = makeRecommendation(0);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse({
          internal_movie_id: recommendation.internal_movie_id,
          movie_id: recommendation.movie_id,
          title: recommendation.title,
          year: recommendation.year,
          release_date: "1991-05-01",
          genres: recommendation.genres,
          tmdb_id: recommendation.tmdb_id,
          poster_path: recommendation.poster_path,
          backdrop_path: recommendation.backdrop_path,
          ratings_count: 1200,
          rating_mean: 7.8,
          weighted_rating: 7.6,
          user_rating: null,
          user_list_status: null,
          summary: "A rich catalogue summary for Film One.",
        }),
      ),
    );

    render(
      <MovieDetails
        open
        recommendation={recommendation}
        model="M5"
        onClose={vi.fn()}
      />,
    );

    expect(
      await screen.findByText("A rich catalogue summary for Film One."),
    ).toBeVisible();
    expect(screen.getByText("May 1, 1991")).toBeVisible();
    expect(
      screen.getByText(/Estimated from your FilmFinder ratings/),
    ).toBeVisible();
    expect(screen.getByText("Pick #1")).toBeVisible();
  });
});
