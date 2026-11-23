import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FilmDetailDialog } from "@/components/films/FilmDetailDialog";
import type { FilmDetail } from "@/types/films";

const DETAIL: FilmDetail = {
  internal_movie_id: 1,
  movie_id: "film-one",
  title: "Film One",
  year: 1999,
  release_date: "1999-05-01",
  genres: ["Drama"],
  tmdb_id: 5001,
  poster_path: "/poster-1.jpg",
  backdrop_path: "/backdrop-1.jpg",
  ratings_count: 1200,
  rating_mean: 7.8,
  weighted_rating: 7.6,
  user_rating: null,
  user_list_status: null,
  summary: "A detailed summary from the catalogue.",
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetch() {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    if (url.endsWith("/api/v1/films/1")) {
      return Promise.resolve(jsonResponse(DETAIL));
    }
    if (url.endsWith("/api/v1/me/ratings/1")) {
      const body = init?.body
        ? JSON.parse(String(init.body))
        : { rating: null };
      return Promise.resolve(
        jsonResponse({
          internal_movie_id: 1,
          movie_id: "film-one",
          title: "Film One",
          rating: body.rating,
          ratings_count: 10,
        }),
      );
    }
    if (url.endsWith("/api/v1/swipes/1")) {
      return Promise.resolve(
        jsonResponse({
          swipe: {
            internal_movie_id: 1,
            movie_id: "film-one",
            title: "Film One",
            year: 1999,
            genres: ["Drama"],
            tmdb_id: 5001,
            poster_path: "/poster-1.jpg",
            backdrop_path: "/backdrop-1.jpg",
            action: "interested",
            updated_at: "2026-01-01T00:00:00Z",
          },
          event_id: 1,
        }),
      );
    }
    return Promise.reject(new Error(`Unexpected fetch: ${url}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  document.cookie = "ff_csrf=test-csrf; path=/";
});

afterEach(() => {
  document.cookie = "ff_csrf=; Max-Age=0; path=/";
  vi.unstubAllGlobals();
});

describe("FilmDetailDialog", () => {
  it("shows catalogue details and allows rating", async () => {
    installFetch();
    const user = userEvent.setup();
    render(
      <FilmDetailDialog
        open
        filmId={1}
        isAuthenticated
        onClose={vi.fn()}
      />,
    );

    expect(
      await screen.findByRole("heading", { name: "Film One" }),
    ).toBeVisible();
    expect(
      await screen.findByText(
        "A detailed summary from the catalogue.",
      ),
    ).toBeVisible();
    expect(screen.getByText("Average rating")).toBeVisible();

    await user.click(
      screen.getByRole("radio", {
        name: "Rate 4.5 stars out of 5 (9/10)",
      }),
    );
    await waitFor(() =>
      expect(screen.getByText("9/10")).toBeVisible(),
    );
  });

  it("invites anonymous visitors to authenticate", async () => {
    installFetch();
    render(
      <FilmDetailDialog
        open
        filmId={1}
        isAuthenticated={false}
        onClose={vi.fn()}
      />,
    );
    expect(
      await screen.findByRole("link", { name: "Create an account" }),
    ).toHaveAttribute("href", "/signup?next=%2Ffilms");
    expect(
      screen.getByRole("link", { name: "Log in" }),
    ).toHaveAttribute("href", "/login?next=%2Ffilms");
  });
});
