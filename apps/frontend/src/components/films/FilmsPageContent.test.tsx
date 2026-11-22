import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FilmsPageContent } from "@/components/films/FilmsPageContent";
import type { FilmListItem } from "@/types/films";

const navigation = vi.hoisted(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  prefetch: vi.fn(),
  params: new URLSearchParams(),
}));

const auth = vi.hoisted(() => ({
  user: null as
    | {
        id: number;
        email: string;
        email_verified: boolean;
        created_at: string;
      }
    | null,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: navigation.push,
    replace: navigation.replace,
    prefetch: navigation.prefetch,
  }),
  useSearchParams: () => navigation.params,
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: auth.user,
    loading: false,
    signIn: vi.fn(),
    signOut: vi.fn(),
    removeAccount: vi.fn(),
    refresh: vi.fn(),
  }),
}));

const FILM_ONE: FilmListItem = {
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
  user_rating: 8,
  user_list_status: "watchlist",
};

const FILM_TWO: FilmListItem = {
  ...FILM_ONE,
  internal_movie_id: 2,
  movie_id: "film-two",
  title: "Film Two",
  year: 2001,
  tmdb_id: 5002,
  poster_path: null,
  user_rating: null,
  user_list_status: null,
  ratings_count: 800,
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetch() {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    void init;
    if (url.endsWith("/api/v1/films/facets")) {
      return Promise.resolve(
        jsonResponse({
          genres: ["Drama", "Comedy"],
          temporal_limit: 2019,
          total: 52,
        }),
      );
    }
    if (url.endsWith("/api/v1/me/profile-summary")) {
      return Promise.resolve(
        jsonResponse({
          ratings_count: 10,
          watched_count: 10,
          watchlist_count: 1,
          do_not_watch_count: 0,
          seen_count: 1,
          recommended_min_ratings: 10,
          can_generate: true,
        }),
      );
    }
    if (url.includes("/api/v1/films?")) {
      return Promise.resolve(
        jsonResponse({
          items: [FILM_ONE, FILM_TWO],
          page: 1,
          page_size: 50,
          total: 52,
          total_pages: 2,
          temporal_limit: 2019,
        }),
      );
    }
    if (url.endsWith("/api/v1/films/1")) {
      return Promise.resolve(
        jsonResponse({
          ...FILM_ONE,
          summary: "A detailed summary from the catalogue.",
        }),
      );
    }
    if (url.includes("/api/v1/recommendations/from-saved-ratings")) {
      return Promise.resolve(jsonResponse({ status: "pending" }, 202));
    }
    return Promise.reject(new Error(`Unexpected fetch: ${url}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  auth.user = {
    id: 1,
    email: "member@example.com",
    email_verified: true,
    created_at: "2026-01-01T00:00:00Z",
  };
  navigation.push.mockReset();
  navigation.replace.mockReset();
  navigation.prefetch.mockReset();
  navigation.params = new URLSearchParams();
  document.cookie = "ff_csrf=test-csrf; path=/";
});

afterEach(() => {
  document.cookie = "ff_csrf=; Max-Age=0; path=/";
  vi.unstubAllGlobals();
});

describe("FilmsPageContent", () => {
  it("renders the catalogue and profile progress", async () => {
    installFetch();
    render(<FilmsPageContent />);

    expect(
      screen.getByText(/films released up to 2019/i),
    ).toBeVisible();
    expect(
      await screen.findByRole("heading", { name: "Film One" }),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Film Two" }),
    ).toBeVisible();
    expect(
      screen.getByText(/ready to discover your recommendations/i),
    ).toBeVisible();
    expect(screen.getByText(/52 films/)).toBeVisible();
    expect(
      screen.getByRole("button", { name: "View Film One (1999)" }),
    ).toBeVisible();
  });

  it("keeps sort, pagination and search in the URL", async () => {
    installFetch();
    const user = userEvent.setup();
    render(<FilmsPageContent />);
    await screen.findByText("Film One");

    await user.selectOptions(
      screen.getByLabelText("Sort films"),
      "rating",
    );
    expect(navigation.push).toHaveBeenCalledWith(
      expect.stringContaining("sort=rating"),
      expect.anything(),
    );

    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(navigation.push).toHaveBeenCalledWith(
      expect.stringContaining("page=2"),
      expect.anything(),
    );

    await user.type(
      screen.getByLabelText("Search films by title"),
      "alien",
    );
    await waitFor(() =>
      expect(navigation.replace).toHaveBeenCalledWith(
        expect.stringContaining("q=alien"),
        expect.anything(),
      ),
    );
  });

  it("generates recommendations for a ready profile", async () => {
    installFetch();
    const user = userEvent.setup();
    render(<FilmsPageContent />);
    await screen.findByText("Film One");
    await user.click(
      screen.getByRole("button", {
        name: /Generate My Recommendations/,
      }),
    );
    await waitFor(() =>
      expect(navigation.push).toHaveBeenCalledWith("/discover"),
    );
  });

  it("invites anonymous visitors to sign in", async () => {
    auth.user = null;
    installFetch();
    render(<FilmsPageContent />);
    expect(
      await screen.findByText("Build your movie profile"),
    ).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Create an account" }),
    ).toHaveAttribute("href", "/signup?next=%2Ffilms");
  });
});
