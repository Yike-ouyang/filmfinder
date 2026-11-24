import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TasteProfilePageContent } from "@/components/taste-profile/TasteProfilePageContent";
import { loadBatch } from "@/lib/storage";
import {
  clearTasteProfileDraft,
  loadTasteProfileDraft,
  saveTasteProfileDraft,
} from "@/lib/tasteProfileDraft";
import { makeResponse } from "@/test/fixtures";
import type { TasteProfileMovie } from "@/types/tasteProfile";

const navigation = vi.hoisted(() => ({
  push: vi.fn(),
  prefetch: vi.fn(),
}));

const auth = vi.hoisted(() => ({
  user: null as
    | {
        id: number;
        email: string;
        email_verified: boolean;
        is_admin: boolean;
        created_at: string;
      }
    | null,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
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

const movies: TasteProfileMovie[] = Array.from(
  { length: 20 },
  (_, index) => ({
    internal_movie_id: 1001 + index,
    movie_id: `taste-film-${index + 1}`,
    title: `Taste Film ${index + 1}`,
    year: 1990 + index,
    genres: ["Drama"],
    tmdb_id: 5001 + index,
    poster_path: `/poster-${index + 1}.jpg`,
    backdrop_path: `/backdrop-${index + 1}.jpg`,
    user_rating: null,
  }),
);

function moviesPayload(
  overrides: Partial<{
    recommended_min_ratings: number;
    movies: TasteProfileMovie[];
  }> = {},
) {
  return {
    movies: overrides.movies ?? movies,
    min_ratings: 3,
    recommended_min_ratings: overrides.recommended_min_ratings ?? 10,
  };
}

function profileSummary(ratingsCount: number) {
  return {
    ratings_count: ratingsCount,
    watched_count: ratingsCount,
    watchlist_count: 0,
    do_not_watch_count: 0,
    seen_count: 0,
    recommended_min_ratings: 10,
    can_generate: ratingsCount >= 10,
  };
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetchMock(payload = moviesPayload()) {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    void init;
    if (url.endsWith("/api/v1/taste-profile/movies")) {
      return Promise.resolve(jsonResponse(payload));
    }
    if (url.endsWith("/api/v1/me/profile-summary")) {
      return Promise.resolve(jsonResponse(profileSummary(4)));
    }
    if (url.includes("/api/v1/taste-profile/recommendations")) {
      return Promise.resolve(jsonResponse(makeResponse(5)));
    }
    return Promise.reject(new Error(`Unexpected fetch: ${url}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

async function startQuiz() {
  const user = userEvent.setup();
  render(<TasteProfilePageContent />);
  await user.click(
    await screen.findByRole("button", { name: /Start rating/ }),
  );
  await screen.findByText("Taste Film 1");
  return user;
}

function rate(key: string) {
  fireEvent.keyDown(window, { key });
}

function skip() {
  fireEvent.keyDown(window, { key: "h" });
}

async function answerWithKeyboard(ratings: string[], skips: number) {
  for (const rating of ratings) {
    act(() => {
      rate(rating);
    });
  }
  for (let index = 0; index < skips; index += 1) {
    act(() => {
      skip();
    });
  }
}

function postCalls(fetchMock: ReturnType<typeof installFetchMock>) {
  return fetchMock.mock.calls.filter(([url]) =>
    String(url).includes("/api/v1/taste-profile/recommendations"),
  ) as [string, RequestInit][];
}

beforeEach(() => {
  navigation.push.mockReset();
  navigation.prefetch.mockReset();
  auth.user = null;
  window.sessionStorage.clear();
  document.cookie = "ff_csrf=test-csrf; path=/";
});

afterEach(() => {
  document.cookie = "ff_csrf=; Max-Age=0; path=/";
  clearTasteProfileDraft();
  vi.unstubAllGlobals();
});

describe("TasteProfilePageContent", () => {
  it("rates with five stars and half stars then submits valid answers", async () => {
    const fetchMock = installFetchMock(
      moviesPayload({ recommended_min_ratings: 3 }),
    );
    const user = await startQuiz();

    await user.click(
      screen.getByRole("radio", {
        name: "Rate 4.5 stars out of 5 (9/10)",
      }),
    );
    await screen.findByText("Taste Film 2");
    await user.click(
      screen.getByRole("radio", {
        name: "Rate 2.5 stars out of 5 (5/10)",
      }),
    );
    await screen.findByText("Taste Film 3");
    await user.click(
      screen.getByRole("radio", {
        name: "Rate 0.5 stars out of 5 (1/10)",
      }),
    );
    await screen.findByText("Taste Film 4");

    for (let index = 4; index <= 20; index += 1) {
      await user.click(
        screen.getByRole("button", { name: /Haven't Seen/ }),
      );
      if (index < 20) {
        await screen.findByText(`Taste Film ${index + 1}`);
      }
    }

    await waitFor(() =>
      expect(navigation.push).toHaveBeenCalledWith("/discover"),
    );

    const calls = postCalls(fetchMock);
    expect(calls).toHaveLength(1);
    const body = JSON.parse(String(calls[0][1].body));
    expect(body.answers).toHaveLength(20);
    expect(
      body.answers.filter(
        (answer: { rating: number | null }) => answer.rating !== null,
      ),
    ).toEqual([
      { internal_movie_id: 1001, rating: 9 },
      { internal_movie_id: 1002, rating: 5 },
      { internal_movie_id: 1003, rating: 1 },
    ]);
    expect(
      body.answers.filter(
        (answer: { rating: number | null }) => answer.rating === null,
      ),
    ).toHaveLength(17);
    expect(loadBatch().status).toBe("ready");
    expect(loadTasteProfileDraft()).not.toBeNull();
  });

  it("counts propositions rather than ratings and supports undo", async () => {
    installFetchMock();
    const user = await startQuiz();

    await user.click(
      screen.getByRole("radio", {
        name: "Rate 3.5 stars out of 5 (7/10)",
      }),
    );
    await screen.findByText("Taste Film 2");
    expect(screen.getByLabelText("2 / 20 movies")).toBeVisible();

    await user.click(
      screen.getByRole("button", {
        name: /Haven't Seen/,
      }),
    );
    await screen.findByText("Taste Film 3");
    expect(screen.getByLabelText("3 / 20 movies")).toBeVisible();

    await user.click(
      screen.getByRole("button", {
        name: /Undo the previous rating/,
      }),
    );
    await screen.findByText("Taste Film 2");
    expect(screen.getByLabelText("2 / 20 movies")).toBeVisible();

    await user.click(
      screen.getByRole("radio", {
        name: "Rate 2 stars out of 5 (4/10)",
      }),
    );
    await screen.findByText("Taste Film 3");
  });

  it("shows an existing library rating on the card", async () => {
    const withRating = movies.map((movie, index) =>
      index === 0 ? { ...movie, user_rating: 7 } : movie,
    );
    installFetchMock(moviesPayload({ movies: withRating }));
    await startQuiz();

    const existing = screen.getByRole("radio", {
      name: "Rate 3.5 stars out of 5 (7/10)",
    });
    expect(existing).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Your library rating")).toBeVisible();
  });

  it("ignores rapid duplicate clicks on the same film", async () => {
    installFetchMock();
    await startQuiz();

    const radio = screen.getByRole("radio", {
      name: "Rate 4.5 stars out of 5 (9/10)",
    });
    act(() => {
      radio.click();
      radio.click();
    });

    await screen.findByText("Taste Film 2");
    expect(screen.getByLabelText("2 / 20 movies")).toBeVisible();
    expect(screen.queryByText("Taste Film 3")).toBeNull();
  });

  it("rejects zero ratings explicitly without calling the backend", async () => {
    const fetchMock = installFetchMock();
    await startQuiz();

    await answerWithKeyboard([], 20);

    expect(
      await screen.findByRole("heading", { name: /No ratings yet/i }),
    ).toBeVisible();
    expect(postCalls(fetchMock)).toHaveLength(0);
  });

  it("asks for a few more ratings below the minimum", async () => {
    const fetchMock = installFetchMock();
    await startQuiz();

    await answerWithKeyboard(["9", "5"], 18);

    expect(
      await screen.findByRole("heading", {
        name: /A few more ratings would help/i,
      }),
    ).toBeVisible();
    expect(postCalls(fetchMock)).toHaveLength(0);
  });

  it("encourages more ratings but lets the user continue", async () => {
    const fetchMock = installFetchMock();
    await startQuiz();

    await answerWithKeyboard(["9", "5", "1"], 17);

    expect(
      await screen.findByRole("heading", {
        name: /Your profile is taking shape/i,
      }),
    ).toBeVisible();
    expect(postCalls(fetchMock)).toHaveLength(0);

    await userEvent
      .setup()
      .click(
        screen.getByRole("button", {
          name: /See my recommendations/,
        }),
      );

    await waitFor(() =>
      expect(navigation.push).toHaveBeenCalledWith("/discover"),
    );
    expect(postCalls(fetchMock)).toHaveLength(1);
  });

  it("accepts a draft for a signed-in account", async () => {
    auth.user = {
      id: 7,
      email: "member@example.com",
      email_verified: true,
      is_admin: false,
      created_at: "2026-01-01T00:00:00Z",
    };
    const fetchMock = installFetchMock(
      moviesPayload({ recommended_min_ratings: 3 }),
    );
    const answers = movies.map((movie, index) => ({
      internal_movie_id: movie.internal_movie_id,
      rating: index < 3 ? 9 : null,
    }));
    saveTasteProfileDraft(answers);

    const user = userEvent.setup();
    render(<TasteProfilePageContent />);
    await screen.findByRole("button", { name: /Start rating/ });
    expect(
      await screen.findByText("Save your Taste Profile to your account"),
    ).toBeVisible();

    await user.click(
      screen.getByRole("button", { name: /Save my profile/ }),
    );
    await waitFor(() =>
      expect(navigation.push).toHaveBeenCalledWith("/discover"),
    );
    const calls = postCalls(fetchMock);
    expect(calls).toHaveLength(1);
    expect(JSON.parse(String(calls[0][1].body))).toEqual({ answers });
    expect(loadBatch().status).toBe("ready");
    expect(loadTasteProfileDraft()).toBeNull();
  });
});
