import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TasteProfilePageContent } from "@/components/taste-profile/TasteProfilePageContent";
import { loadBatch } from "@/lib/storage";
import { saveTasteProfileDraft } from "@/lib/tasteProfileDraft";
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
  }),
);

function moviesPayload() {
  return {
    movies,
    min_ratings: 3,
    rating_scale: { loved: 8, okay: 5, disliked: 3 },
  };
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetchMock() {
  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    void init;
    if (url.endsWith("/api/v1/taste-profile/movies")) {
      return Promise.resolve(jsonResponse(moviesPayload()));
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

async function choose(
  user: ReturnType<typeof userEvent.setup>,
  label: string,
  nextTitle?: string,
) {
  await user.click(screen.getByRole("button", { name: new RegExp(label) }));
  if (nextTitle) {
    await screen.findByText(nextTitle);
  }
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
  vi.unstubAllGlobals();
});

describe("TasteProfilePageContent", () => {
  it("submits three loved films and seventeen no-opinion answers", async () => {
    const fetchMock = installFetchMock();
    const user = await startQuiz();

    await choose(user, "Loved it", "Taste Film 2");
    await choose(user, "Loved it", "Taste Film 3");
    await choose(user, "Loved it", "Taste Film 4");
    for (let index = 4; index <= 19; index += 1) {
      await choose(
        user,
        "No opinion",
        index < 20 ? `Taste Film ${index + 1}` : undefined,
      );
    }
    await choose(user, "No opinion");

    await waitFor(() =>
      expect(navigation.push).toHaveBeenCalledWith("/discover"),
    );

    const postCall = fetchMock.mock.calls.find(([url]) =>
      String(url).includes("/api/v1/taste-profile/recommendations"),
    ) as [string, RequestInit] | undefined;
    expect(postCall).toBeDefined();
    const body = JSON.parse(String(postCall?.[1]?.body));
    expect(body.answers).toHaveLength(20);
    expect(
      body.answers.filter(
        (answer: { rating: number | null }) => answer.rating === 8,
      ),
    ).toHaveLength(3);
    expect(
      body.answers.filter(
        (answer: { rating: number | null }) => answer.rating === null,
      ),
    ).toHaveLength(17);
    expect(loadBatch().status).toBe("ready");
  });

  it("undoes and replaces an answer without losing progress", async () => {
    installFetchMock();
    const user = await startQuiz();

    await choose(user, "Loved it", "Taste Film 2");
    expect(screen.getByLabelText("2 / 20 movies")).toBeVisible();

    await user.click(
      screen.getByRole("button", {
        name: /Undo the previous rating/,
      }),
    );
    await screen.findByText("Taste Film 1");
    expect(screen.getByLabelText("1 / 20 movies")).toBeVisible();

    await choose(user, "It was okay", "Taste Film 2");
    expect(screen.getByLabelText("2 / 20 movies")).toBeVisible();
  });

  it("blocks submission when fewer than three films are rated", async () => {
    const fetchMock = installFetchMock();
    const user = await startQuiz();

    for (let index = 1; index <= 20; index += 1) {
      await choose(
        user,
        "No opinion",
        index < 20 ? `Taste Film ${index + 1}` : undefined,
      );
    }

    expect(
      await screen.findByRole("heading", {
        name: /A few more ratings would help/i,
      }),
    ).toBeVisible();
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/taste-profile/recommendations"),
      ),
    ).toBe(false);

    await user.click(
      screen.getByRole("button", { name: /Review my answers/ }),
    );
    expect(await screen.findByText("Taste Film 20")).toBeVisible();
  });

  it("saves an anonymous draft to a signed-in account", async () => {
    auth.user = {
      id: 7,
      email: "member@example.com",
      email_verified: true,
      created_at: "2026-01-01T00:00:00Z",
    };
    const fetchMock = installFetchMock();
    const answers = movies.map((movie, index) => ({
      internal_movie_id: movie.internal_movie_id,
      rating: (index < 3 ? 8 : null) as 8 | null,
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
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/taste-profile/recommendations"),
      ),
    ).toBe(true);
    expect(loadBatch().status).toBe("ready");
  });

  it("supports keyboard shortcuts", async () => {
    installFetchMock();
    await startQuiz();

    fireEvent.keyDown(window, { key: "1" });
    expect(await screen.findByText("Taste Film 2")).toBeVisible();

    fireEvent.keyDown(window, { key: "Backspace" });
    expect(await screen.findByText("Taste Film 1")).toBeVisible();
  });
});
