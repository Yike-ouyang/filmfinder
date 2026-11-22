import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ListsPageContent } from "@/components/lists/ListsPageContent";

const authState = vi.hoisted(() => ({
  user: null as null | {
    id: number;
    email: string;
    email_verified: boolean;
    created_at: string;
  },
  loading: false,
}));
const swipesApi = vi.hoisted(() => ({
  fetchSwipeLists: vi.fn(),
  removeSwipe: vi.fn(),
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: authState.user,
    loading: authState.loading,
    signIn: vi.fn(),
    signOut: vi.fn(),
    removeAccount: vi.fn(),
    refresh: vi.fn(),
  }),
}));

vi.mock("@/lib/swipes", () => swipesApi);

function item(
  id: number,
  title: string,
  action: "interested" | "not_interested",
) {
  return {
    internal_movie_id: id,
    movie_id: `movie-${id}`,
    title,
    year: 1990 + id,
    genres: ["Drama"],
    tmdb_id: 1000 + id,
    poster_path: `/poster-${id}.jpg`,
    backdrop_path: null,
    action,
    updated_at: "2026-11-21T10:00:00Z",
  };
}

const LISTS = {
  interested: [item(1, "Film One", "interested")],
  not_interested: [item(2, "Film Two", "not_interested")],
  counts: { interested: 1, not_interested: 1 },
};

beforeEach(() => {
  authState.user = null;
  authState.loading = false;
  swipesApi.fetchSwipeLists.mockReset();
  swipesApi.removeSwipe.mockReset();
});

describe("ListsPageContent", () => {
  it("asks anonymous visitors to create an account or log in", () => {
    render(<ListsPageContent />);

    expect(
      screen.getByRole("link", { name: "Create an account" }),
    ).toHaveAttribute("href", "/signup");
    expect(screen.getByRole("link", { name: "Log in" })).toHaveAttribute(
      "href",
      "/login",
    );
    expect(swipesApi.fetchSwipeLists).not.toHaveBeenCalled();
  });

  it("shows interested and not interested tabs", async () => {
    const user = userEvent.setup();
    authState.user = {
      id: 1,
      email: "user@example.com",
      email_verified: true,
      created_at: "2026-11-21T10:00:00Z",
    };
    swipesApi.fetchSwipeLists.mockResolvedValue(LISTS);
    render(<ListsPageContent />);

    expect(await screen.findByText("Film One")).toBeVisible();
    expect(screen.getByRole("tab", { name: /^Watchlist/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await user.click(screen.getByRole("tab", { name: /Do Not Watch/ }));

    expect(screen.getByText("Film Two")).toBeVisible();
    expect(screen.queryByText("Film One")).not.toBeInTheDocument();
  });

  it("removes a film from the active list", async () => {
    const user = userEvent.setup();
    authState.user = {
      id: 1,
      email: "user@example.com",
      email_verified: true,
      created_at: "2026-11-21T10:00:00Z",
    };
    swipesApi.fetchSwipeLists.mockResolvedValue(LISTS);
    swipesApi.removeSwipe.mockResolvedValue(undefined);
    render(<ListsPageContent />);

    await screen.findByText("Film One");
    await user.click(
      screen.getByRole("button", {
        name: "Remove Film One from this list",
      }),
    );

    await waitFor(() =>
      expect(swipesApi.removeSwipe).toHaveBeenCalledWith(1),
    );
    await waitFor(() =>
      expect(screen.queryByText("Film One")).not.toBeInTheDocument(),
    );
  });
});
