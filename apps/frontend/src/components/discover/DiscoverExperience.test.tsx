import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DiscoverExperience } from "@/components/discover/DiscoverExperience";
import { loadBatch } from "@/lib/storage";
import { makeBatch } from "@/test/fixtures";

const navigation = vi.hoisted(() => ({
  push: vi.fn(),
  prefetch: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
}));

beforeEach(() => {
  navigation.push.mockReset();
  navigation.prefetch.mockReset();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("DiscoverExperience", () => {
  it("renders the batch in order without a counter", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<DiscoverExperience initialBatch={makeBatch(2)} />);

    expect(screen.getByText("Film One")).toBeVisible();
    expect(screen.queryByText(/Recommendation \d+ of \d+/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Pass/ }));

    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());
    expect(screen.queryByText(/Recommendation \d+ of \d+/)).not.toBeInTheDocument();
    await waitFor(() =>
      expect(loadBatch().batch?.decisions).toEqual([
        { key: "1001", kind: "passed" },
      ]),
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("undoes the last action and restores the previous card", async () => {
    const user = userEvent.setup();
    render(<DiscoverExperience initialBatch={makeBatch(2)} />);

    await user.click(screen.getByRole("button", { name: /Interested/ }));
    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());

    await user.click(screen.getByRole("button", { name: /Undo/ }));

    await waitFor(() => expect(screen.getByText("Film One")).toBeVisible());
    await waitFor(() =>
      expect(loadBatch().batch?.decisions).toHaveLength(0),
    );
  });

  it("supports keyboard shortcuts outside form fields", async () => {
    render(<DiscoverExperience initialBatch={makeBatch(2)} />);

    fireEvent.keyDown(window, { key: "ArrowRight" });

    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());
  });

  it("supports Backspace undo and Down arrow for already seen", async () => {
    const user = userEvent.setup();
    render(<DiscoverExperience initialBatch={makeBatch(2)} />);

    await user.click(screen.getByRole("button", { name: /Pass/ }));
    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());

    fireEvent.keyDown(window, { key: "Backspace" });
    await waitFor(() => expect(screen.getByText("Film One")).toBeVisible());

    fireEvent.keyDown(window, { key: "ArrowDown" });
    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());
    await waitFor(() =>
      expect(loadBatch().batch?.decisions).toEqual([
        { key: "1001", kind: "seen" },
      ]),
    );
  });

  it("shows the end screen and clears the session for a new import", async () => {
    const user = userEvent.setup();
    render(<DiscoverExperience initialBatch={makeBatch(1)} />);

    await user.click(screen.getByRole("button", { name: /Pass/ }));

    expect(
      await screen.findByRole("heading", {
        name: /reached the end of these discoveries/i,
      }),
    ).toBeVisible();
    expect(screen.getByText("Passed")).toBeVisible();

    await user.click(
      screen.getByRole("button", { name: "Import another profile" }),
    );

    expect(navigation.push).toHaveBeenCalledWith("/");
    expect(loadBatch().status).toBe("missing");
  });

  it("opens accessible details and links to TMDB", async () => {
    const user = userEvent.setup();
    render(<DiscoverExperience initialBatch={makeBatch(1)} />);

    await user.click(
      screen.getByRole("button", { name: "View details for Film One" }),
    );

    const dialog = await screen.findByRole("dialog");
    expect(
      within(dialog).getByRole("heading", { name: "Film One" }),
    ).toBeVisible();
    const tmdbLink = within(dialog).getByRole("link", {
      name: /View on TMDB/,
    });
    expect(tmdbLink).toHaveAttribute(
      "href",
      "https://www.themoviedb.org/movie/5001",
    );

    await user.click(within(dialog).getByRole("button", { name: "Close dialog" }));
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    );
  });

  it("falls back gracefully when a poster is missing", () => {
    const batch = makeBatch(1);
    batch.response.recommendations[0].poster_path = null;
    batch.response.recommendations[0].backdrop_path = null;

    render(<DiscoverExperience initialBatch={batch} />);

    expect(screen.getAllByText("Film One").length).toBeGreaterThanOrEqual(2);
  });
});

describe("DiscoverExperience persistence", () => {
  it("persists a decision before advancing the card", async () => {
    const user = userEvent.setup();
    const persistDecision = vi.fn().mockResolvedValue(undefined);
    render(
      <DiscoverExperience
        initialBatch={makeBatch(2)}
        persistDecision={persistDecision}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Interested/ }));

    await waitFor(() =>
      expect(persistDecision).toHaveBeenCalledWith(
        "interested",
        expect.objectContaining({ title: "Film One" }),
      ),
    );
    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());
  });

  it("keeps the card in place when persistence fails", async () => {
    const user = userEvent.setup();
    const persistDecision = vi.fn().mockRejectedValue(new Error("offline"));
    render(
      <DiscoverExperience
        initialBatch={makeBatch(2)}
        persistDecision={persistDecision}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Pass/ }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Something went wrong",
      ),
    );
    expect(screen.getByText("Film One")).toBeVisible();
  });

  it("persists undo before restoring the previous card", async () => {
    const user = userEvent.setup();
    const persistDecision = vi.fn().mockResolvedValue(undefined);
    const persistUndo = vi.fn().mockResolvedValue(undefined);
    render(
      <DiscoverExperience
        initialBatch={makeBatch(2)}
        persistDecision={persistDecision}
        persistUndo={persistUndo}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Interested/ }));
    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());

    await user.click(screen.getByRole("button", { name: /Undo/ }));

    await waitFor(() =>
      expect(persistUndo).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Film One" }),
      ),
    );
    await waitFor(() => expect(screen.getByText("Film One")).toBeVisible());
  });
});

describe("DiscoverExperience end of set", () => {
  it("offers more recommendations and reports no new candidates", async () => {
    const user = userEvent.setup();
    const onRequestMore = vi.fn().mockResolvedValue({ status: "no_new" });
    render(
      <DiscoverExperience
        initialBatch={makeBatch(1)}
        onRequestMore={onRequestMore}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Pass/ }));
    const more = await screen.findByRole("button", {
      name: "Get more recommendations",
    });
    await user.click(more);

    await waitFor(() =>
      expect(onRequestMore).toHaveBeenCalledTimes(1),
    );
    expect(
      await screen.findByText(/No new recommendations right now/),
    ).toBeVisible();
  });

  it("does not show the refresh action for anonymous sessions", async () => {
    const user = userEvent.setup();
    render(<DiscoverExperience initialBatch={makeBatch(1)} />);

    await user.click(screen.getByRole("button", { name: /Pass/ }));

    expect(
      screen.queryByRole("button", { name: "Get more recommendations" }),
    ).not.toBeInTheDocument();
  });
});
