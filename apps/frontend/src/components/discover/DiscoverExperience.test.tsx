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
  it("renders the batch in order and updates the real counter", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<DiscoverExperience initialBatch={makeBatch(2)} />);

    expect(screen.getByText("Film One")).toBeVisible();
    expect(screen.getByLabelText("Recommendation 1 of 2")).toBeVisible();

    await user.click(screen.getByRole("button", { name: /Pass/ }));

    await waitFor(() => expect(screen.getByText("Film Two")).toBeVisible());
    expect(screen.getByLabelText("Recommendation 2 of 2")).toBeVisible();
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
    expect(screen.getByLabelText("Recommendation 1 of 2")).toBeVisible();
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
        name: /reached the end of your discoveries/i,
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
