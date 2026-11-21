import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  decisionToAction,
  fetchSwipeLists,
  removeSwipe,
  saveSwipe,
  undoSwipe,
} from "@/lib/swipes";

const request = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth", () => ({
  authenticatedRequest: request,
}));

const SWIPE = {
  internal_movie_id: 42,
  movie_id: "movie-42",
  title: "Film Forty Two",
  year: 1999,
  genres: ["Drama"],
  tmdb_id: 100,
  poster_path: "/poster.jpg",
  backdrop_path: "/backdrop.jpg",
  action: "interested",
  updated_at: "2026-11-21T10:00:00Z",
};

beforeEach(() => {
  request.mockReset();
});

describe("swipes client", () => {
  it("maps decision kinds to API actions", () => {
    expect(decisionToAction("interested")).toBe("interested");
    expect(decisionToAction("passed")).toBe("not_interested");
    expect(decisionToAction("seen")).toBe("already_seen");
  });

  it("loads and normalizes both lists", async () => {
    request.mockResolvedValue({
      interested: [SWIPE],
      not_interested: [
        { ...SWIPE, internal_movie_id: 43, action: "not_interested" },
      ],
      counts: { interested: 1, not_interested: 1 },
    });

    const lists = await fetchSwipeLists();

    expect(request).toHaveBeenCalledWith("/api/v1/swipes");
    expect(lists.interested[0].title).toBe("Film Forty Two");
    expect(lists.not_interested[0].action).toBe("not_interested");
    expect(lists.counts).toEqual({ interested: 1, not_interested: 1 });
  });

  it("saves a swipe with PUT", async () => {
    request.mockResolvedValue({ swipe: SWIPE, event_id: 7 });

    const result = await saveSwipe(42, "interested");

    expect(request).toHaveBeenCalledWith("/api/v1/swipes/42", {
      method: "PUT",
      body: JSON.stringify({ action: "interested" }),
    });
    expect(result.event_id).toBe(7);
    expect(result.swipe.internal_movie_id).toBe(42);
  });

  it("undoes and removes a swipe", async () => {
    request.mockResolvedValueOnce({ swipe: null });
    request.mockResolvedValueOnce(null);

    await expect(undoSwipe(42)).resolves.toEqual({ swipe: null });
    await removeSwipe(42);

    expect(request).toHaveBeenNthCalledWith(
      1,
      "/api/v1/swipes/42/undo",
      { method: "POST" },
    );
    expect(request).toHaveBeenNthCalledWith(
      2,
      "/api/v1/swipes/42",
      { method: "DELETE" },
    );
  });

  it("rejects malformed list payloads", async () => {
    request.mockResolvedValue({ interested: "nope" });

    await expect(fetchSwipeLists()).rejects.toMatchObject({
      code: "invalid_response",
    });
  });
});
