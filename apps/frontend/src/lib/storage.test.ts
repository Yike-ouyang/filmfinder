import { afterEach, describe, expect, it, vi } from "vitest";
import { SESSION_TTL_MS, STORAGE_KEY } from "@/lib/constants";
import { makeBatch, makeResponse } from "@/test/fixtures";
import {
  clearBatch,
  loadBatch,
  saveBatch,
  updateDecisions,
} from "@/lib/storage";

afterEach(() => {
  vi.useRealTimers();
});

describe("recommendation storage", () => {
  it("returns missing when no batch exists", () => {
    expect(loadBatch()).toEqual({ status: "missing", batch: null });
  });

  it("preserves the server batch id and decisions", () => {
    const response = makeResponse(2);
    const batch = makeBatch(2);
    expect(
      saveBatch(response, batch.decisions, batch.savedAt, 42),
    ).toBe(true);

    const loaded = loadBatch();
    expect(loaded.batch?.batchId).toBe(42);
    expect(
      updateDecisions(
        { ...batch, batchId: 42 },
        [{ key: "1001", kind: "interested" }],
      ),
    ).toBe(true);
    expect(loadBatch().batch?.batchId).toBe(42);
  });

  it("saves and restores a batch with decisions", () => {
    const response = makeResponse(3);
    expect(saveBatch(response, [{ key: "1001", kind: "interested" }])).toBe(
      true,
    );

    const loaded = loadBatch();
    expect(loaded.status).toBe("ready");
    expect(loaded.batch?.response.recommendations).toHaveLength(3);
    expect(loaded.batch?.decisions).toEqual([
      { key: "1001", kind: "interested" },
    ]);
  });

  it("trims decisions that do not form a valid prefix", () => {
    saveBatch(makeResponse(2), [
      { key: "1001", kind: "interested" },
      { key: "nope", kind: "passed" },
    ]);
    const loaded = loadBatch();
    expect(loaded.batch?.decisions).toEqual([
      { key: "1001", kind: "interested" },
    ]);
  });

  it("updates decisions without changing the saved timestamp", () => {
    const batch = makeBatch(2);
    saveBatch(batch.response, [], batch.savedAt);
    expect(
      updateDecisions(batch, [{ key: "1001", kind: "seen" }]),
    ).toBe(true);
    expect(loadBatch().batch?.savedAt).toBe(batch.savedAt);
    expect(loadBatch().batch?.decisions).toHaveLength(1);
  });

  it("expires stale sessions and clears them", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"));
    saveBatch(makeResponse(1), [], Date.now());
    vi.setSystemTime(Date.now() + SESSION_TTL_MS + 1);

    expect(loadBatch()).toEqual({ status: "expired", batch: null });
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("rejects invalid JSON and incomplete payloads", () => {
    window.sessionStorage.setItem(STORAGE_KEY, "not-json");
    expect(loadBatch().status).toBe("invalid");

    saveBatch(makeResponse(1));
    const raw = JSON.parse(window.sessionStorage.getItem(STORAGE_KEY) ?? "{}");
    raw.response.recommendations = [];
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(raw));
    expect(loadBatch().status).toBe("invalid");
  });

  it("clears the stored batch on demand", () => {
    saveBatch(makeResponse(1));
    clearBatch();
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});
