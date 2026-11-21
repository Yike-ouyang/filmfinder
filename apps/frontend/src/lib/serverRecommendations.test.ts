import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  fetchCurrentRecommendations,
  requestRecommendationRefresh,
} from "@/lib/serverRecommendations";
import { makeResponse } from "@/test/fixtures";

const request = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth", () => ({
  authenticatedRequest: request,
}));

beforeEach(() => {
  request.mockReset();
});

describe("server recommendations client", () => {
  it("loads and normalizes a ready batch", async () => {
    request.mockResolvedValue({
      status: "ready",
      batch_id: 12,
      generated_at: "2026-11-21T10:00:00Z",
      response: makeResponse(2),
    });

    const current = await fetchCurrentRecommendations();

    expect(request).toHaveBeenCalledWith(
      "/api/v1/recommendations/current",
    );
    expect(current.status).toBe("ready");
    expect(current.batchId).toBe(12);
    expect(current.response?.recommendations).toHaveLength(2);
  });

  it("supports a pending batch", async () => {
    request.mockResolvedValue({
      status: "pending",
      batch_id: 13,
      generated_at: null,
      response: null,
    });

    const current = await fetchCurrentRecommendations();

    expect(current.status).toBe("pending");
    expect(current.batchId).toBe(13);
    expect(current.response).toBeNull();
  });

  it("exposes a failed batch", async () => {
    request.mockResolvedValue({
      status: "failed",
      batch_id: 14,
      generated_at: null,
      response: null,
    });

    const current = await fetchCurrentRecommendations();

    expect(current.status).toBe("failed");
    expect(current.batchId).toBe(14);
  });

  it("rejects incomplete ready payloads", async () => {
    request.mockResolvedValue({
      status: "ready",
      batch_id: null,
      generated_at: null,
      response: null,
    });

    await expect(fetchCurrentRecommendations()).rejects.toMatchObject({
      code: "invalid_response",
    });
  });

  it("requests a refresh and returns its status", async () => {
    request.mockResolvedValueOnce({ status: "pending" });
    request.mockResolvedValueOnce({ status: "no_new_candidates" });

    await expect(requestRecommendationRefresh()).resolves.toBe("pending");
    await expect(requestRecommendationRefresh()).resolves.toBe(
      "no_new_candidates",
    );
    expect(request).toHaveBeenCalledWith(
      "/api/v1/recommendations/refresh",
      { method: "POST" },
    );
  });
});
