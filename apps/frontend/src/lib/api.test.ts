import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  errorMessage,
  requestRecommendations,
} from "@/lib/api";
import { makeResponse } from "@/test/fixtures";

const file = new File(["zip-content"], "export.zip", {
  type: "application/zip",
});

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function abortableFetch() {
  return vi.fn((_url: string, init?: RequestInit) => {
    return new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => {
        reject(new DOMException("Aborted", "AbortError"));
      });
    });
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
  document.cookie = "ff_csrf=test-csrf; path=/";
});

afterEach(() => {
  document.cookie = "ff_csrf=; Max-Age=0; path=/";
  vi.unstubAllGlobals();
});

describe("requestRecommendations", () => {
  it("posts multipart form data to the documented endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeResponse()));
    vi.stubGlobal("fetch", fetchMock);

    const result = await requestRecommendations({ file, limit: 25 });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://localhost:8000/api/v1/recommendations/from-letterboxd?limit=25",
    );
    expect(init?.method).toBe("POST");
    expect(init?.credentials).toBe("include");
    expect(new Headers(init?.headers).get("X-CSRF-Token")).toBe("test-csrf");
    expect(init?.body).toBeInstanceOf(FormData);
    const body = init?.body as FormData;
    expect(body.get("file")).toBeInstanceOf(File);
    expect((body.get("file") as File).name).toBe("export.zip");
    expect(result.recommendations).toHaveLength(3);
    expect(result.recommendation_info.model).toBe("M5");
  });

  it("fetches a CSRF token before uploading when the cookie is missing", async () => {
    document.cookie = "ff_csrf=; Max-Age=0; path=/";
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      void init;
      if (url.endsWith("/api/v1/auth/csrf")) {
        return Promise.resolve(
          jsonResponse({ csrf_token: "server-csrf-token" }),
        );
      }
      return Promise.resolve(jsonResponse(makeResponse()));
    });
    vi.stubGlobal("fetch", fetchMock);

    await requestRecommendations({ file });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    const uploadInit = fetchMock.mock.calls[1][1] as RequestInit;
    expect(new Headers(uploadInit.headers).get("X-CSRF-Token")).toBe(
      "server-csrf-token",
    );
  });

  it("uses the default limit of 100", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(makeResponse()));
    vi.stubGlobal("fetch", fetchMock);

    await requestRecommendations({ file });

    expect(fetchMock.mock.calls[0][0]).toContain("limit=100");
  });

  it("maps backend error codes to readable errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "no_matched_ratings",
              message: "Traceback: private backend detail",
            },
          },
          422,
        ),
      ),
    );

    const error = await requestRecommendations({ file }).catch(
      (caught: unknown) => caught,
    );
    expect(error).toMatchObject({
      code: "no_matched_ratings",
      status: 422,
    });
    expect(error).toBeInstanceOf(ApiError);
    expect(errorMessage(error)).toContain("could not match");
    expect(errorMessage(error)).not.toContain("Traceback");
  });

  it("maps 503 responses without the envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("unavailable", { status: 503 })),
    );
    await expect(requestRecommendations({ file })).rejects.toMatchObject({
      code: "engine_unavailable",
    });
  });

  it("rejects empty recommendation batches", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse(makeResponse(0))),
    );
    await expect(requestRecommendations({ file })).rejects.toMatchObject({
      code: "no_recommendations",
    });
  });

  it("rejects malformed recommendation entries", async () => {
    const payload = makeResponse(1) as unknown as {
      recommendations: unknown[];
    };
    payload.recommendations = [{ title: "Missing fields" }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(payload)));

    await expect(requestRecommendations({ file })).rejects.toMatchObject({
      code: "invalid_response",
    });
  });

  it("maps network failures to backend unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    await expect(requestRecommendations({ file })).rejects.toMatchObject({
      code: "backend_unavailable",
    });
  });

  it("times out stalled requests", async () => {
    vi.stubGlobal("fetch", abortableFetch());
    await expect(
      requestRecommendations({ file, timeoutMs: 5 }),
    ).rejects.toMatchObject({ code: "timeout" });
  });

  it("supports caller cancellation", async () => {
    vi.stubGlobal("fetch", abortableFetch());
    const controller = new AbortController();
    const promise = requestRecommendations({
      file,
      signal: controller.signal,
    });
    controller.abort();

    await expect(promise).rejects.toMatchObject({ code: "cancelled" });
  });
});
