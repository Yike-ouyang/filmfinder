import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchCurrentUser,
  forgotPassword,
  getCsrfToken,
  login,
  resetPassword,
  signup,
  verifyEmail,
} from "@/lib/auth";

const USER = {
  id: 1,
  email: "user@example.com",
  email_verified: true,
  is_admin: false,
  created_at: "2026-11-21T10:00:00Z",
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function setCsrfCookie(token: string) {
  document.cookie = `ff_csrf=${token}; path=/`;
}

afterEach(() => {
  document.cookie = "ff_csrf=; Max-Age=0; path=/";
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.unstubAllGlobals();
});

describe("auth client", () => {
  it("reads the CSRF token from the cookie", async () => {
    setCsrfCookie("cookie-token");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(getCsrfToken()).resolves.toBe("cookie-token");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("fetches a CSRF token when the cookie is missing", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse({ csrf_token: "server-token" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getCsrfToken()).resolves.toBe("server-token");
    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://localhost:8000/api/v1/auth/csrf",
    );
    expect(fetchMock.mock.calls[0][1]).toMatchObject({
      credentials: "include",
    });
  });

  it("sends login credentials with CSRF and session cookies", async () => {
    setCsrfCookie("csrf-login");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(USER));
    vi.stubGlobal("fetch", fetchMock);

    const user = await login("user@example.com", "correct horse battery");

    expect(user.email).toBe("user@example.com");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/v1/auth/login");
    expect(init).toMatchObject({
      method: "POST",
      credentials: "include",
      cache: "no-store",
    });
    expect(new Headers(init?.headers).get("X-CSRF-Token")).toBe("csrf-login");
    expect(JSON.parse(String(init?.body))).toEqual({
      email: "user@example.com",
      password: "correct horse battery",
    });
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
  });

  it("maps backend authentication errors", async () => {
    setCsrfCookie("csrf-error");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          {
            error: {
              code: "invalid_credentials",
              message: "backend detail",
            },
          },
          401,
        ),
      ),
    );

    await expect(login("user@example.com", "wrong")).rejects.toMatchObject({
      code: "invalid_credentials",
      status: 401,
    });
  });

  it("normalizes signup, verification and reset messages", async () => {
    setCsrfCookie("csrf-messages");
    const fetchMock = vi.fn().mockImplementation(() =>
      Promise.resolve(
        jsonResponse({ message: "ok", verification_required: false }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      signup("user@example.com", "correct horse battery"),
    ).resolves.toEqual({ message: "ok", verification_required: false });
    await expect(verifyEmail("verification-token-value")).resolves.toEqual({
      message: "ok",
    });
    await expect(forgotPassword("user@example.com")).resolves.toEqual({
      message: "ok",
    });
    await expect(
      resetPassword("reset-token-value", "new correct horse battery"),
    ).resolves.toEqual({ message: "ok" });
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("returns null for an anonymous session", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 401 })),
    );

    await expect(fetchCurrentUser()).resolves.toBeNull();
  });

  it("maps network failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    await expect(fetchCurrentUser()).rejects.toMatchObject({
      code: "backend_unavailable",
    });
  });
});
