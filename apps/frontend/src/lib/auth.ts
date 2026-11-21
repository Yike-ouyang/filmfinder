import { API_BASE_URL, ApiError, getCsrfToken } from "@/lib/api";
import type {
  AuthUser,
  CsrfResponse,
  MessageResponse,
  SignupResponse,
} from "@/types/auth";

const AUTH_BASE = `${API_BASE_URL}/api/v1/auth`;
const UNSAFE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeUser(value: unknown): AuthUser {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  const id = value.id;
  const email = value.email;
  const emailVerified = value.email_verified;
  const createdAt = value.created_at;
  if (
    typeof id !== "number" ||
    typeof email !== "string" ||
    typeof emailVerified !== "boolean" ||
    typeof createdAt !== "string"
  ) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    id,
    email,
    email_verified: emailVerified,
    created_at: createdAt,
  };
}

async function readError(response: Response): Promise<ApiError> {
  let code = "backend_error";
  try {
    const payload: unknown = await response.json();
    if (isRecord(payload) && isRecord(payload.error)) {
      const value = payload.error.code;
      if (typeof value === "string") {
        code = value;
      }
    }
  } catch {
    code = response.status === 401 ? "not_authenticated" : "backend_error";
  }
  return new ApiError(code, response.status);
}

export async function authenticatedRequest(
  path: string,
  init: RequestInit = {},
): Promise<unknown> {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }
  if (UNSAFE_METHODS.has(method)) {
    headers.set("X-CSRF-Token", await getCsrfToken());
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      method,
      headers,
      credentials: "include",
      cache: "no-store",
    });
  } catch {
    throw new ApiError("backend_unavailable", null);
  }

  if (!response.ok) {
    throw await readError(response);
  }
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    throw new ApiError("invalid_response", response.status);
  }
  return response.json();
}

function authFetch(
  path: string,
  init: RequestInit = {},
): Promise<unknown> {
  return authenticatedRequest(`/api/v1/auth${path}`, init);
}

export async function signup(
  email: string,
  password: string,
): Promise<SignupResponse> {
  const payload = await authFetch("/signup", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  if (
    !isRecord(payload) ||
    typeof payload.message !== "string" ||
    typeof payload.verification_required !== "boolean"
  ) {
    throw new ApiError("invalid_response", 201);
  }
  return {
    message: payload.message,
    verification_required: payload.verification_required,
  };
}

export async function login(
  email: string,
  password: string,
): Promise<AuthUser> {
  return normalizeUser(
    await authFetch("/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  );
}

export async function logout(): Promise<void> {
  await authFetch("/logout", { method: "POST" });
}

export async function fetchCurrentUser(): Promise<AuthUser | null> {
  let response: Response;
  try {
    response = await fetch(`${AUTH_BASE}/me`, {
      method: "GET",
      headers: { Accept: "application/json" },
      credentials: "include",
      cache: "no-store",
    });
  } catch {
    throw new ApiError("backend_unavailable", null);
  }
  if (response.status === 401) {
    return null;
  }
  if (!response.ok) {
    throw await readError(response);
  }
  return normalizeUser(await response.json());
}

export async function verifyEmail(token: string): Promise<MessageResponse> {
  return messageResponse(
    await authFetch("/verify-email", {
      method: "POST",
      body: JSON.stringify({ token }),
    }),
  );
}

export async function resendVerification(
  email: string,
): Promise<MessageResponse> {
  return messageResponse(
    await authFetch("/resend-verification", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),
  );
}

export async function forgotPassword(
  email: string,
): Promise<MessageResponse> {
  return messageResponse(
    await authFetch("/forgot-password", {
      method: "POST",
      body: JSON.stringify({ email }),
    }),
  );
}

export async function resetPassword(
  token: string,
  password: string,
): Promise<MessageResponse> {
  return messageResponse(
    await authFetch("/reset-password", {
      method: "POST",
      body: JSON.stringify({ token, password }),
    }),
  );
}

export async function deleteAccount(password: string): Promise<void> {
  await authFetch("/account", {
    method: "DELETE",
    body: JSON.stringify({ password }),
  });
}

function messageResponse(payload: unknown): MessageResponse {
  if (!isRecord(payload) || typeof payload.message !== "string") {
    throw new ApiError("invalid_response", 200);
  }
  return { message: payload.message };
}

export type { CsrfResponse };
export { getCsrfToken };
