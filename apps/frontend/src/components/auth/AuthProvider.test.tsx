import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  AuthProvider,
  useAuth,
} from "@/components/auth/AuthProvider";

const authApi = vi.hoisted(() => ({
  fetchCurrentUser: vi.fn(),
  login: vi.fn(),
  logout: vi.fn(),
  deleteAccount: vi.fn(),
}));

vi.mock("@/lib/auth", () => authApi);

const USER = {
  id: 1,
  email: "user@example.com",
  email_verified: true,
  created_at: "2026-11-21T10:00:00Z",
};

function Probe() {
  const { user, loading, signIn, signOut } = useAuth();
  return (
    <div>
      <p>{loading ? "loading" : user ? user.email : "anonymous"}</p>
      <button type="button" onClick={() => signIn("user@example.com", "pass")}>
        Sign in
      </button>
      <button type="button" onClick={() => signOut()}>
        Sign out
      </button>
    </div>
  );
}

beforeEach(() => {
  authApi.fetchCurrentUser.mockReset();
  authApi.login.mockReset();
  authApi.logout.mockReset();
  authApi.deleteAccount.mockReset();
});

describe("AuthProvider", () => {
  it("restores an existing session", async () => {
    authApi.fetchCurrentUser.mockResolvedValue(USER);
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );

    expect(await screen.findByText("user@example.com")).toBeVisible();
  });

  it("exposes an anonymous state", async () => {
    authApi.fetchCurrentUser.mockResolvedValue(null);
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );

    expect(await screen.findByText("anonymous")).toBeVisible();
  });

  it("updates state after sign in and sign out", async () => {
    const user = userEvent.setup();
    authApi.fetchCurrentUser.mockResolvedValue(null);
    authApi.login.mockResolvedValue(USER);
    authApi.logout.mockResolvedValue(undefined);
    render(
      <AuthProvider>
        <Probe />
      </AuthProvider>,
    );

    await screen.findByText("anonymous");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    await waitFor(() =>
      expect(screen.getByText("user@example.com")).toBeVisible(),
    );

    await user.click(screen.getByRole("button", { name: "Sign out" }));
    await waitFor(() => expect(screen.getByText("anonymous")).toBeVisible());
  });
});
