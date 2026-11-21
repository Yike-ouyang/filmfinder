import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AccountPage from "@/app/account/page";

const navigation = vi.hoisted(() => ({ push: vi.fn() }));
const authState = vi.hoisted(() => ({
  user: null as null | {
    id: number;
    email: string;
    email_verified: boolean;
    created_at: string;
  },
  loading: false,
  signOut: vi.fn(),
  removeAccount: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: authState.user,
    loading: authState.loading,
    signIn: vi.fn(),
    signOut: authState.signOut,
    removeAccount: authState.removeAccount,
    refresh: vi.fn(),
  }),
}));

beforeEach(() => {
  navigation.push.mockReset();
  authState.user = null;
  authState.loading = false;
  authState.signOut.mockReset();
  authState.removeAccount.mockReset();
});

describe("AccountPage", () => {
  it("asks anonymous visitors to sign in", () => {
    render(<AccountPage />);

    expect(
      screen.getByRole("heading", { name: "Sign in required" }),
    ).toBeVisible();
    expect(screen.getByRole("link", { name: "Log in" })).toHaveAttribute(
      "href",
      "/login",
    );
  });

  it("shows the account and logs out", async () => {
    const user = userEvent.setup();
    authState.user = {
      id: 1,
      email: "user@example.com",
      email_verified: true,
      created_at: "2026-11-21T10:00:00Z",
    };
    authState.signOut.mockResolvedValue(undefined);
    render(<AccountPage />);

    expect(screen.getByText("user@example.com")).toBeVisible();
    expect(screen.getByText("Email verified")).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Log out" }));

    await waitFor(() => expect(authState.signOut).toHaveBeenCalledTimes(1));
    expect(navigation.push).toHaveBeenCalledWith("/");
  });

  it("deletes the account with the password", async () => {
    const user = userEvent.setup();
    authState.user = {
      id: 1,
      email: "user@example.com",
      email_verified: true,
      created_at: "2026-11-21T10:00:00Z",
    };
    authState.removeAccount.mockResolvedValue(undefined);
    render(<AccountPage />);

    await user.type(
      screen.getByLabelText("Confirm with your password"),
      "correct horse battery",
    );
    await user.click(
      screen.getByRole("button", { name: "Delete account permanently" }),
    );

    await waitFor(() =>
      expect(authState.removeAccount).toHaveBeenCalledWith(
        "correct horse battery",
      ),
    );
    expect(navigation.push).toHaveBeenCalledWith("/");
  });
});
