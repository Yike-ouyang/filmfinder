import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LoginPage from "@/app/login/page";
import { ApiError } from "@/lib/api";

const navigation = vi.hoisted(() => ({ push: vi.fn() }));
const auth = vi.hoisted(() => ({ signIn: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => navigation,
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: null,
    loading: false,
    signIn: auth.signIn,
    signOut: vi.fn(),
    removeAccount: vi.fn(),
    refresh: vi.fn(),
  }),
}));

vi.mock("@/lib/auth", () => ({
  resendVerification: vi.fn().mockResolvedValue({ message: "ok" }),
}));

beforeEach(() => {
  navigation.push.mockReset();
  auth.signIn.mockReset();
});

describe("LoginPage", () => {
  it("signs in and redirects to discover", async () => {
    const user = userEvent.setup();
    auth.signIn.mockResolvedValue({
      id: 1,
      email: "user@example.com",
      email_verified: true,
      created_at: "2026-11-21T10:00:00Z",
    });
    render(<LoginPage />);

    await user.type(screen.getByLabelText("Email"), "user@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Log in" }));

    await waitFor(() =>
      expect(auth.signIn).toHaveBeenCalledWith(
        "user@example.com",
        "correct horse battery",
      ),
    );
    expect(navigation.push).toHaveBeenCalledWith("/discover");
  });

  it("offers a resend action when the email is not verified", async () => {
    const user = userEvent.setup();
    auth.signIn.mockRejectedValue(new ApiError("email_not_verified", 403));
    render(<LoginPage />);

    await user.type(screen.getByLabelText("Email"), "user@example.com");
    await user.type(screen.getByLabelText("Password"), "correct horse battery");
    await user.click(screen.getByRole("button", { name: "Log in" }));

    expect(
      await screen.findByRole("button", {
        name: "Resend verification email",
      }),
    ).toBeVisible();
    expect(
      screen.getByText(/Verify your email address/),
    ).toBeVisible();
  });
});
