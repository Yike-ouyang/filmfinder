import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SignupPage from "@/app/signup/page";

const authApi = vi.hoisted(() => ({
  signup: vi.fn(),
  resendVerification: vi.fn(),
}));

vi.mock("@/lib/auth", () => authApi);

async function fillAndSubmit() {
  const user = userEvent.setup();
  render(<SignupPage />);
  await user.type(screen.getByLabelText("Email"), "user@example.com");
  await user.type(screen.getByLabelText(/^Password/), "correct horse battery");
  await user.type(
    screen.getByLabelText(/^Confirm password/),
    "correct horse battery",
  );
  await user.click(screen.getByRole("button", { name: "Create account" }));
  return user;
}

beforeEach(() => {
  authApi.signup.mockReset();
  authApi.resendVerification.mockReset();
});

describe("SignupPage", () => {
  it("shows the account created screen when verification is disabled", async () => {
    authApi.signup.mockResolvedValue({
      message: "Your account is ready. You can now log in.",
      verification_required: false,
    });

    await fillAndSubmit();

    expect(
      await screen.findByRole("heading", { name: "Account created" }),
    ).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("Your account is ready");
    const loginLinks = screen.getAllByRole("link", { name: "Log in" });
    expect(loginLinks.length).toBeGreaterThan(0);
    expect(loginLinks[0]).toHaveAttribute("href", "/login");
    expect(
      screen.queryByRole("button", { name: "Resend verification email" }),
    ).not.toBeInTheDocument();
  });

  it("shows the verification screen when required", async () => {
    authApi.signup.mockResolvedValue({
      message: "If this email can be used, a verification link has been sent.",
      verification_required: true,
    });

    await fillAndSubmit();

    expect(
      await screen.findByRole("heading", { name: "Check your inbox" }),
    ).toBeVisible();
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Resend verification email" }),
      ).toBeVisible(),
    );
  });
});
