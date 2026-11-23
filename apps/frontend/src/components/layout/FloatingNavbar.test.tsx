import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FloatingNavbar } from "@/components/layout/FloatingNavbar";

const navigation = vi.hoisted(() => ({
  pathname: "/",
  push: vi.fn(),
}));
const authState = vi.hoisted(() => ({
  user: null as null | { email: string; is_admin?: boolean },
  loading: false,
  signOut: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => navigation.pathname,
  useRouter: () => ({ push: navigation.push }),
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: authState.user,
    loading: authState.loading,
    signIn: vi.fn(),
    signOut: authState.signOut,
    removeAccount: vi.fn(),
    refresh: vi.fn(),
  }),
}));

beforeEach(() => {
  navigation.pathname = "/";
  navigation.push.mockReset();
  authState.user = null;
  authState.loading = false;
  authState.signOut.mockReset();
});

describe("FloatingNavbar auth state", () => {
  it("shows a login link for anonymous visitors", () => {
    render(<FloatingNavbar />);
    expect(screen.getByRole("link", { name: "Log in" })).toHaveAttribute(
      "href",
      "/login",
    );
    expect(screen.getByRole("link", { name: "My Lists" })).toHaveAttribute(
      "href",
      "/lists",
    );
  });

  it("shows the admin link only for administrators", () => {
    authState.user = { email: "admin@example.com", is_admin: true };
    const { unmount } = render(<FloatingNavbar />);
    expect(screen.getByRole("link", { name: "Admin" })).toHaveAttribute(
      "href",
      "/admin",
    );
    unmount();
    authState.user = { email: "user@example.com", is_admin: false };
    render(<FloatingNavbar />);
    expect(screen.queryByRole("link", { name: "Admin" })).toBeNull();
  });

  it("shows account controls for signed-in users", async () => {
    const user = userEvent.setup();
    authState.user = { email: "user@example.com" };
    authState.signOut.mockResolvedValue(undefined);
    render(<FloatingNavbar />);

    expect(screen.getByRole("link", { name: "Account" })).toHaveAttribute(
      "href",
      "/account",
    );
    await user.click(screen.getByRole("button", { name: "Log out" }));

    await waitFor(() => expect(authState.signOut).toHaveBeenCalledTimes(1));
    expect(navigation.push).toHaveBeenCalledWith("/");
  });
});
