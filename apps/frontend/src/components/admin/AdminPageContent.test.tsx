import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AdminPageContent } from "@/components/admin/AdminPageContent";

const auth = vi.hoisted(() => ({
  user: null as
    | {
        id: number;
        email: string;
        email_verified: boolean;
        is_admin: boolean;
        created_at: string;
      }
    | null,
}));

vi.mock("@/components/auth/AuthProvider", () => ({
  useAuth: () => ({
    user: auth.user,
    loading: false,
    signIn: vi.fn(),
    signOut: vi.fn(),
    removeAccount: vi.fn(),
    refresh: vi.fn(),
  }),
}));

const OVERVIEW = {
  generated_at: "2026-11-22T12:00:00Z",
  totals: {
    accounts: 3,
    verified_profiles: 2,
    ratings: 12,
    rating_users: 2,
    logins: 18,
    login_users: 3,
    recommendation_batches: 5,
    recommendation_users: 2,
    letterboxd_imports: 4,
    import_users: 2,
    interested_films: 7,
    interested_users: 2,
    not_interested_films: 3,
    not_interested_users: 1,
  },
  last_7_days: { registrations: 1, active_users: 2, logins: 6 },
  last_30_days: { registrations: 2, active_users: 3, logins: 14 },
  imports_tracking_since: "2026-11-20T10:00:00Z",
};

const ACTIVITY = {
  period: "7d",
  start_date: "2026-11-16",
  end_date: "2026-11-22",
  points: Array.from({ length: 7 }, (_, index) => ({
    date: `2026-11-${String(16 + index).padStart(2, "0")}`,
    registrations: index === 6 ? 1 : 0,
    active_users: index % 2,
    recommendations: index === 5 ? 1 : 0,
  })),
};

const USERS = {
  items: [
    {
      id: 1,
      email: "admin@example.com",
      created_at: "2026-11-20T10:00:00Z",
      email_verified: true,
      is_admin: true,
      last_login_at: "2026-11-22T09:00:00Z",
      ratings_count: 0,
      recommendation_count: 0,
      interested_count: 0,
      not_interested_count: 0,
    },
    {
      id: 2,
      email: "user1@example.com",
      created_at: "2026-11-12T10:00:00Z",
      email_verified: true,
      is_admin: false,
      last_login_at: "2026-11-20T09:00:00Z",
      ratings_count: 2,
      recommendation_count: 1,
      interested_count: 1,
      not_interested_count: 1,
    },
  ],
  page: 1,
  page_size: 50,
  total: 2,
  total_pages: 1,
};

const DETAIL = {
  ...USERS.items[1],
  watched_count: 2,
  last_activity_at: "2026-11-20T09:00:00Z",
  last_rating_at: "2026-11-19T09:00:00Z",
  last_recommendation_at: "2026-11-20T09:00:00Z",
  last_import_at: null,
};

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function installFetch() {
  const fetchMock = vi.fn((url: string) => {
    if (url.endsWith("/api/v1/admin/overview")) {
      return Promise.resolve(jsonResponse(OVERVIEW));
    }
    if (url.includes("/api/v1/admin/activity")) {
      return Promise.resolve(jsonResponse(ACTIVITY));
    }
    if (url.endsWith("/api/v1/admin/users/2")) {
      return Promise.resolve(jsonResponse(DETAIL));
    }
    if (url.includes("/api/v1/admin/users")) {
      return Promise.resolve(jsonResponse(USERS));
    }
    return Promise.reject(new Error(`Unexpected fetch: ${url}`));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  auth.user = {
    id: 1,
    email: "admin@example.com",
    email_verified: true,
    is_admin: true,
    created_at: "2026-11-20T10:00:00Z",
  };
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("AdminPageContent", () => {
  it("renders overview cards, SVG charts and the user table", async () => {
    installFetch();
    render(<AdminPageContent />);

    expect(await screen.findByText("Total accounts")).toBeVisible();
    expect(screen.getByText("Recommendation generations")).toBeVisible();
    expect(
      screen.getByRole("img", { name: /Registrations per day/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Active users per day/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Recommendations per day/ }),
    ).toBeInTheDocument();
    expect(await screen.findByText("user1@example.com")).toBeVisible();
    expect(screen.getByText(/Tracked since/)).toBeVisible();
  });

  it("changes the activity period and opens a read-only user detail", async () => {
    const fetchMock = installFetch();
    const user = userEvent.setup();
    render(<AdminPageContent />);
    await screen.findByText("user1@example.com");

    await user.click(screen.getByRole("button", { name: "30 days" }));
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).includes("period=30d"),
        ),
      ).toBe(true),
    );

    await user.click(
      screen.getByRole("button", { name: "View user1@example.com" }),
    );
    const dialog = await screen.findByRole("dialog");
    expect(
      await screen.findByRole("heading", {
        name: "user1@example.com",
      }),
    ).toBeVisible();
    expect(screen.getByText("User #2")).toBeVisible();
    expect(dialog).toBeVisible();
    expect(screen.queryByText(/password hash/i)).not.toBeInTheDocument();
  });

  it("denies ordinary accounts and prompts anonymous visitors", async () => {
    auth.user = {
      id: 2,
      email: "user@example.com",
      email_verified: true,
      is_admin: false,
      created_at: "2026-11-20T10:00:00Z",
    };
    installFetch();
    const { unmount } = render(<AdminPageContent />);
    expect(
      await screen.findByText("Administrator access required"),
    ).toBeVisible();
    unmount();

    auth.user = null;
    render(<AdminPageContent />);
    expect(await screen.findByText("Admin dashboard")).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Log in" }),
    ).toHaveAttribute("href", "/login?next=%2Fadmin");
  });
});
