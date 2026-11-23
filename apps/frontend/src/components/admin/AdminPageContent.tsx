"use client";

import Link from "next/link";
import {
  Activity,
  BadgeCheck,
  ChevronLeft,
  ChevronRight,
  Eye,
  Film,
  LoaderCircle,
  LogIn,
  RefreshCw,
  Search,
  ShieldAlert,
  Sparkles,
  Star,
  ThumbsUp,
  UserPlus,
  Users,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/components/auth/AuthProvider";
import { ActivityBarChart } from "@/components/admin/ActivityBarChart";
import { AdminUserDetailDialog } from "@/components/admin/AdminUserDetailDialog";
import { errorMessage } from "@/lib/api";
import {
  fetchAdminActivity,
  fetchAdminOverview,
  fetchAdminUsers,
} from "@/lib/admin";
import type {
  AdminActivity,
  AdminOverview,
  AdminPeriod,
  AdminUsers,
} from "@/types/admin";

const PERIODS: { key: AdminPeriod; label: string }[] = [
  { key: "7d", label: "7 days" },
  { key: "30d", label: "30 days" },
  { key: "all", label: "Since launch" },
];

function formatDate(value: string | null): string {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

function formatDateTime(value: string | null): string {
  if (!value) {
    return "Never";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
}

function CenteredPanel({
  icon,
  title,
  body,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  children?: React.ReactNode;
}) {
  return (
    <main
      id="main"
      className="relative flex min-h-svh items-center justify-center overflow-hidden bg-background px-5 py-24"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_80%_at_50%_0%,#1b1b1b_0%,#0d0d0d_45%,#080808_80%)]"
      />
      <div className="relative z-10 w-full max-w-md rounded-3xl border border-white/10 bg-black/45 px-6 py-8 text-center shadow-[0_40px_120px_-30px_rgba(0,0,0,0.95)] backdrop-blur-xl">
        <span className="mx-auto grid h-12 w-12 place-items-center rounded-full border border-white/10 bg-white/[0.04] text-white/55">
          {icon}
        </span>
        <h1 className="mt-5 text-2xl font-semibold tracking-[-0.03em] text-white">
          {title}
        </h1>
        <p className="mt-3 text-sm leading-relaxed text-white/50">{body}</p>
        {children ? <div className="mt-6">{children}</div> : null}
      </div>
    </main>
  );
}

function StatCard({
  icon,
  label,
  value,
  detail,
  note,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail?: string;
  note?: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-4">
      <div className="flex items-center gap-2 text-white/45">
        {icon}
        <p className="text-[10px] font-medium uppercase tracking-[0.14em]">
          {label}
        </p>
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight text-white">
        {value}
      </p>
      {detail ? (
        <p className="mt-1 text-[11px] text-white/45">{detail}</p>
      ) : null}
      {note ? (
        <p className="mt-1 text-[10px] leading-relaxed text-white/30">
          {note}
        </p>
      ) : null}
    </div>
  );
}

function AccessDenied() {
  return (
    <CenteredPanel
      icon={<ShieldAlert aria-hidden="true" className="h-5 w-5" />}
      title="Administrator access required"
      body="This dashboard is restricted to FilmFinder administrators. Your account does not have permission to view it."
    >
      <Link
        href="/"
        className="inline-flex h-10 items-center justify-center rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-90"
      >
        Back to FilmFinder
      </Link>
    </CenteredPanel>
  );
}

function LoginPrompt() {
  return (
    <CenteredPanel
      icon={<LogIn aria-hidden="true" className="h-5 w-5" />}
      title="Admin dashboard"
      body="Sign in with an administrator account to view private usage statistics."
    >
      <Link
        href="/login?next=%2Fadmin"
        className="inline-flex h-10 items-center justify-center rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-90"
      >
        Log in
      </Link>
    </CenteredPanel>
  );
}

export function AdminPageContent() {
  const { user, loading: authLoading } = useAuth();
  const [overview, setOverview] = useState<AdminOverview | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [overviewAttempt, setOverviewAttempt] = useState(0);
  const [activity, setActivity] = useState<AdminActivity | null>(null);
  const [activityLoading, setActivityLoading] = useState(true);
  const [activityError, setActivityError] = useState<string | null>(null);
  const [activityAttempt, setActivityAttempt] = useState(0);
  const [period, setPeriod] = useState<AdminPeriod>("7d");
  const [users, setUsers] = useState<AdminUsers | null>(null);
  const [usersLoading, setUsersLoading] = useState(true);
  const [usersError, setUsersError] = useState<string | null>(null);
  const [usersAttempt, setUsersAttempt] = useState(0);
  const [page, setPage] = useState(1);
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [selectedUserId, setSelectedUserId] = useState<number | null>(
    null,
  );

  const isAdmin = Boolean(user?.is_admin);

  useEffect(() => {
    if (!isAdmin) {
      return;
    }
    let active = true;
    fetchAdminOverview()
      .then((payload) => {
        if (active) {
          setOverview(payload);
          setOverviewError(null);
        }
      })
      .catch((error) => {
        if (active) {
          setOverviewError(errorMessage(error));
        }
      })
      .finally(() => {
        if (active) {
          setOverviewLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [isAdmin, overviewAttempt]);

  useEffect(() => {
    if (!isAdmin) {
      return;
    }
    let active = true;
    const timer = window.setTimeout(() => {
      fetchAdminActivity(period)
        .then((payload) => {
          if (active) {
            setActivity(payload);
            setActivityError(null);
          }
        })
        .catch((error) => {
          if (active) {
            setActivityError(errorMessage(error));
          }
        })
        .finally(() => {
          if (active) {
            setActivityLoading(false);
          }
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [activityAttempt, isAdmin, period]);

  useEffect(() => {
    if (!isAdmin) {
      return;
    }
    let active = true;
    const timer = window.setTimeout(() => {
      fetchAdminUsers({ page, pageSize: 50, search })
        .then((payload) => {
          if (active) {
            setUsers(payload);
            setUsersError(null);
          }
        })
        .catch((error) => {
          if (active) {
            setUsersError(errorMessage(error));
          }
        })
        .finally(() => {
          if (active) {
            setUsersLoading(false);
          }
        });
    }, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [isAdmin, page, search, usersAttempt]);

  useEffect(() => {
    if (searchInput === search) {
      return;
    }
    const timer = window.setTimeout(() => {
      setPage(1);
      setUsersLoading(true);
      setSearch(searchInput.trim());
    }, 300);
    return () => window.clearTimeout(timer);
  }, [search, searchInput]);

  const refresh = useCallback(() => {
    setOverviewLoading(true);
    setActivityLoading(true);
    setUsersLoading(true);
    setOverviewAttempt((value) => value + 1);
    setActivityAttempt((value) => value + 1);
    setUsersAttempt((value) => value + 1);
  }, []);

  if (authLoading) {
    return (
      <main
        id="main"
        className="flex min-h-svh items-center justify-center bg-background"
      >
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin text-white/50"
        />
      </main>
    );
  }

  if (!user) {
    return <LoginPrompt />;
  }

  if (!user.is_admin) {
    return <AccessDenied />;
  }

  return (
    <main
      id="main"
      className="relative min-h-svh overflow-hidden bg-background px-5 pb-20 pt-28 sm:px-8"
    >
      <div
        aria-hidden="true"
        className="absolute inset-0 bg-[radial-gradient(100%_70%_at_50%_0%,#1a1a1a_0%,#0d0d0d_46%,#080808_82%)]"
      />
      <div className="relative z-10 mx-auto w-full max-w-7xl">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-[0.32em] text-white/45">
              Private dashboard
            </p>
            <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em] text-white sm:text-5xl">
              Administration
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-white/50">
              Account growth, activity and catalogue engagement. All timestamps
              are UTC. Sessions, passwords and raw IP addresses are never
              exposed.
            </p>
          </div>
          <button
            type="button"
            onClick={refresh}
            className="inline-flex h-10 items-center gap-2 rounded-full border border-white/10 px-4 text-xs font-medium text-white/60 transition-colors hover:bg-white/5 hover:text-white"
          >
            <RefreshCw
              aria-hidden="true"
              className="h-3.5 w-3.5"
            />
            Refresh
          </button>
        </div>

        {overviewError ? (
          <div className="mt-6 rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-3 text-xs text-danger">
            {overviewError}
          </div>
        ) : null}

        <section aria-labelledby="overview-title" className="mt-8">
          <h2 id="overview-title" className="sr-only">
            Overview
          </h2>
          {overviewLoading && !overview ? (
            <div className="flex items-center gap-2 text-xs text-white/45">
              <LoaderCircle
                aria-hidden="true"
                className="h-4 w-4 animate-spin"
              />
              Loading overview...
            </div>
          ) : overview ? (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              <StatCard
                icon={<Users aria-hidden="true" className="h-4 w-4" />}
                label="Total accounts"
                value={overview.totals.accounts.toLocaleString("en-US")}
                detail={`${overview.totals.verified_profiles.toLocaleString("en-US")} verified`}
              />
              <StatCard
                icon={<UserPlus aria-hidden="true" className="h-4 w-4" />}
                label="Registrations"
                value={`${overview.last_7_days.registrations.toLocaleString("en-US")} / ${overview.last_30_days.registrations.toLocaleString("en-US")}`}
                detail="Last 7 days / last 30 days"
              />
              <StatCard
                icon={<Activity aria-hidden="true" className="h-4 w-4" />}
                label="Active users"
                value={`${overview.last_7_days.active_users.toLocaleString("en-US")} / ${overview.last_30_days.active_users.toLocaleString("en-US")}`}
                detail="Last 7 days / last 30 days"
                note="Users with at least one successful login in the period."
              />
              <StatCard
                icon={<LogIn aria-hidden="true" className="h-4 w-4" />}
                label="Successful logins"
                value={overview.totals.logins.toLocaleString("en-US")}
                detail={`${overview.totals.login_users.toLocaleString("en-US")} distinct users`}
              />
              <StatCard
                icon={<Star aria-hidden="true" className="h-4 w-4" />}
                label="Films rated"
                value={overview.totals.ratings.toLocaleString("en-US")}
                detail={`${overview.totals.rating_users.toLocaleString("en-US")} users`}
              />
              <StatCard
                icon={<Sparkles aria-hidden="true" className="h-4 w-4" />}
                label="Recommendation generations"
                value={overview.totals.recommendation_batches.toLocaleString("en-US")}
                detail={`${overview.totals.recommendation_users.toLocaleString("en-US")} users`}
                note="Ready recommendation batches only."
              />
              <StatCard
                icon={<Film aria-hidden="true" className="h-4 w-4" />}
                label="Letterboxd imports"
                value={overview.totals.letterboxd_imports.toLocaleString("en-US")}
                detail={`${overview.totals.import_users.toLocaleString("en-US")} signed-in users`}
                note={
                  overview.imports_tracking_since
                    ? `Tracked since ${formatDate(overview.imports_tracking_since)}. Earlier imports are not counted.`
                    : "Tracking starts with the first recorded import."
                }
              />
              <StatCard
                icon={<Eye aria-hidden="true" className="h-4 w-4" />}
                label="Watchlist / Do Not Watch"
                value={`${overview.totals.interested_films.toLocaleString("en-US")} / ${overview.totals.not_interested_films.toLocaleString("en-US")}`}
                detail={`${overview.totals.interested_users.toLocaleString("en-US")} / ${overview.totals.not_interested_users.toLocaleString("en-US")} users`}
              />
            </div>
          ) : null}
        </section>

        <section aria-labelledby="activity-title" className="mt-10">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2
                id="activity-title"
                className="text-lg font-semibold tracking-[-0.02em] text-white"
              >
                Activity
              </h2>
              <p className="mt-1 text-xs text-white/40">
                Daily counts in UTC. Active users are distinct users with a
                successful login that day.
              </p>
            </div>
            <div
              role="group"
              aria-label="Activity period"
              className="flex items-center gap-1 rounded-full border border-white/10 bg-black/35 p-1"
            >
              {PERIODS.map((option) => (
                <button
                  key={option.key}
                  type="button"
                  onClick={() => {
                    setActivityLoading(true);
                    setPeriod(option.key);
                  }}
                  aria-pressed={period === option.key}
                  className={
                    period === option.key
                      ? "rounded-full bg-white px-3 py-1.5 text-[11px] font-medium text-black"
                      : "rounded-full px-3 py-1.5 text-[11px] font-medium text-white/50 transition-colors hover:bg-white/5 hover:text-white"
                  }
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {activityError ? (
            <div className="mt-4 flex items-center justify-between gap-3 rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-3 text-xs text-danger">
              <span>{activityError}</span>
              <button
                type="button"
                onClick={() => {
                  setActivityLoading(true);
                  setActivityAttempt((value) => value + 1);
                }}
                className="shrink-0 text-white/70 underline underline-offset-2"
              >
                Retry
              </button>
            </div>
          ) : null}

          {activityLoading && !activity ? (
            <div className="mt-4 flex items-center gap-2 text-xs text-white/45">
              <LoaderCircle
                aria-hidden="true"
                className="h-4 w-4 animate-spin"
              />
              Loading activity...
            </div>
          ) : activity ? (
            <div className="mt-4 grid gap-4 lg:grid-cols-3">
              <ActivityBarChart
                title="Registrations per day"
                definition="Accounts created that day."
                points={activity.points}
                metric="registrations"
                color="#f5f5f5"
              />
              <ActivityBarChart
                title="Active users per day"
                definition="Distinct users with at least one successful login that day."
                points={activity.points}
                metric="active_users"
                color="#f5c518"
              />
              <ActivityBarChart
                title="Recommendations per day"
                definition="Ready recommendation batches completed that day."
                points={activity.points}
                metric="recommendations"
                color="#7fb8ff"
              />
            </div>
          ) : null}
        </section>

        <section aria-labelledby="users-title" className="mt-12">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2
                id="users-title"
                className="text-lg font-semibold tracking-[-0.02em] text-white"
              >
                Users
              </h2>
              <p className="mt-1 text-xs text-white/40">
                Read-only account overview. Counts reflect current saved data.
              </p>
            </div>
            {users ? (
              <p className="text-xs text-white/40">
                {users.total.toLocaleString("en-US")} accounts
              </p>
            ) : null}
          </div>

          <label className="relative mt-4 block max-w-md">
            <span className="sr-only">Search users by email</span>
            <Search
              aria-hidden="true"
              className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-white/35"
            />
            <input
              type="search"
              value={searchInput}
              onChange={(event) => setSearchInput(event.target.value)}
              placeholder="Search users by email"
              className="h-10 w-full rounded-full border border-white/10 bg-black/35 pl-11 pr-4 text-xs text-white placeholder:text-white/35 focus:border-white/30 focus:outline-none"
            />
          </label>

          {usersError ? (
            <div className="mt-4 flex items-center justify-between gap-3 rounded-2xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-4 py-3 text-xs text-danger">
              <span>{usersError}</span>
              <button
                type="button"
                onClick={() => {
                  setUsersLoading(true);
                  setUsersAttempt((value) => value + 1);
                }}
                className="shrink-0 text-white/70 underline underline-offset-2"
              >
                Retry
              </button>
            </div>
          ) : null}

          {usersLoading && !users ? (
            <div className="mt-4 flex items-center gap-2 text-xs text-white/45">
              <LoaderCircle
                aria-hidden="true"
                className="h-4 w-4 animate-spin"
              />
              Loading users...
            </div>
          ) : users && users.items.length > 0 ? (
            <>
              <div className="mt-4 overflow-x-auto rounded-2xl border border-white/10">
                <table className="min-w-full text-left text-xs">
                  <thead className="bg-white/[0.03] text-[10px] uppercase tracking-[0.12em] text-white/40">
                    <tr>
                      <th className="px-4 py-3 font-medium">Email</th>
                      <th className="px-4 py-3 font-medium">Joined</th>
                      <th className="px-4 py-3 font-medium">Verified</th>
                      <th className="px-4 py-3 font-medium">Last login</th>
                      <th className="px-4 py-3 text-right font-medium">Ratings</th>
                      <th className="px-4 py-3 text-right font-medium">Recs</th>
                      <th className="px-4 py-3 text-right font-medium">Watchlist</th>
                      <th className="px-4 py-3 text-right font-medium">Do Not Watch</th>
                      <th className="px-4 py-3" />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.06] text-white/60">
                    {users.items.map((item) => (
                      <tr
                        key={item.id}
                        className="transition-colors hover:bg-white/[0.025]"
                      >
                        <td className="px-4 py-3">
                          <span className="break-all text-white/75">
                            {item.email}
                          </span>
                          {item.is_admin ? (
                            <span className="ml-2 rounded-full border border-white/15 px-2 py-0.5 text-[9px] uppercase tracking-[0.1em] text-white/50">
                              Admin
                            </span>
                          ) : null}
                        </td>
                        <td className="px-4 py-3">
                          {formatDate(item.created_at)}
                        </td>
                        <td className="px-4 py-3">
                          {item.email_verified ? (
                            <span className="inline-flex items-center gap-1 text-white/60">
                              <BadgeCheck
                                aria-hidden="true"
                                className="h-3.5 w-3.5"
                              />
                              Yes
                            </span>
                          ) : (
                            "No"
                          )}
                        </td>
                        <td className="px-4 py-3">
                          {formatDateTime(item.last_login_at)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">
                          {item.ratings_count}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">
                          {item.recommendation_count}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">
                          {item.interested_count}
                        </td>
                        <td className="px-4 py-3 text-right font-mono">
                          {item.not_interested_count}
                        </td>
                        <td className="px-4 py-3 text-right">
                          <button
                            type="button"
                            onClick={() => setSelectedUserId(item.id)}
                            aria-label={`View ${item.email}`}
                            className="inline-flex h-8 items-center justify-center rounded-full border border-white/10 px-3 text-[10px] font-medium text-white/55 transition-colors hover:bg-white/5 hover:text-white"
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                <p className="text-xs text-white/40">
                  Page {users.page} of {users.total_pages}
                </p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setUsersLoading(true);
                      setPage((value) => Math.max(1, value - 1));
                    }}
                    disabled={users.page <= 1 || usersLoading}
                    className="inline-flex h-9 items-center gap-1 rounded-full border border-white/10 px-3 text-[11px] text-white/60 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-30"
                  >
                    <ChevronLeft aria-hidden="true" className="h-3.5 w-3.5" />
                    Previous
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setUsersLoading(true);
                      setPage((value) => value + 1);
                    }}
                    disabled={
                      users.page >= users.total_pages || usersLoading
                    }
                    className="inline-flex h-9 items-center gap-1 rounded-full border border-white/10 px-3 text-[11px] text-white/60 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-30"
                  >
                    Next
                    <ChevronRight aria-hidden="true" className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </>
          ) : users ? (
            <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.02] px-6 py-12 text-center">
              <ThumbsUp
                aria-hidden="true"
                className="mx-auto h-5 w-5 text-white/25"
              />
              <p className="mt-3 text-sm text-white/55">
                {search ? "No users match this search." : "No accounts yet."}
              </p>
            </div>
          ) : null}
        </section>
      </div>

      <AdminUserDetailDialog
        open={selectedUserId !== null}
        userId={selectedUserId}
        onClose={() => setSelectedUserId(null)}
      />
    </main>
  );
}
