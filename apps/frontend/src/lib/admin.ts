import { ApiError } from "@/lib/api";
import { authenticatedRequest } from "@/lib/auth";
import type {
  AdminActivity,
  AdminActivityPoint,
  AdminOverview,
  AdminPeriod,
  AdminTotals,
  AdminUserDetail,
  AdminUserItem,
  AdminUsers,
  AdminWindow,
} from "@/types/admin";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value)
    ? value
    : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNullableString(value: unknown): string | null {
  return value === null || value === undefined ? null : asString(value);
}

function asBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function requireInteger(value: unknown): number {
  const result = asInteger(value);
  if (result === null) {
    throw new ApiError("invalid_response", 200);
  }
  return result;
}

function normalizeWindow(value: unknown): AdminWindow {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    registrations: requireInteger(value.registrations),
    active_users: requireInteger(value.active_users),
    logins: requireInteger(value.logins),
  };
}

function normalizeTotals(value: unknown): AdminTotals {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    accounts: requireInteger(value.accounts),
    verified_profiles: requireInteger(value.verified_profiles),
    ratings: requireInteger(value.ratings),
    rating_users: requireInteger(value.rating_users),
    logins: requireInteger(value.logins),
    login_users: requireInteger(value.login_users),
    recommendation_batches: requireInteger(
      value.recommendation_batches,
    ),
    recommendation_users: requireInteger(value.recommendation_users),
    letterboxd_imports: requireInteger(value.letterboxd_imports),
    import_users: requireInteger(value.import_users),
    interested_films: requireInteger(value.interested_films),
    interested_users: requireInteger(value.interested_users),
    not_interested_films: requireInteger(value.not_interested_films),
    not_interested_users: requireInteger(
      value.not_interested_users,
    ),
  };
}

export function normalizeAdminOverview(value: unknown): AdminOverview {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  const generated = asString(value.generated_at);
  if (!generated) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    generated_at: generated,
    totals: normalizeTotals(value.totals),
    last_7_days: normalizeWindow(value.last_7_days),
    last_30_days: normalizeWindow(value.last_30_days),
    imports_tracking_since: asNullableString(
      value.imports_tracking_since,
    ),
  };
}

function normalizeActivityPoint(value: unknown): AdminActivityPoint {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  const date = asString(value.date);
  if (!date) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    date,
    registrations: requireInteger(value.registrations),
    active_users: requireInteger(value.active_users),
    recommendations: requireInteger(value.recommendations),
  };
}

export function normalizeAdminActivity(value: unknown): AdminActivity {
  if (!isRecord(value) || !Array.isArray(value.points)) {
    throw new ApiError("invalid_response", 200);
  }
  const period = value.period;
  if (period !== "7d" && period !== "30d" && period !== "all") {
    throw new ApiError("invalid_response", 200);
  }
  const start = asString(value.start_date);
  const end = asString(value.end_date);
  if (!start || !end) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    period,
    start_date: start,
    end_date: end,
    points: value.points.map(normalizeActivityPoint),
  };
}

function normalizeUser(value: unknown): AdminUserItem {
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  const email = asString(value.email);
  const createdAt = asString(value.created_at);
  const verified = asBoolean(value.email_verified);
  const admin = asBoolean(value.is_admin);
  if (!email || !createdAt || verified === null || admin === null) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    id: requireInteger(value.id),
    email,
    created_at: createdAt,
    email_verified: verified,
    is_admin: admin,
    last_login_at: asNullableString(value.last_login_at),
    ratings_count: requireInteger(value.ratings_count),
    recommendation_count: requireInteger(value.recommendation_count),
    interested_count: requireInteger(value.interested_count),
    not_interested_count: requireInteger(value.not_interested_count),
  };
}

export function normalizeAdminUsers(value: unknown): AdminUsers {
  if (!isRecord(value) || !Array.isArray(value.items)) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    items: value.items.map(normalizeUser),
    page: requireInteger(value.page),
    page_size: requireInteger(value.page_size),
    total: requireInteger(value.total),
    total_pages: requireInteger(value.total_pages),
  };
}

export function normalizeAdminUserDetail(
  value: unknown,
): AdminUserDetail {
  const base = normalizeUser(value);
  if (!isRecord(value)) {
    throw new ApiError("invalid_response", 200);
  }
  return {
    ...base,
    watched_count: requireInteger(value.watched_count),
    last_activity_at: asNullableString(value.last_activity_at),
    last_rating_at: asNullableString(value.last_rating_at),
    last_recommendation_at: asNullableString(
      value.last_recommendation_at,
    ),
    last_import_at: asNullableString(value.last_import_at),
  };
}

export async function fetchAdminOverview(): Promise<AdminOverview> {
  return normalizeAdminOverview(
    await authenticatedRequest("/api/v1/admin/overview"),
  );
}

export async function fetchAdminActivity(
  period: AdminPeriod,
): Promise<AdminActivity> {
  return normalizeAdminActivity(
    await authenticatedRequest(
      `/api/v1/admin/activity?period=${encodeURIComponent(period)}`,
    ),
  );
}

export async function fetchAdminUsers(options: {
  page: number;
  pageSize: number;
  search: string;
}): Promise<AdminUsers> {
  const query = new URLSearchParams({
    page: String(options.page),
    page_size: String(options.pageSize),
  });
  if (options.search.trim()) {
    query.set("search", options.search.trim());
  }
  return normalizeAdminUsers(
    await authenticatedRequest(
      `/api/v1/admin/users?${query.toString()}`,
    ),
  );
}

export async function fetchAdminUser(
  userId: number,
): Promise<AdminUserDetail> {
  return normalizeAdminUserDetail(
    await authenticatedRequest(`/api/v1/admin/users/${userId}`),
  );
}
