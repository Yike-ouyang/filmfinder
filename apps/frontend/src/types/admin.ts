export type AdminPeriod = "7d" | "30d" | "all";

export interface AdminTotals {
  accounts: number;
  verified_profiles: number;
  ratings: number;
  rating_users: number;
  logins: number;
  login_users: number;
  recommendation_batches: number;
  recommendation_users: number;
  letterboxd_imports: number;
  import_users: number;
  interested_films: number;
  interested_users: number;
  not_interested_films: number;
  not_interested_users: number;
}

export interface AdminWindow {
  registrations: number;
  active_users: number;
  logins: number;
}

export interface AdminOverview {
  generated_at: string;
  totals: AdminTotals;
  last_7_days: AdminWindow;
  last_30_days: AdminWindow;
  imports_tracking_since: string | null;
}

export interface AdminActivityPoint {
  date: string;
  registrations: number;
  active_users: number;
  recommendations: number;
}

export interface AdminActivity {
  period: AdminPeriod;
  start_date: string;
  end_date: string;
  points: AdminActivityPoint[];
}

export interface AdminUserItem {
  id: number;
  email: string;
  created_at: string;
  email_verified: boolean;
  is_admin: boolean;
  last_login_at: string | null;
  ratings_count: number;
  recommendation_count: number;
  interested_count: number;
  not_interested_count: number;
}

export interface AdminUsers {
  items: AdminUserItem[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface AdminUserDetail extends AdminUserItem {
  watched_count: number;
  last_activity_at: string | null;
  last_rating_at: string | null;
  last_recommendation_at: string | null;
  last_import_at: string | null;
}
