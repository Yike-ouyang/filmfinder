"use client";

import { LoaderCircle, RotateCcw, ShieldCheck } from "lucide-react";
import { useEffect, useId, useState } from "react";
import { Dialog } from "@/components/ui/Dialog";
import { errorMessage } from "@/lib/api";
import { fetchAdminUser } from "@/lib/admin";
import type { AdminUserDetail } from "@/types/admin";

interface AdminUserDetailDialogProps {
  open: boolean;
  userId: number | null;
  onClose: () => void;
}

function formatDate(value: string | null): string {
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

function DetailContent({
  userId,
  titleId,
}: {
  userId: number;
  titleId: string;
}) {
  const [user, setUser] = useState<AdminUserDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    fetchAdminUser(userId)
      .then((payload) => {
        if (!controller.signal.aborted) {
          setUser(payload);
          setError(null);
        }
      })
      .catch((caught) => {
        if (!controller.signal.aborted) {
          setError(errorMessage(caught));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [attempt, userId]);

  if (loading && !user) {
    return (
      <div
        role="status"
        className="flex min-h-[18rem] flex-col items-center justify-center gap-3 text-white/50"
      >
        <LoaderCircle aria-hidden="true" className="h-5 w-5 animate-spin" />
        <p className="text-xs">Loading user details...</p>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="flex min-h-[18rem] flex-col items-center justify-center gap-4 px-6 text-center">
        <h2 id={titleId} className="text-xl font-semibold text-white">
          User unavailable
        </h2>
        <p className="text-sm text-white/50">
          {error ?? "This user could not be loaded."}
        </p>
        <button
          type="button"
          onClick={() => {
            setLoading(true);
            setError(null);
            setAttempt((value) => value + 1);
          }}
          className="inline-flex h-9 items-center gap-2 rounded-full bg-white px-4 text-xs font-medium text-black"
        >
          <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
          Try again
        </button>
      </div>
    );
  }

  const items = [
    ["Joined", formatDate(user.created_at)],
    ["Email verified", user.email_verified ? "Yes" : "No"],
    ["Last login", formatDate(user.last_login_at)],
    ["Last activity", formatDate(user.last_activity_at)],
    ["Films rated", user.ratings_count.toLocaleString("en-US")],
    ["Films watched", user.watched_count.toLocaleString("en-US")],
    [
      "Recommendations generated",
      user.recommendation_count.toLocaleString("en-US"),
    ],
    ["Watchlist", user.interested_count.toLocaleString("en-US")],
    ["Do Not Watch", user.not_interested_count.toLocaleString("en-US")],
    ["Last rating", formatDate(user.last_rating_at)],
    [
      "Last recommendation",
      formatDate(user.last_recommendation_at),
    ],
    ["Last Letterboxd import", formatDate(user.last_import_at)],
  ];

  return (
    <div className="p-7 pt-16 sm:p-8 sm:pt-16">
      <p className="text-[10px] font-medium uppercase tracking-[0.3em] text-white/45">
        User details
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <h2
          id={titleId}
          className="break-all text-xl font-semibold text-white sm:text-2xl"
        >
          {user.email}
        </h2>
        {user.is_admin ? (
          <span className="inline-flex items-center gap-1 rounded-full border border-white/15 bg-white/5 px-2.5 py-1 text-[10px] text-white/65">
            <ShieldCheck aria-hidden="true" className="h-3 w-3" />
            Admin
          </span>
        ) : null}
      </div>
      <p className="mt-2 text-xs text-white/40">User #{user.id}</p>

      <dl className="mt-6 grid grid-cols-1 gap-x-6 gap-y-4 sm:grid-cols-2">
        {items.map(([label, value]) => (
          <div
            key={label}
            className="border-b border-white/[0.06] pb-3"
          >
            <dt className="text-[10px] uppercase tracking-[0.14em] text-white/40">
              {label}
            </dt>
            <dd className="mt-1 text-sm text-white/70">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-6 text-[11px] leading-relaxed text-white/35">
        This view is read-only. Passwords, sessions, tokens and IP addresses
        are never exposed.
      </p>
    </div>
  );
}

export function AdminUserDetailDialog({
  open,
  userId,
  onClose,
}: AdminUserDetailDialogProps) {
  const titleId = useId();
  return (
    <Dialog
      open={open}
      onClose={onClose}
      titleId={titleId}
      className="max-w-2xl"
    >
      {open && userId !== null ? (
        <DetailContent key={userId} userId={userId} titleId={titleId} />
      ) : null}
    </Dialog>
  );
}
