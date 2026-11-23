"use client";

import Link from "next/link";
import { ArrowRight, LoaderCircle, Sparkles } from "lucide-react";
import type { ProfileSummary } from "@/types/films";

interface ProfileProgressProps {
  summary: ProfileSummary | null;
  loading: boolean;
  generating: boolean;
  error: string | null;
  isAuthenticated: boolean;
  onGenerate: () => void;
}

function AuthPrompt() {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/30 px-5 py-4">
      <p className="text-sm font-medium text-white">
        Build your movie profile
      </p>
      <p className="mt-1 text-xs leading-relaxed text-white/50">
        Create an account or sign in to rate films and save your lists.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Link
          href="/signup?next=%2Ffilms"
          className="inline-flex h-9 items-center rounded-full bg-white px-4 text-xs font-medium text-black transition-opacity hover:opacity-90"
        >
          Create an account
        </Link>
        <Link
          href="/login?next=%2Ffilms"
          className="inline-flex h-9 items-center rounded-full border border-white/10 px-4 text-xs font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
        >
          Log in
        </Link>
      </div>
    </div>
  );
}

export function ProfileProgress({
  summary,
  loading,
  generating,
  error,
  isAuthenticated,
  onGenerate,
}: ProfileProgressProps) {
  if (loading && !summary) {
    return (
      <div className="flex items-center gap-2 text-xs text-white/45">
        <LoaderCircle
          aria-hidden="true"
          className="h-4 w-4 animate-spin"
        />
        Loading your profile...
      </div>
    );
  }
  if (!isAuthenticated) {
    return <AuthPrompt />;
  }
  if (!summary) {
    return null;
  }
  const enough = summary.ratings_count >= summary.recommended_min_ratings;
  return (
    <div className="rounded-2xl border border-white/10 bg-black/30 px-5 py-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="flex items-center gap-2 text-sm font-medium text-white">
            <Sparkles aria-hidden="true" className="h-4 w-4 text-white/45" />
            {enough
              ? "Great! You're ready to discover your recommendations."
              : `You've rated ${summary.ratings_count} ${
                  summary.ratings_count === 1 ? "movie" : "movies"
                }.`}
          </p>
          <p className="mt-1 text-xs leading-relaxed text-white/45">
            {enough
              ? "Your ratings are saved. Generate a fresh set of picks whenever you like."
              : `We recommend rating at least ${summary.recommended_min_ratings} for better results.`}
          </p>
        </div>
        <button
          type="button"
          onClick={onGenerate}
          disabled={!enough || generating}
          className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-full bg-white px-5 text-xs font-medium text-black transition-opacity hover:opacity-90 disabled:pointer-events-none disabled:opacity-30"
        >
          {generating ? (
            <LoaderCircle
              aria-hidden="true"
              className="h-4 w-4 animate-spin"
            />
          ) : (
            <Sparkles aria-hidden="true" className="h-3.5 w-3.5" />
          )}
          Generate My Recommendations
          {!generating ? (
            <ArrowRight aria-hidden="true" className="h-3.5 w-3.5" />
          ) : null}
        </button>
      </div>
      {error ? (
        <p className="mt-3 text-xs text-danger">{error}</p>
      ) : null}
    </div>
  );
}
