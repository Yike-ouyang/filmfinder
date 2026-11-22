"use client";

import Link from "next/link";
import {
  CalendarDays,
  ExternalLink,
  Film,
  ListPlus,
  ThumbsDown,
  Trash2,
} from "lucide-react";
import type { ReactNode } from "react";
import { RatingStars } from "@/components/films/RatingStars";
import { TmdbMedia } from "@/components/ui/TmdbMedia";
import { getTmdbMovieUrl } from "@/lib/tmdb";
import { cn } from "@/lib/cn";
import type { EditableFilmListStatus } from "@/lib/filmLists";
import type { FilmDetail } from "@/types/films";

interface FilmDetailBodyProps {
  film: FilmDetail;
  titleId?: string;
  isAuthenticated: boolean;
  authNextPath?: string;
  onRatingChange?: (rating: number) => void;
  onRatingDelete?: () => void;
  onListChange?: (status: EditableFilmListStatus) => void;
  onListRemove?: () => void;
  busy?: boolean;
  error?: string | null;
  extra?: ReactNode;
  footer?: ReactNode;
  readOnlyActions?: boolean;
}

function PosterFallback({ title, year }: { title: string; year: number | null }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 bg-[radial-gradient(120%_100%_at_30%_0%,#242424,#101010_55%,#0a0a0a)] p-4 text-center">
      <Film aria-hidden="true" className="h-7 w-7 text-white/25" />
      <p className="text-sm leading-tight text-white/65">{title}</p>
      {year ? <p className="text-xs text-white/40">{year}</p> : null}
    </div>
  );
}

function formatReleaseDate(film: FilmDetail): string {
  if (film.release_date) {
    const parsed = new Date(`${film.release_date}T00:00:00Z`);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toLocaleDateString("en-US", {
        year: "numeric",
        month: "long",
        day: "numeric",
        timeZone: "UTC",
      });
    }
  }
  return film.year ? String(film.year) : "Release date unavailable";
}

function AuthPrompt({ nextPath }: { nextPath: string }) {
  const safeNext = encodeURIComponent(nextPath);
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-4">
      <p className="text-xs leading-relaxed text-white/55">
        Create a FilmFinder account or sign in to rate this film and save it
        to your lists.
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Link
          href={`/signup?next=${safeNext}`}
          className="inline-flex h-9 items-center rounded-full bg-white px-4 text-xs font-medium text-black transition-opacity hover:opacity-90"
        >
          Create an account
        </Link>
        <Link
          href={`/login?next=${safeNext}`}
          className="inline-flex h-9 items-center rounded-full border border-white/10 px-4 text-xs font-medium text-white/70 transition-colors hover:bg-white/5 hover:text-white"
        >
          Log in
        </Link>
      </div>
    </div>
  );
}

export function FilmDetailBody({
  film,
  titleId,
  isAuthenticated,
  authNextPath = "/films",
  onRatingChange,
  onRatingDelete,
  onListChange,
  onListRemove,
  busy = false,
  error = null,
  extra,
  footer,
  readOnlyActions = false,
}: FilmDetailBodyProps) {
  const tmdbUrl = getTmdbMovieUrl(film.tmdb_id);
  const seen =
    film.user_list_status === "seen" || film.user_rating !== null;
  const listStatus =
    film.user_list_status === "watchlist" ||
    film.user_list_status === "do_not_watch"
      ? film.user_list_status
      : null;

  return (
    <div className="p-7 pt-16 sm:p-8 sm:pt-16">
      {extra}
      <div className="mt-5 flex gap-5">
        <div className="h-48 w-[8rem] shrink-0 overflow-hidden rounded-xl border border-white/10 bg-surface">
          <TmdbMedia
            path={film.poster_path}
            variant="poster"
            alt={`${film.title} poster`}
            sizes="128px"
            fallback={
              <PosterFallback title={film.title} year={film.year} />
            }
          />
        </div>
        <div className="min-w-0 flex-1">
          <h2
            id={titleId}
            className="text-balance text-2xl font-semibold leading-tight tracking-[-0.03em] text-white"
          >
            {film.title}
          </h2>
          <p className="mt-2 flex items-center gap-1.5 text-sm text-white/55">
            <CalendarDays aria-hidden="true" className="h-3.5 w-3.5" />
            {formatReleaseDate(film)}
          </p>
          {film.genres.length > 0 ? (
            <ul className="mt-4 flex flex-wrap gap-1.5" aria-label="Genres">
              {film.genres.map((genre) => (
                <li
                  key={genre}
                  className="rounded-full border border-white/10 px-2.5 py-1 text-[10px] text-white/55"
                >
                  {genre}
                </li>
              ))}
            </ul>
          ) : null}
          <div className="mt-5 space-y-1">
            <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-white/45">
              Average rating
            </p>
            {film.ratings_count > 0 ? (
              <div className="flex items-center gap-2">
                <RatingStars
                  value={film.rating_mean}
                  readOnly
                  size="sm"
                  label="Average rating"
                />
                <span className="text-xs text-white/55">
                  {film.rating_mean.toFixed(1)}/10
                  <span className="ml-1 text-white/30">
                    ({film.ratings_count.toLocaleString("en-US")})
                  </span>
                </span>
              </div>
            ) : (
              <p className="text-xs text-white/40">
                Average rating unavailable
              </p>
            )}
          </div>
        </div>
      </div>

      <div className="mt-6 border-t border-white/10 pt-5">
        <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-white/45">
          Summary
        </p>
        <p className="mt-2 text-sm leading-relaxed text-white/60">
          {film.summary?.trim()
            ? film.summary
            : "No summary is available for this film."}
        </p>
      </div>

      {error ? (
        <p className="mt-4 rounded-xl border border-[#e7a6a6]/20 bg-[#e7a6a6]/[0.06] px-3 py-2 text-xs text-danger">
          {error}
        </p>
      ) : null}

      {readOnlyActions ? null : (
      <div className="mt-6 border-t border-white/10 pt-5">
        {!isAuthenticated ? (
          <AuthPrompt nextPath={authNextPath} />
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-white/45">
                  Your rating
                </p>
                <div className="mt-1.5 flex items-center gap-3">
                  <RatingStars
                    value={film.user_rating}
                    onChange={onRatingChange}
                    disabled={busy || !onRatingChange}
                    size="lg"
                    label="Your rating"
                  />
                  {film.user_rating !== null ? (
                    <span className="text-xs text-white/55">
                      {film.user_rating}/10
                    </span>
                  ) : (
                    <span className="text-xs text-white/35">Not rated</span>
                  )}
                </div>
              </div>
              {film.user_rating !== null && onRatingDelete ? (
                <button
                  type="button"
                  onClick={onRatingDelete}
                  disabled={busy}
                  className="inline-flex h-9 items-center gap-1.5 rounded-full border border-white/10 px-3 text-[11px] font-medium text-white/55 transition-colors hover:bg-white/5 hover:text-white disabled:opacity-40"
                >
                  <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
                  Remove rating
                </button>
              ) : null}
            </div>

            {seen && film.user_rating === null ? (
              <p className="mt-4 inline-flex items-center gap-2 rounded-full border border-white/10 px-3 py-1.5 text-[11px] text-white/55">
                <ListPlus aria-hidden="true" className="h-3.5 w-3.5" />
                Marked as seen
              </p>
            ) : seen ? null : (
              <div className="mt-5 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => onListChange?.("watchlist")}
                  disabled={busy || !onListChange}
                  className={cn(
                    "inline-flex h-9 items-center gap-1.5 rounded-full border px-4 text-xs font-medium transition-colors disabled:opacity-40",
                    listStatus === "watchlist"
                      ? "border-white bg-white text-black"
                      : "border-white/15 text-white/70 hover:bg-white/5 hover:text-white",
                  )}
                >
                  <ListPlus aria-hidden="true" className="h-3.5 w-3.5" />
                  {listStatus === "watchlist"
                    ? "In Watchlist"
                    : "Add to Watchlist"}
                </button>
                <button
                  type="button"
                  onClick={() => onListChange?.("do_not_watch")}
                  disabled={busy || !onListChange}
                  className={cn(
                    "inline-flex h-9 items-center gap-1.5 rounded-full border px-4 text-xs font-medium transition-colors disabled:opacity-40",
                    listStatus === "do_not_watch"
                      ? "border-[#e7a6a6]/40 bg-[#e7a6a6]/10 text-[#e7a6a6]"
                      : "border-white/15 text-white/70 hover:bg-white/5 hover:text-white",
                  )}
                >
                  <ThumbsDown aria-hidden="true" className="h-3.5 w-3.5" />
                  Do Not Watch
                </button>
                {listStatus ? (
                  <button
                    type="button"
                    onClick={onListRemove}
                    disabled={busy}
                    className="inline-flex h-9 items-center rounded-full px-3 text-xs font-medium text-white/35 transition-colors hover:text-white/70 disabled:opacity-40"
                  >
                    Remove from list
                  </button>
                ) : null}
              </div>
            )}
          </>
        )}
      </div>
      )}

      {footer}

      <div className="mt-6 border-t border-white/10 pt-5">
        {tmdbUrl ? (
          <a
            href={tmdbUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-10 items-center gap-2 rounded-full border border-white/15 px-4 text-xs font-medium text-white/70 transition-colors hover:border-white/30 hover:bg-white/5 hover:text-white"
          >
            View on TMDB
            <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
          </a>
        ) : (
          <p className="text-xs text-white/45">
            A TMDB page is not available for this film.
          </p>
        )}
      </div>
    </div>
  );
}
